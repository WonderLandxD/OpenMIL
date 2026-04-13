import os
import json
import csv
import argparse
import contextlib
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

from slide_encoder_models.model_registry import create_slide_encoder


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TileFeatDataset(Dataset):
    def __init__(self, data_dir: str, json_path: str, pfm_name: str, split: str):
        self.data_dir = data_dir
        self.pfm_name = pfm_name
        with open(json_path, "r") as f:
            data = json.load(f)
        self.items = data[split]
        self.num_bins = data.get("num_bins", 4)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        data_path = item["path"].replace("<PFM_NAME>", self.pfm_name)
        pth = torch.load(os.path.join(self.data_dir, data_path), map_location="cpu", weights_only=False)
        return {
            "feats": pth["feats"],
            "coords": pth["coords"],
            "patch_size_lv0": pth["patch_size_level0"],
            "survival_time": torch.tensor(item["survival_time"], dtype=torch.float32),
            "censorship": torch.tensor(item["censorship"], dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.long),  # discretized bin
        }


class NLLSurvLoss(nn.Module):
    """Negative log-likelihood survival loss (discrete-time)."""
    def __init__(self):
        super().__init__()

    def forward(self, hazards, labels, censorship):
        S = torch.cumprod(1 - hazards, dim=1)
        S_padded = torch.cat([torch.ones(S.shape[0], 1, device=S.device), S], dim=1)
        # uncensored: log(f(t)) = log(S(t-1)) + log(h(t))
        # censored:   log(S(t))
        uncensored_loss = -(torch.log(S_padded.gather(1, labels.unsqueeze(1)).clamp(min=1e-7))
                           + torch.log(hazards.gather(1, labels.unsqueeze(1)).clamp(min=1e-7)))
        censored_loss = -torch.log(S.gather(1, labels.unsqueeze(1)).clamp(min=1e-7))
        loss = (1 - censorship.unsqueeze(1)) * uncensored_loss + censorship.unsqueeze(1) * censored_loss
        return loss.mean()


class SlideEncoderForDownstreamSurv(nn.Module):
    def __init__(self, model_name: str, dim_in: int, dim_hidden: int, num_bins: int):
        super().__init__()
        if model_name in [
            "gigapath",
            "chief",
            "madeleine",
            "prism",
            "titan",
            "feather_uni_v1",
            "feather_uni_v2",
            "feather_conch_v1_5",
        ]:
            self.model = create_slide_encoder(model_name, num_classes=num_bins)
        else:
            self.model = create_slide_encoder(model_name, dim_in=dim_in, dim_hidden=dim_hidden, num_classes=num_bins)
        self.surv_loss = NLLSurvLoss()

    def forward(self, feats, coords=None, patch_size_lv0=None, survival_time=None, censorship=None, label=None):
        logits = self.model({"feats": feats, "coords": coords, "patch_size_lv0": patch_size_lv0})
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -S.sum(dim=1)  # higher risk = lower survival
        loss = self.surv_loss(hazards, label, censorship)
        return {"loss": loss, "risk": risk, "survival_time": survival_time, "censorship": censorship}


def compute_metrics(risk: torch.Tensor, survival_time: torch.Tensor, censorship: torch.Tensor):
    risk_np = risk.numpy()
    time_np = survival_time.numpy()
    event_np = (1 - censorship).numpy().astype(bool)

    # C-index
    try:
        c_index = concordance_index_censored(event_np, time_np, risk_np)[0]
    except ValueError:
        c_index = float("nan")

    # Time-dependent AUC
    try:
        surv_arr = np.array(list(zip(event_np, time_np)), dtype=[("event", bool), ("time", float)])
        times = np.percentile(time_np[event_np], [25, 50, 75])
        times = times[(times > time_np.min()) & (times < time_np.max())]
        if len(times) > 0:
            auc_values, mean_auc = cumulative_dynamic_auc(surv_arr, surv_arr, risk_np, times)
        else:
            mean_auc = float("nan")
    except (ValueError, IndexError):
        mean_auc = float("nan")

    return {"c_index": float(c_index), "auc": float(mean_auc)}


def run_epoch(model, loader, device, optimizer=None, grad_accum_steps=1, dtype="fp32"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses = []
    all_risk = []
    all_time = []
    all_censor = []

    if is_train:
        optimizer.zero_grad()

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype)
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()

    with torch.set_grad_enabled(is_train):
        from tqdm import tqdm
        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Training" if is_train else "Validation"):
            feats = batch["feats"].to(device)
            coords = batch["coords"].to(device)
            patch_size_lv0 = batch["patch_size_lv0"].to(device)
            survival_time = batch["survival_time"].to(device)
            censorship = batch["censorship"].to(device)
            label = batch["label"].to(device)

            with autocast_ctx:
                out = model(feats, coords, patch_size_lv0, survival_time, censorship, label)
            loss = out["loss"]
            losses.append(loss.item())
            all_risk.append(out["risk"].detach().cpu())
            all_time.append(out["survival_time"].detach().cpu())
            all_censor.append(out["censorship"].detach().cpu())

            if is_train:
                (loss / grad_accum_steps).backward()
                if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

    risk = torch.cat(all_risk, dim=0)
    survival_time = torch.cat(all_time, dim=0)
    censorship = torch.cat(all_censor, dim=0)
    return float(np.mean(losses)), risk, survival_time, censorship


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MIL for survival prediction")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--pfm_name", type=str, default="pfm_name")
    parser.add_argument("--slide_name", type=str, default="slide_name")
    parser.add_argument("--job_dir", type=str, default="./results_surv")
    parser.add_argument("--dataset_name", type=str, default=None, help="Custom dataset name for output directory. If not set, uses json filename.")
    parser.add_argument("--seed", type=int, default=2077)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--best_metrics",
        type=str,
        default="c_index",
        choices=["c_index", "auc"],
    )
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    train_set = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, "train")
    val_set = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, "val")
    test_splits = ["test"]
    test_sets = {s: TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, s) for s in test_splits}

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loaders = {
        s: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        for s, ds in test_sets.items()
    }

    in_dim = train_set[0]["feats"].shape[1]
    num_bins = train_set.num_bins
    model = SlideEncoderForDownstreamSurv(args.slide_name, in_dim, 512, num_bins).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_name = args.dataset_name if args.dataset_name else os.path.basename(args.json_path).split(".")[0]
    output_dir = os.path.join(args.job_dir, dataset_name, args.pfm_name, args.slide_name, str(args.seed), "benchmark")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    val_csv = os.path.join(output_dir, "val_metrics.csv")
    with open(val_csv, "w") as f:
        csv.writer(f).writerow(["epoch", "loss", "c_index", "auc"])

    best_score = -float("inf")
    best_epoch = -1
    no_improve = 0
    min_delta = 1e-3

    for epoch in range(args.epochs):
        train_loss, _, _, _ = run_epoch(
            model, train_loader, device, optimizer=optimizer, grad_accum_steps=max(1, args.grad_accum_steps), dtype=args.dtype
        )
        val_loss, val_risk, val_time, val_censor = run_epoch(model, val_loader, device, dtype=args.dtype)
        val_metrics = compute_metrics(val_risk, val_time, val_censor)

        with open(val_csv, "a") as f:
            csv.writer(f).writerow(
                [epoch + 1, val_loss, val_metrics["c_index"], val_metrics["auc"]]
            )

        print(
            f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_{args.best_metrics}={val_metrics[args.best_metrics]:.4f}"
        )

        score = val_metrics[args.best_metrics]
        if score > (best_score + min_delta):
            best_score = score
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_{args.best_metrics}.pth"))
            with open(os.path.join(output_dir, "best_val_metrics.json"), "w") as f:
                json.dump({"epoch": best_epoch, **val_metrics}, f, indent=2)
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(torch.load(os.path.join(output_dir, f"best_{args.best_metrics}.pth"), map_location=device))
    model.eval()

    test_csv = os.path.join(output_dir, "test_metrics.csv")
    with open(test_csv, "w") as f:
        csv.writer(f).writerow(["split", "c_index", "auc"])

    all_test_metrics = {}
    for split_name, loader in test_loaders.items():
        _, test_risk, test_time, test_censor = run_epoch(model, loader, device, dtype=args.dtype)
        metrics = compute_metrics(test_risk, test_time, test_censor)
        all_test_metrics[split_name] = metrics
        with open(test_csv, "a") as f:
            csv.writer(f).writerow([split_name, metrics["c_index"], metrics["auc"]])
        print(
            f"[{split_name}] c_index={metrics['c_index']:.4f} auc={metrics['auc']:.4f}"
        )

    with open(os.path.join(output_dir, "all_test_metrics.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_metric": args.best_metrics, **all_test_metrics}, f, indent=2)