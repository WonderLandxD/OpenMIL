#!/usr/bin/env python3
"""KNN on slide-level embeddings from pretrained slide encoders (WSI / bag-of-patch features)."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from slide_encoder_models.model_registry import create_slide_encoder

try:
    import faiss
except ImportError:
    faiss = None

# Only HuggingFace / pretrained slide models (not abmil-style MIL trained from scratch here)
PRETRAINED_SLIDE_ENCODERS = (
    "titan",
    "prism",
    "gigapath",
    "madeleine",
    "chief",
    "care",
    "tangle_v2",
    "feather_uni_v1",
    "feather_uni_v2",
    "feather_conch_v1_5",
)


class TileFeatDataset(Dataset):
    def __init__(self, data_dir: str, json_path: str, pfm_name: str, split: str):
        self.data_dir = data_dir
        self.pfm_name = pfm_name
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = data[split]
        all_labels = sorted(
            {
                it["label"]
                for v in data.values()
                if isinstance(v, list)
                for it in v
                if isinstance(it, dict) and "label" in it
            }
        )
        self.label2idx = {label: i for i, label in enumerate(all_labels)}

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
            "labels": self.label2idx[item["label"]],
        }


def parse_args():
    p = argparse.ArgumentParser(description="KNN on pretrained slide-encoder embeddings (WSI)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--json-path", required=True)
    p.add_argument("--pfm-name", default="uni_v2")
    p.add_argument("--save-root", default="./knn_results")
    p.add_argument("--slide-encoder", required=True, choices=PRETRAINED_SLIDE_ENCODERS)
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--backend", default="auto", choices=["auto", "faiss", "sklearn"])
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--save-features", action="store_true", default=True)
    p.add_argument("--no-save-features", dest="save_features", action="store_false")
    p.add_argument("--reuse-features", action="store_true")
    return p.parse_args()


def build_save_dir(save_root, pfm_name: str, slide_encoder, k, metric, key: str):
    return Path(save_root) / pfm_name / key / slide_encoder / f"k{k}_{metric}"


def build_feature_dir(save_root, pfm_name: str, slide_encoder, key: str):
    return Path(save_root) / pfm_name / key / slide_encoder / "features"


def get_feature_paths(feature_dir):
    return {
        "train_features": feature_dir / "train_features.npy",
        "train_labels": feature_dir / "train_labels.npy",
        "test_features": feature_dir / "test_features.npy",
        "test_labels": feature_dir / "test_labels.npy",
    }


def save_results(save_dir, args, test_items, predictions, acc, bal_acc, macro_f1, report, idx2label):
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "json_path": args.json_path,
        "data_dir": args.data_dir,
        "pfm_name": args.pfm_name,
        "slide_encoder": args.slide_encoder,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "k": args.k,
        "metric": args.metric,
        "backend": args.backend,
        "dtype": args.dtype,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "num_test": len(test_items),
    }
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(save_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(save_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for item, pred_idx in zip(test_items, predictions):
            rec = {
                "path": item["path"],
                "label": item["label"],
                "prediction": idx2label[int(pred_idx)],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_feature_cache(feature_dir, train_x, train_y, test_x, test_y):
    feature_dir.mkdir(parents=True, exist_ok=True)
    paths = get_feature_paths(feature_dir)
    np.save(paths["train_features"], train_x)
    np.save(paths["train_labels"], train_y)
    np.save(paths["test_features"], test_x)
    np.save(paths["test_labels"], test_y)


def load_feature_cache(feature_dir):
    paths = get_feature_paths(feature_dir)
    if not all(x.exists() for x in paths.values()):
        return None
    return (
        np.load(paths["train_features"]),
        np.load(paths["train_labels"]),
        np.load(paths["test_features"]),
        np.load(paths["test_labels"]),
    )


def resolve_backend(backend):
    if backend == "sklearn":
        return "sklearn"
    if faiss is None:
        if backend == "faiss":
            raise ImportError("faiss is not installed")
        return "sklearn"
    return "faiss"


def get_faiss_gpu_id(device):
    if not device.startswith("cuda"):
        return None
    return int(device.split(":")[-1]) if ":" in device else 0


def vote_by_neighbors(indices, scores, train_label_ids, num_classes, metric):
    neighbor_label_ids = train_label_ids[indices]
    if metric == "cosine":
        weights = np.clip((scores + 1.0) * 0.5, 1e-12, None)
    else:
        weights = 1.0 / np.clip(scores, 1e-12, None)
    votes = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    rows = np.arange(indices.shape[0])
    for col in range(indices.shape[1]):
        np.add.at(votes, (rows, neighbor_label_ids[:, col]), weights[:, col])
    return votes.argmax(axis=1)


def run_faiss_knn(train_x, train_y, test_x, k, metric, device):
    train_x = np.ascontiguousarray(train_x.astype(np.float32))
    test_x = np.ascontiguousarray(test_x.astype(np.float32))
    classes, train_label_ids = np.unique(train_y, return_inverse=True)
    dim = train_x.shape[1]
    k = min(k, len(train_x))
    index = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
    gpu_id = get_faiss_gpu_id(device)
    if gpu_id is not None and hasattr(faiss, "StandardGpuResources"):
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    index.add(train_x)
    scores, indices = index.search(test_x, k)
    pred_ids = vote_by_neighbors(indices, scores, train_label_ids, len(classes), metric)
    return classes[pred_ids]


def run_sklearn_knn(train_x, train_y, test_x, k, metric):
    knn = KNeighborsClassifier(
        n_neighbors=min(k, len(train_x)),
        metric=metric,
        weights="distance",
        algorithm="brute" if metric == "cosine" else "auto",
    )
    knn.fit(train_x, train_y)
    return knn.predict(test_x)


def autocast_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return None


@torch.inference_mode()
def extract_slide_embeddings(model, loader, device, dtype_str):
    amp = autocast_dtype(dtype_str)
    use_amp = device.startswith("cuda") and amp is not None
    feats_list, labels_list = [], []
    for batch in tqdm(loader, desc="Slides", leave=False):
        feats = batch["feats"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        ps = batch["patch_size_lv0"].to(device, non_blocking=True)
        labels = batch["labels"]
        ctx = torch.autocast("cuda", dtype=amp) if use_amp else contextlib.nullcontext()
        with ctx:
            out = model({"feats": feats, "coords": coords, "patch_size_lv0": ps})
        out = out.float()
        if out.dim() == 1:
            out = out.unsqueeze(0)
        out = F.normalize(out, dim=-1)
        feats_list.append(out.cpu().numpy())
        labels_list.append(labels.numpy() if torch.is_tensor(labels) else labels)
    return np.concatenate(feats_list, axis=0), np.concatenate(labels_list, axis=0)


def main():
    args = parse_args()
    train_ds = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, args.train_split)
    test_ds = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, args.test_split)
    idx2label = {v: k for k, v in train_ds.label2idx.items()}

    backend = resolve_backend(args.backend)
    key = Path(args.json_path).stem
    save_dir = build_save_dir(args.save_root, args.pfm_name, args.slide_encoder, args.k, args.metric, key)
    feature_dir = build_feature_dir(args.save_root, args.pfm_name, args.slide_encoder, key)

    cached = load_feature_cache(feature_dir) if args.reuse_features else None
    if cached is not None:
        train_x, train_y, test_x, test_y = cached
        print(f"Loaded cached features: {feature_dir}")
    else:
        model = create_slide_encoder(args.slide_encoder, num_classes=0).to(args.device)
        model.eval()
        loader_kw = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        )
        train_loader = DataLoader(train_ds, **loader_kw)
        test_loader = DataLoader(test_ds, **loader_kw)
        print(f"slide_encoder={args.slide_encoder} train={len(train_ds)} test={len(test_ds)} device={args.device}")
        train_x, train_y = extract_slide_embeddings(model, train_loader, args.device, args.dtype)
        test_x, test_y = extract_slide_embeddings(model, test_loader, args.device, args.dtype)
        if args.save_features:
            save_feature_cache(feature_dir, train_x, train_y, test_x, test_y)
            print(f"Saved features: {feature_dir}")

    args.backend = backend
    print(f"KNN backend: {backend}")
    if backend == "faiss":
        pred = run_faiss_knn(train_x, train_y, test_x, args.k, args.metric, args.device)
    else:
        pred = run_sklearn_knn(train_x, train_y, test_x, args.k, args.metric)

    acc = accuracy_score(test_y, pred)
    bal_acc = balanced_accuracy_score(test_y, pred)
    macro_f1 = f1_score(test_y, pred, average="macro")
    n_cls = len(train_ds.label2idx)
    target_names = [str(idx2label[i]) for i in range(n_cls)]
    report = classification_report(test_y, pred, labels=list(range(n_cls)), target_names=target_names, digits=4, zero_division=0)
    save_results(save_dir, args, test_ds.items, pred, acc, bal_acc, macro_f1, report, idx2label)

    print(f"Accuracy: {acc:.4f}  Balanced-Acc: {bal_acc:.4f}  Macro-F1: {macro_f1:.4f}")
    print(f"Saved: {save_dir}")
    print(report)


if __name__ == "__main__":
    main()
