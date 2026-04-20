#!/usr/bin/env python3
"""Linear probe on slide embeddings from pretrained slide encoders (WSI)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from knn_benchmark import (
    PRETRAINED_SLIDE_ENCODERS,
    TileFeatDataset,
    build_feature_dir,
    extract_slide_embeddings,
    load_feature_cache,
    save_feature_cache,
)
from slide_encoder_models.model_registry import create_slide_encoder


def parse_args():
    p = argparse.ArgumentParser(description="Linear probe on pretrained slide-encoder embeddings (WSI)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--json-path", required=True)
    p.add_argument("--pfm-name", default="uni_v2")
    p.add_argument("--save-root", default="./linear_probe_results")
    p.add_argument("--slide-encoder", required=True, choices=PRETRAINED_SLIDE_ENCODERS)
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--save-features", action="store_true", default=True)
    p.add_argument("--no-save-features", dest="save_features", action="store_false")
    p.add_argument("--reuse-features", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_save_dir(save_root: str, pfm_name: str, slide_encoder: str, num_folds: int, patience: int, key: str) -> Path:
    return Path(save_root) / pfm_name / key / slide_encoder / f"lp_f{num_folds}_pat{patience}"


def save_results(save_dir, args, test_items, fold_predictions, fold_metrics, summary_metrics, report, idx2label):
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "json_path": args.json_path,
        "data_dir": args.data_dir,
        "pfm_name": args.pfm_name,
        "slide_encoder": args.slide_encoder,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "dtype": args.dtype,
        "epochs": args.epochs,
        "patience": args.patience,
        "num_folds": args.num_folds,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "seed": args.seed,
        "num_test": len(test_items),
        "fold_metrics": fold_metrics,
        "summary": summary_metrics,
    }
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(save_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    for fold_idx, predictions in enumerate(fold_predictions):
        with open(save_dir / f"predictions_fold{fold_idx}.jsonl", "w", encoding="utf-8") as f:
            for sample, pred_id in zip(test_items, predictions):
                rec = {
                    "path": sample["path"],
                    "label": sample["label"],
                    "prediction": idx2label[int(pred_id)],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_tensor_loader(features, label_ids, batch_size, shuffle, device):
    ds = TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(label_ids).long())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )


@torch.inference_mode()
def _predict_ids(linear, features, device):
    linear.eval()
    x = torch.from_numpy(features).float().to(device, non_blocking=True)
    logits = linear(x)
    return logits.argmax(dim=1).cpu().numpy()


def train_linear_probe(train_features, train_label_ids, val_features, val_label_ids, in_dim, num_classes, args):
    linear = nn.Linear(in_dim, num_classes).to(args.device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = _make_tensor_loader(
        train_features,
        train_label_ids,
        batch_size=min(args.train_batch_size, len(train_features)),
        shuffle=True,
        device=args.device,
    )

    best_state = None
    best_val_bal_acc = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(args.epochs):
        linear.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for feats, labels in train_loader:
            feats = feats.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            logits = linear(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            total += feats.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        val_pred_ids = _predict_ids(linear, val_features, args.device)
        val_bal_acc = balanced_accuracy_score(val_label_ids, val_pred_ids)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} acc={acc:.4f} val_bal_acc={val_bal_acc:.4f}"
        )

        if val_bal_acc > best_val_bal_acc + 1e-12:
            best_val_bal_acc = float(val_bal_acc)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in linear.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best_epoch={best_epoch}, best_val_bal_acc={best_val_bal_acc:.4f})"
                )
                break

    if best_state is not None:
        linear.load_state_dict(best_state, strict=True)
    return linear


def compute_scalar_metrics(y_true_ids, y_pred_ids):
    return {
        "accuracy": float(accuracy_score(y_true_ids, y_pred_ids)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_ids, y_pred_ids)),
        "macro_f1": float(f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)),
    }


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    set_seed(args.seed)

    train_ds = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, args.train_split)
    test_ds = TileFeatDataset(args.data_dir, args.json_path, args.pfm_name, args.test_split)
    idx2label = {v: k for k, v in train_ds.label2idx.items()}
    key = Path(args.json_path).stem

    save_dir = build_save_dir(
        args.save_root, args.pfm_name, args.slide_encoder, args.num_folds, args.patience, key
    )
    feature_dir = build_feature_dir(args.save_root, args.pfm_name, args.slide_encoder, key)

    cached = load_feature_cache(feature_dir) if args.reuse_features else None
    if cached is not None:
        train_features, train_y, test_features, test_y = cached
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
        print(
            f"slide_encoder={args.slide_encoder} train={len(train_ds)} test={len(test_ds)} "
            f"device={args.device}"
        )
        train_features, train_y = extract_slide_embeddings(model, train_loader, args.device, args.dtype)
        test_features, test_y = extract_slide_embeddings(model, test_loader, args.device, args.dtype)
        if args.save_features:
            save_feature_cache(feature_dir, train_features, train_y, test_features, test_y)
            print(f"Saved features: {feature_dir}")

    train_label_ids = train_y.astype(np.int64)
    test_label_ids = test_y.astype(np.int64)
    num_classes = len(train_ds.label2idx)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    fold_predictions = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_features, train_label_ids)):
        print(f"Fold [{fold_idx + 1}/{args.num_folds}]")
        linear = train_linear_probe(
            train_features=train_features[tr_idx],
            train_label_ids=train_label_ids[tr_idx],
            val_features=train_features[va_idx],
            val_label_ids=train_label_ids[va_idx],
            in_dim=train_features.shape[1],
            num_classes=num_classes,
            args=args,
        )

        test_pred_ids = _predict_ids(linear, test_features, args.device)
        fold_predictions.append(test_pred_ids.astype(np.int64))

        metrics = compute_scalar_metrics(test_label_ids, test_pred_ids)
        metrics["fold"] = int(fold_idx)
        fold_metrics.append(metrics)

        print(
            f"Fold {fold_idx}: acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )

    accs = [m["accuracy"] for m in fold_metrics]
    bal_accs = [m["balanced_accuracy"] for m in fold_metrics]
    macro_f1s = [m["macro_f1"] for m in fold_metrics]
    acc_mean, acc_std = mean_std(accs)
    bal_mean, bal_std = mean_std(bal_accs)
    f1_mean, f1_std = mean_std(macro_f1s)

    summary_metrics = {
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "balanced_accuracy_mean": bal_mean,
        "balanced_accuracy_std": bal_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
    }

    target_names = [str(idx2label[i]) for i in range(num_classes)]
    per_fold_reports = []
    for fold_idx, predictions in enumerate(fold_predictions):
        pred_ids = predictions.astype(np.int64)
        fold_report = classification_report(
            test_label_ids,
            pred_ids,
            labels=list(range(num_classes)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
        per_fold_reports.append((fold_idx, fold_report))

    report_lines = [
        "## Linear probe (StratifiedKFold on train; eval on held-out test)\n",
        f"json_path: {args.json_path}\n",
        f"slide_encoder: {args.slide_encoder}  pfm_name: {args.pfm_name}\n",
        f"epochs: {args.epochs}  patience: {args.patience}  num_folds: {args.num_folds}  "
        f"lr: {args.lr}  weight_decay: {args.weight_decay}\n",
        "\n## Fold metrics (test)\n",
    ]
    for m in fold_metrics:
        report_lines.append(
            f"fold={m['fold']}  acc={m['accuracy']:.4f}  "
            f"bal_acc={m['balanced_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}\n"
        )
    report_lines.append("\n## Mean ± Std (test over folds)\n")
    report_lines.append(f"accuracy: {acc_mean:.4f} ± {acc_std:.4f}\n")
    report_lines.append(f"balanced_accuracy: {bal_mean:.4f} ± {bal_std:.4f}\n")
    report_lines.append(f"macro_f1: {f1_mean:.4f} ± {f1_std:.4f}\n")
    report_lines.append("\n## Per-fold classification reports (test)\n")
    for fold_idx, fold_report in per_fold_reports:
        report_lines.append(f"\n### fold {fold_idx}\n")
        report_lines.append(fold_report)
        if not fold_report.endswith("\n"):
            report_lines.append("\n")

    report = "".join(report_lines)
    save_results(save_dir, args, test_ds.items, fold_predictions, fold_metrics, summary_metrics, report, idx2label)

    print(f"Accuracy (mean±std): {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Balanced-Acc (mean±std): {bal_mean:.4f} ± {bal_std:.4f}")
    print(f"Macro-F1 (mean±std): {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Saved: {save_dir}")


if __name__ == "__main__":
    main()
