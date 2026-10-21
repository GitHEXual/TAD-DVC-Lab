"""Stage: evaluate one trained model on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from common import ROOT, ManifestDataset, load_checkpoint, load_params, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model key from params.yaml")
    parser.add_argument("--splits", default="data/splits", help="directory with manifests")
    return parser.parse_args()


@torch.no_grad()
def predict(model, loader):
    preds, targets = [], []
    for images, labels in loader:
        logits = model(images)
        preds.extend(logits.argmax(1).tolist())
        targets.extend(labels.tolist())
    return targets, preds


def main() -> None:
    args = parse_args()
    key = args.model
    params = load_params()
    splits = ROOT / args.splits
    classes = json.loads((splits / "classes.json").read_text(encoding="utf-8"))["classes"]

    model, meta = load_checkpoint(ROOT / "models" / f"{key}.pt")
    model.eval()
    img_size = meta["img_size"]

    test_ds = ManifestDataset(splits / "test.csv", classes, img_size)
    test_loader = DataLoader(
        test_ds, batch_size=params["train"]["batch_size"], shuffle=False, num_workers=0
    )
    targets, preds = predict(model, test_loader)

    report = classification_report(
        targets, preds, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "model": key,
        "arch": meta["arch"],
        "pretrained": meta["pretrained"],
        "img_size": img_size,
        "test_size": len(targets),
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
    }
    save_json(ROOT / "reports" / "metrics" / f"{key}.json", metrics)

    per_class = {
        name: {
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1": round(report[name]["f1-score"], 4),
            "support": int(report[name]["support"]),
        }
        for name in classes
    }
    save_json(ROOT / "reports" / "per_class" / f"{key}.json", per_class)

    print(
        f"[{key}] acc={metrics['accuracy']} macro_f1={metrics['macro_f1']} "
        f"weighted_f1={metrics['weighted_f1']}"
    )
    for name in classes:
        row = per_class[name]
        print(f"  {name:10} f1={row['f1']:.3f} recall={row['recall']:.3f} n={row['support']}")


if __name__ == "__main__":
    main()
