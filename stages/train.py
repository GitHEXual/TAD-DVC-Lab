"""Stage: train one model on the train split, validate on the val split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    ROOT,
    ManifestDataset,
    build_model,
    load_params,
    model_hparams,
    save_checkpoint,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model key from params.yaml")
    parser.add_argument("--splits", default="data/splits", help="directory with manifests")
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(targets, preds), f1_score(
        targets, preds, average="macro"
    )


def main() -> None:
    args = parse_args()
    params = load_params()
    key = args.model
    spec = params["models"][key]
    arch, pretrained = spec["arch"], bool(spec.get("pretrained", False))
    hp = model_hparams(params, key)
    seed_everything(params["train"]["seed"])

    splits = ROOT / args.splits
    classes = json.loads((splits / "classes.json").read_text(encoding="utf-8"))["classes"]
    img_size = params["preprocess"]["img_size"]
    batch_size = hp["batch_size"]
    num_workers = params["train"]["num_workers"]

    train_manifest = splits / "train.csv"
    if params["train"].get("data", "raw") == "aug":
        train_manifest = ROOT / "data" / "augmented" / "train_aug.csv"
    print(f"[{key}] train manifest: {train_manifest.relative_to(ROOT)}")

    train_ds = ManifestDataset(train_manifest, classes, img_size)
    val_ds = ManifestDataset(splits / "val.csv", classes, img_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    torch.set_num_threads(params["train"].get("threads") or 8)
    device = torch.device("cpu")
    model = build_model(arch, len(classes), pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    if hp["freeze_backbone"] and pretrained:
        for param in model.parameters():
            param.requires_grad = False
        head = model.fc if arch == "resnet18" else model.classifier
        for param in head.parameters():
            param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    history = []
    best = {"epoch": -1, "val_macro_f1": -1.0}
    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_macro_f1": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        save_json(ROOT / "reports" / "history" / f"{key}.json", history)
        print(
            f"[{key}] epoch {epoch}/{hp['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}",
            flush=True,
        )

        if val_f1 > best["val_macro_f1"]:
            best = {"epoch": epoch, "val_macro_f1": round(val_f1, 4), "val_acc": round(val_acc, 4)}
            save_checkpoint(
                ROOT / "models" / f"{key}.pt",
                model,
                {
                    "model": key,
                    "arch": arch,
                    "pretrained": pretrained,
                    "num_classes": len(classes),
                    "classes": classes,
                    "img_size": img_size,
                    "params": hp,
                },
            )

    save_json(ROOT / "reports" / "history" / f"{key}.json", history)
    print(f"[{key}] best {best} -> models/{key}.pt")


if __name__ == "__main__":
    main()
