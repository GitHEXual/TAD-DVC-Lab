"""Shared helpers for DVC pipeline stages."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_params(path: str = "params.yaml") -> dict:
    import yaml

    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def read_manifest(path: Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [(r["path"], r["label"]) for r in csv.DictReader(f)]


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_transforms(img_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class ManifestDataset(Dataset):
    """Reads a CSV manifest with `path` and `label` columns."""

    def __init__(self, manifest: Path, classes: list[str], img_size: int):
        self.rows = read_manifest(manifest)
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.transform = build_transforms(img_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, label = self.rows[index]
        image = Image.open(ROOT / path).convert("RGB")
        return self.transform(image), self.class_to_idx[label]


class SimpleCNN(nn.Module):
    """Small CNN trained from scratch, designed for 96x96 inputs."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models as tvm

    if arch == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "simplecnn":
        model = SimpleCNN(num_classes)
    else:
        raise ValueError(f"unknown model: {arch}")
    return model


def model_hparams(params: dict, key: str) -> dict:
    """Per-model overrides from `models.<key>`, falling back to `train`."""
    train = params["train"]
    spec = dict(params["models"][key])
    hparams = {
        "lr": spec.get("lr", train["lr"]),
        "weight_decay": spec.get("weight_decay", train["weight_decay"]),
        "epochs": spec.get("epochs", train["epochs"]),
        "batch_size": spec.get("batch_size", train["batch_size"]),
        "dropout": spec.get("dropout", 0.3),
        "freeze_backbone": bool(spec.get("freeze_backbone", True)),
    }
    return hparams


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt["meta"]
    model = build_model(meta["arch"], meta["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, meta
