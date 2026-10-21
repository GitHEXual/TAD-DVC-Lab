"""Stage: generate an augmented training set from the train split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from torchvision import transforms

from common import ROOT, load_params, read_manifest, save_json, seed_everything, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits", help="directory with manifests")
    parser.add_argument("--out", default="data/augmented", help="output directory")
    return parser.parse_args()


def build_augmenter(img_size: int, cfg: dict) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(cfg["crop_scale"], 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(cfg["rotation"]),
            transforms.ColorJitter(
                brightness=cfg["jitter"],
                contrast=cfg["jitter"],
                saturation=cfg["jitter"],
                hue=cfg["hue"],
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    params = load_params()
    cfg = params["augment"]
    img_size = params["preprocess"]["img_size"]
    seed_everything(params["preprocess"]["seed"])

    splits = ROOT / args.splits
    out_dir = ROOT / args.out
    classes = json.loads((splits / "classes.json").read_text(encoding="utf-8"))["classes"]
    train_rows = read_manifest(splits / "train.csv")
    augmenter = build_augmenter(img_size, cfg)

    generated: list[tuple[str, str]] = []
    for index, (rel_path, label) in enumerate(train_rows):
        source = Image.open(ROOT / rel_path).convert("RGB")
        stem = Path(rel_path).stem
        for copy in range(cfg["copies"]):
            seed_everything(cfg["seed"] + index * 100 + copy)
            image = augmenter(source)
            target = out_dir / label / f"{stem}_aug{copy + 1}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, quality=92)
            generated.append((str(target.relative_to(ROOT)), label))

    rows = train_rows + generated
    write_manifest(out_dir / "train_aug.csv", rows)
    save_json(
        out_dir / "augment.json",
        {
            "copies": cfg["copies"],
            "originals": len(train_rows),
            "generated": len(generated),
            "total": len(rows),
            "img_size": img_size,
            "transforms": {
                "random_resized_crop": cfg["crop_scale"],
                "rotation": cfg["rotation"],
                "color_jitter": cfg["jitter"],
                "hue": cfg["hue"],
                "horizontal_flip": True,
            },
        },
    )
    print(f"generated {len(generated)} images, train set: {len(train_rows)} -> {len(rows)}")


if __name__ == "__main__":
    main()
