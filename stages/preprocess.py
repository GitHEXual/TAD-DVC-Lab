"""Stage: scan the raw dataset, build stratified train/val/test manifests."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from common import ROOT, list_images, load_params, save_json, seed_everything, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/data_raw", help="raw dataset directory")
    parser.add_argument("--out", default="data/splits", help="output directory for manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params()
    cfg = params["preprocess"]
    seed = cfg["seed"]
    seed_everything(seed)

    raw_dir = ROOT / args.raw
    out_dir = ROOT / args.out
    classes = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"no class folders found in {raw_dir}")

    paths: list[str] = []
    labels: list[str] = []
    for name in classes:
        for image in list_images(raw_dir / name):
            paths.append(str(image.relative_to(ROOT)))
            labels.append(name)

    # 60/20/20 stratified split: train vs (val+test), then val vs test.
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths,
        labels,
        train_size=cfg["train_size"],
        stratify=labels,
        random_state=seed,
    )
    val_share = cfg["val_size"] / (cfg["val_size"] + cfg["test_size"])
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths,
        rest_labels,
        train_size=val_share,
        stratify=rest_labels,
        random_state=seed,
    )

    splits = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }
    for name, rows in splits.items():
        write_manifest(out_dir / f"{name}.csv", rows)
        print(f"{name:5} {len(rows):5}  {dict(sorted(Counter(l for _, l in rows).items()))}")

    save_json(
        out_dir / "classes.json",
        {
            "classes": classes,
            "img_size": cfg["img_size"],
            "seed": seed,
            "counts": {name: len(rows) for name, rows in splits.items()},
            "total": len(paths),
        },
    )
    print(f"total {len(paths)} images in {len(classes)} classes -> {out_dir}")


if __name__ == "__main__":
    main()
