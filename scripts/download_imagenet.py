#!/usr/bin/env python3
"""
Download ImageNet-1K from HuggingFace and save in torchvision ImageFolder layout:

    /scratch/ag11023/HPML/imagenet/
        train/
            n01440764/  (synset folders)
            ...
        val/
            n01440764/
            ...

Requires:
  - HF token saved via `huggingface-cli login` or HF_TOKEN env var
  - License accepted at https://huggingface.co/datasets/ILSVRC/imagenet-1k

Usage (inside Singularity or venv):
    python scripts/download_imagenet.py
    python scripts/download_imagenet.py --output /scratch/ag11023/HPML/imagenet
    python scripts/download_imagenet.py --split train  # only train
    python scripts/download_imagenet.py --split val    # only val
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/scratch/ag11023/HPML/imagenet",
                   help="Root output directory (will contain train/ and val/)")
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Parallel download workers")
    p.add_argument("--dry-run", action="store_true",
                   help="Download only 200 train + 50 val images to verify pipeline")
    return p.parse_args()


def save_split(dataset, split_dir: Path, num_workers: int):
    """Write HF dataset split to ImageFolder layout using PIL."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    counter = {"n": 0}
    lock = threading.Lock()

    def save_one(item):
        label_idx = item["label"]
        synset = dataset.features["label"].int2str(label_idx)
        class_dir = split_dir / synset
        class_dir.mkdir(exist_ok=True)

        img_path = class_dir / f"{item['__index_level_0__']:08d}.JPEG"
        if not img_path.exists():
            item["image"].save(img_path, format="JPEG", quality=95)

        with lock:
            counter["n"] += 1
            if counter["n"] % 10_000 == 0:
                print(f"  {counter['n']:>7,} / {total:,} saved", flush=True)

    print(f"Saving {total:,} images to {split_dir} with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(save_one, dataset[i]) for i in range(total)]
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"ERROR: {exc}", file=sys.stderr)

    print(f"Done — {split_dir}")


def save_split_streaming(dataset_iter, split_dir: Path, n_samples: int, features):
    """Save from a streaming dataset — fetches only what's needed."""
    split_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, item in enumerate(dataset_iter):
        if saved >= n_samples:
            break
        label_idx = item["label"]
        synset = features["label"].int2str(label_idx)
        class_dir = split_dir / synset
        class_dir.mkdir(exist_ok=True)
        img_path = class_dir / f"{i:08d}.JPEG"
        if not img_path.exists():
            item["image"].save(img_path, format="JPEG", quality=95)
        saved += 1
        if saved % 50 == 0:
            print(f"  {saved} / {n_samples} saved", flush=True)
    print(f"Done — {split_dir} ({saved} images)")


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    output = Path(args.output)

    # Dry-run uses a separate directory so it doesn't pollute the real output
    if args.dry_run:
        output = output.parent / (output.name + "_dryrun")
        print(f"DRY RUN — downloading 200 train + 50 val images to {output}")

    output.mkdir(parents=True, exist_ok=True)

    DRY_RUN_SIZES = {"train": 200, "validation": 50}

    splits = ["train", "validation"] if args.split == "both" else \
             ["validation"] if args.split == "val" else ["train"]

    for split in splits:
        out_folder = "val" if split == "validation" else split
        out_dir = output / out_folder
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"{out_dir} already exists and is non-empty — skipping.")
            continue

        print(f"\n=== Downloading split: {split} → {out_dir} ===")

        if args.dry_run:
            # Streaming — only fetches the exact images needed, no full download
            from datasets import load_dataset as _load
            n = DRY_RUN_SIZES[split]
            print(f"  (dry run: streaming {n} samples)")
            ds_stream = _load("ILSVRC/imagenet-1k", split=split, streaming=True)
            save_split_streaming(iter(ds_stream), out_dir, n, ds_stream.features)
        else:
            ds = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split,
                num_proc=args.num_workers,
            )
            ds = ds.add_column("__index_level_0__", list(range(len(ds))))
            save_split(ds, out_dir, args.num_workers)

    print(f"\nAll done. ImageNet saved to: {output}")


if __name__ == "__main__":
    main()
