"""
Sanity-checks (and optionally restructures) a downloaded benchmark dataset
folder into the layout expected by data/datasets.py:

    root/train/REAL|FAKE/*.png          (cifake)
    root/test/REAL|FAKE/*.png
    root/train/real|fake/*.png          (synthbuster)
    root/test/real|fake/*.png

Usage:
    python data/prepare_data.py --root data/CIFAKE --dataset cifake --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DATASET_SPECS, _list_images  # noqa: E402  (allow `python data/prepare_data.py`)


def check_dataset(root: Path, dataset: str) -> None:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset]

    print(f"Checking dataset '{dataset}' at {root} ...")
    ok = True
    for split in ("train", "test"):
        for cls_dir, label in ((spec.real_dirname, "REAL/AUTHENTIC"), (spec.fake_dirname, "FAKE/AI-GENERATED")):
            folder = root / split / cls_dir
            n = len(_list_images(folder))
            status = "OK" if n > 0 else "MISSING / EMPTY"
            if n == 0:
                ok = False
            print(f"  [{split:5s}] {cls_dir:10s} ({label:18s}) -> {n:6d} images   [{status}]")

    if ok:
        print("Dataset layout looks valid.")
    else:
        print(
            "\nSome expected folders are missing or empty. Expected layout:\n"
            f"  {root}/train/{spec.real_dirname}/*.png\n"
            f"  {root}/train/{spec.fake_dirname}/*.png\n"
            f"  {root}/test/{spec.real_dirname}/*.png\n"
            f"  {root}/test/{spec.fake_dirname}/*.png\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Verify/prepare CIFAKE or Synthbuster dataset layout.")
    parser.add_argument("--root", required=True, help="Path to the dataset root folder.")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_SPECS))
    parser.add_argument("--check", action="store_true", help="Run a sanity check on the folder layout.")
    args = parser.parse_args()

    root = Path(args.root)
    if args.check:
        check_dataset(root, args.dataset)
    else:
        print("Nothing to do. Pass --check to validate the dataset folder layout.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
