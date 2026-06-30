import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from data.datasets import build_dataset, DATASET_SPECS


def _make_dummy_dataset(root: Path, dataset: str, n_per_class: int = 4, size: int = 32):
    spec = DATASET_SPECS[dataset]
    for split in ("train", "test"):
        for cls in (spec.real_dirname, spec.fake_dirname):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"img_{i}.png")


def test_authenticity_dataset_cifake():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_dummy_dataset(root, "cifake", n_per_class=3)
        ds = build_dataset(str(root), "cifake", "train", img_size=64, tourism=False)
        assert len(ds) == 6
        sample = ds[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["label"].item() in (0.0, 1.0)


def test_tourism_dataset_synthbuster():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_dummy_dataset(root, "synthbuster", n_per_class=3)
        ds = build_dataset(str(root), "synthbuster", "test", img_size=64, tourism=True)
        assert len(ds) == 6
        sample = ds[0]
        assert sample["metadata"].shape == (8,)
        assert sample["source"].shape == (6,)
        assert sample["source"].sum().item() == 1.0  # one-hot


def test_dataloader_batches():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_dummy_dataset(root, "cifake", n_per_class=8)
        ds = build_dataset(str(root), "cifake", "train", img_size=64, tourism=True)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 64, 64)
        assert batch["label"].shape == (4,)
        assert batch["metadata"].shape == (4, 8)
        assert batch["source"].shape == (4, 6)


if __name__ == "__main__":
    test_authenticity_dataset_cifake()
    test_tourism_dataset_synthbuster()
    test_dataloader_batches()
    print("All dataloader tests passed.")
