"""
Dataset loaders for the two benchmark datasets used in the paper
(Section 5.1 / Table 2):

    CIFAKE      data/CIFAKE/{train,test}/{REAL,FAKE}/*.png
    Synthbuster data/Synthbuster/{train,test}/{real,fake}/*.png

Also provides `TourismInformationDataset`, a thin wrapper that augments each
image with *synthetic* metadata (M) and source (S) feature vectors so the
Trust Assessment Module (Eq. 39-45) can be trained/evaluated end-to-end even
when the raw benchmark datasets only ship images + binary labels. In a real
deployment, M and S would come from EXIF/geotag verification and platform
source-reliability databases; here they are sampled in a label-correlated
but noisy way so trust estimation has a non-trivial, learnable signal.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_eval_transforms, build_train_transforms

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])


@dataclass
class DatasetSpec:
    name: str
    real_dirname: str
    fake_dirname: str


DATASET_SPECS = {
    "cifake": DatasetSpec("cifake", "REAL", "FAKE"),
    "synthbuster": DatasetSpec("synthbuster", "real", "fake"),
}


class AuthenticityImageDataset(Dataset):
    """Generic binary authenticity dataset: label 1 = Authentic/Real, 0 = AI-generated/Fake.

    Expects the folder layout:
        root/<split>/<real_dirname>/*.{png,jpg,...}
        root/<split>/<fake_dirname>/*.{png,jpg,...}
    """

    def __init__(
        self,
        root: str | os.PathLike,
        dataset: str = "cifake",
        split: str = "train",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        if dataset not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASET_SPECS)}")
        spec = DATASET_SPECS[dataset]
        self.root = Path(root)
        self.split = split
        self.dataset = dataset

        real_dir = self.root / split / spec.real_dirname
        fake_dir = self.root / split / spec.fake_dirname

        real_paths = _list_images(real_dir)
        fake_paths = _list_images(fake_dir)

        rng = random.Random(seed)
        if max_samples_per_class is not None:
            rng.shuffle(real_paths)
            rng.shuffle(fake_paths)
            real_paths = real_paths[:max_samples_per_class]
            fake_paths = fake_paths[:max_samples_per_class]

        self.samples: list[tuple[Path, int]] = (
            [(p, 1) for p in real_paths] + [(p, 0) for p in fake_paths]
        )
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root}/{split}/. "
                f"Expected subfolders '{spec.real_dirname}' and '{spec.fake_dirname}'. "
                f"Run data/prepare_data.py --check to debug your dataset layout."
            )
        rng.shuffle(self.samples)

        self.transform = transform or (
            build_train_transforms(img_size) if split == "train" else build_eval_transforms(img_size)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {e}")
        img = self.transform(img)
        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": str(path),
        }


# Source-type categories used to build the synthetic source vector S.
SOURCE_TYPES = ["official_dmo", "hotel_ota", "travel_agency", "social_media", "ugc", "unknown"]


class TourismInformationDataset(AuthenticityImageDataset):
    """Wraps an AuthenticityImageDataset with synthetic metadata (M) and
    source (S) vectors, producing full Tourism Information Objects
    I = (V, M, S, U) for training/evaluating the Trust Assessment Module.

    Metadata vector (dim=8): [geotag_valid, timestamp_consistent, exif_present,
        exif_consistent, device_fingerprint_match, resolution_typical,
        compression_typical, duplicate_hash_flag]

    Source vector (dim=6): one-hot over SOURCE_TYPES, optionally scaled by a
    historical reliability prior in [0,1] (kept as a one-hot here for
    simplicity; reliability is instead encoded through label-correlated noise
    in metadata/source quality below).
    """

    def __init__(self, *args, metadata_noise: float = 0.25, source_noise: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_noise = metadata_noise
        self.source_noise = source_noise
        self._rng = np.random.default_rng(1234)

    def _sample_metadata(self, label: int) -> np.ndarray:
        # Authentic images tend to have more consistent metadata; fakes tend
        # to have more inconsistencies, but with noise so the signal isn't trivial.
        base = 0.85 if label == 1 else 0.35
        vec = self._rng.normal(base, self.metadata_noise, size=8)
        return np.clip(vec, 0.0, 1.0).astype(np.float32)

    def _sample_source(self, label: int) -> tuple[np.ndarray, str]:
        # Authentic tourism images are more likely (not guaranteed) to come
        # from official/credible sources; fakes skew towards social/UGC.
        if label == 1:
            probs = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]
        else:
            probs = [0.05, 0.10, 0.10, 0.40, 0.30, 0.05]
        if self._rng.random() < self.source_noise:
            probs = [1.0 / len(SOURCE_TYPES)] * len(SOURCE_TYPES)  # noise: fully random source
        idx = self._rng.choice(len(SOURCE_TYPES), p=probs)
        one_hot = np.zeros(len(SOURCE_TYPES), dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot, SOURCE_TYPES[idx]

    def __getitem__(self, idx: int):
        base = super().__getitem__(idx)
        label = int(base["label"].item())
        metadata = self._sample_metadata(label)
        source_vec, source_name = self._sample_source(label)
        base["metadata"] = torch.from_numpy(metadata)
        base["source"] = torch.from_numpy(source_vec)
        base["source_name"] = source_name
        return base


def build_dataset(
    root: str,
    dataset: str = "cifake",
    split: str = "train",
    img_size: int = 224,
    tourism: bool = False,
    **kwargs,
) -> Dataset:
    cls = TourismInformationDataset if tourism else AuthenticityImageDataset
    return cls(root=root, dataset=dataset, split=split, img_size=img_size, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="cifake", choices=list(DATASET_SPECS))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ds = build_dataset(args.root, args.dataset, args.split, tourism=True)
    print(f"Loaded {len(ds)} samples from {args.root}/{args.split}")
    sample = ds[0]
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in sample.items()})
