"""Unified registry for baseline models used in Table 4 / 5 comparisons."""

from __future__ import annotations

from models.baselines.xception import build_xception
from models.baselines.f3net import build_f3net
from models.baselines.swin import build_swin
from models.baselines.aide import build_aide
from models.baselines.iapl import build_iapl
from models.ds_svit import build_ds_svit

REGISTRY = {
    "xception": build_xception,
    "xcp": build_xception,
    "f3net": build_f3net,
    "f3n": build_f3net,
    "swin": build_swin,
    "aide": build_aide,
    "iapl": build_iapl,
    "trusttour": build_ds_svit,
    "tt": build_ds_svit,
    "ds_svit": build_ds_svit,
}


def build_model(name: str, **kwargs):
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(set(REGISTRY.keys()))}")
    return REGISTRY[key](**kwargs)
