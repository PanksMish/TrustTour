"""
Run TrustTour inference on a single image, producing A(I), T(I), R(I), and
the platform decision D(I).

Usage:
    python scripts/infer.py --checkpoint checkpoints/trusttour_best.pth \
        --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from PIL import Image

from data.transforms import build_eval_transforms
from models.decision_module import AdaptiveDecisionModule, DecisionThresholds
from scripts.train import build_trusttour_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/trusttour.yaml")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--tau1", type=float, default=0.35)
    p.add_argument("--tau2", type=float, default=0.65)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Optional metadata/source overrides; default to a "neutral" tourism object.
    p.add_argument("--source", type=str, default="unknown",
                    choices=["official_dmo", "hotel_ota", "travel_agency", "social_media", "ugc", "unknown"])
    return p.parse_args()


SOURCE_TYPES = ["official_dmo", "hotel_ota", "travel_agency", "social_media", "ugc", "unknown"]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    model = build_trusttour_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = build_eval_transforms(cfg["img_size"])
    img = Image.open(args.image).convert("RGB")
    image_t = transform(img).unsqueeze(0).to(device)

    # Neutral metadata vector (no strong evidence either way) and one-hot source.
    metadata = torch.full((1, 8), 0.5, device=device)
    source = torch.zeros((1, 6), device=device)
    source[0, SOURCE_TYPES.index(args.source)] = 1.0

    with torch.no_grad():
        out = model(image_t, metadata, source)

    auth_prob = out["auth_prob"].item()
    trust = out["trust"].item()

    dm = AdaptiveDecisionModule(DecisionThresholds(args.tau1, args.tau2))
    risk = dm.risk(torch.tensor([trust])).item()
    decision = dm.decide(torch.tensor([trust])).item()
    decision_name = {0: "Verified", 1: "Human Review", 2: "Rejected"}[decision]

    print(f"Image:            {args.image}")
    print(f"Authenticity A(I): {auth_prob:.4f}  ({'likely authentic' if auth_prob >= 0.5 else 'likely AI-generated'})")
    print(f"Trust T(I):        {trust:.4f}")
    print(f"Risk R(I):         {risk:.4f}")
    print(f"Decision D(I):     {decision_name}")


if __name__ == "__main__":
    main()
