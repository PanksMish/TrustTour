"""
Evaluate a trained TrustTour checkpoint on a held-out split, reporting
detection metrics (Eq. 60-63), trust-aware decision statistics (Table 7),
and saving ROC/PR + trust-distribution plots (Figures 3 & 5 style).

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/trusttour_best.pth \
        --root data/CIFAKE --dataset cifake --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import build_dataset
from models.decision_module import AdaptiveDecisionModule, DecisionThresholds
from scripts.train import build_trusttour_model
from utils.metrics import detection_metrics, roc_pr_curves, trust_aware_metrics
from utils.plotting import plot_roc_pr, plot_trust_distribution, plot_trust_risk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/trusttour.yaml")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, choices=["cifake", "synthbuster"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--tau1", type=float, default=None)
    p.add_argument("--tau2", type=float, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples_per_class", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    labels, auth_probs, trusts = [], [], []
    for batch in tqdm(loader, desc="evaluate"):
        image = batch["image"].to(device)
        metadata = batch["metadata"].to(device)
        source = batch["source"].to(device)
        label = batch["label"]

        out = model(image, metadata, source)
        labels.append(label.numpy())
        auth_probs.append(out["auth_prob"].cpu().numpy())
        trusts.append(out["trust"].cpu().numpy())

    return (np.concatenate(labels), np.concatenate(auth_probs), np.concatenate(trusts))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    root = args.root or cfg["data"]["root"]
    dataset = args.dataset or cfg["data"]["dataset"]
    img_size = cfg["img_size"]

    ds = build_dataset(root, dataset, args.split, img_size, tourism=True,
                        max_samples_per_class=args.max_samples_per_class)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["data"].get("num_workers", 4))

    model = build_trusttour_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    y_true, y_auth, y_trust = collect_predictions(model, loader, device)

    det_metrics = detection_metrics(y_true, y_auth)
    print("\n=== Detection Performance ===")
    for k, v in det_metrics.items():
        print(f"  {k}: {v:.4f}")

    tau1 = args.tau1 if args.tau1 is not None else cfg["decision"]["tau1"]
    tau2 = args.tau2 if args.tau2 is not None else cfg["decision"]["tau2"]
    dm = AdaptiveDecisionModule(DecisionThresholds(tau1, tau2), impact_factor=cfg["decision"].get("impact_factor", 1.0))

    trust_t = torch.from_numpy(y_trust)
    risk_t = dm.risk(trust_t)
    decisions_t = dm.decide(trust_t)
    risk_np = risk_t.numpy()
    decisions_np = decisions_t.numpy()

    trust_stats = trust_aware_metrics(y_trust, risk_np, decisions_np)
    print("\n=== Trust-aware Decision Statistics ===")
    for k, v in trust_stats.items():
        print(f"  {k}: {v:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = roc_pr_curves(y_true, y_auth)
    curves.update({"auc": det_metrics["AUC"], "ap": det_metrics["AP"]})
    plot_roc_pr({"TrustTour": curves}, out_dir / "roc_pr_curves.png")
    plot_trust_distribution(y_trust, y_true, tau1, tau2, out_dir / "trust_distribution.png")
    plot_trust_risk(y_trust, risk_np, decisions_np, out_dir / "trust_risk.png")

    summary = {
        "detection_metrics": det_metrics,
        "trust_aware_metrics": trust_stats,
        "thresholds": {"tau1": tau1, "tau2": tau2},
        "n_samples": int(len(y_true)),
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved plots and summary to {out_dir}/")


if __name__ == "__main__":
    main()
