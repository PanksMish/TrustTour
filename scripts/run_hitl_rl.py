"""
Run the Human-in-the-Loop Reinforcement Learning threshold-adaptation loop
(Section 3.6 / 4.2, Algorithm 2) on top of a trained TrustTour checkpoint's
trust scores, using dataset ground-truth labels as a stand-in for expert
verification labels H(I).

Usage:
    python scripts/run_hitl_rl.py --checkpoint checkpoints/trusttour_best.pth \
        --root data/CIFAKE --dataset cifake --episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import build_dataset
from rl.hitl_rl import HITLRLAgent, HITLRLConfig
from scripts.evaluate import collect_predictions
from scripts.train import build_trusttour_model
from utils.plotting import plot_rl_convergence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/trusttour.yaml")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, choices=["cifake", "synthbuster"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples_per_class", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)

    root = args.root or cfg["data"]["root"]
    dataset = args.dataset or cfg["data"]["dataset"]

    ds = build_dataset(root, dataset, args.split, cfg["img_size"], tourism=True,
                        max_samples_per_class=args.max_samples_per_class)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["data"].get("num_workers", 4))

    model = build_trusttour_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    y_true, y_auth, y_trust = collect_predictions(model, loader, device)

    trust_t = torch.from_numpy(y_trust).float()
    auth_t = torch.from_numpy(y_auth).float()
    expert_t = torch.from_numpy(y_true).float()  # ground-truth label stands in for expert label H(I)

    rl_cfg = cfg.get("hitl_rl", {})
    agent_cfg = HITLRLConfig(
        lr=rl_cfg.get("lr", 0.01),
        human_cost=rl_cfg.get("human_cost", 0.1),
        trust_lambda=rl_cfg.get("trust_lambda", 0.7),
        tau1_init=cfg["decision"]["tau1"],
        tau2_init=cfg["decision"]["tau2"],
        episodes=args.episodes or rl_cfg.get("episodes", 200),
    )
    agent = HITLRLAgent(agent_cfg)
    result = agent.run(trust_t, auth_t, expert_t, episodes=agent_cfg.episodes)

    print("\n=== HITL-RL Result ===")
    print(f"  Final tau1: {result['tau1']:.4f}")
    print(f"  Final tau2: {result['tau2']:.4f}")
    print(f"  Human Review Rate: {result['human_review_rate']:.4f}")
    print(f"  Decision distribution: {result['decision_distribution']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_rl_convergence(result["reward_history"], out_dir / "hitl_rl_convergence.png")

    with open(out_dir / "hitl_rl_summary.json", "w") as f:
        json.dump({
            "tau1": result["tau1"],
            "tau2": result["tau2"],
            "human_review_rate": result["human_review_rate"],
            "decision_distribution": result["decision_distribution"],
            "final_reward": result["reward_history"][-1] if result["reward_history"] else None,
        }, f, indent=2)
    print(f"Saved HITL-RL results to {out_dir}/")


if __name__ == "__main__":
    main()
