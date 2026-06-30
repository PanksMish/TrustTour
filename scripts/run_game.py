"""
Run the Strategic Optimization Module (Section 3.7 / 4.3, Algorithm 3):
solves the platform-vs-adversary Stackelberg game and reports the optimal
platform strategy (verification intensity, recommendation priority, safety
alert sensitivity) and adversary best response.

Usage:
    python scripts/run_game.py --iterations 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from game.stackelberg import GameConfig, StackelbergGame
from utils.plotting import plot_game_equilibrium


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/trusttour.yaml")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    g_cfg = cfg.get("game", {})
    game_cfg = GameConfig(
        iterations=args.iterations or g_cfg.get("iterations", 60),
        lr_platform=g_cfg.get("lr_platform", 0.05),
        lr_adversary=g_cfg.get("lr_adversary", 0.05),
    )
    game = StackelbergGame(game_cfg)
    result = game.solve(iterations=game_cfg.iterations)

    print("\n=== Stackelberg Equilibrium ===")
    print(f"  Platform strategy (S_P*): {result['platform_strategy']}")
    print(f"  Adversary best response (S_A*): {result['adversary_strategy']}")
    print(f"  U_P*: {result['U_P']:.4f}")
    print(f"  U_A*: {result['U_A']:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_game_equilibrium(result["history"], out_dir / "game_equilibrium.png")

    with open(out_dir / "game_summary.json", "w") as f:
        json.dump({
            "platform_strategy": result["platform_strategy"],
            "adversary_strategy": result["adversary_strategy"],
            "U_P": result["U_P"],
            "U_A": result["U_A"],
        }, f, indent=2)
    print(f"Saved game results to {out_dir}/")


if __name__ == "__main__":
    main()
