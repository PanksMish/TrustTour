"""Plotting helpers that reproduce the style of Figures 3-6 in the paper:
ROC/PR curves, loss/accuracy curves, calibration curve, trust distribution,
trust-risk scatter, and RL/game convergence plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_pr(curves_by_method: dict, out_path: str | Path) -> None:
    """curves_by_method: {method_name: {'fpr':..,'tpr':..,'precision':..,'recall':..,'auc':..,'ap':..}}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, c in curves_by_method.items():
        axes[0].plot(c["fpr"], c["tpr"], label=f"{name} (AUC={c.get('auc', float('nan')):.3f})")
        axes[1].plot(c["recall"], c["precision"], label=f"{name} (AP={c.get('ap', float('nan')):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR)")
    axes[0].set_title("ROC Curves")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_learning_curves(train_loss, val_loss, train_acc, val_acc, val_auc, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = np.arange(1, len(train_loss) + 1)
    axes[0].plot(epochs, train_loss, label="Training loss")
    axes[0].plot(epochs, val_loss, "--", label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title("Loss Convergence")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Training Accuracy")
    axes[1].plot(epochs, val_acc, label="Validation Accuracy")
    axes[1].plot(epochs, val_auc, label="Validation AUC-ROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Learning Curves")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path, n_bins: int = 10) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    mean_pred, frac_pos = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", label="TrustTour")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trust_distribution(trust: np.ndarray, label: np.ndarray, tau1: float, tau2: float,
                             out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(trust[label == 1], bins=30, alpha=0.6, label="Authentic", color="seagreen")
    ax.hist(trust[label == 0], bins=30, alpha=0.6, label="AI-Generated", color="darkorange")
    ax.axvline(1 - tau2, color="gray", linestyle="--", linewidth=1)
    ax.axvline(1 - tau1, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Estimated Trust Score T(I)")
    ax.set_ylabel("Frequency Count")
    ax.set_title("Trust Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trust_risk(trust: np.ndarray, risk: np.ndarray, decisions: np.ndarray, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {0: "seagreen", 1: "goldenrod", 2: "firebrick"}
    names = {0: "Verified", 1: "Human Review", 2: "Rejected"}
    for d in [0, 1, 2]:
        mask = decisions == d
        ax.scatter(risk[mask], trust[mask], s=8, alpha=0.6, color=colors[d], label=names[d])
    ax.set_xlabel("Estimated Risk R(I)")
    ax.set_ylabel("Trust Score T(I)")
    ax.set_title("Trust-Risk Relationship")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rl_convergence(reward_history: list[float], out_path: str | Path, window: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = np.array(reward_history)
    ax.plot(r, alpha=0.3, color="steelblue", label="Reward per episode")
    if len(r) >= window:
        smooth = np.convolve(r, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(r)), smooth, color="navy", label=f"{window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("HITL-RL Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_game_equilibrium(history: dict, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(history["U_P"], label="Platform Utility (Leader)")
    ax.plot(history["U_A"], label="Adversary Utility (Follower)")
    ax.set_xlabel("Policy Iteration")
    ax.set_ylabel("Utility")
    ax.set_title("Stackelberg Game Convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
