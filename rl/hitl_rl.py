"""
Human-in-the-Loop Reinforcement Learning (Section 3.6 / 4.2, Eq. 23-31, 50-51).

State:      s_t = (T'(I), R(I), H(I))                              Eq. (25)
Action:     a in {Verify, Human Review, Reject}                    Eq. (26)
Trust update: T'(I) = lambda*T(I) + (1-lambda)*H(I)                 Eq. (24)
Reward:     r = lambda1*1[D==H] - lambda2*delta - lambda3*C_H       Eq. (50)
Policy update on thresholds: tau_i <- tau_i + eta * dr/dtau_i        Eq. (51)

We implement a lightweight REINFORCE-style policy-gradient agent that treats
(tau1, tau2) as continuous policy parameters and learns them via a Gaussian
policy over threshold adjustments, since the decision module itself is a
deterministic threshold rule, this is the natural action space described in
the paper (the RL "acts" by adjusting verification thresholds, not by
re-training the authenticity classifier).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from models.decision_module import AdaptiveDecisionModule, DecisionThresholds, Decision


@dataclass
class HITLRLConfig:
    lr: float = 0.01           # eta, Eq. (51)
    gamma: float = 0.95        # discount factor, Eq. (27)
    lambda1: float = 1.0       # agreement reward weight
    lambda2: float = 1.0       # verification-error penalty weight
    lambda3: float = 0.5       # human-cost penalty weight
    human_cost: float = 0.1    # C_H, normalized cost of a human review action
    trust_lambda: float = 0.7  # lambda in Eq. (24), automated-vs-human trust blend
    action_std: float = 0.03   # exploration noise on threshold deltas
    tau1_init: float = 0.35
    tau2_init: float = 0.65
    tau_min_gap: float = 0.05
    episodes: int = 200
    batch_size: int = 256
    seed: int = 42


class ThresholdPolicy:
    """Gaussian policy over (delta_tau1, delta_tau2) threshold adjustments."""

    def __init__(self, cfg: HITLRLConfig):
        self.cfg = cfg
        self.tau1 = cfg.tau1_init
        self.tau2 = cfg.tau2_init
        self.rng = np.random.default_rng(cfg.seed)
        # Simple learnable mean-adjustment parameters (the "policy")
        self.mu = np.array([0.0, 0.0])  # mean shift for (tau1, tau2)

    def sample_action(self) -> np.ndarray:
        noise = self.rng.normal(0, self.cfg.action_std, size=2)
        return self.mu + noise

    def apply(self, action: np.ndarray):
        new_tau1 = float(np.clip(self.tau1 + action[0], 0.01, 0.98))
        new_tau2 = float(np.clip(self.tau2 + action[1], 0.02, 0.99))
        if new_tau2 - new_tau1 < self.cfg.tau_min_gap:
            new_tau2 = min(0.99, new_tau1 + self.cfg.tau_min_gap)
        self.tau1, self.tau2 = new_tau1, new_tau2

    def update(self, action: np.ndarray, reward: float):
        """REINFORCE-style mean update: nudge mu towards actions that earned
        above-baseline reward (Eq. 51's eta * dr/dtau gradient, approximated)."""
        grad = action  # d(log N(a|mu,sigma^2))/dmu = (a - mu)/sigma^2 ~ proportional to `action` deviation
        self.mu = self.mu + self.cfg.lr * reward * (action - self.mu)


class HITLRLAgent:
    """Drives the adaptive threshold-tuning loop described in Algorithm 2 /
    Section 3.6, operating on pre-computed trust scores + expert labels
    (i.e., it does not require the visual backbone at training time)."""

    def __init__(self, cfg: HITLRLConfig | None = None):
        self.cfg = cfg or HITLRLConfig()
        self.policy = ThresholdPolicy(self.cfg)
        self.reward_history: list[float] = []
        self.tau_history: list[tuple[float, float]] = []

    def _decision_module(self) -> AdaptiveDecisionModule:
        return AdaptiveDecisionModule(DecisionThresholds(self.policy.tau1, self.policy.tau2))

    def step(self, trust: torch.Tensor, auth_prob: torch.Tensor, expert_label: torch.Tensor) -> float:
        """One RL training step over a batch of trust scores / expert feedback.

        Args:
            trust: T(I) per sample (B,)
            auth_prob: A(I) per sample (B,) used for verification-error delta
            expert_label: ground-truth/expert binary label (1=authentic) (B,)
        Returns:
            mean batch reward (float)
        """
        action = self.policy.sample_action()
        self.policy.apply(action)
        dm = self._decision_module()

        decisions = dm.decide(trust)
        human_cost = torch.where(
            decisions == Decision.HUMAN_REVIEW,
            torch.full_like(trust, self.cfg.human_cost),
            torch.zeros_like(trust),
        )
        rewards = dm.reward(
            decisions, expert_label, auth_prob, human_cost,
            lambda1=self.cfg.lambda1, lambda2=self.cfg.lambda2, lambda3=self.cfg.lambda3,
        )
        mean_reward = float(rewards.mean().item())

        self.policy.update(action, mean_reward)
        self.reward_history.append(mean_reward)
        self.tau_history.append((self.policy.tau1, self.policy.tau2))
        return mean_reward

    def run(self, trust: torch.Tensor, auth_prob: torch.Tensor, expert_label: torch.Tensor,
            episodes: int | None = None, verbose: bool = True) -> dict:
        """Runs the full HITL-RL loop by resampling mini-batches across episodes."""
        episodes = episodes or self.cfg.episodes
        n = trust.shape[0]
        rng = np.random.default_rng(self.cfg.seed)

        for ep in range(episodes):
            idx = rng.choice(n, size=min(self.cfg.batch_size, n), replace=False)
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            r = self.step(trust[idx_t], auth_prob[idx_t], expert_label[idx_t])
            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                print(f"[HITL-RL] episode {ep+1}/{episodes}  reward={r:.4f}  "
                      f"tau1={self.policy.tau1:.3f}  tau2={self.policy.tau2:.3f}")

        final_dm = self._decision_module()
        final_decisions = final_dm.decide(trust)
        return {
            "tau1": self.policy.tau1,
            "tau2": self.policy.tau2,
            "reward_history": self.reward_history,
            "tau_history": self.tau_history,
            "human_review_rate": final_dm.human_review_rate(final_decisions),
            "decision_distribution": final_dm.decision_distribution(final_decisions),
            "final_decisions": final_decisions,
        }
