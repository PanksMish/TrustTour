"""
Adaptive Decision Module (Section 4.2, Eq. 46-51).

    R(I) = 1 - T(I)                                                Eq. (46)
    D(I) = Verified       if R(I) <  tau1
         = Human Review   if tau1 <= R(I) < tau2
         = Rejected        if R(I) >= tau2                          Eq. (47)
    delta = |H(I) - A(I)|                                           Eq. (49)
    r = lambda1 * 1[D == H] - lambda2 * delta - lambda3 * C_H        Eq. (50)
    tau_i <- tau_i + eta * dr/dtau_i                                Eq. (51)  (handled in rl/hitl_rl.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class Decision(IntEnum):
    VERIFIED = 0
    HUMAN_REVIEW = 1
    REJECTED = 2

    def __str__(self) -> str:
        return {0: "Verified", 1: "Human Review", 2: "Rejected"}[int(self)]


@dataclass
class DecisionThresholds:
    tau1: float = 0.35
    tau2: float = 0.65

    def validate(self):
        assert 0.0 < self.tau1 < self.tau2 < 1.0, "Require 0 < tau1 < tau2 < 1"


class AdaptiveDecisionModule:
    """Stateless functional module operating on trust scores (kept as a plain
    class, not nn.Module, since tau1/tau2 are updated by the RL agent rather
    than via backprop)."""

    def __init__(self, thresholds: DecisionThresholds | None = None, impact_factor: float = 1.0):
        self.thresholds = thresholds or DecisionThresholds()
        self.thresholds.validate()
        self.omega = impact_factor  # severity/impact factor, Eq. (17), omega >= 1

    def risk(self, trust: torch.Tensor) -> torch.Tensor:
        """R(I) = omega * (1 - T(I)), Eq. (17) generalizes Eq. (46) with omega=1."""
        return (self.omega * (1.0 - trust)).clamp(0.0, 1.0)

    def decide(self, trust: torch.Tensor) -> torch.Tensor:
        """Vectorized threshold decision, Eq. (47). Returns integer tensor of Decision values."""
        r = self.risk(trust)
        tau1, tau2 = self.thresholds.tau1, self.thresholds.tau2
        decisions = torch.full_like(r, fill_value=Decision.HUMAN_REVIEW, dtype=torch.long)
        decisions = torch.where(r < tau1, torch.full_like(decisions, Decision.VERIFIED), decisions)
        decisions = torch.where(r >= tau2, torch.full_like(decisions, Decision.REJECTED), decisions)
        return decisions

    def verification_error(self, expert_label: torch.Tensor, auth_prob: torch.Tensor) -> torch.Tensor:
        """delta = |H(I) - A(I)|, Eq. (49)."""
        return (expert_label - auth_prob).abs()

    def reward(
        self,
        decision: torch.Tensor,
        expert_label: torch.Tensor,
        auth_prob: torch.Tensor,
        human_cost: torch.Tensor | float,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 0.5,
    ) -> torch.Tensor:
        """r = lambda1 * 1[D == H] - lambda2 * delta - lambda3 * C_H, Eq. (50).

        `expert_label` here is the binarized ground-truth/expert decision used
        as the "agreement" target (1 = authentic/verified-correct, 0 = not).
        For samples without human review, human_cost should be 0.
        """
        delta = self.verification_error(expert_label, auth_prob)
        agree = (decision == Decision.VERIFIED).float() * expert_label + \
                (decision == Decision.REJECTED).float() * (1 - expert_label)
        if not torch.is_tensor(human_cost):
            human_cost = torch.full_like(delta, float(human_cost))
        r = lambda1 * agree - lambda2 * delta - lambda3 * human_cost
        return r

    def decision_distribution(self, decisions: torch.Tensor) -> dict:
        n = decisions.numel()
        out = {}
        for d in Decision:
            out[str(d)] = float((decisions == d).sum().item()) / max(n, 1)
        return out

    def human_review_rate(self, decisions: torch.Tensor) -> float:
        """HRR = N_HR / N, Eq. (66)."""
        return float((decisions == Decision.HUMAN_REVIEW).float().mean().item())
