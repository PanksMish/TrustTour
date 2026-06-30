"""Loss functions used for training. Matches Table 3 (Binary Cross-Entropy
Loss) and adds an optional trust-consistency regularizer used when training
the full Trust Assessment Module end-to-end."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label)


def trust_consistency_loss(trust: torch.Tensor, label: torch.Tensor, weight: float = 0.3) -> torch.Tensor:
    """Encourages the fused trust score T(I) to also track the ground-truth
    authenticity label, in addition to the primary authenticity BCE loss on
    A(I) -- keeps metadata/source weighting from drifting away from the
    underlying authenticity signal during early training."""
    target = label.clamp(1e-4, 1 - 1e-4)
    trust = trust.clamp(1e-4, 1 - 1e-4)
    return weight * F.binary_cross_entropy(trust, target)


class TrustTourLoss(nn.Module):
    def __init__(self, trust_weight: float = 0.3):
        super().__init__()
        self.trust_weight = trust_weight

    def forward(self, outputs: dict, label: torch.Tensor) -> dict:
        auth_logit = outputs["auth_logit"]
        trust = outputs["trust"]
        l_auth = bce_loss(auth_logit, label)
        l_trust = trust_consistency_loss(trust, label, self.trust_weight)
        total = l_auth + l_trust
        return {"loss": total, "loss_auth": l_auth, "loss_trust": l_trust}
