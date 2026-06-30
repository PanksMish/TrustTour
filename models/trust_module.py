"""
Trust Assessment Module (Section 4.1, Eq. 39-45).

Given a tourism information object I = (V, M, S, U):
    A(I)  = DS-SViT authenticity confidence                      Eq. (42)
    M_c   = f_m(M)  metadata consistency score                   Eq. (43)
    S_c   = f_s(S)  source credibility score                     Eq. (43)
    T(I)  = (w1*A + w2*M_c + w3*S_c) / (w1+w2+w3)                  Eq. (44)
    w_i   = softmax(c_i)  confidence-driven adaptive weights      Eq. (45)

This module wraps a DS-SViT backbone and small metadata / source encoders to
produce the final tourism trust score T(I) in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrustModuleConfig:
    feature_dim: int = 384
    metadata_dim: int = 8       # e.g. geotag validity, timestamp delta, EXIF consistency, device fingerprint match...
    source_dim: int = 6         # one-hot / embedding of source type (DMO, hotel, OTA, social media, UGC, unknown)
    hidden_dim: int = 64


class MetadataEncoder(nn.Module):
    """f_m(M) -> M_c in [0,1]. Also emits a scalar confidence c_M used in Eq.45."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(hidden_dim // 2, 1)
        self.conf_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, m: torch.Tensor):
        h = self.net(m)
        score = torch.sigmoid(self.score_head(h)).squeeze(-1)
        conf = self.conf_head(h).squeeze(-1)
        return score, conf


class SourceEncoder(nn.Module):
    """f_s(S) -> S_c in [0,1]. S is typically a categorical source-type vector
    (official DMO, hotel/OTA, travel agency, social media, UGC) optionally
    combined with a historical reliability prior."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(hidden_dim // 2, 1)
        self.conf_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, s: torch.Tensor):
        h = self.net(s)
        score = torch.sigmoid(self.score_head(h)).squeeze(-1)
        conf = self.conf_head(h).squeeze(-1)
        return score, conf


class TrustAssessmentModule(nn.Module):
    """Full Trust Assessment Module: visual + metadata + source -> T(I)."""

    def __init__(self, backbone: nn.Module, cfg: TrustModuleConfig | None = None):
        super().__init__()
        self.cfg = cfg or TrustModuleConfig()
        self.backbone = backbone  # DS-SViT; must expose forward(x, return_features=True)
        self.metadata_encoder = MetadataEncoder(self.cfg.metadata_dim, self.cfg.hidden_dim)
        self.source_encoder = SourceEncoder(self.cfg.source_dim, self.cfg.hidden_dim)

        # Visual-evidence confidence head: derives c_A (Eq. 45) from fused features,
        # e.g. via (negative) predictive entropy -> higher confidence = more peaked A(I).
        self.visual_conf_head = nn.Linear(self.cfg.feature_dim, 1)

    @staticmethod
    def entropy_confidence(a: torch.Tensor) -> torch.Tensor:
        """Confidence derived from inverse Shannon entropy of A(I), Eq. (8).
        Maximum entropy (A=0.5) -> low confidence; A near 0 or 1 -> high confidence."""
        eps = 1e-8
        a = a.clamp(eps, 1 - eps)
        h = -(a * torch.log(a) + (1 - a) * torch.log(1 - a))
        h_max = torch.log(torch.tensor(2.0, device=a.device))
        return 1.0 - (h / h_max)  # in [0,1]

    def forward(self, image: torch.Tensor, metadata: torch.Tensor, source: torch.Tensor):
        """
        Args:
            image:    (B,3,H,W)
            metadata: (B, metadata_dim)
            source:   (B, source_dim)
        Returns dict with A(I), M_c, S_c, weights (w1,w2,w3), and T(I).
        """
        auth_prob, logit, fused = self.backbone(image, return_features=True)  # Eq. 40-42
        m_c, c_m_learned = self.metadata_encoder(metadata)                     # Eq. 43
        s_c, c_s_learned = self.source_encoder(source)                        # Eq. 43

        # Confidence signals feeding the adaptive softmax weighting, Eq. (45).
        c_a = self.entropy_confidence(auth_prob) + self.visual_conf_head(fused).squeeze(-1)
        c_m = c_m_learned
        c_s = c_s_learned

        conf_stack = torch.stack([c_a, c_m, c_s], dim=-1)  # (B,3)
        weights = F.softmax(conf_stack, dim=-1)            # Eq. (45): w_i = softmax(c_i)
        w1, w2, w3 = weights[:, 0], weights[:, 1], weights[:, 2]

        # Eq. (44): weighted aggregation (already normalized since softmax sums to 1)
        trust = w1 * auth_prob + w2 * m_c + w3 * s_c
        trust = trust.clamp(0.0, 1.0)

        return {
            "auth_prob": auth_prob,      # A(I)
            "auth_logit": logit,
            "metadata_score": m_c,       # M_c
            "source_score": s_c,         # S_c
            "weights": weights,          # (w1, w2, w3)
            "trust": trust,              # T(I)
            "fused_features": fused,
        }
