"""
Dual-Stream Scaled Vision Transformer (DS-SViT).

Implements:
    F_s = Phi_s(V)        semantic stream (RGB ViT)            Eq. (40)
    F_f = Phi_f(V)        forensic stream (SRM residual ViT)    Eq. (40)
    F   = Psi(F_s, F_f)   cross-attention fusion                Eq. (41)
    A(I) = sigmoid(W_a F + b_a)                                  Eq. (42)

The "Scaled" in DS-SViT refers to a learnable per-stream temperature/scale
applied before fusion, allowing the network to up- or down-weight the
semantic vs. forensic evidence depending on confidence (mirrors the adaptive
weighting idea of Eq. 45 at the feature level).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.srm import ForensicStem


@dataclass
class DSSViTConfig:
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    forensic_in_channels: int = 64
    fusion_heads: int = 6


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, img_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float, drop_rate: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """Psi(F_s, F_f): aligns semantic consistency with forensic manipulation
    evidence via bidirectional cross-attention, then merges via a gated sum."""

    def __init__(self, embed_dim: int, num_heads: int, drop_rate: float = 0.1):
        super().__init__()
        self.s2f = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.f2s = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm_s = nn.LayerNorm(embed_dim)
        self.norm_f = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> torch.Tensor:
        # f_s, f_f: (B, N, D) token sequences (cls + patches)
        s_attn, _ = self.s2f(f_s, f_f, f_f)
        f_attn, _ = self.f2s(f_f, f_s, f_s)
        s_fused = self.norm_s(f_s + s_attn)
        f_fused = self.norm_f(f_f + f_attn)

        s_cls = s_fused[:, 0]
        f_cls = f_fused[:, 0]
        g = self.gate(torch.cat([s_cls, f_cls], dim=-1))
        fused = g * s_cls + (1 - g) * f_cls
        return self.out_norm(fused)  # (B, D)


class DSSViT(nn.Module):
    """Dual-Stream Scaled Vision Transformer."""

    def __init__(self, cfg: DSSViTConfig | None = None):
        super().__init__()
        cfg = cfg or DSSViTConfig()
        self.cfg = cfg

        # Semantic stream (RGB)
        self.semantic_patch = PatchEmbed(3, cfg.embed_dim, cfg.patch_size, cfg.img_size)
        self.semantic_encoder = TransformerEncoder(
            cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.mlp_ratio, cfg.drop_rate
        )

        # Forensic stream (SRM residuals)
        self.forensic_stem = ForensicStem(out_channels=cfg.forensic_in_channels)
        self.forensic_patch = PatchEmbed(
            cfg.forensic_in_channels, cfg.embed_dim, cfg.patch_size, cfg.img_size
        )
        self.forensic_encoder = TransformerEncoder(
            cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.mlp_ratio, cfg.drop_rate
        )

        # Learnable per-stream scale ("Scaled" ViT)
        self.semantic_scale = nn.Parameter(torch.tensor(1.0))
        self.forensic_scale = nn.Parameter(torch.tensor(1.0))

        # Cross-attention fusion: Eq. (41)
        self.fusion = CrossAttentionFusion(cfg.embed_dim, cfg.fusion_heads, cfg.drop_rate)

        # Authenticity head: Eq. (42)  A(I) = sigmoid(W_a F + b_a)
        self.auth_head = nn.Linear(cfg.embed_dim, 1)

        self.feature_dim = cfg.embed_dim

    def encode_semantic(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.semantic_patch(x)
        tok = self.semantic_encoder(tok)
        return tok * self.semantic_scale

    def encode_forensic(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.forensic_stem(x)
        tok = self.forensic_patch(stem)
        tok = self.forensic_encoder(tok)
        return tok * self.forensic_scale

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f_s = self.encode_semantic(x)   # (B, N+1, D)
        f_f = self.encode_forensic(x)   # (B, N+1, D)
        fused = self.fusion(f_s, f_f)   # (B, D)  -- F in Eq. (41)
        logit = self.auth_head(fused).squeeze(-1)  # pre-sigmoid
        auth_prob = torch.sigmoid(logit)  # A(I), Eq. (42)

        if return_features:
            return auth_prob, logit, fused
        return auth_prob, logit


def build_ds_svit(img_size: int = 224, **kwargs) -> DSSViT:
    cfg = DSSViTConfig(img_size=img_size, **kwargs)
    return DSSViT(cfg)


if __name__ == "__main__":
    model = build_ds_svit()
    x = torch.randn(2, 3, 224, 224)
    p, logit = model(x)
    print("auth prob:", p.shape, "logit:", logit.shape)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {n_params:.2f}M")
