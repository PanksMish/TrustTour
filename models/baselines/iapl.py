"""
IAPL baseline — Image-Adaptive Prompt Learning on a Vision Foundation Model
(VFM). Following the spirit of prompt-tuning approaches for cross-generator
generalization: a frozen ViT backbone is augmented with a small set of
learnable prompt tokens whose values are *modulated per-image* by a
lightweight hypernetwork conditioned on a global image embedding, then
prepended to the patch token sequence before the (frozen) transformer
blocks, with only the prompts + classification head trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class ImageAdaptivePromptGenerator(nn.Module):
    """Generates per-image prompt tokens from a coarse global descriptor."""

    def __init__(self, embed_dim: int, num_prompts: int = 8):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.base_prompts = nn.Parameter(torch.randn(num_prompts, embed_dim) * 0.02)
        self.hyper = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_prompts * embed_dim),
        )

    def forward(self, global_desc: torch.Tensor) -> torch.Tensor:
        b = global_desc.shape[0]
        delta = self.hyper(global_desc).view(b, self.num_prompts, self.embed_dim)
        prompts = self.base_prompts.unsqueeze(0) + 0.1 * delta
        return prompts  # (B, num_prompts, D)


class _ScratchViTBackbone(nn.Module):
    """Small from-scratch ViT used when timm isn't available, exposing
    patch-embedding + transformer blocks so prompts can be injected."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        self.patch = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                            batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        for p in self.parameters():
            p.requires_grad = False  # frozen VFM backbone

    def patch_tokens(self, x):
        tok = self.patch(x).flatten(2).transpose(1, 2)
        tok = tok + self.pos_embed
        return tok

    def forward_with_prompts(self, patch_tokens, prompts):
        x = torch.cat([prompts, patch_tokens], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, : prompts.shape[1]].mean(dim=1)  # pooled prompt-token representation


class IAPL(nn.Module):
    """Image-Adaptive Prompt Learning detector on a frozen VFM backbone."""

    def __init__(self, num_classes: int = 1, num_prompts: int = 8, embed_dim: int = 384):
        super().__init__()
        self.backbone = _ScratchViTBackbone(embed_dim=embed_dim)
        self.global_proj = nn.Linear(embed_dim, embed_dim)
        self.prompt_gen = ImageAdaptivePromptGenerator(embed_dim, num_prompts)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes)
        )
        self.feature_dim = embed_dim

    def forward(self, x, return_features: bool = False):
        patch_tok = self.backbone.patch_tokens(x)
        global_desc = self.global_proj(patch_tok.mean(dim=1))
        prompts = self.prompt_gen(global_desc)
        feat = self.backbone.forward_with_prompts(patch_tok, prompts)
        logit = self.head(feat).squeeze(-1)
        prob = torch.sigmoid(logit)
        if return_features:
            return prob, logit, feat
        return prob, logit


def build_iapl(**kwargs) -> IAPL:
    return IAPL(**kwargs)
