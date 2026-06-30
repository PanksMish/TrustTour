"""
AIDE baseline — hybrid semantic (CLIP-style) + frequency-aware AI-generated
image detector, following the spirit of "AIDE" (2025): combine a frozen/
fine-tuned CLIP visual encoder with a lightweight high-frequency patch
statistics branch (patch-wise FFT energy), then fuse for classification.

If `open_clip` or `clip` packages are unavailable, falls back to a torchvision
ResNet50 as the semantic backbone so the model is runnable out-of-the-box.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False

import torchvision.models as tvm


class CLIPSemanticEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", freeze: bool = True):
        super().__init__()
        if _HAS_OPEN_CLIP:
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.encoder = model.visual
            self.feature_dim = model.visual.output_dim
            self._is_clip = True
        else:
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            try:
                backbone = tvm.resnet50(weights=weights)
            except Exception:
                # Falls back to random init if pretrained weights can't be
                # downloaded (e.g. no internet access in this environment).
                backbone = tvm.resnet50(weights=None)
            backbone.fc = nn.Identity()
            self.encoder = backbone
            self.feature_dim = 2048
            self._is_clip = False

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.encoder(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        return feat.float()


class FrequencyStatsBranch(nn.Module):
    """Patch-wise FFT magnitude statistics as a cheap high-frequency cue."""

    def __init__(self, patch_size: int = 16, out_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(patch_size * patch_size, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        b, _, h, w = gray.shape
        ps = self.patch_size
        gray = F.adaptive_avg_pool2d(gray, (h // ps * ps, w // ps * ps))
        patches = gray.unfold(2, ps, ps).unfold(3, ps, ps)  # (B,1,nh,nw,ps,ps)
        patches = patches.contiguous().view(b, -1, ps, ps)
        fft = torch.fft.fft2(patches)
        mag = torch.log1p(torch.abs(fft)).view(b, patches.shape[1], -1)
        energy = mag.mean(dim=1)  # average magnitude spectrum across patches: (B, ps*ps)
        return self.mlp(energy)


class AIDE(nn.Module):
    """AIDE: hybrid semantic (CLIP) + frequency-statistics AI-image detector."""

    def __init__(self, num_classes: int = 1, freeze_clip: bool = True):
        super().__init__()
        self.semantic = CLIPSemanticEncoder(freeze=freeze_clip)
        self.freq = FrequencyStatsBranch(out_dim=128)
        fusion_dim = self.semantic.feature_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self.feature_dim = fusion_dim

    def forward(self, x, return_features: bool = False):
        sem = self.semantic(x)
        freq = self.freq(x)
        feat = torch.cat([sem, freq], dim=-1)
        logit = self.classifier(feat).squeeze(-1)
        prob = torch.sigmoid(logit)
        if return_features:
            return prob, logit, feat
        return prob, logit


def build_aide(**kwargs) -> AIDE:
    return AIDE(**kwargs)
