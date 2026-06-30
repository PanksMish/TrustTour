"""
F3Net (F3N) baseline — frequency-aware forgery detection network.

Re-implements the core idea of F3Net: combine an RGB CNN branch with an
explicit frequency-domain (DCT-based) branch that highlights blocking /
compression / generative-upsampling frequency artifacts, fused before the
classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dct_basis(n: int = 8) -> torch.Tensor:
    """Generates an (n*n, n, n) DCT-II basis used as a fixed conv kernel bank,
    approximating block-wise DCT frequency decomposition."""
    basis = torch.zeros(n * n, n, n)
    for u in range(n):
        for v in range(n):
            for x in range(n):
                for y in range(n):
                    cu = (1 / n) ** 0.5 if u == 0 else (2 / n) ** 0.5
                    cv = (1 / n) ** 0.5 if v == 0 else (2 / n) ** 0.5
                    basis[u * n + v, x, y] = (
                        cu * cv *
                        torch.cos(torch.tensor((2 * x + 1) * u * 3.14159265 / (2 * n))) *
                        torch.cos(torch.tensor((2 * y + 1) * v * 3.14159265 / (2 * n)))
                    )
    return basis


class FrequencyBranch(nn.Module):
    """Block-wise DCT frequency decomposition + learnable frequency filtering,
    mirroring F3Net's Frequency-aware Decomposition (FAD) idea."""

    def __init__(self, block_size: int = 8, out_channels: int = 64):
        super().__init__()
        basis = _dct_basis(block_size).unsqueeze(1)  # (n*n, 1, n, n)
        self.register_buffer("dct_kernels", basis)
        self.block_size = block_size
        n2 = block_size * block_size
        self.freq_gate = nn.Parameter(torch.ones(n2))
        self.proj = nn.Sequential(
            nn.Conv2d(n2 * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        outs = []
        for ch in range(c):
            xc = x[:, ch:ch + 1]
            coeffs = F.conv2d(xc, self.dct_kernels, stride=self.block_size)
            coeffs = coeffs * self.freq_gate.view(1, -1, 1, 1)
            coeffs = F.interpolate(coeffs, size=(h, w), mode="nearest")
            outs.append(coeffs)
        freq = torch.cat(outs, dim=1)  # (B, n*n*3, H, W)
        return self.proj(freq)


class RGBBranch(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class F3Net(nn.Module):
    def __init__(self, num_classes: int = 1, base_channels: int = 64):
        super().__init__()
        self.rgb_branch = RGBBranch(base_channels)
        self.freq_branch = FrequencyBranch(block_size=8, out_channels=base_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        self.feature_dim = 1024

    def forward(self, x, return_features: bool = False):
        rgb = self.rgb_branch(x)
        freq = self.freq_branch(x)
        fused = torch.cat([rgb, freq], dim=1)
        fused = self.fusion(fused)
        feat = self.pool(fused).flatten(1)
        logit = self.fc(feat).squeeze(-1)
        prob = torch.sigmoid(logit)
        if return_features:
            return prob, logit, feat
        return prob, logit


def build_f3net(**kwargs) -> F3Net:
    return F3Net(**kwargs)
