"""
Spatial Rich Model (SRM) residual extraction.

Implements a fixed-weight convolutional layer using classical SRM high-pass
filter kernels, used to expose high-frequency forensic artifacts (upsampling /
transposed-convolution fingerprints) that generative models leave behind.
This feeds the forensic stream of the Dual-Stream Scaled Vision Transformer
(DS-SViT), Eq. (40): F_f = Phi_f(V).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Classical 5x5 SRM kernels (normalized). These are the standard filters used
# in image-forensics literature (Fridrich & Kodovsky) to expose pixel
# co-occurrence residuals invisible to plain RGB classifiers.
def _srm_kernels() -> torch.Tensor:
    k1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0

    k2 = torch.tensor([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=torch.float32) / 12.0

    k3 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 2.0

    return torch.stack([k1, k2, k3], dim=0)  # (3, 5, 5)


class SRMConv(nn.Module):
    """Fixed (non-trainable) SRM residual extraction layer.

    Applies the 3 SRM kernels independently to each of the R, G, B channels,
    producing a 9-channel high-frequency residual map highlighting generative
    upsampling artifacts and local pixel-correlation anomalies.
    """

    def __init__(self, trainable: bool = False):
        super().__init__()
        kernels = _srm_kernels()  # (3, 5, 5)
        # Build a (9, 1, 5, 5) depthwise-style kernel: 3 SRM filters x 3 channels
        weight = kernels.unsqueeze(1).repeat(3, 1, 1, 1)  # (9,1,5,5)
        self.weight = nn.Parameter(weight, requires_grad=trainable)
        self.register_buffer("_groups_channels", torch.tensor(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> repeat each RGB channel for 3 SRM filters via grouped conv
        b, c, h, w = x.shape
        assert c == 3, "SRMConv expects 3-channel RGB input"
        # Expand channels: interleave so each of the 3 SRM kernels sees each channel
        x_rep = x.repeat_interleave(3, dim=1)  # (B, 9, H, W) order: R,R,R,G,G,G,B,B,B
        out = F.conv2d(x_rep, self.weight, padding=2, groups=9)
        out = torch.clamp(out, -3.0, 3.0)
        return out  # (B, 9, H, W)


class ForensicStem(nn.Module):
    """Projects SRM residuals into a CNN-token-friendly feature map before the
    forensic transformer stream patch-embeds it."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.srm = SRMConv(trainable=False)
        self.proj = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.srm(x)
        return self.proj(r)
