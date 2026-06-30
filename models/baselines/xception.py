"""
Xception (XCP) baseline — classical depthwise-separable CNN forensic baseline.
Uses torchvision/timm if available, otherwise falls back to a compact
from-scratch Xception-style implementation so the repo has zero hard
dependency on a specific timm version.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
        self.skip_bn = nn.BatchNorm2d(out_ch)
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.MaxPool2d(3, stride, 1),
        )

    def forward(self, x):
        skip = self.skip_bn(self.skip(x))
        return self.block(x) + skip


class Xception(nn.Module):
    """Compact Xception-style network for binary authenticity classification."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            XceptionBlock(64, 128),
            XceptionBlock(128, 256),
            XceptionBlock(256, 728),
        )
        self.middle = nn.Sequential(*[
            nn.Sequential(
                SeparableConv2d(728, 728), nn.BatchNorm2d(728), nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        self.exit = nn.Sequential(
            XceptionBlock(728, 1024),
            SeparableConv2d(1024, 1536), nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048), nn.BatchNorm2d(2048), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        self.feature_dim = 2048

    def forward(self, x, return_features: bool = False):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.middle(x)
        x = self.exit(x)
        feat = self.pool(x).flatten(1)
        logit = self.fc(feat).squeeze(-1)
        prob = torch.sigmoid(logit)
        if return_features:
            return prob, logit, feat
        return prob, logit


def build_xception(**kwargs) -> Xception:
    return Xception(**kwargs)
