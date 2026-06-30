"""
Swin Transformer (Swin) baseline — hierarchical shifted-window self-attention.

Uses `timm`'s swin_tiny_patch4_window7_224 if timm is installed (recommended,
matches the paper's setting of ImageNet-pretrained backbones); otherwise
falls back to a lightweight from-scratch hierarchical window-attention
transformer so the repository still runs without timm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class TimmSwinWrapper(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True,
                 model_name: str = "swin_tiny_patch4_window7_224"):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        feat = self.backbone(x)
        logit = self.fc(feat).squeeze(-1)
        prob = torch.sigmoid(logit)
        if return_features:
            return prob, logit, feat
        return prob, logit


# ---------------------- lightweight fallback implementation ----------------------

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        # x: (num_windows*B, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        windows = window_partition(x, ws).view(-1, ws * ws, C)
        attn_windows = self.attn(windows).view(-1, ws, ws, C)
        x = window_reverse(attn_windows, ws, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :H, :W, :]
        x = x.reshape(B, N, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        x0, x1, x2, x3 = x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x), H // 2, W // 2


class SwinTinyScratch(nn.Module):
    """Compact 4-stage Swin-style network (fallback when timm unavailable)."""

    def __init__(self, num_classes: int = 1, img_size: int = 224, patch_size: int = 4,
                 embed_dim: int = 96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size: int = 7):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.H = self.W = img_size // patch_size
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i, depth in enumerate(depths):
            blocks = nn.ModuleList([
                SwinBlock(dims[i], num_heads[i], window_size) for _ in range(depth)
            ])
            self.stages.append(blocks)
            self.merges.append(PatchMerging(dims[i]) if i < len(depths) - 1 else None)
        self.norm = nn.LayerNorm(dims[-1])
        self.fc = nn.Linear(dims[-1], num_classes)
        self.feature_dim = dims[-1]

    def forward(self, x, return_features: bool = False):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blocks, merge in zip(self.stages, self.merges):
            for blk in blocks:
                x = blk(x, H, W)
            if merge is not None:
                x, H, W = merge(x, H, W)
        x = self.norm(x)
        feat = x.mean(dim=1)
        logit = self.fc(feat).squeeze(-1)
        prob = torch.sigmoid(logit)
        if return_features:
            return prob, logit, feat
        return prob, logit


def build_swin(num_classes: int = 1, pretrained: bool = True, **kwargs):
    if _HAS_TIMM:
        return TimmSwinWrapper(num_classes=num_classes, pretrained=pretrained)
    return SwinTinyScratch(num_classes=num_classes, **kwargs)
