"""Atrous Spatial Pyramid Pooling for multi-scale lesion context."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPP(nn.Module):
    """ASPP with dilation rates from the paper (1, 6, 12, 18/24)."""

    def __init__(self, in_channels: int, out_channels: int = 256, rates: list[int] | None = None):
        super().__init__()
        rates = rates or [1, 6, 12, 18]
        self.branches = nn.ModuleList()
        # 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        for r in rates:
            if r == 1:
                self.branches.append(ASPPConv(in_channels, out_channels, dilation=1))
            else:
                self.branches.append(ASPPConv(in_channels, out_channels, dilation=r))

        # No BatchNorm after global pool — BN is unstable for batch size 1
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        n_branches = len(self.branches) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))
