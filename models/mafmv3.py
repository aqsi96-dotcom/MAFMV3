"""MAFMv3: Multi-Scale Attention-Based Feature Fusion MobileNetv3.

MobileNetV3 backbone enhanced with CBAM + ASPP.
Three streams are fused for spine lesion classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .aspp import ASPP
from .cbam import CBAM


class MobileNetV3ASPPAttention(nn.Module):
    """Single-stream MobileNetV3 with CBAM on stages and ASPP on the deepest map."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        aspp_rates: list[int] | None = None,
        out_channels: int = 256,
    ):
        super().__init__()
        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = tvm.mobilenet_v3_large(weights=weights)

        if in_channels != 3:
            old = net.features[0][0]
            new = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            net.features[0][0] = new

        self.stage1 = nn.Sequential(*net.features[0:3])
        self.stage2 = nn.Sequential(*net.features[3:7])
        self.stage3 = nn.Sequential(*net.features[7:13])
        self.stage4 = nn.Sequential(*net.features[13:])

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            f1 = self.stage1(dummy)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
            channels = [f1.shape[1], f2.shape[1], f3.shape[1], f4.shape[1]]

        self.cbams = nn.ModuleList([CBAM(ch) for ch in channels])
        self.aspp = ASPP(channels[-1], out_channels=out_channels, rates=aspp_rates)
        self.side_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, out_channels // 4, 1, bias=False),
                    nn.BatchNorm2d(out_channels // 4),
                    nn.ReLU(inplace=True),
                )
                for ch in channels[:-1]
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + (out_channels // 4) * (len(channels) - 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        attended = [cbam(f) for cbam, f in zip(self.cbams, [f1, f2, f3, f4])]
        deep = self.aspp(attended[-1])
        sides = []
        for proj, f in zip(self.side_projs, attended[:-1]):
            s = F.interpolate(proj(f), size=deep.shape[-2:], mode="bilinear", align_corners=False)
            sides.append(s)
        return self.fuse(torch.cat([deep, *sides], dim=1))


class MAFMv3(nn.Module):
    """Full MAFMv3 classifier with three MobileNetV3+CBAM+ASPP streams."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        pretrained: bool = True,
        aspp_rates: list[int] | None = None,
        feature_channels: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        rates = aspp_rates or [1, 6, 12, 18]
        self.stream1 = MobileNetV3ASPPAttention(in_channels, pretrained, rates, feature_channels)
        self.stream2 = MobileNetV3ASPPAttention(in_channels, pretrained, rates, feature_channels)
        self.stream3 = MobileNetV3ASPPAttention(in_channels, pretrained, rates, feature_channels)

        self.head = nn.Sequential(
            nn.Conv2d(feature_channels * 3, feature_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_channels // 2, num_classes),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        x3: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x2 is None:
            x2 = x1
        if x3 is None:
            x3 = x1
        f1 = self.stream1(x1)
        f2 = self.stream2(x2)
        f3 = self.stream3(x3)
        h, w = f1.shape[-2:]
        f2 = F.interpolate(f2, size=(h, w), mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=(h, w), mode="bilinear", align_corners=False)
        return self.classifier(self.head(torch.cat([f1, f2, f3], dim=1)))
