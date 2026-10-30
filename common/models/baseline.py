"""DINOv2 baseline 的底座:DINOv2 + LoRA 编码 + 绝对差分 FPN 解码。

完整 baseline 见 dinov2_baseline.py(另加图内注意力上下文)。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DinoLoRAFPN(nn.Module):
    """接收已构建的编码器,提供 FPN 底座;完整 baseline 见 DinoV2Baseline。

    正常使用传入 DinoV2LoRA。
    编码器需要提供 dim 和 forward_pair(x1, x2)，每期返回四层同网格特征。
    输入图像由 dataCD 预处理；内部只执行送入 ViT 所需的双线性缩放。
    """

    def __init__(self, encoder: nn.Module, mid: int = 256, vit_input: int = 448):
        super().__init__()
        if not isinstance(encoder, nn.Module) or not hasattr(encoder, "dim") or not callable(
            getattr(encoder, "forward_pair", None)
        ):
            raise TypeError("encoder 必须提供 dim 和 forward_pair(x1, x2)")
        if not isinstance(mid, int) or isinstance(mid, bool) or mid <= 0:
            raise ValueError("mid 必须是正整数")
        if not isinstance(vit_input, int) or isinstance(vit_input, bool) or vit_input < 56 or vit_input % 56:
            raise ValueError("vit_input 必须是 56 的正整数倍，以形成可四倍池化的 patch 网格")
        if not isinstance(encoder.dim, int) or encoder.dim <= 0:
            raise ValueError("encoder.dim 必须是正整数")
        self.encoder = encoder
        self.mid = mid
        self.vit_input = vit_input
        self.lateral = nn.ModuleList(nn.Conv2d(encoder.dim, mid, 1) for _ in range(4))
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True))
            for _ in range(4)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(4 * mid, mid, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, 2, 1)
        )

    def encode_pair(self, x1: Tensor, x2: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """把两期影像缩放到 vit_input，再独立编码；不进行额外归一化。"""
        if x1.ndim != 4 or x1.shape[1] != 3 or not x1.is_floating_point():
            raise ValueError("输入必须是浮点 [B, 3, H, W] 张量")
        if x1.shape != x2.shape or x1.device != x2.device or x1.dtype != x2.dtype:
            raise ValueError("两期影像必须具有相同形状、设备和 dtype")
        if x1.shape[0] < 1 or min(x1.shape[-2:]) < 1:
            raise ValueError("输入 batch 和空间尺寸必须非空")
        size = (self.vit_input, self.vit_input)
        x1 = F.interpolate(x1, size=size, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=size, mode="bilinear", align_corners=False)
        return self.encoder.forward_pair(x1, x2)

    def _validate_features(self, features: Sequence[Tensor]) -> None:
        if len(features) != 4:
            raise ValueError("FPN 必须接收四层编码特征")
        shape = features[0].shape
        if len(shape) != 4 or shape[1] != self.encoder.dim:
            raise ValueError("编码特征必须是 [B, encoder.dim, H, W]")
        if min(shape[-2:]) < 4 or shape[-2] % 4 or shape[-1] % 4:
            raise ValueError("编码特征网格高宽必须是 4 的正整数倍")
        if any(feature.shape != shape for feature in features):
            raise ValueError("四层 ViT 特征必须具有相同形状")

    def _rescale(self, features: Sequence[Tensor]) -> list[Tensor]:
        self._validate_features(features)
        return [
            F.interpolate(features[0], scale_factor=2, mode="bilinear", align_corners=False),
            features[1],
            F.avg_pool2d(features[2], 2),
            F.avg_pool2d(features[3], 4),
        ]

    def pyramid_features(self, f1: Sequence[Tensor], f2: Sequence[Tensor]) -> list[Tensor]:
        """四层 token 特征重采样，再 absdiff、横向映射、自顶向下相加与平滑。"""
        f1 = self._rescale(f1)
        f2 = self._rescale(f2)
        if any(a.shape != b.shape for a, b in zip(f1, f2)):
            raise ValueError("两期编码特征形状必须一致")
        lateral = [layer(torch.abs(a - b)) for layer, a, b in zip(self.lateral, f1, f2)]
        merged = [lateral[3]]
        for index in (2, 1, 0):
            merged.append(lateral[index] + F.interpolate(
                merged[-1], size=lateral[index].shape[-2:], mode="bilinear", align_corners=False
            ))
        merged.reverse()
        return [smooth(feature) for smooth, feature in zip(self.smooth, merged)]

    def classify(self, features: Sequence[Tensor], output_size: tuple[int, int]) -> Tensor:
        """四尺度上采样拼接，输出原影像大小的二分类 logits。"""
        if len(features) != 4 or any(feature.ndim != 4 or feature.shape[1] != self.mid for feature in features):
            raise ValueError("分类器必须接收四层 [B, mid, H, W] 特征")
        if len(output_size) != 2 or min(output_size) <= 0:
            raise ValueError("output_size 必须是正的 (H, W)")
        size = features[0].shape[-2:]
        fused = torch.cat([
            feature if feature.shape[-2:] == size else F.interpolate(
                feature, size=size, mode="bilinear", align_corners=False
            )
            for feature in features
        ], dim=1)
        logits = self.classifier(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        f1, f2 = self.encode_pair(x1, x2)
        return self.classify(self.pyramid_features(f1, f2), tuple(x1.shape[-2:]))

    def n_trainable(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
