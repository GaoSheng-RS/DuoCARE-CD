"""Registration-equivariant detector(论文 3.2.3)。

在 DINOv2 baseline 的四层 lateral 上并联零初始化支路,输入两个坐标系下的对齐差分:
    D(2) = |T_s G1 − G2|(T1 重采样到 T2 坐标系,式 7),D(1) = |T_−s G2 − G1|(式 8)
T_s 是按位移 s 的双线性重采样(grid_sample);s 由影像估计后作为外部输入
(2 维,原图像素单位,见 estimate_shifts.py),网络内没有可学习的位移估计。
零初始化保证训练起点与 baseline 一致;s=0 时两条支路退化为原始差分。
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from common.models.dinov2_baseline import DinoV2Baseline, ImagePrediction


def translate_feature(feature: Tensor, shift_grid: Tensor) -> Tensor:
    """feature [B,C,H,W];shift_grid [B,2] 为 (sx, sy),单位 = 本网格像素。
    输出 out(y,x) = feature(y + sy, x + sx),双线性,边界复制。"""
    if feature.ndim != 4 or shift_grid.shape != (feature.shape[0], 2):
        raise ValueError("translate_feature 需要 [B,C,H,W] 特征和 [B,2] 位移")
    B, _, H, W = feature.shape
    xs = torch.arange(W, device=feature.device, dtype=feature.dtype)
    ys = torch.arange(H, device=feature.device, dtype=feature.dtype)
    gx = (xs[None, :] + 0.5 + shift_grid[:, 0:1]) * (2.0 / W) - 1.0
    gy = (ys[None, :] + 0.5 + shift_grid[:, 1:2]) * (2.0 / H) - 1.0
    grid = torch.stack((gx[:, None, :].expand(B, H, W), gy[:, :, None].expand(B, H, W)), dim=-1)
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=False)


class RegistrationEquivariantDetector(DinoV2Baseline):
    def __init__(self, encoder: nn.Module, mid: int = 256, vit_input: int = 448, heads: int = 8):
        super().__init__(encoder=encoder, mid=mid, vit_input=vit_input, heads=heads)
        self.align = nn.ModuleList(nn.Conv2d(2 * encoder.dim, mid, 1) for _ in range(4))
        for conv in self.align:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            # 新支路在全部父模块之后创建,父模块初始化与 baseline 相同;零初始化 ⇒ 初始输出 == baseline

    def pyramid_features_shift(self, f1, f2, shift_px: Tensor, image_size: int):
        f1 = self._rescale(f1)
        f2 = self._rescale(f2)
        if any(a.shape != b.shape for a, b in zip(f1, f2)):
            raise ValueError("两期编码特征形状必须一致")
        lateral = []
        for layer, extra, a, b in zip(self.lateral, self.align, f1, f2):
            grid = a.shape[-1]
            s = shift_px.to(a.dtype) * (grid / float(image_size))
            forward = (translate_feature(a, s) - b).abs()
            backward = (translate_feature(b, -s) - a).abs()
            lateral.append(layer((a - b).abs()) + extra(torch.cat((forward, backward), dim=1)))
        merged = [lateral[3]]
        for index in (2, 1, 0):
            merged.append(lateral[index] + F.interpolate(
                merged[-1], size=lateral[index].shape[-2:], mode="bilinear", align_corners=False))
        merged.reverse()
        return [smooth(feature) for smooth, feature in zip(self.smooth, merged)]

    def predict_with_tokens(self, x1: Tensor, x2: Tensor, shift: Tensor | None = None) -> ImagePrediction:
        if shift is None:
            raise ValueError("前向必须提供位移输入(无位移请传全零)")
        if x1.shape[-1] != x1.shape[-2]:
            raise ValueError("位移换算假设方形输入")
        f1, f2 = self.encode_pair(x1, x2)
        pyramid = self.pyramid_features_shift(f1, f2, shift, image_size=x1.shape[-1])
        tokens = self.context.embed(f1[-1], f2[-1])
        grid = tuple(f1[-1].shape[-2:])
        delta = self.context(tokens).transpose(1, 2).reshape(x1.shape[0], -1, *grid)
        if pyramid[1].shape != delta.shape:
            raise ValueError("上下文残差必须与 FPN 第二层处于同一 token 网格")
        pyramid[1] = pyramid[1] + delta
        logits = self.classify(pyramid, output_size=tuple(x1.shape[-2:]))
        return ImagePrediction(logits=logits, tokens=tokens, grid=grid)

    def forward(self, x1: Tensor, x2: Tensor, shift: Tensor | None = None) -> Tensor:
        return self.predict_with_tokens(x1, x2, shift).logits
