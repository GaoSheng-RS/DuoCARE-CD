"""独立的 DINOv2 + 普通 QKV LoRA；两期影像之间没有交换分支。

只接受本地官方无 register-token 的 ViT-B/14、ViT-L/14 权重。
构造时必须给出 checkpoint，不下载权重，也不允许误用随机冻结骨干。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ARCHITECTURES = {
    "vitb14": {"dim": 768, "depth": 12, "heads": 12},
    "vitl14": {"dim": 1024, "depth": 24, "heads": 16},
}


class LoRALinear(nn.Module):
    """普通 LoRA：W(x) + (alpha / rank) B(A(x))，只作用于当前输入。"""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float | None = None):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("LoRA rank 必须是正整数")
        if alpha is not None and (not math.isfinite(alpha) or alpha <= 0):
            raise ValueError("LoRA alpha 必须是有限正数")
        self.base = base.requires_grad_(False)
        self.scale = float(rank if alpha is None else alpha) / rank
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.B(self.A(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int = 14):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(out.transpose(1, 2).reshape(batch, length, dim))


class Mlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoV2LoRA(nn.Module):
    """冻结 DINOv2，训练每个 block 的普通 QKV LoRA。

    输出四个 block 经最终 LayerNorm 后的 patch 特征，均保持原 token 网格。
    ViT-L 默认下标为 (5, 11, 17, 23)，ViT-B 为 (2, 5, 8, 11)。
    输入必须已由 dataCD 完成归一化；本类不改变图像颜色或数值范围。
    """

    def __init__(
        self,
        arch: str = "vitl14",
        checkpoint: str | Path | None = None,
        rank: int = 16,
        alpha: float | None = None,
        out_indices: Sequence[int] | None = None,
    ):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"不支持的 DINOv2 架构 {arch!r}；可选 {tuple(ARCHITECTURES)}")
        if checkpoint is None:
            raise ValueError("必须指定本地官方 DINOv2 checkpoint，禁止随机冻结骨干")
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到本地 DINOv2 权重：{checkpoint}")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("LoRA rank 必须是正整数")
        if alpha is not None and (not math.isfinite(alpha) or alpha <= 0):
            raise ValueError("LoRA alpha 必须是有限正数")
        cfg = ARCHITECTURES[arch]
        self.arch = arch
        self.dim = cfg["dim"]
        self.depth = cfg["depth"]
        self.patch_size = 14
        self.out_indices = tuple(out_indices) if out_indices is not None else tuple(
            self.depth * n // 4 - 1 for n in (1, 2, 3, 4)
        )
        if (
            len(self.out_indices) != 4
            or any(not isinstance(i, int) or isinstance(i, bool) for i in self.out_indices)
            or tuple(sorted(set(self.out_indices))) != self.out_indices
            or self.out_indices[0] < 0
            or self.out_indices[-1] >= self.depth
        ):
            raise ValueError("out_indices 必须包含四个严格递增且在骨干范围内的 block 下标")
        self.patch_embed = PatchEmbed(self.dim, self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, self.dim))
        self.blocks = nn.ModuleList(Block(self.dim, cfg["heads"]) for _ in range(self.depth))
        self.norm = nn.LayerNorm(self.dim, eps=1e-6)

        # 先严格检查整个官方骨干，再插入 LoRA，避免遗漏权重被静默忽略。
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
            raise ValueError("checkpoint 必须是官方 DINOv2 原始 state_dict")
        state = {key: value for key, value in state.items() if key != "mask_token"}
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        for block in self.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)

    def _prepare(self, x: Tensor) -> tuple[Tensor, int, int]:
        if x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError("DINOv2 输入必须是浮点 [B, 3, H, W] 张量")
        batch, _, height, width = x.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError("DINOv2 输入尺寸不能小于 patch_size=14")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("DINOv2 输入高宽必须是 14 的整数倍")
        gh, gw = height // self.patch_size, width // self.patch_size
        tokens = self.patch_embed(x)
        # 官方无 register 模型使用 offset=0.1 的 bicubic 位置插值。
        if gh == 37 and gw == 37:
            patch_pos = self.pos_embed[:, 1:]
        else:
            grid = self.pos_embed[:, 1:].float().reshape(1, 37, 37, self.dim).permute(0, 3, 1, 2)
            grid = F.interpolate(
                grid,
                scale_factor=((gh + 0.1) / 37, (gw + 0.1) / 37),
                mode="bicubic",
                align_corners=False,
                antialias=False,
            )
            if grid.shape[-2:] != (gh, gw):
                raise RuntimeError("DINOv2 位置插值产生了错误的 token 网格")
            patch_pos = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, self.dim)
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(batch, -1, -1)
        tokens = torch.cat((cls.to(tokens.dtype), tokens + patch_pos.to(tokens.dtype)), dim=1)
        return tokens, gh, gw

    def forward(self, x: Tensor) -> list[Tensor]:
        tokens, gh, gw = self._prepare(x)
        features = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.out_indices:
                patch_tokens = self.norm(tokens)[:, 1:]
                features.append(patch_tokens.transpose(1, 2).reshape(x.shape[0], self.dim, gh, gw))
            if index == self.out_indices[-1]:
                break
        return features

    def forward_pair(self, x1: Tensor, x2: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """两期共享同一编码器，各自独立前向；不存在跨时相 token 交换。"""
        if x1.shape != x2.shape or x1.device != x2.device or x1.dtype != x2.dtype:
            raise ValueError("两期影像必须具有相同形状、设备和 dtype")
        return self(x1), self(x2)

    def n_trainable(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
