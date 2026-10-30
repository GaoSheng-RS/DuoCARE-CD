"""DINOv2 baseline:LoRA 编码 + 绝对差分 FPN + 图内自注意力上下文。

配准等变检测器在此基础上加入对齐差分支路。
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .baseline import DinoLoRAFPN


class SharedImageAttention(nn.Module):
    """最深层双期特征 [F1,F2,|F1-F2|] 上的图内自注意力,输出零初始化残差。"""

    def __init__(self, feature_dim: int, dim: int = 256, heads: int = 8):
        super().__init__()
        if min(feature_dim, dim, heads) <= 0 or dim % heads:
            raise ValueError("feature_dim/dim/heads 必须为正，dim 必须能被 heads 整除")
        self.dim, self.heads = dim, heads
        self.pair = nn.Sequential(
            nn.Conv1d(3 * feature_dim, dim, 1), nn.GELU(), nn.Conv1d(dim, dim, 1)
        )
        self.norm_q, self.norm_k = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def embed(self, f1: Tensor, f2: Tensor) -> Tensor:
        if f1.shape != f2.shape or f1.ndim != 4:
            raise ValueError("双时相特征必须为同形状 [B,C,H,W]")
        pair = torch.cat((f1, f2, (f1 - f2).abs()), dim=1).flatten(2)
        return self.pair(pair).transpose(1, 2)

    def _heads(self, tokens: Tensor) -> Tensor:
        return tokens.reshape(tokens.shape[0], tokens.shape[1], self.heads, -1).transpose(1, 2)

    def project_qk(self, query: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        return self._heads(self.q(self.norm_q(query))), self._heads(self.k(self.norm_k(reference)))

    def attention_weights(self, query: Tensor, reference: Tensor) -> Tensor:
        """[B, heads, Nq, Nr] 注意力权重。"""
        q, k = self.project_qk(query, reference)
        return (q @ k.transpose(-1, -2) / (self.dim // self.heads) ** 0.5).softmax(-1)

    def forward(self, tokens: Tensor) -> Tensor:
        q, k = self.project_qk(tokens, tokens)
        v = self._heads(self.v(self.norm_k(tokens)))
        context = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        context = context.transpose(1, 2).reshape_as(tokens)
        return self.out(context)


@dataclass
class ImagePrediction:
    logits: Tensor
    tokens: Tensor
    grid: tuple[int, int]


class DinoV2Baseline(DinoLoRAFPN):
    """DINOv2 baseline:逐位置比较两期特征,图内注意力残差加到 FPN 第二层。"""

    def __init__(self, encoder: nn.Module, mid: int = 256, vit_input: int = 448, heads: int = 8):
        super().__init__(encoder=encoder, mid=mid, vit_input=vit_input)
        self.context = SharedImageAttention(encoder.dim, dim=mid, heads=heads)

    def predict_with_tokens(self, x1: Tensor, x2: Tensor) -> ImagePrediction:
        f1, f2 = self.encode_pair(x1, x2)
        pyramid = self.pyramid_features(f1, f2)
        tokens = self.context.embed(f1[-1], f2[-1])
        grid = tuple(f1[-1].shape[-2:])
        delta = self.context(tokens).transpose(1, 2).reshape(x1.shape[0], -1, *grid)
        if pyramid[1].shape != delta.shape:
            raise ValueError("上下文残差必须与 FPN 第二层处于同一 token 网格")
        pyramid[1] = pyramid[1] + delta
        logits = self.classify(pyramid, output_size=tuple(x1.shape[-2:]))
        return ImagePrediction(logits=logits, tokens=tokens, grid=grid)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.predict_with_tokens(x1, x2).logits
