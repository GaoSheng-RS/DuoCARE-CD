"""Correspondence-aware detector(论文 3.2.2)。

逐位置比较两期特征得到初始变化证据 X(式 1);
跨期校验:每个位置在另一期全图中搜索最佳外观匹配,匹配分经门控修正证据(式 3–4);
外观引导传播:按双期外观亲和度在位置之间交换校验后的证据(式 5–6),再上采样解码。
外观描述子取 DINOv2 第 12、24 层(APPEARANCE_LEVELS),不含时相差分。
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from common.models.dinov2_baseline import SharedImageAttention

APPEARANCE_LEVELS = (1, 3)
# 编码器四层输出 {6,12,18,24} 中的第 12 与第 24 层


def _heads(tokens: Tensor, heads: int) -> Tensor:
    return tokens.reshape(tokens.shape[0], tokens.shape[1], heads, -1).transpose(1, 2)


def _merge(tokens: Tensor) -> Tensor:
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], -1)


class CrossTemporalVerification(nn.Module):
    """跨期校验(式 3–4):双向全图最佳匹配分 → MLP 门控 → 证据的通道级修正。"""

    def __init__(self, dim: int, epoch_appearance_dim: int, heads: int = 8):
        super().__init__()
        if dim % heads:
            raise ValueError("dim 必须能被 heads 整除")
        self.heads = heads
        self.appearance = nn.Conv2d(epoch_appearance_dim, dim, 1)
        # 两期共用同一投影,匹配在同一空间里进行
        self.norm = nn.LayerNorm(dim)
        self.q, self.k = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm_x = nn.LayerNorm(dim)
        self.hidden = nn.Linear(2 * heads, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # 零初始化:训练起点等价于不做校验

    def match_scores(self, source: Tensor, target: Tensor) -> Tensor:
        """source 每个位置在 target 全图中的最佳匹配分 [B, heads, N]。"""
        q = _heads(self.q(self.norm(source)), self.heads)
        k = _heads(self.k(self.norm(target)), self.heads)
        return (q @ k.transpose(-1, -2) / (q.shape[-1]) ** 0.5).max(dim=-1).values
        # 搜索覆盖整幅特征图,匹配可以出现在原位置之外,不要求两期逐位置对齐

    def forward(self, x: Tensor, a1_cat: Tensor, a2_cat: Tensor) -> Tensor:
        t1 = self.appearance(a1_cat).flatten(2).transpose(1, 2)
        t2 = self.appearance(a2_cat).flatten(2).transpose(1, 2)
        m21 = self.match_scores(t2, t1)
        m12 = self.match_scores(t1, t2)
        match = torch.cat((m21, m12), dim=1).transpose(1, 2)
        # 两个方向、所有 head 的匹配分 m_i
        s = self.out(F.gelu(self.hidden(match)))
        e = self.norm_x(x.flatten(2).transpose(1, 2))
        delta = (s * e).transpose(1, 2).reshape_as(x)
        return x + delta
        # 式 4:X~ = X + γ(m) ⊙ LN(X)


class AppearanceGuidedPropagation(nn.Module):
    """外观引导传播(式 5–6):Q/K 来自双期外观,V 来自校验后的证据。"""

    def __init__(self, dim: int, appearance_dim: int, heads: int = 8):
        super().__init__()
        if dim % heads:
            raise ValueError("dim 必须能被 heads 整除")
        self.heads = heads
        self.appearance = nn.Conv2d(appearance_dim, dim, 1)
        self.norm_q, self.norm_k = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # 零初始化:训练起点等价于不传播

    def forward(self, x: Tensor, appearance_cat: Tensor, return_sim: bool = False):
        a = self.appearance(appearance_cat).flatten(2).transpose(1, 2)
        e = x.flatten(2).transpose(1, 2)
        q = _heads(self.q(self.norm_q(a)), self.heads)
        k = _heads(self.k(self.norm_k(a)), self.heads)
        v = _heads(self.v(self.norm_v(e)), self.heads)
        sim = q @ k.transpose(-1, -2) / (q.shape[-1]) ** 0.5
        # 亲和度 S_ij^h 只由外观计算,证据不参与
        read = _merge(sim.softmax(-1) @ v)
        delta = self.out(read).transpose(1, 2).reshape_as(x)
        if return_sim:
            return x + delta, sim.mean(dim=1)
            # 亲和度损失用 head 平均的 softmax 前亲和度 [B, N, N]
        return x + delta


class CorrespondenceAwareDetector(nn.Module):

    def __init__(self, encoder: nn.Module, *, mid: int = 256, vit_input: int = 448, heads: int = 8):
        super().__init__()
        if not isinstance(encoder, nn.Module) or not hasattr(encoder, "dim") or not callable(
            getattr(encoder, "forward_pair", None)
        ):
            raise TypeError("encoder 必须提供 dim 和 forward_pair(x1, x2)")
        if not isinstance(mid, int) or isinstance(mid, bool) or mid <= 0 or mid % 4:
            raise ValueError("mid 必须是正的 4 的倍数")
        if not isinstance(vit_input, int) or isinstance(vit_input, bool) or vit_input < 56 or vit_input % 14:
            raise ValueError("vit_input 必须是 14 的正整数倍")
        self.encoder = encoder
        self.mid = mid
        self.vit_input = vit_input
        self.lateral = nn.ModuleList(nn.Conv2d(3 * encoder.dim, mid, 1) for _ in range(4))
        self.fuse = nn.Sequential(nn.Conv2d(4 * mid, mid, 1), nn.ReLU(inplace=True))
        self.context = SharedImageAttention(encoder.dim, dim=mid, heads=heads)
        self.smooth = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(mid, mid // 2, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(mid // 2, mid // 4, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(mid // 4, 2, 1)
        self.gate = CrossTemporalVerification(mid, len(APPEARANCE_LEVELS) * encoder.dim, heads=heads)
        self.propagate = AppearanceGuidedPropagation(mid, 2 * len(APPEARANCE_LEVELS) * encoder.dim, heads=heads)
        # 模块创建顺序决定随机初始化序列,与论文训练时一致,勿调换

    def encode_pair(self, x1: Tensor, x2: Tensor):
        if x1.ndim != 4 or x1.shape[1] != 3 or not x1.is_floating_point():
            raise ValueError("输入必须是浮点 [B, 3, H, W] 张量")
        if x1.shape != x2.shape or x1.device != x2.device or x1.dtype != x2.dtype:
            raise ValueError("两期影像必须具有相同形状、设备和 dtype")
        size = (self.vit_input, self.vit_input)
        x1 = F.interpolate(x1, size=size, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=size, mode="bilinear", align_corners=False)
        return self.encoder.forward_pair(x1, x2)

    def evidence_and_appearance(self, x1: Tensor, x2: Tensor):
        """初始变化证据 X(式 1)与两期外观描述子 U^t(式 2)。"""
        f1, f2 = self.encode_pair(x1, x2)
        if len(f1) != 4 or any(a.shape != b.shape for a, b in zip(f1, f2)):
            raise ValueError("需要四层形状一致的双期编码特征")
        lateral = [
            layer(torch.cat((a, b, (a - b).abs()), dim=1))
            for layer, a, b in zip(self.lateral, f1, f2)
        ]
        x = self.fuse(torch.cat(lateral, dim=1))
        tokens = self.context.embed(f1[-1], f2[-1])
        grid = tuple(f1[-1].shape[-2:])
        x = x + self.context(tokens).transpose(1, 2).reshape(x1.shape[0], -1, *grid)
        x = self.smooth(x)
        a1_cat = torch.cat([f1[lvl] for lvl in APPEARANCE_LEVELS], dim=1)
        a2_cat = torch.cat([f2[lvl] for lvl in APPEARANCE_LEVELS], dim=1)
        return x, (a1_cat, a2_cat)

    def decode(self, x: Tensor, output_size) -> Tensor:
        x = self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return F.interpolate(self.head(x), size=output_size, mode="bilinear", align_corners=False)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x, (a1, a2) = self.evidence_and_appearance(x1, x2)
        x = self.gate(x, a1, a2)
        return self.decode(self.propagate(x, torch.cat((a1, a2), dim=1)), tuple(x1.shape[-2:]))

    def forward_train(self, x1: Tensor, x2: Tensor):
        """训练用前向:另返回传播亲和度,供亲和度损失使用。"""
        x, (a1, a2) = self.evidence_and_appearance(x1, x2)
        x = self.gate(x, a1, a2)
        x, sim = self.propagate(x, torch.cat((a1, a2), dim=1), return_sim=True)
        return self.decode(x, tuple(x1.shape[-2:])), sim
