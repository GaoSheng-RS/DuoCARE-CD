"""Correspondence-aware detector 的训练目标(论文 3.2.4,式 9–10):
L_c = L_ce + L_dice + λ_aff · L_aff。

亲和度损失:在 token 网格上给每个位置标注它所在的 GT 变化连通域
(该域占 token ≥50%),纯未变化 token 记为背景;
同一变化域内的位置对为正(z=1),变化位置与背景位置的对为负(z=0),
对 head 平均的 softmax 前亲和度做 BCE。抽样用独立 CPU generator,不扰动增强 RNG。
"""
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from common.objective import ObjectiveOutput, soft_dice_loss, validate_labels

EIGHT = np.ones((3, 3), dtype=np.int32)


def component_maps(labels: torch.Tensor) -> list[np.ndarray]:
    """每个样本的 GT 变化连通域标号图(0=非变化),8 连通。"""
    maps = []
    for sample in labels.detach().cpu().numpy():
        comp, _ = ndimage.label(sample == 1, structure=EIGHT)
        maps.append(comp)
    return maps


def token_component_labels(labels: torch.Tensor, comps: list[np.ndarray], grid: int):
    """token 级标注:连通域 id(≥1)、-1=纯未变化背景、0=其他(不参与抽样)。"""
    out = []
    stride = labels.shape[-1] // grid
    valid = F.avg_pool2d((labels != 255).float().unsqueeze(1), stride)[:, 0]
    fg = F.avg_pool2d((labels == 1).float().unsqueeze(1), stride)[:, 0]
    for b, comp in enumerate(comps):
        token = np.zeros((grid, grid), dtype=np.int64)
        full = valid[b].cpu().numpy() >= 0.999
        token[full & (fg[b].cpu().numpy() == 0)] = -1
        if comp.max() > 0:
            blocks = comp.reshape(grid, stride, grid, stride).transpose(0, 2, 1, 3).reshape(grid, grid, -1)
            for i, j in zip(*np.nonzero(fg[b].cpu().numpy() >= 0.5)):
                values, counts = np.unique(blocks[i, j][blocks[i, j] > 0], return_counts=True)
                if counts.size and counts.max() >= 0.5 * stride * stride:
                    token[i, j] = int(values[np.argmax(counts)])
        out.append(token.ravel())
    return out


def affinity_loss(sim: torch.Tensor, token_labels: list[np.ndarray], generator,
                  pairs_per_class: int = 256) -> torch.Tensor:
    """sim [B,N,N] 为 softmax 前亲和度;正对=同一变化域,负对=变化 token × 背景 token。"""
    losses = []
    for b, token in enumerate(token_labels):
        comp_ids = [np.flatnonzero(token == c) for c in np.unique(token[token > 0])]
        unchanged = np.flatnonzero(token == -1)
        positives, negatives = [], []
        for members in comp_ids:
            if members.size < 2:
                continue
            count = min(pairs_per_class // max(len(comp_ids), 1) + 1, members.size * (members.size - 1))
            left = members[torch.randint(members.size, (count,), generator=generator).numpy()]
            right = members[torch.randint(members.size, (count,), generator=generator).numpy()]
            keep = left != right
            positives.append(np.stack((left[keep], right[keep]), axis=1))
            if unchanged.size:
                neg_right = unchanged[torch.randint(unchanged.size, (int(keep.sum()),), generator=generator).numpy()]
                negatives.append(np.stack((left[keep], neg_right), axis=1))
        if not positives:
            continue
        neg_arrays = []
        if negatives:
            neg_arrays.append(np.concatenate(negatives)[:pairs_per_class])
        pos = np.concatenate(positives)[:pairs_per_class]
        pairs = [pos] + neg_arrays
        targets = [np.ones(len(pos))] + [np.zeros(len(a)) for a in neg_arrays]
        all_pairs = np.concatenate(pairs)
        if len(all_pairs) == 0:
            continue
            # 小连通域抽样碰撞可能让正负对同时为空;空张量上的 BCE 均值是 NaN
        index = torch.from_numpy(all_pairs).to(sim.device)
        target = torch.from_numpy(np.concatenate(targets)).float().to(sim.device)
        logits = sim[b][index[:, 0], index[:, 1]]
        losses.append(F.binary_cross_entropy_with_logits(logits, target))
    if not losses:
        return sim.sum() * 0.0
    return torch.stack(losses).mean()


def batch_objective(model, batch, config, *, generator=None):
    x1, x2, labels, names = batch
    del names
    validate_labels(labels)
    weights = config["loss"]
    logits, sim = model.forward_train(x1, x2)
    cross_entropy = (F.cross_entropy(logits, labels, ignore_index=255)
                     if bool((labels != 255).any()) else logits.sum() * 0.0)
    dice = soft_dice_loss(logits, labels)
    total = cross_entropy + weights["dice_weight"] * dice
    diagnostics = {"ce_loss": float(cross_entropy.detach()), "dice_loss": float(dice.detach())}
    if generator is None:
        raise ValueError("亲和度抽样必须传入独立 CPU generator")
    grid = int(sim.shape[-1] ** 0.5)
    affinity = affinity_loss(sim, token_component_labels(labels, component_maps(labels), grid), generator)
    total = total + weights["affinity_weight"] * affinity
    diagnostics["affinity_loss"] = float(affinity.detach())
    return ObjectiveOutput(total, cross_entropy, affinity, logits, diagnostics)
