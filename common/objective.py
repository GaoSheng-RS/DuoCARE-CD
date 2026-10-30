"""两个检测器共用的训练目标工具:输出结构、标签校验与前景 Dice 损失。"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ObjectiveOutput:
    loss: Tensor
    segmentation_loss: Tensor
    auxiliary_loss: Tensor
    logits: Tensor
    diagnostics: dict[str, int | float]


def validate_labels(labels: Tensor) -> None:
    if labels.ndim != 3 or labels.dtype != torch.long:
        raise ValueError("labels 必须是 [B,H,W] 的 torch.long")
    if not bool(((labels == 0) | (labels == 1) | (labels == 255)).all()):
        raise ValueError("增强后的标签只能为 0/1/255;255 为 ignore")


def soft_dice_loss(logits: Tensor, labels: Tensor) -> Tensor:
    probability = logits.softmax(dim=1)[:, 1]
    valid = (labels != 255).float()
    target = (labels == 1).float()
    p = (probability * valid).flatten(1)
    t = (target * valid).flatten(1)
    dice = (2 * (p * t).sum(1) + 1.0) / (p.sum(1) + t.sum(1) + 1.0)
    return 1.0 - dice.mean()
    # 前景 Dice,只在有效(非 255)像素上计算
