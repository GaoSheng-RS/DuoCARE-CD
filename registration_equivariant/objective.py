"""Registration-equivariant detector 的训练目标:像素交叉熵(ignore 255),前向多一个位移输入。
诊断里记录 batch 的位移统计。"""
import torch.nn.functional as F

from common.objective import ObjectiveOutput, validate_labels


def batch_objective(model, batch, config, *, generator=None):
    x1, x2, labels, names, shift = batch
    validate_labels(labels)
    logits = model(x1, x2, shift)
    segmentation = (F.cross_entropy(logits, labels, ignore_index=255)
                    if bool((labels != 255).any()) else logits.sum() * 0.0)
    zero = logits.sum() * 0.0
    magnitude = shift.norm(dim=1)
    diagnostics = {"shift_mean_px": float(magnitude.mean()),
                   "shift_nonzero_fraction": float((magnitude > 0).float().mean())}
    return ObjectiveOutput(segmentation, segmentation, zero, logits, diagnostics)
