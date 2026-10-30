"""在 WCH-CD 测试集上复现论文结果:两个检测器单独及集成的 Precision / Recall / F1 / IoU。

指标对全部测试像素汇总计算(pooled),阈值 0.5,不做任何后处理。
路径写在文件末尾的 __main__ 里,直接运行。
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from predict import ROOT, load_detectors
from registration_equivariant.data import ShiftedCD, load_shift_table


def scores(tp, fp, fn):
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall), "iou": tp / (tp + fp + fn)}


def evaluate(data_root, backbone, split="test", device="cuda:0", batch_size=8, threshold=0.5):
    config = json.loads((ROOT / "registration_equivariant" / "config.json").read_text(encoding="utf-8"))
    tables = load_shift_table(ROOT / "registration_equivariant" / config["shift_table"])
    dataset = ShiftedCD(str(Path(data_root) / split), mode="eval", return_name=True, shift_table=tables[split])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    correspondence_aware, registration_equivariant = load_detectors(backbone, device)
    counts = {key: [0, 0, 0] for key in ("correspondence_aware", "registration_equivariant", "DuoCARE-CD")}
    with torch.inference_mode():
        for x1, x2, labels, _, shift in loader:
            x1, x2, shift, labels = x1.to(device), x2.to(device), shift.to(device), labels.to(device)
            p_c = correspondence_aware(x1, x2).float().softmax(dim=1)[:, 1]
            p_r = registration_equivariant(x1, x2, shift).float().softmax(dim=1)[:, 1]
            valid, target = labels != 255, labels == 1
            for key, p in (("correspondence_aware", p_c), ("registration_equivariant", p_r),
                           ("DuoCARE-CD", (p_c + p_r) / 2)):
                pred = p >= threshold
                counts[key][0] += int((pred & target & valid).sum())
                counts[key][1] += int((pred & ~target & valid).sum())
                counts[key][2] += int((~pred & target & valid).sum())
    return {key: scores(*value) for key, value in counts.items()}


if __name__ == "__main__":
    data_root = "/media/gs/sd10/RSE/split_v4"
    backbone = "/home/gs/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    device = "cuda:0"
    results = evaluate(data_root, backbone, device=device)
    for key, value in results.items():
        print(f"{key:26s} " + "  ".join(f"{m} {100 * v:.2f}" for m, v in value.items()))
