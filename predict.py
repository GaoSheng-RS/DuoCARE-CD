"""DuoCARE-CD 推理:两个检测器分别预测变化概率,取平均后按 0.5 二值化(论文 3.2.5,式 11)。

用法:在文件末尾 __main__ 里填 T1/T2 影像路径(256×256,0.5 m)、位移和输出路径,直接运行。
位移 shift = (dx, dy),原图像素单位,可用 registration_equivariant/estimate_shifts.py 估计;
未知时传 (0, 0),此时 registration-equivariant detector 的对齐支路退化为原始差分。
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image
import torch

from common.data.transforms import Compose, NormalizePair, ToTensorPair
from common.train import load_trainable_state
from correspondence_aware.train import build_model as build_correspondence_aware
from registration_equivariant.train import build_model as build_registration_equivariant

ROOT = Path(__file__).resolve().parent


def load_detectors(backbone, device):
    """返回 (correspondence-aware, registration-equivariant) 两个已加载权重的检测器。"""
    detectors = []
    for name, build in (("correspondence_aware", build_correspondence_aware),
                        ("registration_equivariant", build_registration_equivariant)):
        config = json.loads((ROOT / name / "config.json").read_text(encoding="utf-8"))
        model = build(config, checkpoint=backbone).float()
        state = torch.load(ROOT / "checkpoints" / f"{name}.pth", map_location="cpu", weights_only=False)
        load_trainable_state(model, state["trainable_and_buffers"])
        detectors.append(model.to(device).eval())
    return tuple(detectors)


def ensemble_probability(correspondence_aware, registration_equivariant, x1, x2, shift):
    """x1/x2 为归一化后的 [B,3,H,W],shift 为 [B,2];返回平均后的变化概率 [B,H,W]。"""
    with torch.inference_mode():
        p_c = correspondence_aware(x1, x2).float().softmax(dim=1)[:, 1]
        p_r = registration_equivariant(x1, x2, shift).float().softmax(dim=1)[:, 1]
    return (p_c + p_r) / 2


def predict_pair(detectors, t1_path, t2_path, shift=(0.0, 0.0), device="cuda:0", threshold=0.5):
    transform = Compose([ToTensorPair(), NormalizePair(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])
    t1 = np.array(Image.open(t1_path).convert("RGB"))
    t2 = np.array(Image.open(t2_path).convert("RGB"))
    x1, x2, _ = transform(t1, t2, np.zeros(t1.shape[:2], dtype=np.int64))
    s = torch.tensor([list(shift)], dtype=torch.float32, device=device)
    probability = ensemble_probability(*detectors, x1[None].to(device), x2[None].to(device), s)[0].cpu().numpy()
    return probability >= threshold, probability


if __name__ == "__main__":
    backbone = "/home/gs/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    # DINOv2 ViT-L/14 官方权重路径
    t1_path = "/media/gs/sd10/RSE/split_v4/test/T1/1006P3_1.png"
    t2_path = "/media/gs/sd10/RSE/split_v4/test/T2/1006P3_1.png"
    shift = (0.0, 0.0)
    # 该影像对的配准位移 (dx, dy)
    out_path = "prediction.png"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detectors = load_detectors(backbone, device)
    mask, probability = predict_pair(detectors, t1_path, t2_path, shift=shift, device=device)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
    print(f"saved {out_path},变化像素 {int(mask.sum())}")
