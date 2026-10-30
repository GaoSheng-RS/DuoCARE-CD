"""Correspondence-aware detector 训练入口。

在 VSCode 里直接 Run 即可;路径和参数写在文件末尾的 __main__ 里。
训练协议:20000 次更新,AdamW(lr 1e-4,wd 0.05),5000/10000 次后 lr 减半,FP32,
每 1000 次在 val 上评估,按 val pooled IoU@0.5 保存 best.pth,结束后在 test 上评估。
"""
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "correspondence_aware"

import math

from common import train as runner
from common.models.dinov2_lora import DinoV2LoRA
from .model import CorrespondenceAwareDetector
from .objective import batch_objective


def validate_config(config):
    if config["detector"] != "correspondence_aware":
        raise ValueError("本入口只接收 correspondence_aware 的配置")
    for key in ("dice_weight", "affinity_weight"):
        if not math.isfinite(config["loss"][key]) or config["loss"][key] < 0:
            raise ValueError(f"loss.{key} 必须为有限非负数")


def build_model(config, *, checkpoint=None):
    settings = config["model"]
    encoder = DinoV2LoRA(arch=settings["arch"], checkpoint=checkpoint or config["checkpoint"],
                         rank=settings["rank"], alpha=settings["alpha"])
    return CorrespondenceAwareDetector(encoder, mid=settings["mid"], vit_input=settings["vit_input"],
                                       heads=settings["heads"])


def source_identity():
    return runner.source_identity(Path(__file__).resolve().parent)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = root / "config.json"
    data_root = "/media/gs/sd10/RSE/split_v4"
    # 数据目录:<data_root>/{train,val,test}/{T1,T2,GT}/,文件名 <站点>_<编号>.png
    backbone = "/home/gs/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    # DINOv2 ViT-L/14 官方权重(无 register token)
    output = root.parent / "runs" / "correspondence_aware"
    device = "cuda:0"
    preflight = 0
    # 设为 2 只跑 2 个 batch 的前向/反向检查,不更新参数
    resume = None
    # 续训时填 output 下的 latest.pth
    runner.launch(sys.modules[__name__], config=config, data_root=data_root, backbone=backbone,
                  output=output, device=device, preflight=preflight, resume=resume)
