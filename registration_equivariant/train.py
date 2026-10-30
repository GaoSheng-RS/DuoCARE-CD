"""Registration-equivariant detector 训练入口。

在 VSCode 里直接 Run 即可;路径和参数写在文件末尾的 __main__ 里。
训练前先用 estimate_shifts.py 生成位移表(仓库已附带 WCH-CD 的位移表)。
训练协议与 correspondence-aware detector 相同;batch 多一个随增强同步变换的位移。
"""
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "registration_equivariant"

import torch

from common import train as runner
from common.metrics import StreamingMetrics, _preserve_evaluation_state
from common.models.dinov2_lora import DinoV2LoRA
from .data import ShiftedCD, load_shift_table
from .model import RegistrationEquivariantDetector
from .objective import batch_objective

_state = {"tables": None, "table_sha": None}


def validate_config(config):
    if config["detector"] != "registration_equivariant":
        raise ValueError("本入口只接收 registration_equivariant 的配置")
    table_path = Path(__file__).resolve().parent / config["shift_table"]
    if not table_path.is_file():
        raise FileNotFoundError(f"位移表不存在:{table_path};请先运行 estimate_shifts.py")
    _state["tables"] = load_shift_table(table_path)
    _state["table_sha"] = runner.file_sha256(table_path)


def build_model(config, *, checkpoint=None):
    settings = config["model"]
    encoder = DinoV2LoRA(arch=settings["arch"], checkpoint=checkpoint or config["checkpoint"],
                         rank=settings["rank"], alpha=settings["alpha"])
    return RegistrationEquivariantDetector(encoder, mid=settings["mid"], vit_input=settings["vit_input"],
                                           heads=settings["heads"])


def build_dataset(dataset_path, mode="train", return_name=True):
    if _state["tables"] is None:
        raise RuntimeError("位移表未加载;validate_config 必须先运行")
    split = Path(dataset_path).name
    if split not in _state["tables"]:
        raise ValueError(f"位移表里没有 split:{split}")
    return ShiftedCD(dataset_path, mode=mode, return_name=return_name, shift_table=_state["tables"][split])


def batch_signature(batch):
    x1, x2, labels, names, shift = batch
    signature = runner.tensor_state_sha256({"x1": x1, "x2": x2, "labels": labels})
    return {"names": list(names), "augmented_tensor_sha256": signature,
            "shift_sha256": runner.tensor_state_sha256({"shift": shift}),
            "shape": list(x1.shape), "valid_pixels": int((labels != 255).sum()),
            "foreground_pixels": int((labels == 1).sum())}


def move_batch(batch, device):
    x1, x2, labels, names, shift = batch
    if x1.dtype != torch.float32 or x2.dtype != torch.float32 or labels.dtype != torch.long:
        raise ValueError("数据管线必须返回 FP32 图像和 int64 标签")
    if shift.dtype != torch.float32 or shift.ndim != 2 or shift.shape[1] != 2:
        raise ValueError("位移必须是 [B,2] 的 FP32 张量")
    if not bool(torch.isfinite(x1).all() & torch.isfinite(x2).all() & torch.isfinite(shift).all()):
        raise FloatingPointError("图像或位移包含 NaN/Inf")
    return x1.to(device), x2.to(device), labels.to(device), names, shift.to(device)


def evaluate(model, loader, device, *, thresholds=None, selected_threshold=None, ap_bins=4096):
    accumulator = StreamingMetrics(thresholds=thresholds, selected_threshold=selected_threshold, ap_bins=ap_bins)
    device = torch.device(device)
    with _preserve_evaluation_state(model, loader), torch.inference_mode():
        for x1, x2, labels, names, shift in loader:
            logits = model(x1.to(device, non_blocking=True), x2.to(device, non_blocking=True),
                           shift.to(device, non_blocking=True))
            accumulator.update(logits.float().softmax(dim=1)[:, 1], labels, names)
    return accumulator.compute()


def source_identity():
    identity = runner.source_identity(Path(__file__).resolve().parent)
    identity["shift_table_sha256"] = _state["table_sha"] or "unloaded"
    return identity


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config = root / "config.json"
    data_root = "/media/gs/sd10/RSE/split_v4"
    # 数据目录:<data_root>/{train,val,test}/{T1,T2,GT}/,文件名 <站点>_<编号>.png
    backbone = "/home/gs/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    # DINOv2 ViT-L/14 官方权重(无 register token)
    output = root.parent / "runs" / "registration_equivariant"
    device = "cuda:0"
    preflight = 0
    # 设为 2 只跑 2 个 batch 的前向/反向检查,不更新参数
    resume = None
    # 续训时填 output 下的 latest.pth
    runner.launch(sys.modules[__name__], config=config, data_root=data_root, backbone=backbone,
                  output=output, device=device, preflight=preflight, resume=resume)
