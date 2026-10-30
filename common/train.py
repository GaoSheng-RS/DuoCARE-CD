"""两个检测器共用的可恢复 FP32 训练循环。

由各检测器的 train.py 调用 run(args, detector);detector 是检测器模块,提供
validate_config / build_model / batch_objective / source_identity,可选提供
build_dataset / batch_signature / move_batch / evaluate 以替换默认实现。
检查点只保存可训练参数和 buffer,加载时必须提供同一份 DINOv2 官方权重。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from .data.dataCD import build_dataCD
from .data.site_batch import SiteBatchSampler, site_of
from . import metrics


FORMAT_VERSION = 1
AUGMENTATION_PROTOCOL = "per_update_cpu_seed_sequence_v1"
SAMPLING_PROTOCOL = "site_batch_single_epoch_20000_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_log(path: Path, value: dict) -> None:
    record = {"time": utc_now(), **value}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def validate_training(config: dict) -> None:
    training = config["training"]
    for name in ("total_updates", "lr", "weight_decay", "lr_milestones", "lr_gamma"):
        if name not in training:
            raise ValueError(f"training.{name} 必须显式给出")
    gamma = training["lr_gamma"]
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0 < gamma < 1:
        raise ValueError("lr_gamma 必须为 (0,1) 内的数")
    if config["sampling"].get("num_workers", 0) != 0:
        raise ValueError("精确恢复协议要求 num_workers=0,避免预取提前消费增强 RNG")


def seed_model(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(include_cuda: bool) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def restore_rng(state: dict, include_cuda: bool) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    cuda = state["torch_cuda"]
    if include_cuda:
        if cuda is None or len(cuda) != torch.cuda.device_count():
            raise ValueError("恢复时 CUDA RNG 设备数量与检查点不同")
        torch.cuda.set_rng_state_all(cuda)


@contextmanager
def augmentation_rng(seed: int, update_index: int) -> Iterator[None]:
    """每次 update 的数据增强独立定种子；辅助损失、评估和恢复均不改变它。"""
    state = rng_state(include_cuda=False)
    step_seed = int(np.random.SeedSequence([seed, 0x415547, update_index]).generate_state(1, dtype=np.uint64)[0])
    random.seed(step_seed)
    np.random.seed(step_seed % 2**32)
    torch.random.default_generator.manual_seed(step_seed)
    try:
        yield
    finally:
        restore_rng(state, include_cuda=False)


def training_batches(dataset, sampler: SiteBatchSampler, start_update: int = 0):
    """只跳过 sampler 索引，不重新读图或回放旧的增强；从下一次未完成更新继续。"""
    if not 0 <= start_update <= len(sampler):
        raise ValueError("恢复的采样位置不在合法 update 范围内")
    if sampler.epoch != 0:
        raise ValueError("训练协议固定 sampler.epoch=0，steps=total_updates")
    indices = itertools.islice(iter(sampler), start_update, None)
    for update_index, batch_indices in enumerate(indices, start=start_update):
        with augmentation_rng(sampler.seed, update_index):
            batch = default_collate([dataset[index] for index in batch_indices])
        yield update_index, batch_indices, batch


def learning_rate(training: dict, completed_updates: int) -> float:
    """完成 5000/10000 次更新后降 lr，分别作用于第 5001/10001 次更新。"""
    decays = sum(completed_updates >= milestone for milestone in training["lr_milestones"])
    return float(training["lr"] * training["lr_gamma"] ** decays)


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError(f"检查点可训练参数/buffer 不匹配：missing={sorted(expected-set(state))}, extra={sorted(set(state)-expected)}")
    result = model.load_state_dict(state, strict=False)
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    if result.unexpected_keys or set(result.missing_keys) != frozen:
        raise ValueError("检查点意外遗漏了非冻结参数，拒绝恢复")


def batch_signature(batch) -> dict:
    x1, x2, labels, names = batch
    signature = tensor_state_sha256({"x1": x1, "x2": x2, "labels": labels})
    return {"names": list(names), "augmented_tensor_sha256": signature,
            "shape": list(x1.shape), "valid_pixels": int((labels != 255).sum()),
            "foreground_pixels": int((labels == 1).sum())}


def move_batch(batch, device: torch.device):
    x1, x2, labels, names = batch
    if x1.dtype != torch.float32 or x2.dtype != torch.float32 or labels.dtype != torch.long:
        raise ValueError("数据管线必须返回 FP32 图像和 int64 标签")
    if not bool(torch.isfinite(x1).all() & torch.isfinite(x2).all()):
        raise FloatingPointError("图像包含 NaN/Inf")
    return x1.to(device), x2.to(device), labels.to(device), names


def gradient_report(model: nn.Module, *, detailed: bool = False) -> dict:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    if missing:
        raise RuntimeError(f"可训练参数未获得梯度：{missing}")
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise RuntimeError("冻结参数意外获得梯度")
    finite = torch.stack([torch.isfinite(parameter.grad).all() for _, parameter in trainable]).all()
    if not bool(finite):
        bad = [name for name, parameter in trainable if not bool(torch.isfinite(parameter.grad).all())]
        raise FloatingPointError(f"参数梯度存在 NaN/Inf：{bad}")
    norms = torch.stack([parameter.grad.float().norm() for _, parameter in trainable])
    norm = float(norms.norm())
    if not math.isfinite(norm):
        raise FloatingPointError("梯度总范数溢出")
    report = {"gradient_l2": norm, "gradient_tensors": len(trainable), "all_gradients_finite": True}
    if detailed:
        report["nonzero_gradient_tensors"] = int((norms > 0).sum())
        report["gradient_norms"] = {name: float(value) for (name, _), value in zip(trainable, norms)}
    return report


def dataset_description(dataset) -> dict:
    names = [sample["name"] for sample in dataset.samples]
    return {"samples": len(names), "sites": len(set(map(site_of, names))),
            "ordered_names_sha256": canonical_sha256(names)}


def source_identity(*packages: Path) -> dict[str, str]:
    """训练代码的 hash,写进检查点身份,防止用改过的代码静默续训。"""
    roots = (Path(__file__).resolve().parent, *packages)
    return {f"{root.name}/{path.relative_to(root)}": file_sha256(path)
            for root in roots for path in sorted(root.rglob("*.py"))}


def resume_identity(config: dict, base_sha256: str, datasets: dict, sources: dict) -> dict:
    logical_config = copy.deepcopy(config)
    # 同字节的代码/权重/数据可以搬到新目录恢复，不依赖绝对路径。
    logical_config.pop("data_root", None)
    logical_config.pop("checkpoint", None)
    return {"format_version": FORMAT_VERSION, "config": logical_config,
            "base_checkpoint_sha256": base_sha256, "datasets": datasets,
            "sources": sources, "augmentation_protocol": AUGMENTATION_PROTOCOL,
            "sampling_protocol": SAMPLING_PROTOCOL}


def save_checkpoint(path: Path, *, model: nn.Module, optimizer, aux_generator,
                    completed_updates: int, best_iou, best_update, best_validation,
                    identity: dict, include_cuda: bool) -> None:
    payload = {
        "format_version": FORMAT_VERSION, "identity": identity,
        "requires_base_checkpoint": True,
        "trainable_and_buffers": trainable_state(model), "optimizer": optimizer.state_dict(),
        "completed_updates": completed_updates, "sampler_epoch": 0,
        "sampler_next_index": completed_updates,
        "rng": rng_state(include_cuda), "aux_generator": aux_generator.get_state(),
        "best_iou": best_iou, "best_update": best_update, "best_validation": best_validation,
        "saved_at": utc_now(),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def restore_checkpoint(path: Path, *, model, optimizer, aux_generator,
                       identity: dict, include_cuda: bool) -> dict:
    # 只加载本入口自己生成、由用户指定的受信任检查点（含 Python/NumPy RNG）。
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("identity") != identity:
        raise ValueError("检查点与当前配置、骨干、数据索引或代码 hash 不匹配，拒绝静默续训")
    if payload["sampler_epoch"] != 0 or payload["sampler_next_index"] != payload["completed_updates"]:
        raise ValueError("检查点的 update 和采样位置不一致")
    load_trainable_state(model, payload["trainable_and_buffers"])
    optimizer.load_state_dict(payload["optimizer"])
    aux_generator.set_state(payload["aux_generator"])
    restore_rng(payload["rng"], include_cuda)
    return payload


def make_eval_loader(dataset, batch_size: int, pin_memory: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      pin_memory=pin_memory, generator=torch.Generator(device="cpu").manual_seed(0))


def run(args: argparse.Namespace, detector) -> None:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("checkpoint", "data_root"):
        if getattr(args, key) is not None:
            config[key] = str(Path(getattr(args, key)).expanduser().resolve())
    validate_training(config)
    detector.validate_config(config)
    if not config.get("checkpoint") or not Path(config["checkpoint"]).is_file():
        raise FileNotFoundError("必须提供真实存在的本地官方 checkpoint")
    if args.preflight < 0 or args.log_every < 1 or args.eval_batch_size < 1:
        raise ValueError("preflight 必须非负，log_every/eval_batch_size 必须为正")
    if args.preflight > config["training"]["total_updates"]:
        raise ValueError("preflight 的 batch 数不能超过总更新数")
    if args.preflight and args.resume:
        raise ValueError("preflight 从固定初始化检查；不能和 resume 同时使用")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("只支持 cpu/cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但此进程无法访问 GPU")
        torch.cuda.set_device(device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.preflight and (output / "manifest.json").exists() and not args.resume:
        raise FileExistsError("输出目录已经有训练 manifest；请显式 --resume 或使用新目录")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)
    if args.resume and Path(args.resume).resolve().parent != output:
        raise ValueError("请在检查点所在输出目录续训，确保 best.pth 与 latest.pth 一起保留")
    training = config["training"]
    sampling = config["sampling"]
    build_dataset = getattr(detector, "build_dataset", build_dataCD)
    batch_signature_ = getattr(detector, "batch_signature", batch_signature)
    move_batch_ = getattr(detector, "move_batch", move_batch)
    evaluate = getattr(detector, "evaluate", metrics.evaluate)
    datasets = {split: build_dataset(str(Path(config["data_root"]) / split),
                                  mode="train" if split == "train" else "eval", return_name=True)
                for split in ("train", "val", "test")}
    descriptions = {split: dataset_description(dataset) for split, dataset in datasets.items()}
    site_sets = {split: set(site_of(sample["name"]) for sample in dataset.samples) for split, dataset in datasets.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = site_sets[left] & site_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} site 泄漏：{sorted(overlap)}")
    sampler = SiteBatchSampler(datasets["train"], M=sampling["M"], K=sampling["K"],
                               site_power=sampling["site_power"], steps=training["total_updates"], seed=sampling["seed"])
    base_digest = file_sha256(config["checkpoint"])
    sources = detector.source_identity()
    identity = resume_identity(config, base_digest, descriptions, sources)
    seed_model(sampling["seed"], device)
    model = detector.build_model(config).float()
    initial_hash = tensor_state_sha256(trainable_state(model))
    model.to(device)
    aux_generator = torch.Generator(device="cpu").manual_seed(sampling["seed"] + 0x415558)
    manifest = {
        "created_at": utc_now(), "command": sys.argv, "config": config,
        "identity": identity, "identity_sha256": canonical_sha256(identity),
        "initial_trainable_sha256": initial_hash,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "augmentation_protocol": AUGMENTATION_PROTOCOL, "sampling_protocol": SAMPLING_PROTOCOL,
        "validation_every": 1000, "lr_boundary": "after completed update 5000/10000; effective at update 5001/10001",
        "checkpoint_contents": "trainable parameters + buffers + AdamW + RNG + sampler position; official frozen backbone required",
        "preflight_batches": args.preflight,
    }
    prefix = "preflight_" if args.preflight else ""
    manifest_path = output / (prefix + "manifest.json")
    if args.resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest["identity"] != identity:
            raise ValueError("现有输出 manifest 与恢复配置不符，保留原记录并拒绝启动")
    else:
        atomic_json(manifest_path, manifest)
    log = output / (prefix + "events.jsonl")
    status_path = output / (prefix + "status.json")
    status = {"state": "preflight" if args.preflight else "running", "completed_updates": 0,
              "detector": config["detector"], "started_at": utc_now()}
    atomic_json(status_path, status)
    append_log(log, {"event": "start", "detector": config["detector"], "initial_trainable_sha256": initial_hash,
                     "datasets": descriptions, "preflight_batches": args.preflight})
    try:
        model.train()
        if args.preflight:
            checks = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            finite_parameters = torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
            if not bool(finite_parameters):
                raise FloatingPointError("模型初始化参数含 NaN/Inf")
            for step, indices, cpu_batch in itertools.islice(training_batches(datasets["train"], sampler), args.preflight):
                signature = batch_signature_(cpu_batch)
                if signature["valid_pixels"] == 0:
                    raise RuntimeError("preflight 遇到全 ignore 的 batch")
                model.zero_grad(set_to_none=True)
                objective = detector.batch_objective(model, move_batch_(cpu_batch, device), config, generator=aux_generator)
                if not bool(torch.isfinite(objective.loss)):
                    raise FloatingPointError("preflight loss 含 NaN/Inf")
                objective.loss.backward()
                report = {"batch": step, "indices": indices, **signature,
                          "loss": float(objective.loss.detach()), "segmentation_loss": float(objective.segmentation_loss.detach()),
                          "auxiliary_loss": float(objective.auxiliary_loss.detach()),
                          "diagnostics": objective.diagnostics, **gradient_report(model, detailed=True)}
                if device.type == "cuda":
                    report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                    report["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
                checks.append(report)
                append_log(log, {"event": "preflight_batch", **{key: val for key, val in report.items() if key != "gradient_norms"}})
                del objective
            model.zero_grad(set_to_none=True)
            unchanged = tensor_state_sha256(trainable_state(model)) == initial_hash
            if not unchanged:
                raise RuntimeError("preflight 意外修改了模型参数/buffer")
            result = {"passed": True, "optimizer_steps": 0, "parameters_unchanged": unchanged,
                      "batches": checks, "initial_trainable_sha256": initial_hash}
            if device.type == "cuda":
                result["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                result["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            atomic_json(output / "preflight.json", result)
            status.update(state="preflight_passed", finished_at=utc_now(), checked_batches=len(checks))
            atomic_json(status_path, status)
            return

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=training["lr"], weight_decay=training["weight_decay"])
        completed = 0
        best_iou, best_update, best_validation = None, None, None
        if args.resume:
            restored = restore_checkpoint(Path(args.resume), model=model, optimizer=optimizer,
                                          aux_generator=aux_generator, identity=identity,
                                          include_cuda=device.type == "cuda")
            completed = restored["completed_updates"]
            best_iou, best_update, best_validation = restored["best_iou"], restored["best_update"], restored["best_validation"]
            append_log(log, {"event": "resume", "checkpoint": str(Path(args.resume).resolve()), "completed_updates": completed})
            status.update(completed_updates=completed, best_iou=best_iou, best_update=best_update)
            atomic_json(status_path, status)
        val_loader = make_eval_loader(datasets["val"], args.eval_batch_size, sampling.get("pin_memory", False))
        started = time.monotonic()
        for step, indices, cpu_batch in training_batches(datasets["train"], sampler, completed):
            lr = learning_rate(training, step)
            set_learning_rate(optimizer, lr)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = move_batch_(cpu_batch, device)
            objective = detector.batch_objective(model, batch, config, generator=aux_generator)
            if not bool(torch.isfinite(objective.loss)):
                raise FloatingPointError(f"update {step + 1}: loss 非有限")
            objective.loss.backward()
            gradients = gradient_report(model)
            optimizer.step()
            completed = step + 1
            log_now = completed <= 3 or completed % args.log_every == 0 or completed in (5000, 5001, 10000, 10001)
            if log_now:
                memory = ({"peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                           "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
                          if device.type == "cuda" else {})
                append_log(log, {"event": "update", "completed_updates": completed, "lr": lr,
                                 "loss": float(objective.loss.detach()), "segmentation_loss": float(objective.segmentation_loss.detach()),
                                 "auxiliary_loss": float(objective.auxiliary_loss.detach()), "diagnostics": objective.diagnostics,
                                 **gradients, **memory, **batch_signature_(cpu_batch), "elapsed_seconds": time.monotonic() - started})
            del objective, batch
            status.update(completed_updates=completed, lr=lr, updated_at=utc_now(), best_iou=best_iou, best_update=best_update)
            if log_now:
                atomic_json(status_path, status)
            if completed % 1000 == 0:
                validation = evaluate(model, val_loader, device)
                atomic_json(output / f"val_{completed:05d}.json", validation)
                iou = validation["pooled_at_0_5"]["iou"]
                if iou is None or not math.isfinite(iou):
                    raise RuntimeError("val 的 pooled IoU 无定义，不能选模")
                improved = best_iou is None or iou > best_iou
                if improved:
                    best_iou, best_update, best_validation = iou, completed, validation
                set_learning_rate(optimizer, learning_rate(training, completed))
                checkpoint_args = dict(model=model, optimizer=optimizer, aux_generator=aux_generator,
                                       completed_updates=completed, best_iou=best_iou, best_update=best_update,
                                       best_validation=best_validation, identity=identity, include_cuda=device.type == "cuda")
                if improved:
                    save_checkpoint(output / "best.pth", **checkpoint_args)
                save_checkpoint(output / "latest.pth", **checkpoint_args)
                append_log(log, {"event": "validation", "completed_updates": completed,
                                 "pooled_iou_at_0_5": iou, "best_iou": best_iou, "best_update": best_update,
                                 "improved": improved, "validation_file": f"val_{completed:05d}.json"})
                status.update(best_iou=best_iou, best_update=best_update, last_checkpoint_update=completed)
                atomic_json(status_path, status)

        best_path = output / "best.pth"
        if not best_path.is_file():
            raise FileNotFoundError("完成训练但没有 best.pth；请在原输出目录 resume，保留最佳检查点")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best["identity"] != identity:
            raise ValueError("best 检查点身份与当前运行不符")
        if best["completed_updates"] != best_update or best["best_iou"] != best_iou:
            raise ValueError("best.pth 与训练记录的最佳 update/IoU 不一致，拒绝用错模型评估 test")
        load_trainable_state(model, best["trainable_and_buffers"])
        selected = best["best_validation"]["threshold_search"]["best_threshold"]
        test_loader = make_eval_loader(datasets["test"], args.eval_batch_size, sampling.get("pin_memory", False))
        test = evaluate(model, test_loader, device, thresholds=(), selected_threshold=selected)
        atomic_json(output / "test_best.json", {"checkpoint_update": best["completed_updates"],
                                                "val_selected_threshold": selected, "metrics": test})
        status.update(state="complete", completed_updates=completed, best_iou=best_iou,
                      best_update=best_update, finished_at=utc_now())
        atomic_json(status_path, status)
        append_log(log, {"event": "complete", **status, "test_file": "test_best.json"})
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}", stopped_at=utc_now())
        atomic_json(status_path, status)
        append_log(log, {"event": status["state"], **status})
        raise


def launch(detector, *, config, data_root, backbone, output, device="cuda:0", preflight=0, resume=None,
           log_every=50, eval_batch_size=8) -> None:
    """训练入口:持锁防止同一输出目录重复启动,然后调用 run。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = argparse.Namespace(config=str(config), output=str(output), checkpoint=str(backbone),
                              data_root=str(data_root), device=device, resume=resume, preflight=preflight,
                              eval_batch_size=eval_batch_size, log_every=log_every)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该输出目录已有训练/预检进程持锁,拒绝重复启动") from exc
        run(args, detector)
