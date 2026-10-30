import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

try:
    from .transforms import (
        ColorJitterPair,
        Compose,
        NormalizePair,
        RandomAffinePair,
        RandomGrayscalePair,
        RandomHorizontalFlipPair,
        RandomRotate90Pair,
        RandomVerticalFlipPair,
        SwapPair,
        ToTensorPair,
    )
except ImportError:
    from transforms import (
        ColorJitterPair,
        Compose,
        NormalizePair,
        RandomAffinePair,
        RandomGrayscalePair,
        RandomHorizontalFlipPair,
        RandomRotate90Pair,
        RandomVerticalFlipPair,
        SwapPair,
        ToTensorPair,
    )

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES


def _scan_image_dir(dir_path: Path) -> Dict[str, Path]:
    sample_map: Dict[str, Path] = {}
    for path in sorted(dir_path.iterdir()):
        if not _is_image_file(path):
            continue
        stem = path.stem
        if stem in sample_map:
            raise ValueError(f"{dir_path} 中存在同名样本 stem 冲突: {stem}")
        sample_map[stem] = path
    return sample_map


class ChangeDetectionStableDataset(Dataset):
    """
    常规变化检测数据集，只读取 query：
    - 数据目录结构：
        root_dir/
          T1/
          T2/
          GT/
    - 不依赖 txt，直接扫描三个目录并自动配对样本
    - 返回：
        pre_img:  [3, H, W]
        post_img: [3, H, W]
        label:    [H, W]
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        return_name: bool = False,
    ):
        super().__init__()
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode 只能是 'train' 或 'eval'，当前为: {mode}")

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.return_name = return_name

        self.t1_dir = self.root_dir / "T1"
        self.t2_dir = self.root_dir / "T2"
        self.gt_dir = self.root_dir / "GT"

        for dir_path in (self.t1_dir, self.t2_dir, self.gt_dir):
            if not dir_path.is_dir():
                raise FileNotFoundError(f"目录不存在: {dir_path}")

        self.samples = self._build_samples()
        self.train_transform = self._build_train_transform()
        self.eval_transform = self._build_eval_transform()

    def _build_samples(self) -> List[Dict[str, str]]:
        t1_map = _scan_image_dir(self.t1_dir)
        t2_map = _scan_image_dir(self.t2_dir)
        gt_map = _scan_image_dir(self.gt_dir)

        common_names = sorted(set(t1_map) & set(t2_map) & set(gt_map))
        if not common_names:
            raise RuntimeError(f"{self.root_dir} 下没有找到可配对的 T1/T2/GT 样本")

        if len(common_names) != len(t1_map) or len(common_names) != len(t2_map) or len(common_names) != len(gt_map):
            missing_t1 = sorted((set(t2_map) & set(gt_map)) - set(t1_map))[:5]
            missing_t2 = sorted((set(t1_map) & set(gt_map)) - set(t2_map))[:5]
            missing_gt = sorted((set(t1_map) & set(t2_map)) - set(gt_map))[:5]
            raise RuntimeError(
                "T1/T2/GT 中的样本名不一致。"
                f" T1={len(t1_map)}, T2={len(t2_map)}, GT={len(gt_map)}, common={len(common_names)}."
                f" 缺失示例: T1<-{missing_t1}, T2<-{missing_t2}, GT<-{missing_gt}"
            )

        return [
            {
                "name": name,
                "t1": os.fspath(t1_map[name]),
                "t2": os.fspath(t2_map[name]),
                "gt": os.fspath(gt_map[name]),
            }
            for name in common_names
        ]

    def _build_train_transform(self):
        return Compose([
            ToTensorPair(),
            RandomHorizontalFlipPair(p=0.3),
            RandomVerticalFlipPair(p=0.3),
            RandomRotate90Pair(p=0.3),
            RandomAffinePair(
                degrees=25,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=10,
                image_interpolation="bilinear",
                fill_img=(0.485, 0.456, 0.406),
                fill_mask=255,
                p=0.3,
                avoid_empty=False,
                num_tentatives=2,
            ),
            RandomGrayscalePair(p=0.05, consistent=True),
            ColorJitterPair(
                brightness=0.1,
                contrast=0.05,
                saturation=0.05,
                hue=None,
                consistent=False,
            ),
            NormalizePair(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            SwapPair(p=0.5),
        ])

    def _build_eval_transform(self):
        return Compose([
            ToTensorPair(),
            NormalizePair(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_sample(self, index: int):
        sample = self.samples[index]
        pre_img = np.array(Image.open(sample["t1"]).convert("RGB"))
        post_img = np.array(Image.open(sample["t2"]).convert("RGB"))
        label = np.array(Image.open(sample["gt"]))
        return pre_img, post_img, label

    def __getitem__(self, index: int):
        sample_name = self.samples[index]["name"]
        pre_img, post_img, label = self._load_raw_sample(index)

        if self.mode == "train":
            pre_img, post_img, label = self.train_transform(pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.eval_transform(pre_img, post_img, label)

        if self.return_name:
            return pre_img, post_img, label, sample_name
        return pre_img, post_img, label


def build_dataCD(
    dataset_path: str,
    mode: str = "train",
    return_name: bool = True,
) -> ChangeDetectionStableDataset:
    """
    构建常规变化检测数据集。

    参数：
    - dataset_path: 目录下直接包含 T1/T2/GT
    - mode: 'train' 或 'eval'
    """
    return ChangeDetectionStableDataset(
        root_dir=dataset_path,
        mode=mode,
        return_name=return_name,
    )
