"""Registration-equivariant detector 的数据层:在共用数据管线上附带"增强后位移"作为 batch 第 5 个元素。

与共用管线的批次逐位相同:包装的几何变换与原版消耗完全相同的随机数序列、
执行完全相同的图像操作,只额外把位移向量 s 按同一几何变换传播:
- 水平翻转 s=(dx,dy) → (−dx, dy);垂直翻转 → (dx, −dy)
- rot90 一次(H→W 方向) → (dy, −dx),k 次迭代
- 仿射(两期同参数)→ s' = L·s,L 为正向仿射的线性部分(平移部分两期相消)
- SwapPair → −s;颜色类变换不动 s
位移单位保持原始影像像素(256),约定 T1(y,x) ≈ T2(y−dy, x−dx)。
"""
import json
from pathlib import Path
import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as Tv1
from torchvision.transforms.functional import InterpolationMode, _get_inverse_affine_matrix

from common.data.dataCD import ChangeDetectionStableDataset
from common.data.transforms import (
    ColorJitterPair,
    Compose,
    NormalizePair,
    RandomGrayscalePair,
    SwapPair,
    ToTensorPair,
)


class GeoState:
    """单线程数据集内共享的可变位移;__getitem__ 开头重置。"""

    def __init__(self):
        self.shift = np.zeros(2, dtype=np.float64)


class TrackedHFlip:
    def __init__(self, p, state):
        self.p, self.state = p, state

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            pre_img = Tv1.hflip(pre_img)
            post_img = Tv1.hflip(post_img)
            label = Tv1.hflip(label.unsqueeze(0)).squeeze(0)
            self.state.shift[0] = -self.state.shift[0]
        return pre_img, post_img, label


class TrackedVFlip:
    def __init__(self, p, state):
        self.p, self.state = p, state

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            pre_img = Tv1.vflip(pre_img)
            post_img = Tv1.vflip(post_img)
            label = Tv1.vflip(label.unsqueeze(0)).squeeze(0)
            self.state.shift[1] = -self.state.shift[1]
        return pre_img, post_img, label


class TrackedRotate90:
    def __init__(self, p, state):
        self.p, self.state = p, state

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            k = random.randint(0, 3)
            pre_img = torch.rot90(pre_img, k, dims=[1, 2])
            post_img = torch.rot90(post_img, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
            dx, dy = self.state.shift
            for _ in range(k % 4):
                dx, dy = dy, -dx
                # 推导:T1'(i,j)=T1(j,W−1−i) ⇒ 新位移 (dx',dy')=(dy,−dx);由单元测试用相位相关验证
            self.state.shift[:] = (dx, dy)
        return pre_img, post_img, label


class TrackedAffine:
    """逐行复刻 RandomAffinePair(dataCD 配置:avoid_empty=False ⇒ tries=1),额外算线性部分。"""

    def __init__(self, degrees, translate, scale, shear, image_interpolation,
                 fill_img, fill_mask, p, avoid_empty, num_tentatives, state):
        self.degrees = degrees if isinstance(degrees, (tuple, list)) else (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear if isinstance(shear, (tuple, list)) else (-shear, shear) if shear else None
        self.fill_img = fill_img
        self.fill_mask = int(fill_mask)
        self.num_tentatives = max(1, int(num_tentatives))
        self.p = float(p)
        self.avoid_empty = bool(avoid_empty)
        self.image_interp = {"bicubic": InterpolationMode.BICUBIC,
                             "bilinear": InterpolationMode.BILINEAR}[image_interpolation]
        self.state = state

    def __call__(self, pre_img, post_img, label):
        if random.random() >= self.p:
            return pre_img, post_img, label
        _, H, W = pre_img.shape
        img_size = [W, H]
        tries = self.num_tentatives if self.avoid_empty else 1
        for _ in range(tries):
            angle, translate, scale, shear = T.RandomAffine.get_params(
                degrees=self.degrees, translate=self.translate, scale_ranges=self.scale,
                shears=self.shear, img_size=img_size,
            )
            lbl_t = Tv1.affine(label.unsqueeze(0).float(), angle=angle, translate=translate, scale=scale,
                               shear=shear, interpolation=InterpolationMode.NEAREST,
                               fill=float(self.fill_mask)).squeeze(0).long()
            if self.avoid_empty and lbl_t.max() == 0:
                continue
            pre_t = Tv1.affine(pre_img, angle=angle, translate=translate, scale=scale, shear=shear,
                               interpolation=self.image_interp, fill=self.fill_img)
            post_t = Tv1.affine(post_img, angle=angle, translate=translate, scale=scale, shear=shear,
                                interpolation=self.image_interp, fill=self.fill_img)
            shear_list = list(shear) if isinstance(shear, (tuple, list)) else [float(shear), 0.0]
            inverse = _get_inverse_affine_matrix([0.0, 0.0], angle, [0.0, 0.0], scale, shear_list)
            a, b, _, d, e, _ = inverse
            det = a * e - b * d
            forward = np.array([[e, -b], [-d, a]]) / det
            # 正向线性部分 = 逆矩阵线性块的逆;平移对两期相同,对相对位移无影响
            self.state.shift[:] = forward @ self.state.shift
            return pre_t, post_t, lbl_t
        return pre_img, post_img, label


class TrackedSwap:
    def __init__(self, p, state):
        self.p, self.state = p, state

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            self.state.shift[:] = -self.state.shift
            return post_img, pre_img, label
        return pre_img, post_img, label


class ShiftedCD(ChangeDetectionStableDataset):
    """共用数据集 + 位移表;train 管线换成跟踪位移的版本(随机数消耗与原版逐次相同)。"""

    def __init__(self, root_dir, mode, return_name, shift_table):
        self._geo = GeoState()
        self.shift_table = shift_table
        super().__init__(root_dir=root_dir, mode=mode, return_name=return_name)
        if not return_name:
            raise ValueError("需要样本名查位移表")
        missing = [s["name"] for s in self.samples if s["name"] not in shift_table]
        if missing:
            raise ValueError(f"位移表缺 {len(missing)} 个样本,例如 {missing[:3]}")

    def _build_train_transform(self):
        return Compose([
            ToTensorPair(),
            TrackedHFlip(p=0.3, state=self._geo),
            TrackedVFlip(p=0.3, state=self._geo),
            TrackedRotate90(p=0.3, state=self._geo),
            TrackedAffine(degrees=25, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10,
                          image_interpolation="bilinear", fill_img=(0.485, 0.456, 0.406), fill_mask=255,
                          p=0.3, avoid_empty=False, num_tentatives=2, state=self._geo),
            RandomGrayscalePair(p=0.05, consistent=True),
            ColorJitterPair(brightness=0.1, contrast=0.05, saturation=0.05, hue=None, consistent=False),
            NormalizePair(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TrackedSwap(p=0.5, state=self._geo),
        ])

    def __getitem__(self, index):
        sample_name = self.samples[index]["name"]
        pre_img, post_img, label = self._load_raw_sample(index)
        self._geo.shift[:] = self.shift_table[sample_name]
        if self.mode == "train":
            pre_img, post_img, label = self.train_transform(pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.eval_transform(pre_img, post_img, label)
        shift = torch.tensor(self._geo.shift, dtype=torch.float32)
        return pre_img, post_img, label, sample_name, shift


def load_shift_table(path):
    tables = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(tables) != {"train", "val", "test"}:
        raise ValueError("位移表必须包含 train/val/test 三个 split")
    return tables
