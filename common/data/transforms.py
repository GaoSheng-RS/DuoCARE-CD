# transforms.py
# 一套适用于遥感变化检测“影像对(T1/T2) + Label”的数据增强
# - 几何变换：对 T1/T2/Label 一致施加，保证像素对齐
# - 颜色变换：可选择“整对一致”或“分别抖动”
# - 纯 PyTorch / Torchvision 实现，便于放到 GPU 上跑（但一般放 CPU）

import random
from typing import Iterable, Tuple, Union, List, Optional

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import torchvision.transforms.functional as Tv1


def _to_tensor_image(img_np) -> torch.Tensor:
    """
    HWC numpy -> CHW torch.float32 in [0,1]
    """
    if isinstance(img_np, torch.Tensor):
        assert img_np.dim() == 3 and img_np.shape[0] in (1, 3), "expect CHW tensor for image"
        return img_np
    assert img_np.ndim == 3 and img_np.shape[2] in (3, 1), "expect HWC with 1 or 3 channels"
    t = torch.from_numpy(img_np).float()
    if t.max() > 1.0:
        t = t / 255.0
    t = t.permute(2, 0, 1)  # CHW
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)  # 灰度扩展到3通道，方便后续颜色操作
    return t


def _to_tensor_label(lbl_np) -> torch.Tensor:
    """
    HW numpy -> HW torch.long，保留原始标签值。
    当前二分类训练约定:
    - 0: 未变化
    - 1: 变化
    - 255: ignore_index
    """
    if isinstance(lbl_np, torch.Tensor):
        lbl = lbl_np.long()
    else:
        assert lbl_np.ndim == 2, "label must be HW"
        lbl = torch.from_numpy(lbl_np).long()

    invalid_mask = (lbl != 0) & (lbl != 1) & (lbl != 255)
    if invalid_mask.any():
        invalid_values = torch.unique(lbl[invalid_mask])[:20].tolist()
        raise ValueError(
            "标签中存在二分类训练未定义的像素值，"
            f"只允许 0/1/255，示例值: {invalid_values}"
        )
    return lbl


def _get_size_with_aspect_ratio(image_size: Tuple[int, int], size: int, max_size: Optional[int] = None):
    """
    与 SAM2 同逻辑：按短边缩放，最长边不超过 max_size
    输入 image_size = (W,H)；返回 (H_new, W_new)
    """
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))
    return (oh, ow)


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, pre_img, post_img, label):
        for t in self.transforms:
            pre_img, post_img, label = t(pre_img, post_img, label)
        return pre_img, post_img, label

    def __repr__(self):
        inner = "\n  ".join([repr(t) for t in self.transforms])
        return f"{self.__class__.__name__}(\n  {inner}\n)"


class ToTensorPair:
    """
    将 numpy 的 HWC -> torch 的 CHW (float32, [0,1])；Label -> torch.long 的 {0,1}
    """

    def __call__(self, pre_img, post_img, label):
        pre_img = _to_tensor_image(pre_img)
        post_img = _to_tensor_image(post_img)
        label = _to_tensor_label(label)  # HW
        return pre_img, post_img, label

    def __repr__(self):
        return "ToTensorPair()"


class NormalizePair:
    """
    对两张图做 ImageNet 标准化；Label 不变
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, pre_img, post_img, label):
        pre_img = Tv1.normalize(pre_img, mean=self.mean, std=self.std)
        post_img = Tv1.normalize(post_img, mean=self.mean, std=self.std)
        return pre_img, post_img, label

    def __repr__(self):
        return f"NormalizePair(mean={self.mean}, std={self.std})"


class RandomHorizontalFlipPair:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            pre_img = Tv1.hflip(pre_img)
            post_img = Tv1.hflip(post_img)
            label = Tv1.hflip(label.unsqueeze(0)).squeeze(0)
        return pre_img, post_img, label

    def __repr__(self):
        return f"RandomHorizontalFlipPair(p={self.p})"


class RandomVerticalFlipPair:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            pre_img = Tv1.vflip(pre_img)
            post_img = Tv1.vflip(post_img)
            label = Tv1.vflip(label.unsqueeze(0)).squeeze(0)
        return pre_img, post_img, label

    def __repr__(self):
        return f"RandomVerticalFlipPair(p={self.p})"


class RandomRotate90Pair:
    """
    随机 0/90/180/270 度旋转（整数倍 90°），几何完全对齐
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            k = random.randint(0, 3)
            pre_img = torch.rot90(pre_img, k, dims=[1, 2])
            post_img = torch.rot90(post_img, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        return pre_img, post_img, label

    def __repr__(self):
        return f"RandomRotate90Pair(p={self.p})"


class RandomResizePair:
    """
    sizes: int 或 (H,W) 或 多个候选（与 SAM2 类似）
    square=True -> 直接变成 size x size
    """

    def __init__(
        self,
        sizes: Union[int, Tuple[int, int], Iterable],
        max_size: Optional[int] = None,
        square: bool = False,
        image_interp: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        if isinstance(sizes, int) or isinstance(sizes, tuple):
            sizes = [sizes]
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.image_interp = image_interp

    def __call__(self, pre_img, post_img, label):
        # 输入 tensor: pre/post: [3,H,W], label: [H,W]
        _, H, W = pre_img.shape
        if self.square:
            size = random.choice(self.sizes)
            target_hw = (size, size)
        else:
            size = random.choice(self.sizes)
            if isinstance(size, (list, tuple)):
                target_hw = (size[0], size[1])  # already (H,W)
            else:
                target_hw = _get_size_with_aspect_ratio((W, H), size, self.max_size)

        pre_img = Tv1.resize(pre_img, target_hw, interpolation=self.image_interp, antialias=True)
        post_img = Tv1.resize(post_img, target_hw, interpolation=self.image_interp, antialias=True)
        # 标签最近邻
        label = Tv1.resize(label.unsqueeze(0), target_hw, interpolation=InterpolationMode.NEAREST).squeeze(0)
        return pre_img, post_img, label

    def __repr__(self):
        return f"RandomResizePair(sizes={self.sizes}, max_size={self.max_size}, square={self.square})"


class RandomCropPair:
    """
    随机裁剪到 (ch, cw)，不足则跳过
    """

    def __init__(self, crop_size: Tuple[int, int]):
        self.ch, self.cw = crop_size

    def __call__(self, pre_img, post_img, label):
        _, H, W = pre_img.shape
        if H < self.ch or W < self.cw:
            return pre_img, post_img, label
        top = random.randint(0, H - self.ch)
        left = random.randint(0, W - self.cw)
        pre_img = pre_img[:, top:top + self.ch, left:left + self.cw]
        post_img = post_img[:, top:top + self.ch, left:left + self.cw]
        label = label[top:top + self.ch, left:left + self.cw]
        return pre_img, post_img, label

    def __repr__(self):
        return f"RandomCropPair(size=({self.ch},{self.cw}))"


class RandomAffinePair:
    """
    对 pre/post/label 使用同一组仿射参数
    - images: 双线性/双三次
    - label : 最近邻，默认用 255 作为 ignore 填充
    - p: 触发概率
    - avoid_empty: 是否避免目标全被拽出画面（默认 False）
    """

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = 25,
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        shear: Optional[Union[float, Tuple[float, float]]] = 20,
        image_interpolation: str = "bilinear",
        fill_img: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # 图像填充值（[0,1]）
        fill_mask: int = 255,  # 标签填充值（ignore）
        num_tentatives: int = 1,  # 仅在 avoid_empty=True 时有意义
        p: float = 0.3,  # 触发概率
        avoid_empty: bool = False,  # ★ 你要的开关（默认 False）
        log_warning: bool = True,
    ):
        self.degrees = degrees if isinstance(degrees, (tuple, list)) else (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear if (isinstance(shear, (tuple, list))) else (-shear, shear) if shear else None
        self.fill_img = fill_img
        self.fill_mask = int(fill_mask)
        self.num_tentatives = max(1, int(num_tentatives))
        self.p = float(p)
        self.avoid_empty = bool(avoid_empty)
        self.log_warning = log_warning

        if image_interpolation == "bicubic":
            self.image_interp = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interp = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError(f"Unknown interpolation: {image_interpolation}")

    def __call__(self, pre_img, post_img, label):
        # 概率：未命中直接返回
        if random.random() >= self.p:
            return pre_img, post_img, label

        _, H, W = pre_img.shape
        img_size = [W, H]

        tries = self.num_tentatives if self.avoid_empty else 1
        for _ in range(tries):
            angle, translate, scale, shear = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

            # 先仿射标签（最近邻 + ignore 填充）
            lbl_t = Tv1.affine(
                label.unsqueeze(0).float(),  # 1xHxW
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=float(self.fill_mask),
            ).squeeze(0).long()

            # 如果要求避免“全背景”，且当前为全 0，则重抽
            if self.avoid_empty and lbl_t.max() == 0:
                if self.log_warning:
                    # 可在此打印/记录一次信息（可选）
                    pass
                continue

            # 仿射两张图（双线性/双三次 + 图像填充）
            pre_t = Tv1.affine(
                pre_img,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=self.image_interp,
                fill=self.fill_img,
            )
            post_t = Tv1.affine(
                post_img,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=self.image_interp,
                fill=self.fill_img,
            )
            return pre_t, post_t, lbl_t

        # 若 avoid_empty=True 且多次重抽仍不满足，就返回原图
        return pre_img, post_img, label

    def __repr__(self):
        return (
            f"RandomAffinePair(degrees={self.degrees}, translate={self.translate}, "
            f"scale={self.scale}, shear={self.shear}, image_interp={self.image_interp}, "
            f"fill_img={self.fill_img}, fill_mask={self.fill_mask}, "
            f"p={self.p}, avoid_empty={self.avoid_empty}, num_tentatives={self.num_tentatives})"
        )


class ColorJitterPair:
    """
    颜色抖动：可选一致/不一致
    - consistent=True: T1/T2 用同一随机参数（模拟“同一场景曝光一致”）
    - consistent=False: T1/T2 各自独立（模拟轻微时变外观）
    """

    def __init__(self, brightness=0.1, contrast=0.05, saturation=0.05, hue=None, consistent: bool = True):
        self.consistent = consistent
        self.brightness = brightness if isinstance(brightness, (tuple, list)) else [max(0, 1 - brightness), 1 + brightness]
        self.contrast = contrast if isinstance(contrast, (tuple, list)) else [max(0, 1 - contrast), 1 + contrast]
        self.saturation = saturation if isinstance(saturation, (tuple, list)) else [max(0, 1 - saturation), 1 + saturation]
        self.hue = hue if (isinstance(hue, (tuple, list)) or hue is None) else [-hue, hue]

    def _apply(self, img: torch.Tensor, params):
        fn_idx, b, c, s, h = params
        out = img
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                out = Tv1.adjust_brightness(out, b)
            elif fn_id == 1 and c is not None:
                out = Tv1.adjust_contrast(out, c)
            elif fn_id == 2 and s is not None:
                out = Tv1.adjust_saturation(out, s)
            elif fn_id == 3 and h is not None:
                out = Tv1.adjust_hue(out, h)
        return out

    def __call__(self, pre_img, post_img, label):
        if self.consistent:
            params = T.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            pre_img = self._apply(pre_img, params)
            post_img = self._apply(post_img, params)
        else:
            p1 = T.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            p2 = T.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            pre_img = self._apply(pre_img, p1)
            post_img = self._apply(post_img, p2)
        return pre_img, post_img, label

    def __repr__(self):
        return f"ColorJitterPair(consistent={self.consistent})"


class RandomGrayscalePair:
    def __init__(self, p: float = 0.05, consistent: bool = True):
        self.p = p
        self.consistent = consistent
        self.to_gray = T.Grayscale(num_output_channels=3)

    def __call__(self, pre_img, post_img, label):
        if self.consistent:
            if random.random() < self.p:
                pre_img = self.to_gray(pre_img)
                post_img = self.to_gray(post_img)
        else:
            if random.random() < self.p:
                pre_img = self.to_gray(pre_img)
            if random.random() < self.p:
                post_img = self.to_gray(post_img)
        return pre_img, post_img, label

    def __repr__(self):
        return f"RandomGrayscalePair(p={self.p}, consistent={self.consistent})"


class SwapPair:
    """
    以概率交换 T1/T2，用于让模型对“时间顺序”更稳健（若任务允许对调）
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, pre_img, post_img, label):
        if random.random() < self.p:
            return post_img, pre_img, label
        return pre_img, post_img, label

    def __repr__(self):
        return f"SwapPair(p={self.p})"
