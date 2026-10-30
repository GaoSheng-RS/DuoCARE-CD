"""估计每个影像对的配准位移,生成 registration-equivariant detector 用的位移表(论文 3.2.3)。

不使用标签:
1. 对加 Hann 窗的灰度影像做相位相关,得到整像素平移 (dx, dy);
2. 按该平移把 T2 移位后,与 T1 在中央有效区域的归一化互相关(NCC)至少提高 0.05,才算可靠估计;
3. 可靠的影像对用自身估计;否则取同一站点其他可靠估计的逐分量中位数(至少 2 个);
   仍没有则记为 (0, 0),此时检测器的对齐支路退化为原始差分。
站点取文件名第一个 "_" 前的部分。位移单位为原图像素,约定 T1(y, x) ≈ T2(y − dy, x − dx)。

附带的 shifts/shift_table.json 是论文所用的表;生成它时,同站点中位数额外排除了
与本影像对裁自同一原始窗口的重叠影像对,部分影像对的位移与本脚本不同,
但对测试集结果没有影响(用本脚本的表,集成 IoU 同为 65.32%)。
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image


def gray(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return g - g.mean()


def hann2d(h, w):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def phase_corr(a, b):
    A, B = np.fft.fft2(a), np.fft.fft2(b)
    R = A * np.conj(B)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-8)).real
    H, W = a.shape
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return int(dx), int(dy)


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def shifted_ncc(a, b, dx, dy, margin):
    """把 b 平移 (dx, dy) 后与 a 在去掉 margin 的中央区域算 NCC;越界返回 None。"""
    H, W = a.shape
    ys, xs = slice(margin, H - margin), slice(margin, W - margin)
    ys2, xs2 = slice(margin - dy, H - margin - dy), slice(margin - dx, W - margin - dx)
    if ys2.start < 0 or xs2.start < 0 or ys2.stop > H or xs2.stop > W:
        return None
    return ncc(a[ys, xs], b[ys2, xs2])


def estimate_pair(t1_path, t2_path, margin=48, min_gain=0.05):
    """返回 (dx, dy, reliable)。"""
    a, b = gray(t1_path), gray(t2_path)
    window = hann2d(*a.shape)
    dx, dy = phase_corr(a * window, b * window)
    n0 = shifted_ncc(a, b, 0, 0, margin)
    n1 = shifted_ncc(a, b, dx, dy, margin)
    reliable = (n0 is not None and n1 is not None and n1 - n0 >= min_gain
                and abs(dx) <= margin and abs(dy) <= margin)
    return dx, dy, bool(reliable)


def _estimate(args):
    split_dir, name = args
    return name, estimate_pair(split_dir / "T1" / name, split_dir / "T2" / name)


def shift_table(split_dir, workers=8):
    """一个 split 目录 → {样本名(不含扩展名): [dx, dy]}。"""
    split_dir = Path(split_dir)
    names = sorted(p.name for p in (split_dir / "T1").iterdir())
    with Pool(workers) as pool:
        results = pool.map(_estimate, [(split_dir, name) for name in names], chunksize=16)
    estimates = {Path(name).stem: (dx, dy, ok) for name, (dx, dy, ok) in results}
    by_site = {}
    for stem, (dx, dy, ok) in estimates.items():
        if ok:
            by_site.setdefault(stem.split("_")[0], []).append((stem, dx, dy))
    table, counts = {}, {"self": 0, "site": 0, "zero": 0}
    for stem, (dx, dy, ok) in estimates.items():
        if ok:
            table[stem] = [float(dx), float(dy)]
            counts["self"] += 1
            continue
        donors = [(x, y) for other, x, y in by_site.get(stem.split("_")[0], []) if other != stem]
        if len(donors) >= 2:
            median = np.median(np.array(donors, dtype=np.float64), axis=0)
            table[stem] = [float(median[0]), float(median[1])]
            counts["site"] += 1
        else:
            table[stem] = [0.0, 0.0]
            counts["zero"] += 1
    return table, counts


if __name__ == "__main__":
    data_root = Path("/media/gs/sd10/RSE/split_v4")
    # 数据目录:<data_root>/{train,val,test}/{T1,T2,GT}/
    output = Path(__file__).resolve().parent / "shifts" / "shift_table_estimated.json"
    # 想用新估计的表训练时,把 config.json 的 shift_table 改成这个文件
    workers = 8
    tables = {}
    for split in ("train", "val", "test"):
        tables[split], counts = shift_table(data_root / split, workers)
        print(f"{split}: {len(tables[split])} 对,来源 {counts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tables), encoding="utf-8")
    print("写出", output)
