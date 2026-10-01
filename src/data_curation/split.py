#!/usr/bin/env python3
"""划分 train/val/test，可选先按 dHash 去重（避免近重复帧跨 split 泄漏）。

用法：
    python split.py                     # 默认 0.8/0.1/0.1，复制文件
    python split.py --dedup-dist 6      # 先去掉 dHash 距离<=6 的近重复图
    python split.py --ratio 0.7 0.15 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 src/ 加入 import 路径
from common import LABELED_IMAGES, LABELED_LABELS, DATASETS_DIR, IMG_EXTS


def dedup(images: list[Path], dist: int) -> list[Path]:
    """去掉 dHash 距离<=dist 的近重复图，保留每组第一张。"""
    from PIL import Image
    import numpy as np

    def _dhash(p: Path):
        g = Image.open(p).convert("L").resize((9, 8), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
        return (a[:, 1:] > a[:, :-1]).flatten()

    hashes = [_dhash(p) for p in images]
    kept = []
    for i in range(len(images)):
        if not any((hashes[i] != hashes[j]).sum() <= dist for j in kept):
            kept.append(i)
    return [images[i] for i in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=LABELED_IMAGES)
    ap.add_argument("--labels", type=Path, default=LABELED_LABELS)
    ap.add_argument("--out", type=Path, default=DATASETS_DIR)
    ap.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="train val test 比例")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup-dist", type=float, default=None, help="dHash 距离<=此值判为重复并去掉")
    args = ap.parse_args()

    images = sorted(p for p in args.images.glob("*") if p.suffix.lower() in IMG_EXTS)
    images = [p for p in images if (args.labels / (p.stem + ".txt")).exists()]

    if args.dedup_dist is not None:
        n0 = len(images)
        images = dedup(images, args.dedup_dist)
        print(f"去重：{n0} -> {len(images)} 张（去掉 {n0 - len(images)} 张近重复）")

    random.seed(args.seed)
    random.shuffle(images)
    tr, va, te = args.ratio
    n = len(images)
    i_va = int(n * tr)
    i_te = int(n * (tr + va))
    splits = {"train": images[:i_va], "val": images[i_va:i_te], "test": images[i_te:]}

    for name, lst in splits.items():
        img_dir = args.out / "images" / name
        lbl_dir = args.out / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for ip in lst:
            shutil.copy2(ip, img_dir / ip.name)
            shutil.copy2(args.labels / (ip.stem + ".txt"), lbl_dir / (ip.stem + ".txt"))
        print(f"{name}: {len(lst)} 张")

    print(f"完成 -> {args.out}/images/{{train,val,test}} + labels/{{train,val,test}}")


if __name__ == "__main__":
    main()
