#!/usr/bin/env python3
"""划分 train/val（可选 test，本项目最终只划分 train/val）

用法：
    python split.py                              # 默认 0.85/0.15（train/val，无 test）
    python split.py --ratio 0.8 0.2              # 自定义 train/val
    python split.py --ratio 0.8 0.1 0.1          # train/val/test
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 src/ 加入 import 路径
from common import LABELED_IMAGES, LABELED_LABELS, DATASETS_DIR, IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=LABELED_IMAGES)
    ap.add_argument("--labels", type=Path, default=LABELED_LABELS)
    ap.add_argument("--out", type=Path, default=DATASETS_DIR)
    ap.add_argument("--ratio", type=float, nargs="+", default=[0.85, 0.15], help="train val [test] 比例")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = sorted(p for p in args.images.glob("*") if p.suffix.lower() in IMG_EXTS)
    images = [p for p in images if (args.labels / (p.stem + ".txt")).exists()]

    random.seed(args.seed)
    random.shuffle(images)
    n = len(images)
    tr, va = args.ratio[0], args.ratio[1]
    i_va = int(n * tr)
    if len(args.ratio) > 2:
        te = args.ratio[2]
        i_te = int(n * (tr + va))
        splits = {"train": images[:i_va], "val": images[i_va:i_te], "test": images[i_te:]}
    else:
        splits = {"train": images[:i_va], "val": images[i_va:]}

    for name, lst in splits.items():
        img_dir = args.out / "images" / name
        lbl_dir = args.out / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for ip in lst:
            shutil.copy2(ip, img_dir / ip.name)
            shutil.copy2(args.labels / (ip.stem + ".txt"), lbl_dir / (ip.stem + ".txt"))
        print(f"{name}: {len(lst)} 张")

    print(f"完成 -> {args.out}/images/ + labels/")


if __name__ == "__main__":
    main()
