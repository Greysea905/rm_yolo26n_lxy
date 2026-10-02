#!/usr/bin/env python3
"""低质量图片筛选：模糊 / 过暗过曝 → 移入 quarantine（不直接删）。

（重复图请用 group_dups.py 处理，本脚本只管模糊/过暗。）

用法：
    python filter.py --dry-run    # 看指标分布
    python filter.py              # 把可疑图移到 quarantine
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 src/ 加入 import 路径
from common import LABELED_IMAGES, LABELED_LABELS, QUARANTINE_DIR, IMG_EXTS

try:
    import cv2

    def blur_score(a: np.ndarray) -> float:
        """Laplacian 方差，越小越模糊。"""
        return float(cv2.Laplacian(a, cv2.CV_64F).var())

except ImportError:  # 没有 cv2 时用梯度能量近似
    def blur_score(a: np.ndarray) -> float:
        gy, gx = np.gradient(a.astype(np.float32))
        return float((gx ** 2 + gy ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=LABELED_IMAGES)
    ap.add_argument("--labels", type=Path, default=LABELED_LABELS)
    ap.add_argument("--out", type=Path, default=QUARANTINE_DIR)
    ap.add_argument("--blur-thresh", type=float, default=30.0, help="模糊度低于此值判为模糊")
    ap.add_argument("--bright-low", type=float, default=15.0, help="平均亮度低于此值判为过暗")
    ap.add_argument("--bright-high", type=float, default=245.0, help="平均亮度高于此值判为过曝")
    ap.add_argument("--dry-run", action="store_true", help="只打印分布，不移动")
    args = ap.parse_args()

    paths = sorted(p for p in args.images.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        print("没有找到图片，检查 --images 路径"); return

    blur_scores, brights = [], []
    for p in paths:
        img = Image.open(p).convert("L")
        a = np.asarray(img)
        w, h = img.size
        scale = 512 / max(w, 1)
        small = np.asarray(img.resize((512, max(1, int(h * scale))), Image.LANCZOS))
        blur_scores.append(blur_score(small))
        brights.append(float(a.mean()))

    blur_scores = np.array(blur_scores)
    brights = np.array(brights)

    print(f"共 {len(paths)} 张")
    print(f"模糊度(越小越糊): min={blur_scores.min():.1f}  p5={np.percentile(blur_scores, 5):.1f}  "
          f"median={np.median(blur_scores):.1f}")
    print(f"亮度(0-255): min={brights.min():.1f}  p1={np.percentile(brights, 1):.1f}  "
          f"median={np.median(brights):.1f}  p99={np.percentile(brights, 99):.1f}  max={brights.max():.1f}")

    reasons: dict[Path, list[str]] = {}
    for p, b, br in zip(paths, blur_scores, brights):
        r = []
        if b < args.blur_thresh:
            r.append("blur")
        if br < args.bright_low:
            r.append("too_dark")
        if br > args.bright_high:
            r.append("too_bright")
        if r:
            reasons[p] = r

    if reasons:
        kinds = sorted({r[0] for r in reasons.values()})
        print(f"可疑图: {len(reasons)} 张（原因: {', '.join(kinds)}）")
    else:
        print("可疑图: 0 张")

    if args.dry_run:
        print("[dry-run] 未移动任何文件"); return

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    lines = []
    for p, rlist in reasons.items():
        shutil.move(str(p), args.out / "images" / p.name)
        lp = args.labels / (p.stem + ".txt")
        if lp.exists():
            shutil.move(str(lp), args.out / "labels" / lp.name)
        lines.append(f"{p.name}\t{','.join(rlist)}")
    (args.out / "report.txt").write_text("\n".join(lines))
    print(f"已移动 {len(reasons)} 张到 {args.out}，理由见 report.txt")


if __name__ == "__main__":
    main()
