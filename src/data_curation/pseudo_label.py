#!/usr/bin/env python3
"""伪标签：用训练好的模型给未标注图打标签，输出到 datasets/{images,labels}/pseudo/

进行了可视化检查和在 labelimg 人工调整，确认无误后已并入 train 重训模型

用法：
    python pseudo_label.py                     # 默认用 runs/rm_car/weights/best.pt
    python pseudo_label.py --conf 0.7          # 提高阈值
    python pseudo_label.py --weights <路径>    # 指定权重文件
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 src/ 加入 import 路径
from common import DATASETS_DIR, RUNS_DIR, UNLABELED_IMAGES, IMG_EXTS, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=UNLABELED_IMAGES)
    ap.add_argument("--weights", type=str, default=str(RUNS_DIR / "rm_car" / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值，低于此值的框丢弃")
    ap.add_argument("--name", type=str, default="pseudo", help="输出子目录名（用于对比不同阈值）")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    out_img = DATASETS_DIR / "images" / args.name
    out_lbl = DATASETS_DIR / "labels" / args.name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)
    images = sorted(p for p in args.images.glob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"共 {len(images)} 张未标注图，阈值 conf>{args.conf}")

    n_with_box = 0
    n_empty = 0
    n_boxes = 0
    for ip in images:
        results = model.predict(str(ip), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        r = results[0]
        orig_h, orig_w = r.orig_shape

        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls = int(b.cls[0])
                xc = (x1 + x2) / 2 / orig_w
                yc = (y1 + y2) / 2 / orig_h
                bw = (x2 - x1) / orig_w
                bh = (y2 - y1) / orig_h
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                n_boxes += 1
            n_with_box += 1
        else:
            n_empty += 1

        shutil.copy2(ip, out_img / ip.name)
        (out_lbl / (ip.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"完成：{n_with_box} 张有框，{n_empty} 张无框（可能漏检，重点抽查），共 {n_boxes} 个框")
    print(f"输出 -> {out_img} / {out_lbl}")


if __name__ == "__main__":
    main()
