#!/usr/bin/env python3
"""推理：对一批图片跑检测，打印 bbox list。

用法：
    python run.py --weights runs/rm_car/weights/best.pt --source datasets/images/test
    python run.py --source datasets/images/test --save   # 顺便保存画框结果图
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import DATASETS_DIR, RUNS_DIR, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(RUNS_DIR / "rm_car" / "weights" / "best.pt"))
    ap.add_argument("--source", type=str, default=str(DATASETS_DIR / "images" / "test"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true", help="保存画框结果图")
    args = ap.parse_args()

    model = load_model(args.weights)
    results = model.predict(args.source, imgsz=args.imgsz, conf=args.conf, save=args.save)

    print(f"共 {len(results)} 张图")
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            print(f"{Path(r.path).name}: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] conf={conf:.3f} cls={cls}")


if __name__ == "__main__":
    main()
