#!/usr/bin/env python3
"""在验证集上评估模型，输出 mAP50 / mAP50-95"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import DATASETS_DIR, CONFIG_DIR, RUNS_DIR, load_yaml, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(RUNS_DIR / "rm_car" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    dcfg = load_yaml(CONFIG_DIR / "data.yaml")
    dcfg["path"] = str(DATASETS_DIR)

    model = load_model(args.weights)
    metrics = model.val(data=dcfg, imgsz=args.imgsz, device=args.device)
    print(f"\nmAP50    = {metrics.box.map50:.4f}")
    print(f"mAP50-95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
