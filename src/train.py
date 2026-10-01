#!/usr/bin/env python3
"""训练入口：读 config/train.yaml 里的超参，调用 ultralytics 训练。

用法：
    python train.py                          # 用 yaml 里的超参
    python train.py --epochs 100 --imgsz 1280   # 命令行覆盖最常用的几项
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, DATASETS_DIR, CONFIG_DIR, RUNS_DIR, load_yaml, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, default=CONFIG_DIR / "train.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    for key in ("epochs", "imgsz", "batch", "device"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v

    # data.yaml 里的 path 绝对化，避免相对路径歧义
    dcfg = load_yaml(PROJECT_ROOT / cfg["data"])
    dcfg["path"] = str(DATASETS_DIR)

    model = load_model(PROJECT_ROOT / cfg["model"])
    model.train(
        data=dcfg,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg.get("workers", 8),
        project=str(RUNS_DIR),
        name=cfg.get("name", "rm_car"),
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
