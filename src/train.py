#!/usr/bin/env python3
"""训练入口：读 config/train.yaml 里的超参，调用 ultralytics 训练"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, DATASETS_DIR, CONFIG_DIR, RUNS_DIR, load_yaml, dump_yaml, load_model


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

    dcfg = load_yaml(PROJECT_ROOT / cfg["data"])
    data_tmp = RUNS_DIR / "_data.yaml"
    dump_yaml(
        {"path": str(DATASETS_DIR), "train": dcfg["train"], "val": dcfg["val"], "names": dcfg["names"]},
        data_tmp,
    )

    model = load_model(PROJECT_ROOT / cfg["model"])

    # 基础参数
    train_args = {
        "data": str(data_tmp),
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "device": cfg["device"],
        "workers": cfg.get("workers", 8),
        "project": str(RUNS_DIR),
        "name": cfg.get("name", "rm_car"),
        "exist_ok": False,  # 目录已存在时自动加序号（rm_car2/rm_car3），避免覆盖旧结果
    }
    # 可选超参
    for key in ("lr0", "lrf", "cos_lr", "warmup_epochs", "patience", "save_period", "close_mosaic",
                "mosaic", "optimizer", "weight_decay", "hsv_h", "hsv_s", "hsv_v", "fliplr"):
        if key in cfg:
            train_args[key] = cfg[key]

    model.train(**train_args)


if __name__ == "__main__":
    main()
