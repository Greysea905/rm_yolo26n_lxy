#!/usr/bin/env python3
"""labelImg 编辑工作区：把「images + labels」合并到同一目录供 labelImg 编辑，再发布回原位置。

三步：
    1) prepare ：把 --images 和 --labels 里的图+标注合并到 data/labelimg_edit/（labelImg 要求同目录）
    2) 编辑    ：在 labelImg 里改框 / 删图（删图 = 直接删 labelimg_edit 里的 .jpg + .txt）
    3) publish ：把 data/labelimg_edit/ 拆分回 --images 和 --labels（删除自动生效）

--images / --labels 默认指 data/labeled，但可指向任意目录（如 datasets/images/train）。

用法（默认 data/labeled）：
    python labelimg_prep.py               # prepare
    labelImg data/labelimg_edit data/labelimg_edit/classes.txt data/labelimg_edit   # 编辑
    python labelimg_prep.py --publish     # publish 回 data/labeled

用法（编辑 datasets 的 train，改完放回原位置）：
    python labelimg_prep.py --images datasets/images/train --labels datasets/labels/train
    labelImg data/labelimg_edit data/labelimg_edit/classes.txt data/labelimg_edit
    python labelimg_prep.py --images datasets/images/train --labels datasets/labels/train --publish

注意：data/labelimg_edit/ 是共享中间工作区，每次 prepare 都会先清空；一次只处理一个集合，
      publish 发布完再 prepare 下一个，否则上一个未发布的改动会被清掉。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 src/ 加入 import 路径
from common import LABELED_IMAGES, LABELED_LABELS, PROJECT_ROOT, IMG_EXTS

EDIT_DIR = PROJECT_ROOT / "data" / "labelimg_edit"
CLASSES = ["car"]   # 单类（按实际类别名改）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=LABELED_IMAGES)
    ap.add_argument("--labels", type=Path, default=LABELED_LABELS)
    ap.add_argument("--edit-dir", type=Path, default=EDIT_DIR)
    ap.add_argument("--publish", action="store_true", help="用 edit-dir 重建 images/ + labels/（含删除）")
    ap.add_argument("--dry-run", action="store_true", help="配合 --publish：只看会发布多少，不动")
    args = ap.parse_args()

    if args.publish:
        imgs = sorted(f for f in args.edit_dir.glob("*") if f.suffix.lower() in IMG_EXTS)
        txts = sorted(f for f in args.edit_dir.glob("*.txt") if f.name != "classes.txt")

        # 孤儿检查（图缺标注 / 标注缺图）
        img_stems = {f.stem for f in imgs}
        txt_stems = {f.stem for f in txts}
        if img_stems - txt_stems:
            print(f"⚠️ {len(img_stems - txt_stems)} 张图缺标注: {sorted(img_stems - txt_stems)[:10]} …")
        if txt_stems - img_stems:
            print(f"⚠️ {len(txt_stems - img_stems)} 个标注缺图: {sorted(txt_stems - img_stems)[:10]} …")

        print(f"将发布 {len(imgs)} 图 + {len(txts)} 标注")
        if args.dry_run:
            return

        # 清空并重建（原始数据仍在 task_4/toolkit/data/ 有备份，可放心）
        if args.images.exists():
            shutil.rmtree(args.images)
        if args.labels.exists():
            shutil.rmtree(args.labels)
        args.images.mkdir(parents=True, exist_ok=True)
        args.labels.mkdir(parents=True, exist_ok=True)
        for f in imgs:
            shutil.copy2(f, args.images / f.name)
        for f in txts:
            shutil.copy2(f, args.labels / f.name)
        print(f"已发布 → {args.images} / {args.labels}")
        return

    # 准备：清空旧工作区，再合并 images + labels（避免残留上一次的图）
    if args.edit_dir.exists():
        shutil.rmtree(args.edit_dir)
    args.edit_dir.mkdir(parents=True, exist_ok=True)
    (args.edit_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    n = 0
    for ip in sorted(args.images.glob("*")):
        lp = args.labels / (ip.stem + ".txt")
        if not lp.exists():
            continue
        shutil.copy2(ip, args.edit_dir / ip.name)
        shutil.copy2(lp, args.edit_dir / (ip.stem + ".txt"))
        n += 1
    print(f"已合并 {n} 对（图 + txt）到 {args.edit_dir}")
    print(f"下一步: labelImg {args.edit_dir}")


if __name__ == "__main__":
    main()
