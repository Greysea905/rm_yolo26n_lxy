#!/usr/bin/env python3
"""数据集标注可视化：画框叠加图 + 统计图 + HTML 画廊。

用途：检查标注质量（框是否贴合、有无漏标/误标、目标尺寸分布）。

用法：
    python visualize.py                    # 默认路径，随机采样 60 张
    python visualize.py --n 200            # 采样 200 张
    python visualize.py --all              # 全部
    python visualize.py --out data/visualized  # 指定输出目录
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 把 src/ 加入 import 路径
from common import LABELED_IMAGES, LABELED_LABELS, VISUALIZED_DIR, parse_label, iter_image_label

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # 没装 matplotlib 时仍能画框，只是不出统计图
    HAS_MPL = False

CLASS_COLORS = ["#00E676", "#FF3D00", "#00B0FF", "#FFC400", "#E040FB", "#B2FF59"]


def draw_boxes(img: Image.Image, boxes) -> Image.Image:
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for c, x, y, w, h in boxes:
        x1, y1 = (x - w / 2) * W, (y - h / 2) * H
        x2, y2 = (x + w / 2) * W, (y + h / 2) * H
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, W // 400))
        draw.text((x1, max(0, y1 - 16)), f"cls{c}", fill=color)
    return img


def make_stats(all_boxes, per_img_counts, out: Path):
    """生成统计图并保存。"""
    if not HAS_MPL:
        print("  [跳过统计图] 未安装 matplotlib")
        return
    boxes = np.array(all_boxes)  # [N, 4] = w, h, area, ar
    w, h = boxes[:, 0], boxes[:, 1]
    area = w * h

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(area, bins=60, color="#4C9AFF")
    axes[0, 0].set_title(f"Box area (median {np.median(area):.4f}, mean {area.mean():.4f})")
    axes[0, 0].set_xlabel("normalized area")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(w, bins=60, color="#36B37E")
    axes[0, 1].set_title("Box width (normalized)")
    axes[0, 1].set_xlabel("width")

    axes[1, 0].hist(h, bins=60, color="#FFAB00")
    axes[1, 0].set_title("Box height (normalized)")
    axes[1, 0].set_xlabel("height")

    axes[1, 1].hist(per_img_counts, bins=range(0, max(per_img_counts) + 2), color="#FF5630")
    axes[1, 1].set_title("Boxes per image")
    axes[1, 1].set_xlabel("boxes per image")
    fig.tight_layout()
    fig.savefig(out / "stats.png", dpi=130)
    plt.close(fig)
    print(f"  [统计图] {out / 'stats.png'}")


def main():
    ap = argparse.ArgumentParser(description="标注可视化")
    ap.add_argument("--images", type=Path, default=LABELED_IMAGES)
    ap.add_argument("--labels", type=Path, default=LABELED_LABELS)
    ap.add_argument("--out", type=Path, default=VISUALIZED_DIR)
    ap.add_argument("--n", type=int, default=60, help="采样张数")
    ap.add_argument("--all", action="store_true", help="处理全部")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = args.out
    overlay_dir = out / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pairs = iter_image_label(args.images, args.labels)
    if args.all:
        sample = pairs
    else:
        sample = random.sample(pairs, min(args.n, len(pairs)))

    # 1) 画框叠加图
    all_boxes, per_img = [], []
    for ip, lp in sample:
        img = Image.open(ip).convert("RGB")
        boxes = parse_label(lp)
        draw_boxes(img, boxes)
        img.save(overlay_dir / (ip.stem + ".jpg"))
        all_boxes.extend((b[2], b[3], b[2] * b[3], b[2] / max(b[3], 1e-6)) for b in boxes)
        per_img.append(len(boxes))
    print(f"已生成 {len(sample)} 张叠加图 -> {overlay_dir}")

    # 2) 统计图（用全量标签，反映整个数据集）
    full_boxes, full_per_img = [], []
    for _, lp in iter_image_label(args.images, args.labels):
        boxes = parse_label(lp)
        full_per_img.append(len(boxes))
        full_boxes.extend((b[2], b[3], b[2] * b[3], b[2] / max(b[3], 1e-6)) for b in boxes)
    make_stats(full_boxes, full_per_img, out)

    # 3) HTML 画廊
    rows = []
    for ip, _ in sample:
        stem = ip.stem
        rows.append(
            f'<div class="card"><img src="overlay/{stem}.jpg" loading="lazy">'
            f'<div class="cap">{stem}</div></div>'
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>标注可视化</title>
<style>
body{{font-family:sans-serif;background:#111;color:#eee;margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
.card img{{width:100%;display:block;border-radius:6px}}
.cap{{padding:4px 6px;font-size:13px;color:#aaa}}
</style></head><body>
<h2>标注可视化（共 {len(sample)} 张）</h2>
<div class="grid">{''.join(rows)}</div>
</body></html>"""
    (out / "gallery.html").write_text(html)
    print(f"已生成画廊 -> {out / 'gallery.html'}（浏览器打开）")


if __name__ == "__main__":
    main()
