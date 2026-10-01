"""项目共享工具：路径解析 + YOLO 标注解析 + 配置/模型加载。

所有脚本从这里 import，保持各脚本简短、可复用。

导入方式：
  - src/ 下的脚本（train/val/run）直接 `from common import ...`
  - src/data_curation/ 下的脚本先加两行把 src/ 加进 import 路径：
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    再 `from common import ...`
"""
from __future__ import annotations

from pathlib import Path

# ---- 路径（项目根 = 本文件上一级，即 src/common.py 的上一级）----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LABELED_IMAGES = DATA_DIR / "labeled" / "images"
LABELED_LABELS = DATA_DIR / "labeled" / "labels"
UNLABELED_IMAGES = DATA_DIR / "unlabeled" / "images"
VISUALIZED_DIR = DATA_DIR / "visualized"
QUARANTINE_DIR = DATA_DIR / "quarantine"

DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---- YOLO 标注解析 ----
def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """解析 YOLO 标注，返回 [(class, xc, yc, w, h)，归一化]。"""
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split()
        c = int(float(p[0]))
        x, y, w, h = (float(v) for v in p[1:5])
        boxes.append((c, x, y, w, h))
    return boxes


def iter_image_label(images: Path, labels: Path) -> list[tuple[Path, Path]]:
    """按文件名把图片和标注配对，返回 [(image_path, label_path)]。"""
    pairs = []
    for ip in sorted(images.glob("*")):
        if ip.suffix.lower() not in IMG_EXTS:
            continue
        lp = labels / (ip.stem + ".txt")
        if lp.exists():
            pairs.append((ip, lp))
    return pairs


# ---- 配置 / 模型加载（懒加载，避免 data_curation 脚本依赖 ultralytics）----
def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(weights: str | Path):
    """加载 YOLO 模型。weights 为 .pt 路径或模型名（找不到会在线下载）。"""
    from ultralytics import YOLO
    return YOLO(str(weights))
