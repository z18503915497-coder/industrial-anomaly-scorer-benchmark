# ============================================================
# 数据集管理模块
# 功能: 加载 MVTec AD / VisA，按实验条件分组
# ============================================================

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image


def load_mvtec_ad_category(
    root: str,
    category: str,
) -> Dict[str, List[Path]]:
    """加载 MVTec AD 单个类别的所有图像路径。

    MVTec AD 目录结构:
      {root}/{category}/
        train/good/   ← 正常训练样本
        test/good/    ← 正常测试样本
        test/{defect_type}/  ← 缺陷测试样本
        ground_truth/{defect_type}/  ← 像素级标注

    Args:
        root: MVTec AD 根目录
        category: 类别名称

    Returns:
        {
            "train_good": [Path, ...],
            "test_good": [Path, ...],
            "test_defect": {
                "defect_type_1": {
                    "images": [Path, ...],
                    "masks": [Path, ...],
                },
                ...
            }
        }
    """
    cat_dir = Path(root) / category

    if not cat_dir.exists():
        raise FileNotFoundError(f"类别目录不存在: {cat_dir}")

    result = {}

    # 训练集正常样本
    train_good_dir = cat_dir / "train" / "good"
    if train_good_dir.exists():
        result["train_good"] = sorted(train_good_dir.glob("*.png"))

    # 测试集正常样本
    test_good_dir = cat_dir / "test" / "good"
    if test_good_dir.exists():
        result["test_good"] = sorted(test_good_dir.glob("*.png"))

    # 测试集缺陷样本
    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"
    result["test_defect"] = {}

    if test_dir.exists():
        for defect_dir in sorted(test_dir.iterdir()):
            if defect_dir.is_dir() and defect_dir.name != "good":
                defect_type = defect_dir.name
                images = sorted(defect_dir.glob("*.png"))

                # 对应的 ground truth mask
                masks = []
                gt_defect_dir = gt_dir / defect_type
                if gt_defect_dir.exists():
                    masks = sorted(gt_defect_dir.glob("*.png"))

                if images:
                    result["test_defect"][defect_type] = {
                        "images": images,
                        "masks": masks,
                    }

    return result


def compute_defect_area_percent(mask_path: Path) -> float:
    """基于 GT mask 自身尺寸计算缺陷面积百分比。

    Args:
        mask_path: GT mask 路径（二值图像，非零像素=缺陷区域）

    Returns:
        缺陷面积百分比 [0, 100]
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    total_pixels = mask.size
    defect_pixels = (mask > 127).sum()
    ratio = (defect_pixels / total_pixels) * 100
    return float(ratio)


def compute_rms_contrast(image: np.ndarray) -> float:
    """计算图像的 RMS 对比度。

    RMS contrast = std(pixel_intensities) / mean(pixel_intensities)
    如果均值为 0，返回 0。
    """
    image = image.astype(np.float64)
    mean_val = image.mean()
    if mean_val == 0:
        return 0.0
    return image.std() / mean_val


def group_by_area(
    defect_dict: Dict,
    levels: List[str],
    small_max: float,
    medium_max: float,
    image_size: tuple = (224, 224),
    seed: int = 42,
    max_per_condition: int = 15,
    print_stats: bool = True,
) -> Dict[str, List[Dict]]:
    """基于 GT mask 的真实缺陷面积百分比分组。

    分类规则（对应论文实验设计）:
      small:  ratio < small_max   （默认 1%）
      medium: small_max <= ratio <= medium_max   （默认 1%-5%）
      large:  ratio > medium_max   （默认 >5%）

    Args:
        defect_dict: load_mvtec_ad_category() 返回的 test_defect
        levels: ["small", "medium", "large"]
        small_max: "small" 的最大面积(%)
        medium_max: "medium" 的最大面积(%)
        seed: 随机种子
        max_per_condition: 每个条件最多保留的样本数
        print_stats: 是否打印每类统计

    Returns:
        {"small": [{image, mask, area, defect_type}, ...], "medium": [...], "large": [...]}
    """
    rng = random.Random(seed)
    grouped = defaultdict(list)

    for defect_type, data in defect_dict.items():
        images = data["images"]
        masks = data["masks"]

        for img_path, mask_path in zip(images, masks):
            area = compute_defect_area_percent(mask_path)  # 基于 mask 自身尺寸
            if area < small_max:
                level = "small"
            elif area <= medium_max:
                level = "medium"
            else:
                level = "large"

            grouped[level].append(
                {"image": img_path, "mask": mask_path, "area": area, "defect_type": defect_type}
            )

    # 统计输出
    if print_stats:
        total = sum(len(v) for v in grouped.values())
        print(f"  [面积分布] total={total}  ", end="")
        for lv in levels:
            items = grouped.get(lv, [])
            areas = [it["area"] for it in items]
            rng_str = f"{min(areas):.2f}-{max(areas):.2f}%" if areas else "—"
            print(f"{lv}={len(items)}({rng_str})  ", end="")
        print()

    # 每层随机采样
    result = {}
    for level in levels:
        items = grouped.get(level, [])
        if len(items) > max_per_condition:
            items = rng.sample(items, max_per_condition)
        result[level] = items

    return result


def group_by_area_and_contrast(
    defect_dict: Dict,
    area_levels: List[str],
    small_max: float,
    medium_max: float,
    contrast_levels: List[str],
    seed: int = 42,
    max_per_condition: int = 15,
    print_stats: bool = True,
) -> Dict[str, Dict[str, List[Dict]]]:
    """将缺陷样本同时按 defect_area 和 texture_contrast 交叉分组。

    返回结构:
      {area_level: {contrast_level: [{image, mask, area, defect_type, contrast}, ...]}}

    面积阈值沿用 group_by_area 的规则；
    对比度用 RMS contrast，取该类别所有缺陷样本的 median split。
    """
    rng = random.Random(seed)

    # ---- 第一步：收集所有缺陷样本 + 计算面积/对比度 ----
    raw = []
    for defect_type, data in defect_dict.items():
        for img_path, mask_path in zip(data["images"], data["masks"]):
            area = compute_defect_area_percent(mask_path)
            img_gray = np.array(Image.open(img_path).convert("L"))
            contrast = compute_rms_contrast(img_gray)
            raw.append({
                "image": img_path, "mask": mask_path,
                "area": area, "contrast": contrast,
                "defect_type": defect_type,
            })

    if not raw:
        return {lv: {cl: [] for cl in contrast_levels} for lv in area_levels}

    # ---- 第二步：按面积分桶 ----
    area_buckets = defaultdict(list)
    for s in raw:
        area = s["area"]
        if area < small_max:
            area_buckets["small"].append(s)
        elif area <= medium_max:
            area_buckets["medium"].append(s)
        else:
            area_buckets["large"].append(s)

    # ---- 第三步：每个面积桶内按对比度 median split ----
    result = {}
    for area_lv in area_levels:
        bucket = area_buckets.get(area_lv, [])
        result[area_lv] = {}

        if len(bucket) <= 1:
            # 样本太少，全放第一个 contrast level
            for cl in contrast_levels:
                result[area_lv][cl] = bucket if cl == contrast_levels[0] else []
        else:
            contrasts = [s["contrast"] for s in bucket]
            median_c = float(np.median(contrasts))
            low_group, high_group = [], []
            for s in bucket:
                if s["contrast"] >= median_c:
                    high_group.append(s)
                else:
                    low_group.append(s)
            result[area_lv]["low"] = low_group
            result[area_lv]["high"] = high_group

    # ---- 第四步：采样 + 统计 ----
    if print_stats:
        print(f"  [面积×对比度分布]", end="")
        for area_lv in area_levels:
            for cl in contrast_levels:
                items = result.get(area_lv, {}).get(cl, [])
                areas = [it["area"] for it in items]
                rng_str = f"[{min(areas):.2f}-{max(areas):.2f}%]" if areas else "—"
                print(f" {area_lv}/{cl}={len(items)}{rng_str}", end="")
        print()

    for area_lv in area_levels:
        for cl in contrast_levels:
            items = result[area_lv][cl]
            if len(items) > max_per_condition:
                result[area_lv][cl] = rng.sample(items, max_per_condition)

    return result


def group_by_contrast(
    images: List[Path],
    levels: List[str],
    seed: int = 42,
    max_per_condition: int = 15,
) -> Dict[str, List[Path]]:
    """按 RMS 对比度分组（用于正常样本或异常样本的背景分析）。

    使用中位数分割: >= median → "high", < median → "low"
    """
    rng = random.Random(seed)

    # 计算每张图的对比度
    contrasts = []
    for p in images:
        img = np.array(Image.open(p).convert("L"))
        contrasts.append(compute_rms_contrast(img))

    median_contrast = np.median(contrasts)

    grouped = {"low": [], "high": []}
    for p, c in zip(images, contrasts):
        if c >= median_contrast:
            grouped["high"].append(p)
        else:
            grouped["low"].append(p)

    result = {}
    for level in levels:
        items = grouped.get(level, [])
        if len(items) > max_per_condition:
            items = rng.sample(items, max_per_condition)
        result[level] = items

    return result


def load_all_categories(
    root: str,
    categories: List[str],
) -> Dict[str, Dict]:
    """加载多个类别的数据。"""
    data = {}
    for cat in categories:
        data[cat] = load_mvtec_ad_category(root, cat)
    return data
