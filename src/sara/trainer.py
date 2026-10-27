"""SP Trainer: 在 auxiliary categories 上训练 Semantic Projector.

从 auxiliary categories 的 normal + defect images 提取 patch features,
训练 SP 使 calibrated text embeddings 对齐 visual normality patterns.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm

from src.sara.sp import SemanticProjector, contrastive_alignment_loss
from src.sara.pipeline import SARAPipeline


def collect_patches_from_categories(
    clip_model,
    data_root: Path,
    categories: List[str],
    max_images: int = 100,
    max_patches_per_cat: int = 5000,
    protocol: str = "strict_zero_shot",
) -> tuple:
    """从 auxiliary categories 收集 normal patch features.

    SP 训练约束:
      - strict_zero_shot: 仅使用 auxiliary train_good (normal images)
      - few_shot_adaptation: 同上（SP 始终不接触 target/defect 数据）

    Returns:
        normal_patches: (N, 512) — auxiliary normal patches (no defect masks used)
    """
    from src.dataset import load_mvtec_ad_category

    all_normal = []

    for cat in categories:
        data = load_mvtec_ad_category(str(data_root), cat)
        n_per_cat = 0

        # Only use train_good — no test data, no defect masks
        for img_path in data.get("train_good", [])[:max_images]:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            _, patch_feat = clip_model.encode_image(image)
            patches = patch_feat.cpu().numpy()[0]  # (h, w, D)
            all_normal.append(patches.reshape(-1, patches.shape[-1]))
            n_per_cat += patches.shape[0] * patches.shape[1]
            if n_per_cat >= max_patches_per_cat:
                break

    normal = np.concatenate(all_normal, axis=0) if all_normal else np.zeros((0, 512))
    # SP training does not use defect patches under strict zero-shot
    return normal, None


def train_sp_on_auxiliary(
    sara: SARAPipeline,
    normal_patches: np.ndarray,
    normal_text: List[str],
    anomaly_text: List[str],
    epochs: int = 50,
    batch_size: int = 128,
    verbose: bool = True,
) -> List[float]:
    """Simple wrapper: delegates to SARAPipeline.train_sp()."""
    return sara.train_sp(
        normal_patches,
        normal_text, anomaly_text,
        epochs=epochs, batch_size=batch_size, verbose=verbose,
    )
