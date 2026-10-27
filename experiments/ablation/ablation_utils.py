"""Utilities for Stage 3A SARA ablation experiments.

This module is intentionally independent of existing experiment scripts. It
reuses model classes where safe, but implements protocol bookkeeping,
variant wiring, and paper-standard metrics locally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from skimage import measure


VARIANT_ORDER = [
    "text_raw",
    "text_sp",
    "vne_only",
    "vne_text_raw_fixed",
    "vne_text_sp_fixed",
    "vne_text_raw_af",
    "full_sara",
    "full_sara_refine",
]

VARIANT_CONFIGS = {
    "text_raw": dict(use_text=True, use_sp=False, use_vne=False, fusion="none", refine=False),
    "text_sp": dict(use_text=True, use_sp=True, use_vne=False, fusion="none", refine=False),
    "vne_only": dict(use_text=False, use_sp=False, use_vne=True, fusion="none", refine=False),
    "vne_text_raw_fixed": dict(use_text=True, use_sp=False, use_vne=True, fusion="fixed", refine=False),
    "vne_text_sp_fixed": dict(use_text=True, use_sp=True, use_vne=True, fusion="fixed", refine=False),
    "vne_text_raw_af": dict(use_text=True, use_sp=False, use_vne=True, fusion="af", refine=False),
    "full_sara": dict(use_text=True, use_sp=True, use_vne=True, fusion="af", refine=False),
    "full_sara_refine": dict(use_text=True, use_sp=True, use_vne=True, fusion="af", refine=True),
}

MVTEC_AD_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


@dataclass
class Sample:
    category: str
    split: str
    label: int
    image_path: Path
    mask_path: Optional[Path]
    defect_type: str
    image_sha256: str
    mask_sha256: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_sha256(path: Optional[Path]) -> str:
    return sha256_file(path) if path and path.exists() else ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(path: str | Path, root: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or project_root()) / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_list(value: Optional[str], default: Sequence[str]) -> List[str]:
    if value is None or value == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_variants(variants: Sequence[str]) -> List[str]:
    unknown = [v for v in variants if v not in VARIANT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}")
    return list(variants)


def fixed_fusion_weights(cfg: dict) -> Tuple[float, float]:
    vw = float(cfg["fusion"].get("fixed_visual_weight", 0.5))
    tw = float(cfg["fusion"].get("fixed_text_weight", 0.5))
    total = vw + tw
    if total <= 0:
        raise ValueError("fixed fusion weights must sum to a positive value")
    return vw / total, tw / total


def load_category_samples(data_root: Path, category: str, image_size: Tuple[int, int]) -> List[Sample]:
    from src.dataset import load_mvtec_ad_category

    data = load_mvtec_ad_category(str(data_root), category)
    samples: List[Sample] = []
    for img_path in data.get("test_good", []):
        samples.append(Sample(
            category=category,
            split="test_good",
            label=0,
            image_path=Path(img_path),
            mask_path=None,
            defect_type="good",
            image_sha256=sha256_file(Path(img_path)),
            mask_sha256="",
        ))
    for defect_type, info in data.get("test_defect", {}).items():
        images = list(info.get("images", []))
        masks = list(info.get("masks", []))
        if len(images) != len(masks):
            raise ValueError(
                f"{category}/{defect_type}: image-mask count mismatch "
                f"({len(images)} vs {len(masks)})"
            )
        for img_path, mask_path in zip(images, masks):
            samples.append(Sample(
                category=category,
                split="test_defect",
                label=1,
                image_path=Path(img_path),
                mask_path=Path(mask_path),
                defect_type=defect_type,
                image_sha256=sha256_file(Path(img_path)),
                mask_sha256=sha256_file(Path(mask_path)),
            ))
    if not samples:
        raise ValueError(f"No test samples found for category: {category}")
    return samples


def write_sample_manifest(samples: Sequence[Sample], path: Path) -> str:
    rows = []
    for idx, s in enumerate(samples):
        rows.append({
            "sample_id": idx,
            "category": s.category,
            "split": s.split,
            "label": s.label,
            "defect_type": s.defect_type,
            "image_path": str(s.image_path),
            "mask_path": str(s.mask_path) if s.mask_path else "",
            "image_sha256": s.image_sha256,
            "mask_sha256": s.mask_sha256,
        })
    write_dict_csv(rows, path)
    return sha256_text(json.dumps(rows, sort_keys=True))


def manifest_signature(samples: Sequence[Sample]) -> str:
    payload = [
        (str(s.image_path), str(s.mask_path) if s.mask_path else "", s.label, s.image_sha256, s.mask_sha256)
        for s in samples
    ]
    return sha256_text(json.dumps(payload, sort_keys=True))


def assert_same_manifest(reference: Sequence[Sample], other: Sequence[Sample]) -> None:
    if manifest_signature(reference) != manifest_signature(other):
        raise RuntimeError("Sample manifest mismatch across variants")


def write_dict_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_mask(mask_path: Optional[Path], image_size: Tuple[int, int]) -> np.ndarray:
    if mask_path is None:
        return np.zeros(image_size, dtype=np.uint8)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((image_size[1], image_size[0]), Image.NEAREST)
    return np.array(mask)


def upsample_score_map(score_map: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    score_t = torch.from_numpy(score_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(score_t, size=image_size, mode="bilinear", align_corners=False)
    return out.squeeze().detach().cpu().numpy().astype(np.float32)


def topk_mean(score_map: np.ndarray, frac: float) -> float:
    flat = np.asarray(score_map, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("empty score map")
    k = max(1, int(math.ceil(flat.size * frac)))
    return float(np.partition(flat, -k)[-k:].mean())


def safe_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) == 0 or len(set(labels)) < 2:
        return 0.5
    value = float(roc_auc_score(labels, scores))
    if not np.isfinite(value):
        raise ValueError("AUROC is not finite")
    return value


def global_pixel_auroc(gt_masks: Sequence[np.ndarray], pred_maps: Sequence[np.ndarray]) -> float:
    y_true = np.concatenate([(m > 127).astype(np.uint8).reshape(-1) for m in gt_masks])
    y_score = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in pred_maps])
    if not np.all(np.isfinite(y_score)):
        raise ValueError("pixel scores contain NaN or Inf")
    return safe_roc_auc(y_true.tolist(), y_score.tolist())


def aupro_0_30(gt_masks: Sequence[np.ndarray], pred_maps: Sequence[np.ndarray], max_fpr: float = 0.30) -> float:
    """Paper-standard AUPRO over FPR [0, 0.30].

    FPR is computed on all normal pixels. PRO is averaged over connected
    defect regions and integrated over the same FPR grid, normalized by 0.30.
    """
    regions = []
    normal_scores = []
    for gt, pred in zip(gt_masks, pred_maps):
        gt_bin = (gt > 127).astype(np.uint8)
        pred = np.asarray(pred, dtype=np.float64)
        normal_scores.append(pred[gt_bin == 0].reshape(-1))
        labeled = measure.label(gt_bin, connectivity=2)
        for region in measure.regionprops(labeled):
            region_mask = labeled == region.label
            regions.append(pred[region_mask].reshape(-1))

    if not regions:
        return 0.0
    normal_all = np.concatenate(normal_scores) if normal_scores else np.array([], dtype=np.float64)
    if normal_all.size == 0:
        return 0.0

    all_scores = np.concatenate([normal_all] + regions)
    thresholds = np.unique(all_scores)
    if thresholds.size > 512:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 512))
        thresholds = np.unique(thresholds)
    thresholds = thresholds[::-1]

    fprs, pros = [], []
    for thr in thresholds:
        fp = np.count_nonzero(normal_all >= thr)
        fpr = fp / max(1, normal_all.size)
        if fpr > max_fpr:
            continue
        overlaps = [np.count_nonzero(region_scores >= thr) / max(1, region_scores.size)
                    for region_scores in regions]
        fprs.append(float(fpr))
        pros.append(float(np.mean(overlaps)))

    if not fprs:
        return 0.0
    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    if fprs[0] > 0:
        fprs = np.insert(fprs, 0, 0.0)
        pros = np.insert(pros, 0, pros[0])
    if fprs[-1] < max_fpr:
        fprs = np.append(fprs, max_fpr)
        pros = np.append(pros, pros[-1])
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    value = float(integrate(pros, fprs) / max_fpr)
    if not np.isfinite(value):
        raise ValueError("AUPRO is not finite")
    return max(0.0, min(1.0, value))


def compute_category_metrics(samples: Sequence[Sample], pred_maps: Sequence[np.ndarray],
                             gt_masks: Sequence[np.ndarray], top_fracs: Sequence[float],
                             max_fpr: float) -> dict:
    labels = [s.label for s in samples]
    scores_top = {frac: [topk_mean(p, frac) for p in pred_maps] for frac in top_fracs}
    metrics = {
        "image_auroc_top1_raw": safe_roc_auc(labels, scores_top[0.01]),
        "image_auroc_top5_raw": safe_roc_auc(labels, scores_top[0.05]),
        "pixel_auroc_global": global_pixel_auroc(gt_masks, pred_maps),
        "aupro_0_30": aupro_0_30(gt_masks, pred_maps, max_fpr=max_fpr),
        "n_normal": int(sum(1 for x in labels if x == 0)),
        "n_defect": int(sum(1 for x in labels if x == 1)),
    }
    validate_metric_ranges(metrics)
    return metrics


def validate_metric_ranges(metrics: dict) -> None:
    for key, value in metrics.items():
        if key.startswith("n_"):
            if int(value) < 0:
                raise ValueError(f"invalid count for {key}: {value}")
            continue
        if not np.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"metric {key} out of [0,1]: {value}")


def reference_normalizer(scores: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores) + 1e-8),
    }


def normalize_visual_scores(scores: np.ndarray, stats: dict) -> np.ndarray:
    denom = stats["max"] - stats["min"]
    if denom <= 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - stats["min"]) / denom).astype(np.float32)


def adaptive_fusion_no_per_image_minmax(visual_norm: np.ndarray, text_score: np.ndarray,
                                        theta: float, gamma: float) -> np.ndarray:
    x = np.clip((visual_norm - theta) / max(gamma, 1e-8), -50.0, 50.0)
    alpha = 1.0 / (1.0 + np.exp(-x))
    return (alpha * visual_norm + (1.0 - alpha) * text_score).astype(np.float32)


def fixed_fusion(visual_norm: np.ndarray, text_score: np.ndarray,
                 visual_weight: float, text_weight: float) -> np.ndarray:
    return (visual_weight * visual_norm + text_weight * text_score).astype(np.float32)


def get_sp_checkpoint_path(cfg: dict, category: str) -> Path:
    sp_dir = resolve_path(cfg["checkpoints"]["sp_dir"])
    pattern = cfg["checkpoints"].get("sp_pattern", "sp_{category}.pt")
    return sp_dir / pattern.format(category=category)


def git_status_summary(root: Path) -> dict:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"is_git_repo": False, "status": "not_a_git_repository"}
    return {"is_git_repo": True, "status": "git_repository_present"}


def environment_summary() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def code_file_hashes(files: Iterable[Path]) -> Dict[str, str]:
    out = {}
    for path in files:
        if path.exists():
            out[str(path)] = sha256_file(path)
    return out


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_failures(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def append_failure(path: Path, failure: dict) -> None:
    failures = load_failures(path)
    failures.append(failure)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
