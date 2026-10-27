"""Stage 6 strong visual-reference baseline experiment.

Compares nearest-reference visual memory against VNE K=1 using the same target
train/good reference patch pool, reference normalization, and Stage 3A metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from experiments.ablation.ablation_utils import (
    aupro_0_30,
    code_file_hashes,
    environment_summary,
    git_status_summary,
    global_pixel_auroc,
    load_category_samples,
    load_mask,
    load_yaml,
    normalize_visual_scores,
    now_utc,
    project_root,
    reference_normalizer,
    resolve_path,
    safe_roc_auc,
    set_seed,
    sha256_file,
    topk_mean,
    upsample_score_map,
    validate_metric_ranges,
    write_dict_csv,
    write_yaml,
)
from src.dataset import load_mvtec_ad_category
from src.sara.vne import VisualNormalcyEncoder


METHODS = ["nearest_reference", "vne_k1"]
BASELINE_METHOD = "vne_k1"
REQUIRED_COLUMNS = [
    "category",
    "method",
    "image_auroc",
    "pixel_auroc_global",
    "aupro_0_30",
    "normal_image_count",
    "defect_image_count",
    "pixel_count",
    "reference_image_count",
    "reference_patch_count",
    "status",
]
DELTA_COLUMNS = [
    "row_type",
    "category",
    "method",
    "baseline",
    "image_auroc_delta",
    "pixel_auroc_global_delta",
    "aupro_0_30_delta",
    "image_auroc_delta_improved_count",
    "image_auroc_delta_tied_count",
    "image_auroc_delta_degraded_count",
    "pixel_auroc_global_delta_improved_count",
    "pixel_auroc_global_delta_tied_count",
    "pixel_auroc_global_delta_degraded_count",
    "aupro_0_30_delta_improved_count",
    "aupro_0_30_delta_tied_count",
    "aupro_0_30_delta_degraded_count",
]


class NearestReferenceScorer:
    """Nearest target-normal patch memory with fixed chunked cosine search."""

    def __init__(self, chunk_size: int = 1024) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self.reference: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> None:
        if features.size == 0:
            raise ValueError("empty reference features")
        self.reference = l2_normalize_rows(features.astype(np.float32))

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("nearest reference scorer is not fitted")
        return nearest_reference_scores(features, self.reference, self.chunk_size)


def l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (features / denom).astype(np.float32)


def nearest_reference_scores(test_features: np.ndarray, reference_features: np.ndarray,
                             chunk_size: int) -> np.ndarray:
    """Return 1 - max cosine similarity using fixed-size test chunks."""
    test = l2_normalize_rows(test_features.astype(np.float32))
    ref = l2_normalize_rows(reference_features.astype(np.float32))
    chunks = []
    for start in range(0, test.shape[0], int(chunk_size)):
        chunk = test[start:start + int(chunk_size)]
        max_sim = np.max(np.matmul(chunk, ref.T), axis=1)
        chunks.append(1.0 - max_sim)
    return np.concatenate(chunks).astype(np.float32)


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vne_reference_baseline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    fh = logging.FileHandler(path, encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def parse_categories(value: str, all_categories: Sequence[str]) -> List[str]:
    if not value or value == "all":
        return list(all_categories)
    categories = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [cat for cat in categories if cat not in all_categories]
    if unknown:
        raise ValueError(f"unknown categories: {unknown}")
    return categories


def parse_methods(value: str, default: Sequence[str]) -> List[str]:
    methods = list(default) if not value else [item.strip() for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    return methods


def resolve_config(args: argparse.Namespace) -> dict:
    cfg_path = resolve_path(args.config)
    cfg = load_yaml(cfg_path)
    all_categories = cfg["dataset"]["all_categories"]
    cfg["dataset"]["categories"] = parse_categories(
        args.categories or cfg["dataset"].get("categories", "all"),
        all_categories,
    )
    cfg["methods"] = parse_methods(args.methods, cfg.get("methods", METHODS))
    if args.device:
        cfg["model"]["device"] = args.device
    cfg["output"]["dir"] = args.output_dir or cfg["output"]["dir"]
    cfg["_config_path"] = str(cfg_path)
    return cfg


def output_dir(cfg: dict) -> Path:
    return resolve_path(cfg["output"]["dir"])


def dry_run(cfg: dict, out_dir: Path, resume: bool) -> None:
    print(f"Protocol: {cfg['experiment']['protocol_name']}")
    print("Protocol note: not a no-reference zero-shot setting; target train/good is used only as VNE visual reference.")
    print(f"Model: OpenCLIP {cfg['model']['name']} pretrained={cfg['model']['pretrained']}")
    print(f"Image size: {cfg['dataset']['image_size']}")
    print(f"Seed: {cfg['experiment']['seed']}")
    print(f"Data root: {resolve_path(cfg['dataset']['root'])}")
    print(f"Categories: {', '.join(cfg['dataset']['categories'])}")
    print(f"Methods: {', '.join(cfg['methods'])}")
    print(f"Similarity chunk size: {cfg['reference']['similarity_chunk_size']}")
    print(f"Output dir: {out_dir}")
    print(f"Resume: {resume}")
    print("Dry-run only: data samples and CLIP model are not loaded; no results are written.")


def collect_reference_pool(model, cfg: dict, category: str) -> tuple[np.ndarray, int]:
    data = load_mvtec_ad_category(str(resolve_path(cfg["dataset"]["root"])), category)
    max_ref = int(cfg["dataset"]["max_normal_reference"])
    selected = list(data.get("train_good", []))[:max_ref]
    patches = []
    for img_path in selected:
        image = Image.open(img_path).convert("RGB")
        _, patch_feat = model.encode_image(image)
        patch_np = patch_feat.detach().cpu().numpy()[0].reshape(-1, patch_feat.shape[-1])
        patches.append(patch_np)
    if not patches:
        raise ValueError(f"No target train/good reference images found for {category}")
    return np.concatenate(patches, axis=0), len(selected)


def build_visual_scorer(method: str, cfg: dict, reference_patches: np.ndarray):
    if method == "nearest_reference":
        scorer = NearestReferenceScorer(chunk_size=int(cfg["reference"]["similarity_chunk_size"]))
    elif method == "vne_k1":
        scorer = VisualNormalcyEncoder(
            n_components=int(cfg["vne"]["n_components"]),
            covariance_type=str(cfg["vne"].get("covariance_type", "full")),
        )
    else:
        raise ValueError(f"unsupported method: {method}")
    scorer.fit(reference_patches)
    stats = reference_normalizer(scorer.score_samples(reference_patches))
    return scorer, stats


def predict_maps_for_category(model, cfg: dict, category: str, method: str,
                              samples: Sequence, reference_patches: np.ndarray) -> List[np.ndarray]:
    image_size = tuple(cfg["dataset"]["image_size"])
    scorer, stats = build_visual_scorer(method, cfg, reference_patches)
    pred_maps: List[np.ndarray] = []
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        _, patch_feat = model.encode_image(image)
        patch_np = patch_feat.detach().cpu().numpy()[0]
        h, w, d = patch_np.shape
        patch_flat = patch_np.reshape(-1, d)
        raw_scores = scorer.score_samples(patch_flat)
        norm_scores = normalize_visual_scores(raw_scores, stats)
        pred_maps.append(upsample_score_map(norm_scores.reshape(h, w).astype(np.float32), image_size))
    return pred_maps


def compute_main_metrics(samples: Sequence, pred_maps: Sequence[np.ndarray],
                         gt_masks: Sequence[np.ndarray], cfg: dict) -> dict:
    labels = [sample.label for sample in samples]
    image_scores = [topk_mean(pred, float(cfg["metrics"]["image_topk"])) for pred in pred_maps]
    metrics = {
        "image_auroc": safe_roc_auc(labels, image_scores),
        "pixel_auroc_global": global_pixel_auroc(gt_masks, pred_maps),
        "aupro_0_30": aupro_0_30(gt_masks, pred_maps, max_fpr=float(cfg["metrics"]["aupro_max_fpr"])),
    }
    validate_metric_ranges(metrics)
    return metrics


def valid_completed_row(row: dict, expected_reference: dict | None = None) -> bool:
    if row.get("status") != "ok":
        return False
    try:
        validate_metric_ranges({
            "image_auroc": float(row["image_auroc"]),
            "pixel_auroc_global": float(row["pixel_auroc_global"]),
            "aupro_0_30": float(row["aupro_0_30"]),
        })
        if int(row["normal_image_count"]) < 0 or int(row["defect_image_count"]) <= 0:
            return False
        if int(row["reference_image_count"]) <= 0 or int(row["reference_patch_count"]) <= 0:
            return False
        if expected_reference is not None:
            category = row.get("category", "")
            expected = expected_reference.get(category)
            if expected is not None:
                return (
                    int(row["reference_image_count"]) == int(expected["reference_image_count"])
                    and int(row["reference_patch_count"]) == int(expected["reference_patch_count"])
                )
        return True
    except Exception:
        return False


def load_completed_rows(path: Path, expected_reference: dict | None = None) -> Dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        (row.get("category", ""), row.get("method", "")): row
        for row in rows
        if valid_completed_row(row, expected_reference=expected_reference)
    }


def append_json_list(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8") or "[]")
    rows.append(item)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def make_failure(category: str, method: str, exc: Exception) -> dict:
    return {
        "category": category,
        "method": method,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def aggregate_rows(rows: Sequence[dict], methods: Sequence[str]) -> dict:
    out = {}
    for method in methods:
        ok_rows = [row for row in rows if row["method"] == method and row["status"] == "ok"]
        entry = {"valid_category_count": len(ok_rows)}
        for src_key, dst_key in [
            ("image_auroc", "image_auroc"),
            ("pixel_auroc_global", "pixel_auroc"),
            ("aupro_0_30", "aupro"),
        ]:
            values = [float(row[src_key]) for row in ok_rows]
            entry[f"mean_{dst_key}"] = float(np.mean(values)) if values else None
            entry[f"std_{dst_key}"] = float(np.std(values)) if values else None
        out[method] = entry
    return out


def deltas_vs_vne_k1(rows: Sequence[dict]) -> List[dict]:
    by_category: Dict[str, Dict[str, dict]] = {}
    for row in rows:
        if row.get("status") == "ok":
            by_category.setdefault(row["category"], {})[row["method"]] = row
    out: List[dict] = []
    for category, method_rows in sorted(by_category.items()):
        base = method_rows.get(BASELINE_METHOD)
        row = method_rows.get("nearest_reference")
        if not base or not row:
            continue
        out.append({
            "row_type": "category_delta",
            "category": category,
            "method": "nearest_reference",
            "baseline": BASELINE_METHOD,
            "image_auroc_delta": float(row["image_auroc"]) - float(base["image_auroc"]),
            "pixel_auroc_global_delta": float(row["pixel_auroc_global"]) - float(base["pixel_auroc_global"]),
            "aupro_0_30_delta": float(row["aupro_0_30"]) - float(base["aupro_0_30"]),
        })
    return out


def summarize_deltas(delta_rows: Sequence[dict]) -> List[dict]:
    item = {
        "row_type": "summary",
        "category": "",
        "method": "nearest_reference",
        "baseline": BASELINE_METHOD,
    }
    for key in ["image_auroc_delta", "pixel_auroc_global_delta", "aupro_0_30_delta"]:
        values = [float(row[key]) for row in delta_rows]
        item[f"{key}_improved_count"] = sum(1 for v in values if v > 0.0)
        item[f"{key}_tied_count"] = sum(1 for v in values if v == 0.0)
        item[f"{key}_degraded_count"] = sum(1 for v in values if v < 0.0)
    return [item]


def write_delta_outputs(rows: Sequence[dict], path: Path) -> None:
    delta_rows = deltas_vs_vne_k1(rows)
    summary_rows = summarize_deltas(delta_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DELTA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in delta_rows + summary_rows:
            writer.writerow({field: row.get(field, "") for field in DELTA_COLUMNS})


def run_manifest(cfg: dict, out_dir: Path) -> dict:
    root = project_root()
    code_files = [
        Path(__file__),
        root / "experiments" / "vne_reference_baseline" / "vne_reference_baseline.yaml",
        root / "experiments" / "vne_ablation" / "run_vne_ablation.py",
        root / "experiments" / "ablation" / "ablation_utils.py",
        root / "src" / "sara" / "vne.py",
    ]
    return {
        "time_utc": now_utc(),
        "stage": cfg["experiment"]["stage"],
        "protocol": cfg["experiment"]["protocol_name"],
        "protocol_note": "not a no-reference zero-shot setting because target train/good is used only as VNE visual reference",
        "categories": cfg["dataset"]["categories"],
        "methods": cfg["methods"],
        "seed": cfg["experiment"]["seed"],
        "model": cfg["model"],
        "image_size": cfg["dataset"]["image_size"],
        "data_root": str(resolve_path(cfg["dataset"]["root"])),
        "output_dir": str(out_dir),
        "normal_reference_source": "target train/good only",
        "reference_image_limit": cfg["dataset"]["max_normal_reference"],
        "similarity_chunk_size": cfg["reference"]["similarity_chunk_size"],
        "excluded_from_training_or_tuning": ["target test", "defect images", "GT masks"],
        "metrics": {
            "image_auroc": "category-level AUROC over normal+defect test images using top-1% raw score",
            "pixel_auroc_global": "concatenate all normal+defect pixels for one category-level AUROC",
            "aupro_0_30": "connected GT defect regions, FPR from normal pixels, integrate 0-0.30 and normalize by 0.30",
            "normalization": "reference_minmax from the shared target train/good reference patch pool",
        },
        "method_definitions": {
            "nearest_reference": "1 - max cosine similarity(test_patch, reference_patches), chunked over test patches",
            "vne_k1": "VisualNormalcyEncoder with n_components=1 fit on the same reference patch pool",
        },
        "git": git_status_summary(root),
        "environment": environment_summary(),
        "code_file_sha256": code_file_hashes(code_files),
        "config_sha256": sha256_file(Path(cfg["_config_path"])),
    }


def ensure_output_schema(rows: Iterable[dict]) -> List[dict]:
    normalized = []
    for row in rows:
        out = {key: row.get(key, "") for key in REQUIRED_COLUMNS}
        for key, value in row.items():
            if key not in out:
                out[key] = value
        normalized.append(out)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/vne_reference_baseline/vne_reference_baseline.yaml")
    parser.add_argument("--categories", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(args)
    out_dir = output_dir(cfg)
    if args.dry_run:
        dry_run(cfg, out_dir, args.resume)
        return

    set_seed(int(cfg["experiment"]["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "run.log")
    write_yaml(cfg, out_dir / "resolved_config.yaml")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest(cfg, out_dir), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    per_category_path = out_dir / "per_category_metrics.csv"
    completed = load_completed_rows(per_category_path) if args.resume else {}
    rows: List[dict] = list(completed.values())
    failures_path = out_dir / "failures.json"
    if not failures_path.exists():
        failures_path.write_text("[]", encoding="utf-8")

    logger.info("Loading CLIP model")
    from src.models.clip_wrapper import CLIPWrapper

    model = CLIPWrapper(cfg["model"]["name"], cfg["model"]["pretrained"], cfg["model"]["device"])
    image_size = tuple(cfg["dataset"]["image_size"])

    for category in cfg["dataset"]["categories"]:
        logger.info("Loading samples and shared reference pool for %s", category)
        samples = load_category_samples(resolve_path(cfg["dataset"]["root"]), category, image_size)
        gt_masks = [load_mask(sample.mask_path, image_size) for sample in samples]
        normal_count = sum(1 for sample in samples if sample.label == 0)
        defect_count = sum(1 for sample in samples if sample.label == 1)
        pixel_count = int(sum(mask.size for mask in gt_masks))
        reference_patches, reference_image_count = collect_reference_pool(model, cfg, category)
        reference_patch_count = int(reference_patches.shape[0])

        for method in cfg["methods"]:
            key = (category, method)
            if key in completed:
                logger.info("Skipping completed %s/%s", category, method)
                continue
            try:
                logger.info("Running %s/%s", category, method)
                pred_maps = predict_maps_for_category(model, cfg, category, method, samples, reference_patches)
                metrics = compute_main_metrics(samples, pred_maps, gt_masks, cfg)
                row = {
                    "category": category,
                    "method": method,
                    "image_auroc": metrics["image_auroc"],
                    "pixel_auroc_global": metrics["pixel_auroc_global"],
                    "aupro_0_30": metrics["aupro_0_30"],
                    "normal_image_count": normal_count,
                    "defect_image_count": defect_count,
                    "pixel_count": pixel_count,
                    "reference_image_count": reference_image_count,
                    "reference_patch_count": reference_patch_count,
                    "status": "ok",
                }
                rows = [r for r in rows if not (r["category"] == category and r["method"] == method)]
                rows.append(row)
                write_dict_csv(ensure_output_schema(rows), per_category_path)
            except Exception as exc:
                append_json_list(failures_path, make_failure(category, method, exc))
                rows.append({
                    "category": category,
                    "method": method,
                    "image_auroc": "",
                    "pixel_auroc_global": "",
                    "aupro_0_30": "",
                    "normal_image_count": normal_count,
                    "defect_image_count": defect_count,
                    "pixel_count": pixel_count,
                    "reference_image_count": reference_image_count,
                    "reference_patch_count": reference_patch_count,
                    "status": "failed",
                })
                write_dict_csv(ensure_output_schema(rows), per_category_path)
                logger.exception("Failed %s/%s", category, method)

    aggregate = aggregate_rows(rows, cfg["methods"])
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_delta_outputs(rows, out_dir / "deltas_vs_vne_k1.csv")


if __name__ == "__main__":
    main()
