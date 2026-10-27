"""Read-only audit of the near-zero A1 reference normalization range.

Part 1 summarizes the frozen CSV fields.
Part 2 checks, on CPU synthetic data, whether min-max normalization with a
near-zero or large reference range changes any rank-based metric used in the
paper (Image AUROC, Global Pixel AUROC, AUPRO@0.30).
No formal data, runner, or frozen result is modified.
"""
import csv
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
CSV_PATH = os.path.join(ROOT, "merged_results_v3v4v5_27cat_135_20261025",
                        "EXPERIMENT_AB_MERGED_135_RESULTS.csv")
PROJ = os.path.abspath(os.path.join(ROOT, ".."))

sys.path.insert(0, PROJ)
from experiments.ablation.ablation_utils import (  # noqa: E402
    aupro_0_30, global_pixel_auroc, safe_roc_auc, topk_mean,
)

METHODS = ["A0_vne_k1", "A1_nearest_full", "A2_nearest_coreset5",
           "A3_nearest_coreset10", "A4_nearest_random5"]


def load_rows():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(rows):
    out = {}
    for ds in ("MVTec", "VisA"):
        out[ds] = {}
        for m in METHODS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            rr = np.array([float(r["ref_raw_range"]) for r in sel])
            b0 = np.array([int(r["test_norm_below0"]) for r in sel])
            a1 = np.array([int(r["test_norm_above1"]) for r in sel])
            out[ds][m] = {
                "n": len(sel),
                "ref_range_min": float(rr.min()),
                "ref_range_median": float(np.median(rr)),
                "ref_range_max": float(rr.max()),
                "denominator_above_1e-12": bool((rr > 1e-12).all()),
                "below0_total": int(b0.sum()),
                "above1_total": int(a1.sum()),
                "above1_max_per_category": int(a1.max()),
            }
    return out


def minmax(x, lo, hi):
    return (x - lo) / (hi - lo)


def synthetic_metric_check(seed=42):
    rng = np.random.RandomState(seed)
    n_images = 40
    h = w = 64
    masks, raw_maps = [], []
    labels = []
    for i in range(n_images):
        mask = np.zeros((h, w), dtype=np.uint8)
        # half the images contain one or two compact defect regions
        if i % 2 == 0:
            cy, cx = rng.randint(16, 48, size=2)
            yy, xx = np.ogrid[:h, :w]
            mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= 25] = 255
            if i % 4 == 0:
                cy2, cx2 = rng.randint(16, 48, size=2)
                mask[(yy - cy2) ** 2 + (xx - cx2) ** 2 <= 16] = 255
        base = rng.gamma(shape=2.0, scale=1.0, size=(h, w))
        signal = mask.astype(np.float64) * rng.uniform(0.15, 0.8)
        raw_maps.append((base + signal).astype(np.float64))
        masks.append(mask)
        labels.append(int((mask > 127).any()))

    def metrics(maps):
        img_scores = [topk_mean(m, 0.01) for m in maps]
        return {
            "image_auroc": float(safe_roc_auc(labels, img_scores)),
            "pixel_auroc": float(global_pixel_auroc(masks, maps)),
            "aupro": float(aupro_0_30(masks, maps)),
            "image_scores": img_scores,
        }

    raw = metrics(raw_maps)
    results = {}
    for name, scale in [("range_1e-6", 1e-6), ("range_1e-2", 1e-2), ("range_1e2", 1e2)]:
        # Simulate a reference range by a positive affine transform of the raw
        # scores; the transform is exactly what min-max normalization applies.
        shifted = [m * scale for m in raw_maps]
        norm = metrics(shifted)
        # float32 cast, as in normalize_visual_scores
        shifted32 = [((m * scale).astype(np.float32)).astype(np.float64) for m in raw_maps]
        norm32 = metrics(shifted32)
        raw_rank = np.argsort(np.argsort(raw["image_scores"]))
        for tag, mm in [("float64", norm), ("float32_after_scale", norm32)]:
            rank = np.argsort(np.argsort(mm["image_scores"]))
            inversions = int(np.sum(raw_rank != rank))
            results[f"{name}_{tag}"] = {
                "image_auroc": mm["image_auroc"],
                "pixel_auroc": mm["pixel_auroc"],
                "aupro": mm["aupro"],
                "image_rank_inversions_vs_raw": inversions,
                "max_abs_metric_diff": float(max(
                    abs(mm["image_auroc"] - raw["image_auroc"]),
                    abs(mm["pixel_auroc"] - raw["pixel_auroc"]),
                    abs(mm["aupro"] - raw["aupro"]),
                )),
            }
    results["raw"] = {
        "image_auroc": raw["image_auroc"],
        "pixel_auroc": raw["pixel_auroc"],
        "aupro": raw["aupro"],
    }
    return results


def main():
    rows = load_rows()
    out = {
        "source_csv": CSV_PATH,
        "ref_range_summary": summarize(rows),
        "code_path": {
            "reference_normalizer": "experiments/ablation/ablation_utils.py:353",
            "normalize_visual_scores": "experiments/ablation/ablation_utils.py:363",
            "nearest_reference_scores": "experiments/vne_reference_baseline/run_vne_reference_baseline.py:112",
            "dtype_chain": "raw nearest score float32 -> reference_normalizer float64 -> "
                           "(score-min)/range float64 -> astype(float32)",
            "clipping": "none; values outside [0,1] are retained",
            "degenerate_guard": "denom <= 1e-12 returns zeros",
        },
        "synthetic_rank_check": synthetic_metric_check(),
    }
    with open(os.path.join(BASE, "NORMALIZATION_AUDIT.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
