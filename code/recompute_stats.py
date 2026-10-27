"""Independent recomputation of the locked 27-category A/B statistics.

Input: frozen merged CSV (135 rows). Output: JSON + plain-text summary.
No values in the frozen input are modified.
"""
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "..", "data", "EXPERIMENT_AB_MERGED_135_RESULTS.csv")
OUT_DIR = os.path.abspath(os.path.join(BASE, "..", "outputs", "recompute"))
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["A0_vne_k1", "A1_nearest_full", "A2_nearest_coreset5",
           "A3_nearest_coreset10", "A4_nearest_random5"]
DATASETS = ["MVTec", "VisA"]
METRICS = ["image_auroc_top1", "pixel_auroc_global", "aupro_0_30"]


def load():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("image_auroc_top1", "pixel_auroc_global", "aupro_0_30",
                  "storage_native_bytes", "storage_min_inference_bytes",
                  "b2_prepare_s", "b2_coreset_selection_s",
                  "b3_scorer_only_per_image_ms", "b3_scorer_only_median_s",
                  "b4_map_pipeline_s", "ref_raw_range"):
            if r[k] not in ("", None):
                try:
                    r[k + "__f"] = float(r[k])
                except ValueError:
                    pass
    return rows


def paired(rows, ds, m1, m2, metric):
    by_unit = defaultdict(dict)
    for r in rows:
        if r["dataset"] == ds:
            by_unit[r["category"]][r["method"]] = r[metric + "__f"]
    cats = sorted(by_unit)
    d = np.array([by_unit[c][m1] - by_unit[c][m2] for c in cats], dtype=float)
    return cats, d


def wilcoxon_rule(d):
    # Locked rule: the exact/asymptotic branch is decided on the original
    # paired sample. Any zero difference or tied absolute difference sends the
    # test to the asymptotic branch; zeros are then discarded by
    # zero_method="wilcox". The frozen implementation used correction=True.
    n_original = len(d)
    nonzero = d[np.abs(d) > 1e-12]
    n_nonzero = len(nonzero)
    zeros = n_original - n_nonzero
    ties = n_nonzero - len(np.unique(np.abs(nonzero)))
    exact = (n_original <= 25 and zeros == 0 and ties == 0)
    res = stats.wilcoxon(d, zero_method="wilcox", correction=True,
                         alternative="two-sided",
                         method="exact" if exact else "asymptotic")
    return (float(res.statistic), float(res.pvalue),
            ("exact" if exact else "asymptotic"), n_nonzero, ties, zeros)


def bootstrap_ci(d, n_boot=20000, seed=20261025):
    # Match the locked analysis: legacy RandomState, one draw at a time.
    rng = np.random.RandomState(seed)
    boot = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


def bh(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, pvals[i] * m / (rank + 1))
        adj[i] = running
    return adj.tolist()


def main():
    rows = load()
    out = {"n_rows": len(rows), "n_unique_units": len({r["unit"] for r in rows})}

    # Integrity
    combos = Counter((r["dataset"], r["category"], r["method"]) for r in rows)
    out["integrity"] = {
        "duplicate_combos": [k for k, v in combos.items() if v > 1],
        "missing_methods": [],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "datasets": dict(Counter(r["dataset"] for r in rows)),
        "batch_counts": dict(Counter(r["source_batch"] for r in rows)),
        "warnings": [
            {k: r[k] for k in ("dataset", "category", "method", "aupro_repro_warning", "reproduction")}
            for r in rows if r["aupro_repro_warning"].strip()
        ],
        "b5_status": dict(Counter(r["b5_status"] for r in rows)),
        "b3_n_images": sorted({int(r["b3_n_images"]) for r in rows}),
        "b3_cap_applied_counts": dict(Counter(r["b3_cap_applied"] for r in rows)),
        "b2_algorithm_counts": dict(Counter(r["b2_algorithm"] for r in rows)),
        "n_ref_patches": sorted({int(r["n_ref_patches"]) for r in rows}),
    }
    for ds in DATASETS:
        for cat in sorted({r["category"] for r in rows if r["dataset"] == ds}):
            ms = {r["method"] for r in rows if r["dataset"] == ds and r["category"] == cat}
            for m in METHODS:
                if m not in ms:
                    out["integrity"]["missing_methods"].append(f"{ds}:{cat}:{m}")

    # Descriptives
    desc = {}
    for ds in DATASETS:
        desc[ds] = {}
        for m in METHODS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            d = {}
            for metric in METRICS:
                vals = np.array([r[metric + "__f"] for r in sel])
                d[metric] = {"mean": round(float(vals.mean()), 6),
                             "sd_ddof0": round(float(vals.std(ddof=0)), 6),
                             "median": round(float(np.median(vals)), 6),
                             "min": round(float(vals.min()), 6),
                             "max": round(float(vals.max()), 6),
                             "n": int(len(vals))}
            for k in ("storage_native_bytes", "storage_min_inference_bytes",
                      "b2_prepare_s", "b2_coreset_selection_s",
                      "b3_scorer_only_median_s", "b3_scorer_only_per_image_ms",
                      "b4_map_pipeline_s", "ref_raw_range"):
                vals = np.array([r[k + "__f"] for r in sel])
                d[k] = {"mean": round(float(vals.mean()), 6),
                        "median": round(float(np.median(vals)), 6),
                        "min": round(float(vals.min()), 6),
                        "max": round(float(vals.max()), 6)}
            desc[ds][m] = d
    out["descriptives"] = desc

    # Primary: A0 vs A2 AUPRO
    primary = {}
    praw = []
    for ds in DATASETS:
        cats, d = paired(rows, ds, "A0_vne_k1", "A2_nearest_coreset5", "aupro_0_30")
        stat, p, method, n, ties, zeros = wilcoxon_rule(d)
        lo, hi = bootstrap_ci(d)
        wins = int((d > 1e-12).sum()); loss = int((d < -1e-12).sum())
        entry = {"n": len(d), "mean_delta": round(float(d.mean()), 9),
                 "median_delta": round(float(np.median(d)), 9),
                 "ci95_percentile": [round(lo, 9), round(hi, 9)],
                 "wins_losses_ties": [wins, loss, len(d) - wins - loss],
                 "wilcoxon_stat": stat, "wilcoxon_p": p, "method": method,
                 "n_nonzero": n, "n_abs_ties": ties, "n_zeros": zeros,
                 "per_category": {c: round(float(v), 9) for c, v in zip(cats, d)}}
        primary[ds] = entry
        praw.append(p)
    holm_p = holm(praw)
    for ds, hp in zip(DATASETS, holm_p):
        primary[ds]["holm_p"] = hp
    out["primary_A0_vs_A2_aupro"] = primary

    # Secondary family (10 tests), BH
    secondary = []
    for ds in DATASETS:
        specs = [("A2_nearest_coreset5", "image_auroc_top1", "A0_vs_A2_image"),
                 ("A2_nearest_coreset5", "pixel_auroc_global", "A0_vs_A2_pixel"),
                 ("A4_nearest_random5", "aupro_0_30", "A0_vs_A4_aupro"),
                 ("A1_nearest_full", "aupro_0_30", "A0_vs_A1_aupro"),
                 ("A3_nearest_coreset10", "aupro_0_30", "A0_vs_A3_aupro")]
        for other, metric, label in specs:
            cats, d = paired(rows, ds, "A0_vne_k1", other, metric)
            stat, p, method, n, ties, zeros = wilcoxon_rule(d)
            lo, hi = bootstrap_ci(d)
            wins = int((d > 1e-12).sum()); loss = int((d < -1e-12).sum())
            secondary.append({"dataset": ds, "comparison": label, "metric": metric,
                              "n": len(d), "mean_delta": round(float(d.mean()), 9),
                              "median_delta": round(float(np.median(d)), 9),
                              "ci95": [round(lo, 9), round(hi, 9)],
                              "wins_losses_ties": [wins, loss, len(d) - wins - loss],
                              "wilcoxon_stat": stat, "raw_p": p, "method": method,
                              "n_nonzero": n, "n_abs_ties": ties, "n_zeros": zeros})
    q = bh([s["raw_p"] for s in secondary])
    for s, qq in zip(secondary, q):
        s["bh_q"] = qq
    out["secondary_family"] = secondary

    # Pooled 27 descriptive (A0 vs A2)
    pooled = {}
    for metric in METRICS:
        diffs = []
        for ds in DATASETS:
            _, d = paired(rows, ds, "A0_vne_k1", "A2_nearest_coreset5", metric)
            diffs.extend(d.tolist())
        diffs = np.array(diffs)
        pooled[metric] = {"mean_delta": round(float(diffs.mean()), 9),
                          "median_delta": round(float(np.median(diffs)), 9),
                          "a0_better": int((diffs > 1e-12).sum()),
                          "a2_better": int((diffs < -1e-12).sum()),
                          "tie": int((np.abs(diffs) <= 1e-12).sum())}
    out["pooled_27_descriptive"] = pooled

    # Image regressions and PCB subset
    reg = []
    for ds in DATASETS:
        cats, d = paired(rows, ds, "A0_vne_k1", "A2_nearest_coreset5", "image_auroc_top1")
        for c, v in zip(cats, d):
            if v < -1e-12:
                reg.append({"dataset": ds, "category": c, "delta": round(float(v), 9)})
    out["image_regressions_A0_vs_A2"] = reg
    out["image_regression_count"] = len(reg)

    pcb = []
    for ds in DATASETS:
        cats, d = paired(rows, ds, "A0_vne_k1", "A2_nearest_coreset5", "image_auroc_top1")
        for c, v in zip(cats, d):
            if c.startswith("pcb"):
                pcb.append({"category": c, "delta_image": round(float(v), 9)})
    out["pcb_image_deltas"] = pcb

    # Storage ratios
    st = {}
    for ds in DATASETS:
        m = {r["method"]: r for r in rows if r["dataset"] == ds}
        a0_min = float(m["A0_vne_k1"]["storage_min_inference_bytes"])
        a0_nat = float(m["A0_vne_k1"]["storage_native_bytes"])
        st[ds] = {
            "A0_native": a0_nat, "A0_min": a0_min,
            "A1": float(m["A1_nearest_full"]["storage_min_inference_bytes"]),
            "A2": float(m["A2_nearest_coreset5"]["storage_min_inference_bytes"]),
            "A3": float(m["A3_nearest_coreset10"]["storage_min_inference_bytes"]),
            "A4": float(m["A4_nearest_random5"]["storage_min_inference_bytes"]),
            "A1_over_A0min": round(float(m["A1_nearest_full"]["storage_min_inference_bytes"]) / a0_min, 4),
            "A0min_over_A2": round(a0_min / float(m["A2_nearest_coreset5"]["storage_min_inference_bytes"]), 4),
            "A1_over_A2": round(float(m["A1_nearest_full"]["storage_min_inference_bytes"]) / float(m["A2_nearest_coreset5"]["storage_min_inference_bytes"]), 4),
            "A0nat_over_A2": round(a0_nat / float(m["A2_nearest_coreset5"]["storage_min_inference_bytes"]), 4),
        }
    out["storage"] = st

    # Cost cross-over per dataset (B2 difference and per-image B3 difference)
    cross = {}
    for ds in DATASETS:
        m = {r["method"]: r for r in rows if r["dataset"] == ds}
        b2_0 = statistics.mean(float(r["b2_prepare_s"]) for r in rows if r["dataset"] == ds and r["method"] == "A0_vne_k1")
        b2_2 = statistics.mean(float(r["b2_prepare_s"]) for r in rows if r["dataset"] == ds and r["method"] == "A2_nearest_coreset5")
        s0 = statistics.mean(float(r["b3_scorer_only_per_image_ms"]) for r in rows if r["dataset"] == ds and r["method"] == "A0_vne_k1") / 1000.0
        s2 = statistics.mean(float(r["b3_scorer_only_per_image_ms"]) for r in rows if r["dataset"] == ds and r["method"] == "A2_nearest_coreset5") / 1000.0
        dp = b2_0 - b2_2
        dsec = s0 - s2
        nstar = (-dp / dsec) if dp * dsec < 0 else None
        cross[ds] = {"B2_A0_mean_s": round(b2_0, 6), "B2_A2_mean_s": round(b2_2, 6),
                     "delta_prepare_s": round(dp, 6),
                     "B3_per_image_A0_s": round(s0, 9), "B3_per_image_A2_s": round(s2, 9),
                     "delta_per_image_s": round(dsec, 9),
                     "break_even_images": None if nstar is None else int(math.ceil(nstar))}
    out["cost_cross_over"] = cross

    out["source_csv_sha256_note"] = "see MANIFEST_SHA256.txt"

    with open(os.path.join(OUT_DIR, "RECOMPUTED_STATS.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    # Human-readable summary
    lines = []
    lines.append(f"rows={out['n_rows']} unique_units={out['n_unique_units']}")
    lines.append(f"duplicates={out['integrity']['duplicate_combos']} missing={len(out['integrity']['missing_methods'])}")
    lines.append(f"status={out['integrity']['status_counts']} datasets={out['integrity']['datasets']}")
    lines.append(f"warnings({len(out['integrity']['warnings'])}):")
    for w in out["integrity"]["warnings"]:
        lines.append(f"  {w}")
    lines.append(f"b5_status={out['integrity']['b5_status']}")
    lines.append(f"b3_n_images={out['integrity']['b3_n_images']} cap_counts={out['integrity']['b3_cap_applied_counts']}")
    lines.append("\nPRIMARY A0-A2 AUPRO")
    for ds in DATASETS:
        e = primary[ds]
        lines.append(f"  {ds}: n={e['n']} mean={e['mean_delta']:+.6f} median={e['median_delta']:+.6f} "
                     f"CI=[{e['ci95_percentile'][0]:+.6f},{e['ci95_percentile'][1]:+.6f}] "
                     f"W/L/T={e['wins_losses_ties']} p={e['wilcoxon_p']:.9g} ({e['method']}) holm={e['holm_p']:.9g}")
    lines.append("\nSECONDARY")
    for s in secondary:
        lines.append(f"  {s['dataset']:6s} {s['comparison']:22s} mean={s['mean_delta']:+.6f} "
                     f"W/L/T={s['wins_losses_ties']} raw_p={s['raw_p']:.6g} bh_q={s['bh_q']:.6g}")
    lines.append("\nPOOLED 27 DESCRIPTIVE")
    for metric, v in pooled.items():
        lines.append(f"  {metric}: mean={v['mean_delta']:+.6f} median={v['median_delta']:+.6f} "
                     f"better/tie/worse={v['a0_better']}/{v['tie']}/{v['a2_better']}")
    lines.append(f"\nIMAGE REGRESSIONS: {out['image_regression_count']}")
    for r in reg:
        lines.append(f"  {r['dataset']}:{r['category']} {r['delta']:+.6f}")
    lines.append("\nSTORAGE")
    for ds in DATASETS:
        lines.append(f"  {ds}: {st[ds]}")
    lines.append("\nCROSS-OVER")
    for ds in DATASETS:
        lines.append(f"  {ds}: {cross[ds]}")
    lines.append("\nDESCRIPTIVES")
    for ds in DATASETS:
        for m in METHODS:
            d = desc[ds][m]
            lines.append(f"  {ds} {m}: img={d['image_auroc_top1']['mean']:.6f} "
                         f"pix={d['pixel_auroc_global']['mean']:.6f} aupro={d['aupro_0_30']['mean']:.6f} "
                         f"B2med={d['b2_prepare_s']['median']:.6f} "
                         f"B3med_ms={d['b3_scorer_only_per_image_ms']['median']:.6f} "
                         f"B4med={d['b4_map_pipeline_s']['median']:.6f} "
                         f"store_min={d['storage_min_inference_bytes']['min']:.0f}")

    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "RECOMPUTED_STATS_SUMMARY.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
