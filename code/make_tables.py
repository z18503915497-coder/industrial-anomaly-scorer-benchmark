"""Build manuscript tables and derived ratios from the frozen 135-row CSV."""
import csv
import json
import os
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "..", "data", "EXPERIMENT_AB_MERGED_135_RESULTS.csv")
OUT = os.path.abspath(os.path.join(BASE, "..", "tables"))
os.makedirs(OUT, exist_ok=True)

METHODS = [("A0_vne_k1", "A0 VNE-K1"),
           ("A1_nearest_full", "A1 Nearest-Full"),
           ("A2_nearest_coreset5", "A2 Coreset-5%"),
           ("A3_nearest_coreset10", "A3 Coreset-10%"),
           ("A4_nearest_random5", "A4 Random-5%")]
DATASETS = ["MVTec", "VisA"]
METRICS = ["image_auroc_top1", "pixel_auroc_global", "aupro_0_30"]


def load():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("image_auroc_top1", "pixel_auroc_global", "aupro_0_30",
                  "storage_native_bytes", "storage_min_inference_bytes",
                  "b2_prepare_s", "b2_coreset_selection_s",
                  "b3_scorer_only_per_image_ms", "b4_map_pipeline_s",
                  "b5_e2e_median_s"):
            if r[k] not in ("", None):
                r[k + "_f"] = float(r[k])
    return rows


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def fmt(x, n=3):
    return f"{x:.{n}f}"


def main():
    rows = load()
    stats = json.load(open(os.path.join(BASE, "..", "outputs", "recompute", "RECOMPUTED_STATS.json"), encoding="utf-8"))

    # Table 1: aggregate accuracy
    t1 = []
    for mid, label in METHODS:
        rec = [label]
        for ds in DATASETS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == mid]
            for metric in METRICS:
                v = np.array([r[metric + "_f"] for r in sel])
                rec.append(f"{v.mean():.3f} ± {v.std(ddof=0):.3f}")
        t1.append(rec)
    write_csv("table1_aggregate_accuracy.csv",
              ["Method"] + [f"{ds} {m}" for ds in DATASETS for m in
                            ("Image AUROC", "Pixel AUROC", "AUPRO@0.30")], t1)

    # Table 2: primary (with rank-biserial effect size)
    t2 = []
    for ds in DATASETS:
        e = stats["primary_A0_vs_A2_aupro"][ds]
        t2.append([ds, e["n"], f"{e['mean_delta']:+.6f}", f"{e['median_delta']:+.6f}",
                   f"[{e['ci95_percentile'][0]:+.6f}, {e['ci95_percentile'][1]:+.6f}]",
                   "/".join(str(x) for x in e["wins_losses_ties"]),
                   f"{e['wilcoxon_p']:.2e}", f"{e['holm_p']:.2e}", e["method"]])
    write_csv("table2_primary.csv",
              ["Dataset", "n", "Mean delta", "Median delta", "95% CI", "W/L/T",
               "Wilcoxon p", "Holm p", "Wilcoxon method"], t2)

    # Table 3: secondary
    comp_label = {
        "A0_vs_A2_image": "A0 vs A2, Image AUROC",
        "A0_vs_A2_pixel": "A0 vs A2, Pixel AUROC",
        "A0_vs_A4_aupro": "A0 vs A4, AUPRO",
        "A0_vs_A1_aupro": "A0 vs A1, AUPRO",
        "A0_vs_A3_aupro": "A0 vs A3, AUPRO",
    }
    t3 = []
    for s in stats["secondary_family"]:
        t3.append([s["dataset"], comp_label[s["comparison"]],
                   f"{s['mean_delta']:+.6f}", f"{s['median_delta']:+.6f}",
                   "/".join(str(x) for x in s["wins_losses_ties"]),
                   f"{s['raw_p']:.6g}", f"{s['bh_q']:.6g}"])
    write_csv("table3_secondary.csv",
              ["Dataset", "Comparison", "Mean delta", "Median delta", "W/L/T",
               "Raw p", "BH q"], t3)

    # Table 4: per-category A0 vs A2
    by = defaultdict(dict)
    for r in rows:
        by[(r["dataset"], r["category"])][r["method"]] = r
    t4 = []
    for ds in DATASETS:
        cats = sorted(c for (d, c) in by if d == ds)
        for c in cats:
            a0 = by[(ds, c)]["A0_vne_k1"]; a2 = by[(ds, c)]["A2_nearest_coreset5"]
            t4.append([ds, c,
                       f"{a0['aupro_0_30_f']:.4f}", f"{a2['aupro_0_30_f']:.4f}",
                       f"{a0['aupro_0_30_f'] - a2['aupro_0_30_f']:+.4f}",
                       f"{a0['image_auroc_top1_f']:.4f}", f"{a2['image_auroc_top1_f']:.4f}",
                       f"{a0['image_auroc_top1_f'] - a2['image_auroc_top1_f']:+.4f}",
                       f"{a0['pixel_auroc_global_f']:.4f}", f"{a2['pixel_auroc_global_f']:.4f}",
                       f"{a0['pixel_auroc_global_f'] - a2['pixel_auroc_global_f']:+.4f}"])
    write_csv("table4_per_category.csv",
              ["Dataset", "Category", "A0 AUPRO", "A2 AUPRO", "Δ AUPRO",
               "A0 Image", "A2 Image", "Δ Image", "A0 Pixel", "A2 Pixel", "Δ Pixel"], t4)

    # Appendix A: full per-category accuracy for all five scorers
    for table_name, metric in [
        ("tableA1_aupro.csv", "aupro_0_30"),
        ("tableA2_image.csv", "image_auroc_top1"),
        ("tableA3_pixel.csv", "pixel_auroc_global"),
    ]:
        out_rows = []
        for ds in DATASETS:
            cats = sorted(c for (d, c) in by if d == ds)
            for c in cats:
                rec = [ds, c]
                for mid, _ in METHODS:
                    rec.append(f"{by[(ds, c)][mid][metric + '_f']:.4f}")
                out_rows.append(rec)
        write_csv(table_name, ["Dataset", "Category"] + [lab for _, lab in METHODS], out_rows)

    # Table 5: storage + preparation
    t5 = []
    for ds in DATASETS:
        for mid, label in METHODS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == mid]
            native = int(sel[0]["storage_native_bytes"])
            minimum = int(sel[0]["storage_min_inference_bytes"])
            b2 = np.array([r["b2_prepare_s_f"] for r in sel])
            sel_s = np.array([r["b2_coreset_selection_s_f"] for r in sel])
            t5.append([ds, label, f"{native:,}", f"{minimum:,}",
                       f"{minimum / 1003520:.3f}", f"{b2.mean():.4f}",
                       f"{np.median(b2):.4f}", f"{sel_s.mean():.4f}"])
    write_csv("table5_storage_preparation.csv",
              ["Dataset", "Method", "Native state (B)", "Minimal inference state (B)",
               "Relative to A2", "B2 mean (s)", "B2 median (s)", "Selection included (s)"], t5)

    # Table 6: scoring and end-to-end
    t6 = []
    for ds in DATASETS:
        for mid, label in METHODS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == mid]
            b3 = np.array([r["b3_scorer_only_per_image_ms_f"] for r in sel])
            b4 = np.array([r["b4_map_pipeline_s_f"] for r in sel])
            b5 = np.array([r["b5_e2e_median_s_f"] for r in sel if r["b5_e2e_median_s"]])
            t6.append([ds, label, f"{b3.mean():.3f}", f"{b3.std(ddof=0):.3f}",
                       f"{b4.mean():.3f}", f"{b4.std(ddof=0):.3f}",
                       "not run" if b5.size == 0 else f"{1000 * b5.mean():.2f}",
                       "not run" if b5.size == 0 else str(b5.size)])
    write_csv("table6_timing.csv",
              ["Dataset", "Method", "B3 mean (ms/image)", "B3 SD", "B4 mean (s/category)",
               "B4 SD", "B5 end-to-end mean (ms/image)", "B5 n categories"], t6)

    # B5 details
    b5rows = []
    for ds in DATASETS:
        for mid, label in METHODS:
            vals = np.array([r["b5_e2e_median_s_f"] * 1000 for r in rows
                             if r["dataset"] == ds and r["method"] == mid and r["b5_e2e_median_s"]])
            if vals.size:
                b5rows.append([ds, label, len(vals), f"{vals.mean():.2f}",
                               f"{np.median(vals):.2f}", f"{vals.min():.2f}", f"{vals.max():.2f}"])
    write_csv("b5_end_to_end.csv",
              ["Dataset", "Method", "n categories", "Mean (ms/image)", "Median (ms/image)",
               "Min", "Max"], b5rows)

    # Derived ratios
    ratios = {}
    for ds in DATASETS:
        d = {}
        for mid, _ in METHODS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == mid]
            d[mid] = {
                "b2_mean_s": float(np.mean([r["b2_prepare_s_f"] for r in sel])),
                "b3_mean_ms": float(np.mean([r["b3_scorer_only_per_image_ms_f"] for r in sel])),
                "b4_mean_s": float(np.mean([r["b4_map_pipeline_s_f"] for r in sel])),
                "storage_min": int(sel[0]["storage_min_inference_bytes"]),
                "storage_native": int(sel[0]["storage_native_bytes"]),
            }
            b5 = [r["b5_e2e_median_s_f"] * 1000 for r in sel if r["b5_e2e_median_s"]]
            if b5:
                d[mid]["b5_mean_ms"] = float(np.mean(b5))
        ratios[ds] = d
    derived = {}
    for ds in DATASETS:
        d = ratios[ds]
        derived[ds] = {
            "A2_prepare_over_A0": d["A2_nearest_coreset5"]["b2_mean_s"] / d["A0_vne_k1"]["b2_mean_s"],
            "A0_prepare_over_A1": d["A0_vne_k1"]["b2_mean_s"] / d["A1_nearest_full"]["b2_mean_s"],
            "A0_prepare_over_A4": d["A0_vne_k1"]["b2_mean_s"] / d["A4_nearest_random5"]["b2_mean_s"],
            "A2_scorer_over_A0": d["A2_nearest_coreset5"]["b3_mean_ms"] / d["A0_vne_k1"]["b3_mean_ms"],
            "A1_scorer_over_A0": d["A1_nearest_full"]["b3_mean_ms"] / d["A0_vne_k1"]["b3_mean_ms"],
            "A0_b5_over_A2": d["A0_vne_k1"]["b5_mean_ms"] / d["A2_nearest_coreset5"]["b5_mean_ms"],
            "A1_b5_over_A0": d["A1_nearest_full"]["b5_mean_ms"] / d["A0_vne_k1"]["b5_mean_ms"],
            "A1_storage_over_A0": d["A1_nearest_full"]["storage_min"] / d["A0_vne_k1"]["storage_min"],
            "A0_storage_over_A2": d["A0_vne_k1"]["storage_min"] / d["A2_nearest_coreset5"]["storage_min"],
            "A0_native_over_A2": d["A0_vne_k1"]["storage_native"] / d["A2_nearest_coreset5"]["storage_min"],
        }
    with open(os.path.join(OUT, "derived_ratios.json"), "w", encoding="utf-8") as fh:
        json.dump({"per_dataset": ratios, "derived": derived}, fh, indent=2)

    print(json.dumps(derived, indent=2))


if __name__ == "__main__":
    main()
