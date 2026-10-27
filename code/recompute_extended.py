"""Extended CPU-only recomputation from frozen CSVs.

Covers:
  - formal primary A0 vs A2 on AUPRO (full 27 categories, Holm over two datasets);
  - post hoc 21-class sensitivity (exclude six pilot categories, Holm over two datasets);
  - historical PCA090/097/099 vs A0 (same-run historical, post hoc, uncorrected);
  - historical PCA090/097/099 vs A2 (cross-batch, post hoc, uncorrected);
  - descriptive float64 PCA state storage.

No GPU, no training, no modification of frozen inputs. The output labels each
comparison with its inferential boundary.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(BASE, "..", "data"))
OUT = os.path.abspath(os.path.join(BASE, "..", "outputs", "recompute_extended"))
os.makedirs(OUT, exist_ok=True)

CSV135 = os.path.join(DATA, "EXPERIMENT_AB_MERGED_135_RESULTS.csv")
PCAM = os.path.join(DATA, "STAGE3A_MVTEC_QUICKGELU_PER_CATEGORY_RESULTS.csv")
PCAV = os.path.join(DATA, "STAGE3B_VISA_QUICKGELU_PER_CATEGORY_RESULTS.csv")

PILOT = {("MVTec", "bottle"), ("MVTec", "grid"), ("MVTec", "screw"),
         ("VisA", "candle"), ("VisA", "pcb3"), ("VisA", "macaroni1")}
PCA_CHAINS = {"PCA090": "pca_residual_0p90",
              "PCA097": "pca_residual_0p97",
              "PCA099": "pca_residual_0p99"}


def locked_wilcoxon(d):
    """Match the frozen branch rule: exact only if n<=25, no zero, no tied |d|."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    nonzero = d[np.abs(d) > 1e-12]
    zeros = n - len(nonzero)
    ties = len(nonzero) - len(np.unique(np.abs(nonzero)))
    exact = (n <= 25 and zeros == 0 and ties == 0)
    res = wilcoxon(d, zero_method="wilcox", correction=True, alternative="two-sided",
                   method="exact" if exact else "asymptotic")
    return float(res.statistic), float(res.pvalue), ("exact" if exact else "asymptotic"), \
           int(zeros), int(ties)


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out = {}
    running = 0.0
    m = len(items)
    for i, (key, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        out[key] = running
    return out


def paired_stats(d, label):
    d = np.asarray(d, dtype=float)
    stat, p, method, zeros, ties = locked_wilcoxon(d)
    return {
        "label": label,
        "n": int(len(d)),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "wins_losses_ties": [int((d > 1e-12).sum()), int((d < -1e-12).sum()),
                             int((np.abs(d) <= 1e-12).sum())],
        "wilcoxon_stat": stat,
        "wilcoxon_p_uncorrected": p,
        "wilcoxon_method": method,
        "n_zeros": zeros,
        "n_abs_ties": ties,
    }


def main():
    df = pd.read_csv(CSV135)
    a0 = df[df.method == "A0_vne_k1"].set_index(["dataset", "category"])["aupro_0_30"]
    a2 = df[df.method == "A2_nearest_coreset5"].set_index(["dataset", "category"])["aupro_0_30"]

    result = {
        "source": "frozen CSVs; CPU recomputation; no GPU, no training",
        "formal_primary": {},
        "post_hoc_21class": {},
        "historical_pca": {},
        "storage": {},
    }

    # Formal primary: full 27 categories, A0 vs A2 AUPRO, Holm over two datasets.
    pvals = {}
    for ds in ["MVTec", "VisA"]:
        idx = a0.loc[ds].index
        d = (a0.loc[ds] - a2.loc[ds]).dropna()
        result["formal_primary"][ds] = paired_stats(d, "formal primary, full 27, A0-A2, AUPRO")
        pvals[ds] = result["formal_primary"][ds]["wilcoxon_p_uncorrected"]
    holm_p = holm(pvals)
    for ds in pvals:
        result["formal_primary"][ds]["holm_p"] = holm_p[ds]
    result["formal_primary"]["boundary"] = (
        "formal primary for the 27-category extension; pilot-informed prespecified; "
        "Holm over the two datasets; not an independent blinded confirmation")

    # Post hoc 21-class sensitivity: exclude six pilot categories.
    pvals = {}
    for ds in ["MVTec", "VisA"]:
        cats = [c for c in a0.loc[ds].index if (ds, c) not in PILOT]
        d = (a0.loc[ds].loc[cats] - a2.loc[ds].loc[cats]).dropna()
        result["post_hoc_21class"][ds] = paired_stats(
            d, "post hoc 21-class, A0-A2, AUPRO; not blind; full 27 remains formal primary")
        pvals[ds] = result["post_hoc_21class"][ds]["wilcoxon_p_uncorrected"]
    holm_p = holm(pvals)
    for ds in pvals:
        result["post_hoc_21class"][ds]["holm_p"] = holm_p[ds]
    result["post_hoc_21class"]["boundary"] = (
        "post hoc sensitivity excluding six pilot categories; the 21 categories were observed "
        "in the earlier QuickGELU experiment, not an unseen sample; Holm over the two datasets; "
        "does not replace the formal full-27 primary")

    # Historical PCA: same-run A0 vs cross-batch A2.
    pca = pd.concat([pd.read_csv(PCAM), pd.read_csv(PCAV)], ignore_index=True)
    pca["dataset"] = pca["dataset"].replace({"MVTec AD": "MVTec"})
    for ds in ["MVTec", "VisA"]:
        sub = pca[pca.dataset == ds].set_index("category")
        for label, chain in PCA_CHAINS.items():
            if chain not in sub.chain.values:
                continue
            pca_vals = sub[sub.chain == chain]["aupro_0_30"]
            a0_vals = sub[sub.chain == "vne_k1"]["aupro_0_30"]
            common0 = pca_vals.index.intersection(a0_vals.index)
            d0 = (pca_vals.loc[common0] - a0_vals.loc[common0]).dropna()
            common2 = pca_vals.index.intersection(a2.loc[ds].index)
            d2 = (pca_vals.loc[common2] - a2.loc[ds].loc[common2]).dropna()
            result["historical_pca"].setdefault(label, {})[ds] = {
                "vs_A0_same_run": paired_stats(
                    d0, "same-run historical, post hoc, uncorrected; not an equivalence test"),
                "vs_A2_cross_batch": paired_stats(
                    d2, "cross-batch, post hoc, uncorrected; not an equivalence test"),
            }
    result["historical_pca"]["boundary"] = (
        "PCA batch is separate from the 135-record freeze. PCA-vs-A0 is same-run historical; "
        "PCA-vs-A2 is cross-batch. All comparisons are post hoc, uncorrected, and not "
        "equivalence tests; no PCA preparation/B3/B5 measurements exist.")

    # Descriptive float64 PCA state storage.
    for ds in ["MVTec", "VisA"]:
        sub = pca[pca.dataset == ds]
        result["storage"][ds] = {}
        for label, chain in PCA_CHAINS.items():
            vals = sub[sub.chain == chain]["storage_bytes"].astype(float)
            result["storage"][ds][label] = {
                "n": int(len(vals)),
                "mean_bytes": float(vals.mean()),
                "min_bytes": float(vals.min()),
                "max_bytes": float(vals.max()),
                "dtype_note": "float64 components_+mean_ per historical protocol",
            }
        result["storage"][ds]["A0_min"] = int(df[(df.dataset == ds) &
                                                (df.method == "A0_vne_k1")].storage_min_inference_bytes.iloc[0])
        result["storage"][ds]["A2"] = int(df[(df.dataset == ds) &
                                            (df.method == "A2_nearest_coreset5")].storage_min_inference_bytes.iloc[0])

    with open(os.path.join(OUT, "EXTENDED_STATS.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    lines = ["Extended CPU recomputation from frozen CSVs", ""]
    lines.append("FORMAL PRIMARY (full 27, A0-A2 AUPRO, Holm two datasets)")
    for ds in ["MVTec", "VisA"]:
        r = result["formal_primary"][ds]
        lines.append("  %s: n=%d mean=%.9f median=%.9f W/L/T=%s p=%.10f Holm=%.10f (%s)" %
                     (ds, r["n"], r["mean_delta"], r["median_delta"],
                      r["wins_losses_ties"], r["wilcoxon_p_uncorrected"], r["holm_p"],
                      r["wilcoxon_method"]))
    lines.append("")
    lines.append("POST HOC 21-CLASS (exclude six pilots; not blind; full 27 remains formal)")
    for ds in ["MVTec", "VisA"]:
        r = result["post_hoc_21class"][ds]
        lines.append("  %s: n=%d mean=%.9f W/L/T=%s p=%.10f Holm=%.10f (%s)" %
                     (ds, r["n"], r["mean_delta"], r["wins_losses_ties"],
                      r["wilcoxon_p_uncorrected"], r["holm_p"], r["wilcoxon_method"]))
    lines.append("")
    lines.append("HISTORICAL PCA (same-run vs A0; cross-batch vs A2; uncorrected; not equivalence)")
    for label in PCA_CHAINS:
        if label not in result["historical_pca"]:
            continue
        for ds in ["MVTec", "VisA"]:
            a = result["historical_pca"][label][ds]["vs_A0_same_run"]
            b = result["historical_pca"][label][ds]["vs_A2_cross_batch"]
            lines.append("  %s %s vs A0: mean=%+.6f W/L/T=%s p=%.6f; vs A2: mean=%+.6f W/L/T=%s p=%.6f" %
                         (ds, label, a["mean_delta"], a["wins_losses_ties"], a["wilcoxon_p_uncorrected"],
                          b["mean_delta"], b["wins_losses_ties"], b["wilcoxon_p_uncorrected"]))
    lines.append("")
    lines.append("PCA STORAGE (float64 components_+mean_)")
    for ds in ["MVTec", "VisA"]:
        for label in PCA_CHAINS:
            r = result["storage"][ds][label]
            lines.append("  %s %s: mean=%.1f B range=%.0f-%.0f B" %
                         (ds, label, r["mean_bytes"], r["min_bytes"], r["max_bytes"]))
        lines.append("  %s A0_min=%d B; A2=%d B" %
                     (ds, result["storage"][ds]["A0_min"], result["storage"][ds]["A2"]))
    with open(os.path.join(OUT, "EXTENDED_STATS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    print("wrote", OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
