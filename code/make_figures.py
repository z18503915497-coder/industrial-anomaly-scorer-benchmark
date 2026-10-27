"""Generate manuscript figures from the frozen 135-row CSV."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "..", "data", "EXPERIMENT_AB_MERGED_135_RESULTS.csv")
FIG = os.path.abspath(os.path.join(BASE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

METHODS = ["A0_vne_k1", "A1_nearest_full", "A2_nearest_coreset5",
           "A3_nearest_coreset10", "A4_nearest_random5"]
LABELS = {"A0_vne_k1": "A0 VNE-K1", "A1_nearest_full": "A1 Nearest-Full",
          "A2_nearest_coreset5": "A2 Coreset-5%", "A3_nearest_coreset10": "A3 Coreset-10%",
          "A4_nearest_random5": "A4 Random-5%"}
COLORS = {"A0_vne_k1": "#1b4965", "A1_nearest_full": "#6c757d",
          "A2_nearest_coreset5": "#e07a5f", "A3_nearest_coreset10": "#3d9970",
          "A4_nearest_random5": "#b08968"}
MARKERS = {"A0_vne_k1": "o", "A1_nearest_full": "s", "A2_nearest_coreset5": "^",
           "A3_nearest_coreset10": "D", "A4_nearest_random5": "v"}
POS, NEG = "#2a9d8f", "#c1121f"


def load():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("image_auroc_top1", "pixel_auroc_global", "aupro_0_30",
                  "storage_min_inference_bytes", "b2_prepare_s",
                  "b3_scorer_only_per_image_ms", "b5_e2e_median_s"):
            if r[k] not in ("", None):
                r[k + "_f"] = float(r[k])
    return rows


def fig1_schematic():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, text, fc="#f7f7f7", ec="#333333", fs=9.0, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                           linewidth=0.8, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, lw=0.8, color="#333333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=lw, color=color))

    # inputs
    box(1, 45, 14, 8, "Reference images\n≤50 train/good", "#eef3f8", fs=9.0)
    box(1, 7, 14, 8, "Test image", "#eef3f8", fs=9.0)

    # shared encoder
    box(17, 7, 18, 46,
        "Frozen OpenCLIP\nViT-B-16\nquickgelu\n\n196×512\nL2-normalized",
        "#e8f0ea", fs=8.8)
    arrow(15, 49, 17, 49)
    arrow(15, 11, 17, 11)

    # feature pools
    box(37, 45, 15, 8, "Reference patches\nN ≤ 9,800", "#eef3f8", fs=9.0)
    box(37, 7, 15, 8, "Query patches\n196 per image", "#eef3f8", fs=9.0)
    arrow(35, 49, 37, 49)
    arrow(35, 11, 37, 11)

    # scorer input node
    box(54, 26, 8, 9, "fit /\nscore", "#f7f7f7", fs=9.0)
    arrow(52, 49, 58, 35)
    arrow(52, 11, 58, 26)

    # scorers
    details = [
        ("A0 Gaussian NLL\n1.05 MB", "A0_vne_k1"),
        ("A1 Full bank\n20.07 MB", "A1_nearest_full"),
        ("A2 Coreset 5%\n1.00 MB", "A2_nearest_coreset5"),
        ("A3 Coreset 10%\n2.01 MB", "A3_nearest_coreset10"),
        ("A4 Random 5%\n1.00 MB", "A4_nearest_random5"),
    ]
    ys = [47, 40, 33, 26, 19]
    for (txt, mid), y in zip(details, ys):
        box(64, y - 3, 22, 6, txt, "#fbfbfb", fs=9.0)
        arrow(62, 30.5, 64, y)

    # downstream
    box(88, 16, 12, 35,
        "Shared\ndownstream\n\nnormalize\n\n14×14\n\n224×224\n\ntop-1%\n\nmetrics",
        "#f7f7f7", fs=8.0)
    for y in ys:
        arrow(86, y, 88, y)

    ax.text(1, 2.5,
            "State sizes are minimal inference arrays per category; the shared backbone is excluded.",
            fontsize=7.2, color="#444444")

    fig.savefig(os.path.join(FIG, "figure1_pipeline.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_diff(rows, metric, fname, xlim, title_metric, highlight_pcb=False):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax, ds in zip(axes, ["MVTec", "VisA"]):
        cats = sorted({r["category"] for r in rows if r["dataset"] == ds})
        diffs = []
        for c in cats:
            a0 = next(r for r in rows if r["dataset"] == ds and r["category"] == c and r["method"] == "A0_vne_k1")
            a2 = next(r for r in rows if r["dataset"] == ds and r["category"] == c and r["method"] == "A2_nearest_coreset5")
            diffs.append((c, a0[metric + "_f"] - a2[metric + "_f"]))
        diffs.sort(key=lambda t: t[1])
        names = [d[0] for d in diffs]; vals = [d[1] for d in diffs]
        y = np.arange(len(names))
        for yi, (n, v) in zip(y, diffs):
            color = POS if v > 0 else NEG
            ax.plot([0, v], [yi, yi], color=color, linewidth=1.0, alpha=0.75)
            ax.scatter([v], [yi], color=color, s=22, zorder=3)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y)
        labels = [n.upper() if highlight_pcb and n.startswith("pcb") else n for n in names]
        ax.set_yticklabels(labels, fontsize=7)
        for tick, n in zip(ax.get_yticklabels(), names):
            if highlight_pcb and n.startswith("pcb"):
                tick.set_color(NEG); tick.set_fontweight("bold")
        ax.set_xlim(*xlim)
        ax.set_xlabel(f"Δ {title_metric} (A0 − A2)")
        pos = sum(1 for v in vals if v > 0); neg = sum(1 for v in vals if v < 0)
        ax.set_title(f"{ds}: {pos} positive, {neg} negative", fontsize=8.5)
        ax.grid(axis="x", color="#dddddd", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_storage(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    panels = [("MVTec", "aupro_0_30", "AUPRO@0.30", "(a)"),
              ("VisA", "aupro_0_30", "AUPRO@0.30", "(b)"),
              ("MVTec", "image_auroc_top1", "Image AUROC", "(c)"),
              ("VisA", "image_auroc_top1", "Image AUROC", "(d)")]
    handles, labels = [], []
    for ax, (ds, metric, ylab, plabel) in zip(axes.flat, panels):
        ax.axvspan(0.95e6, 1.12e6, color="#e9ecef", alpha=0.7, zorder=0)
        for mid in METHODS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == mid]
            x = sel[0]["storage_min_inference_bytes_f"]
            y = np.mean([r[metric + "_f"] for r in sel])
            ysd = np.std([r[metric + "_f"] for r in sel], ddof=0)
            h = ax.errorbar(x, y, yerr=ysd, fmt=MARKERS[mid], color=COLORS[mid],
                            markersize=7, capsize=2.5, linewidth=0.8, elinewidth=0.7,
                            markeredgecolor="white", markeredgewidth=0.4, zorder=3,
                            label=LABELS[mid])
            if plabel == "(a)":
                handles.append(h); labels.append(LABELS[mid])
        ax.set_xscale("log")
        ax.set_xlabel("Minimal inference state (bytes, log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{plabel} {ds}", fontsize=9, loc="left")
        ax.grid(color="#e5e5e5", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.axvline(1.050628e6, color="#555555", linewidth=0.7, linestyle=(0, (3, 2)))
        ax.text(1.08e6, ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "≈1 MB", fontsize=6.8, color="#555555")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "figure4_storage_accuracy.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_cost(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    ds_colors = {"MVTec": "#1b4965", "VisA": "#e07a5f"}
    # Panel A: B2 preparation mean, log scale
    ax = axes[0]
    x = np.arange(len(METHODS)); width = 0.36
    for off, ds, hatch in [(-width/2, "MVTec", ""), (width/2, "VisA", "//")]:
        vals = []
        for mid in METHODS:
            vals.append(np.mean([r["b2_prepare_s_f"] for r in rows
                                 if r["dataset"] == ds and r["method"] == mid]))
        ax.bar(x + off, vals, width, label=ds, color=ds_colors[ds],
               edgecolor="#333333", linewidth=0.4, hatch=hatch, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Preparation time (s/category, log scale)")
    ax.set_title("(a) One-time preparation (B2)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5, alpha=0.7); ax.set_axisbelow(True)

    # Panel B: B5 end-to-end per image
    ax = axes[1]
    methods3 = ["A0_vne_k1", "A1_nearest_full", "A2_nearest_coreset5"]
    x = np.arange(len(methods3)); width = 0.36
    for off, ds, hatch in [(-width/2, "MVTec", ""), (width/2, "VisA", "//")]:
        vals = []
        for mid in methods3:
            vals.append(np.mean([r["b5_e2e_median_s_f"] * 1000 for r in rows
                                 if r["dataset"] == ds and r["method"] == mid and r["b5_e2e_median_s"]]))
        ax.bar(x + off, vals, width, label=ds, color=ds_colors[ds],
               edgecolor="#333333", linewidth=0.4, hatch=hatch, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in methods3], fontsize=7.5)
    ax.set_ylabel("End-to-end time (ms/image)")
    ax.set_title("(b) End-to-end per image (B5)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5, alpha=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figure5_cost_profile.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows = load()
    fig1_schematic()
    fig_diff(rows, "aupro_0_30", "figure2_aupro_differences.png",
             (-0.075, 0.165), "AUPRO@0.30")
    fig_diff(rows, "image_auroc_top1", "figure3_image_differences.png",
             (-0.155, 0.095), "Image AUROC", highlight_pcb=True)
    fig_storage(rows)
    fig_cost(rows)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
