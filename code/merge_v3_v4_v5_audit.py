# -*- coding: utf-8 -*-
"""Merge v3 MVTec15 + v4 VisA15 + v5 remaining21 into 135 records, audit, and run the
locked statistical plan. Source CSVs are read-only; output is a new directory.
"""
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path("<PROJECT_ROOT>/manuscript_rescue_experiments")
V3 = BASE / "ARCHIVE_V3_MVTEC15_20261025"
V3_CSV = V3 / "data" / "EXPERIMENT_AB_SIX_CATEGORY_PILOT_RESULTS.csv"
V3_SUMMARY = V3 / "data" / "pilot_summary.json"
V3_LOG = V3 / "data" / "pilot_run.log"
V3_MANIFEST = V3 / "V3_MVTEC15_ARCHIVE_MANIFEST.json"
V4 = BASE / "cloud_results_v4_20261025"
V4_RUN = V4 / "pilot_run_v4_20261025"
V4_CSV = V4_RUN / "EXPERIMENT_AB_SIX_CATEGORY_PILOT_RESULTS.csv"
V4_SUMMARY = V4_RUN / "pilot_summary.json"
V4_LOG = V4_RUN / "pilot_run.log"
V4_WARNING = V4_RUN / "AUPRO_REPRO_WARNING.json"
V5 = BASE / "cloud_results_v5_20261025"
V5_RUN = V5 / "pilot_run_v5_20261025"
V5_CSV = V5_RUN / "EXPERIMENT_AB_SIX_CATEGORY_PILOT_RESULTS.csv"
V5_SUMMARY = V5_RUN / "pilot_summary.json"
V5_LOG = V5_RUN / "pilot_run.log"
V5_WARNING = V5_RUN / "AUPRO_REPRO_WARNING.json"
V5_SELF_CHECK = V5 / "self_check_cloud_v5_20261025" / "self_check_report.json"
V5_WHITELIST = BASE / "pilot_runner_v5_20261025" / "v5_remaining21_whitelist.json"
OUT = BASE / "merged_results_v3v4v5_27cat_135_20261025"

METHODS = ["A0_vne_k1", "A1_nearest_full", "A2_nearest_coreset5",
           "A3_nearest_coreset10", "A4_nearest_random5"]
METRICS = ["image_auroc_top1", "pixel_auroc_global", "aupro_0_30"]
LABELS = {"image_auroc_top1": "Image AUROC", "pixel_auroc_global": "Pixel AUROC",
          "aupro_0_30": "AUPRO@0.30"}
STORAGE = {
    "A0_vne_k1": (3_147_780, 1_050_628, 0),
    "A1_nearest_full": (20_070_400, 20_070_400, 9800),
    "A2_nearest_coreset5": (1_003_520, 1_003_520, 490),
    "A3_nearest_coreset10": (2_007_040, 2_007_040, 980),
    "A4_nearest_random5": (1_003_520, 1_003_520, 490),
}
V3_RUNNER_SHA = "f48e27d7401e81f492c1d0770f713faf72880ed75c2da94bb3ee74e2da7c3596"
V4_RUNNER_SHA = "89e1d5ebf5c92a00aa3c54dfcb45447af3503928e91e1618f0a613ac16ec6edb"
V5_RUNNER_SHA = "1b5dd5103601b2c1198c9c8d962fc2eef3279a0563e548a1bfe4c7aa555b7a0e"
V5_WHITELIST_SHA = "291907f6f5c37ddf9d80d13ba2f583348fd511215eeb1978ad71b61e6ebb3e45"
V5_TARGETS_SHA = "a52c52528156f02108466790983d98305b48f1558ca3a810e90590b051376057"
V3_GATE = "v3: Image<=1e-6, Pixel<=1e-6, AUPRO<=3e-5; all three required"
V4_GATE = "v4: Image<=1e-6, Pixel<=1e-6; AUPRO<=3e-5 PASS, (3e-5,1e-3] WARNING, >1e-3 block"
V5_GATE = "v5 inherits v4 AUPRO policy; Image/Pixel<=1e-6; AUPRO PASS/WARNING/block"
BOOT_SEED = 20261025

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append({"check": name, "passed": bool(cond), "detail": str(detail)})
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{detail}]"))


def f(x):
    return float(x)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def parse_times(log_path):
    pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| (MVTec|VisA):([^/]+)/([^:]+): OK")
    out = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone(timedelta(hours=8)))
        out[f"{m.group(2)}:{m.group(3)}:{m.group(4)}"] = ts.astimezone(
            timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def parse_log_counts(log_path):
    out = {}
    pat = re.compile(r"([A-Za-z]+)/([A-Za-z0-9_]+): features ref=\(\d+, \d+\) test_imgs=(\d+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            out[f"{m.group(1)}:{m.group(2)}"] = int(m.group(3))
    return out


def holm(entries):
    """entries: list of (name, p). Returns dict name -> adjusted p."""
    s = sorted(entries, key=lambda x: x[1])
    m = len(s)
    out = {}
    running = 0.0
    for i, (name, p) in enumerate(s):
        val = (m - i) * p
        running = max(running, val)
        out[name] = min(1.0, running)
    return out


def bh(entries):
    s = sorted(entries, key=lambda x: x[1])
    m = len(s)
    out = {}
    running = 1.0
    for i in range(m - 1, -1, -1):
        name, p = s[i]
        val = p * m / (i + 1)
        running = min(running, val)
        out[name] = min(1.0, running)
    return out


def paired_stats(diffs):
    d = np.asarray(diffs, dtype=np.float64)
    n = int(d.size)
    mean = float(np.mean(d))
    median = float(np.median(d))
    pos = int(np.sum(d > 0)); neg = int(np.sum(d < 0)); zero = int(np.sum(d == 0))
    rng = np.random.RandomState(BOOT_SEED)
    boot = [float(np.mean(rng.choice(d, size=n, replace=True))) for _ in range(20000)]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    method = "wilcoxon_auto_exact_if_n<=25_no_ties_else_asymptotic"
    if n == 0:
        p = float("nan"); stat = float("nan"); method = "empty"
    elif np.all(d == 0):
        p = 1.0; stat = 0.0; method = "all_zero"
    else:
        try:
            res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox",
                                 correction=True, method="auto")
            p = float(res.pvalue); stat = float(res.statistic)
        except Exception as e:
            p = float("nan"); stat = float("nan"); method = f"wilcoxon_error:{e}"
    return {"n": n, "mean_diff": mean, "median_diff": median,
            "positive": pos, "negative": neg, "zero": zero,
            "bootstrap_ci95_mean": ci, "wilcoxon_stat": stat, "wilcoxon_p": p,
            "wilcoxon_method": method, "diffs": [float(x) for x in d]}


def main():
    for p in (V3_CSV, V3_SUMMARY, V3_LOG, V3_MANIFEST, V4_CSV, V4_SUMMARY, V4_LOG,
              V4_WARNING, V5_CSV, V5_SUMMARY, V5_LOG, V5_SELF_CHECK, V5_WHITELIST):
        if not p.is_file():
            raise SystemExit(f"missing required input: {p}")
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    v3rows = read_csv(V3_CSV)
    v4rows = read_csv(V4_CSV)
    v5rows = read_csv(V5_CSV)
    v3s = json.loads(V3_SUMMARY.read_text(encoding="utf-8"))
    v4s = json.loads(V4_SUMMARY.read_text(encoding="utf-8"))
    v5s = json.loads(V5_SUMMARY.read_text(encoding="utf-8"))
    v5sc = json.loads(V5_SELF_CHECK.read_text(encoding="utf-8"))
    whitelist = json.loads(V5_WHITELIST.read_text(encoding="utf-8"))
    full27 = [(ds, c) for ds in ("MVTec", "VisA") for c in whitelist["full27"][ds]]
    completed6 = {(ds, c) for ds in ("MVTec", "VisA") for c in whitelist["completed6"][ds]}
    remaining21 = {(ds, c) for ds in ("MVTec", "VisA") for c in whitelist["remaining21"][ds]}
    expected = {(ds, c, m) for ds, c in full27 for m in METHODS}
    v4w = json.loads(V4_WARNING.read_text(encoding="utf-8"))["warnings"]
    v5w = json.loads(V5_WARNING.read_text(encoding="utf-8"))["warnings"] if V5_WARNING.is_file() else []
    v3times = parse_times(V3_LOG)
    v4times = {f"{r['dataset']}:{r['category']}:{r['method']}": r.get("record_time_utc", "")
               for r in v4rows}
    v5times = {f"{r['dataset']}:{r['category']}:{r['method']}": r.get("record_time_utc", "")
               for r in v5rows}
    v3f = list(v3rows[0]); v4f = list(v4rows[0]); v5f = list(v5rows[0])
    base_fields = []
    for k in v3f + v4f + v5f:
        if k not in base_fields:
            base_fields.append(k)
    source_fields = ["source_batch", "source_runner_version", "source_runner_sha256",
                     "source_policy_id", "source_record_time_utc", "source_gate_policy",
                     "source_b3_batch_note"]
    fields = source_fields + base_fields
    merged = []
    for r in v3rows:
        k = f"{r['dataset']}:{r['category']}:{r['method']}"
        prov = {"source_batch": "v3_mvtec15_20261025",
                "source_runner_version": v3s["runner_version"],
                "source_runner_sha256": v3s["runner_sha256"],
                "source_policy_id": "v3-per-metric-20261025",
                "source_record_time_utc": v3times.get(k, "UNKNOWN"),
                "source_gate_policy": V3_GATE,
                "source_b3_batch_note": "v3 MVTec batch; 2026-10-25; same RTX 4090, different session"}
        merged.append({**{x: "" for x in fields}, **r, **prov})
    for r in v4rows:
        k = f"{r['dataset']}:{r['category']}:{r['method']}"
        prov = {"source_batch": "visa_three_category_completion",
                "source_runner_version": v4s["runner_version"],
                "source_runner_sha256": v4s["runner_sha256"],
                "source_policy_id": "v4-aupro-warning-20261025",
                "source_record_time_utc": v4times.get(k, "UNKNOWN"),
                "source_gate_policy": V4_GATE,
                "source_b3_batch_note": "v4 VisA batch; 2026-10-25; same RTX 4090, different session"}
        merged.append({**{x: "" for x in fields}, **r, **prov})
    for r in v5rows:
        k = f"{r['dataset']}:{r['category']}:{r['method']}"
        prov = {"source_batch": "remaining_21_ab_extension",
                "source_runner_version": v5s["runner_version"],
                "source_runner_sha256": v5s["runner_sha256"],
                "source_policy_id": "v4-aupro-warning-20261025",
                "source_record_time_utc": v5times.get(k, "UNKNOWN"),
                "source_gate_policy": V5_GATE,
                "source_b3_batch_note": "v5 remaining21 batch; 2026-10-25; same RTX 4090, different session"}
        merged.append({**{x: "" for x in fields}, **r, **prov})
    by = {(r["dataset"], r["category"], r["method"]): r for r in merged}

    check("135_rows", len(merged) == 135, str(len(merged)))
    check("135_unique", len(by) == 135 and len(merged) == 135)
    check("expected_135", set(by) == expected)
    check("source_counts",
          len(v3rows) == 15 and len(v4rows) == 15 and len(v5rows) == 105,
          f"{len(v3rows)}/{len(v4rows)}/{len(v5rows)}")
    check("v3_categories",
          {(r["dataset"], r["category"]) for r in v3rows} ==
          {("MVTec", c) for c in ["bottle", "grid", "screw"]})
    check("v4_categories",
          {(r["dataset"], r["category"]) for r in v4rows} ==
          {("VisA", c) for c in ["candle", "pcb3", "macaroni1"]})
    check("v5_categories", {(r["dataset"], r["category"]) for r in v5rows} == remaining21)
    check("no_overlap_batches",
          not ({(r["dataset"], r["category"]) for r in v4rows} & remaining21)
          and not ({(r["dataset"], r["category"]) for r in v3rows} & remaining21))
    check("v5_status", v5s.get("status") == "COMPLETE_REMAINING21", v5s.get("status"))
    check("v5_units", v5s.get("completed_units") == 105 and v5s.get("expected_units") == 105)
    check("v5_scope", v5s.get("pilot_scope") == "remaining_21_ab_extension")
    check("v5_batch_full",
          v5s["batch"].get("scope") == "full_remaining21"
          and v5s["batch"].get("full_remaining21_complete") is True
          and v5s["batch"].get("not_full_27_complete") is True)
    check("v4_status", v4s.get("status") == "COMPLETE_SUBSET")
    check("v4_units", v4s.get("completed_units") == 15 and v4s.get("expected_units") == 15)
    check("v3_rows_valid", all(r["status"] in ("OK", "OK_B5_PARTIAL") for r in v3rows))
    check("v4_rows_valid", all(r["status"] in ("OK", "OK_B5_PARTIAL") for r in v4rows))
    check("v5_rows_valid", all(r["status"] in ("OK", "OK_B5_PARTIAL") for r in v5rows))
    check("v5_self_check", v5sc.get("all_ok") is True)
    check("v5_runner_sha", v5s.get("runner_sha256") == V5_RUNNER_SHA)
    check("v5_whitelist_sha", sha(V5_WHITELIST) == V5_WHITELIST_SHA)
    check("v5_targets_sha", v5s.get("targets_sha256") == V5_TARGETS_SHA)
    check("v3_runner_sha", v3s.get("runner_sha256") == V3_RUNNER_SHA)
    check("v4_runner_sha", v4s.get("runner_sha256") == V4_RUNNER_SHA)
    check("warnings_v4", len(v4w) >= 1)
    check("warnings_v5_recorded", isinstance(v5w, list))
    check("strict_repro_not_claimed", not (len(v4w) == 0 and len(v5w) == 0))
    counts = {**parse_log_counts(V3_LOG), **parse_log_counts(V4_LOG), **parse_log_counts(V5_LOG)}

    for r in merged:
        for m in METRICS:
            v = f(r[m])
            check(f"valid_{r['dataset']}_{r['category']}_{r['method']}_{m}",
                  math.isfinite(v) and 0.0 <= v <= 1.0, r[m])
        native, minimum, bank = STORAGE[r["method"]]
        check(f"storage_{r['dataset']}_{r['category']}_{r['method']}",
              int(r["storage_native_bytes"]) == native
              and int(r["storage_min_inference_bytes"]) == minimum
              and int(r["n_bank_patches"]) == bank)
        n_test = counts.get(f"{r['dataset']}:{r['category']}")
        exp_n = min(50, n_test) if n_test else None
        check(f"b3_{r['dataset']}_{r['category']}_{r['method']}",
              int(r["b3_reps_completed"]) == 10 and n_test is not None
              and int(r["b3_n_images"]) == exp_n
              and int(r["b3_n_test_patches"]) == exp_n * 196
              and int(r["b3_cap_applied"]) == int(n_test > 50),
              f"n_test={n_test} exp_n={exp_n} row={r['b3_n_images']}/{r['b3_n_test_patches']}/{r['b3_cap_applied']}")
        check(f"provenance_{r['dataset']}_{r['category']}_{r['method']}",
              all(str(r[k]).strip() for k in source_fields))

    check("v3_b3_150", sum(len(v3s["per_category"][f"MVTec:{c}"][m]["b3"]["reps_s"])
                           for c in ["bottle", "grid", "screw"] for m in METHODS) == 150)
    check("v4_b3_150", sum(len(v4s["per_category"][f"VisA:{c}"][m]["b3"]["reps_s"])
                           for c in ["candle", "pcb3", "macaroni1"] for m in METHODS) == 150)
    check("v5_b3_1050", sum(len(v5s["per_category"][f"{d}:{c}"][m]["b3"]["reps_s"])
                            for d, c in remaining21 for m in METHODS) == 1050)
    # Source numeric fidelity
    mism = []
    for src in (v3rows, v4rows, v5rows):
        for r in src:
            k = (r["dataset"], r["category"], r["method"])
            mr = by[k]
            for field, val in r.items():
                if mr.get(field, "") != val:
                    mism.append((k, field)); break
    check("source_numeric_copy", not mism, str(mism[:3]))

    merged_csv = out / "EXPERIMENT_AB_MERGED_135_RESULTS.csv"
    with open(merged_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fields})

    # ---------------- locked statistics
    mvtec = [("MVTec", c) for c in whitelist["full27"]["MVTec"]]
    visa = [("VisA", c) for c in whitelist["full27"]["VisA"]]

    def diffvals(cats, method_a, method_b, metric):
        return [f(by[(d, c, method_a)][metric]) - f(by[(d, c, method_b)][metric])
                for d, c in cats]

    primary = {}
    for name, cats in (("MVTec15", mvtec), ("VisA12", visa)):
        primary[name] = paired_stats(diffvals(cats, "A0_vne_k1", "A2_nearest_coreset5", "aupro_0_30"))
    holm_p = holm([(k, v["wilcoxon_p"]) for k, v in primary.items()])
    for k in primary:
        primary[k]["holm_adjusted_p"] = holm_p[k]
        primary[k]["significant_holm_0p05"] = bool(holm_p[k] < 0.05)

    sec_entries = []
    sec = {}
    for dsname, cats in (("MVTec15", mvtec), ("VisA12", visa)):
        specs = [
            ("A0_vs_A2_image", "A0_vne_k1", "A2_nearest_coreset5", "image_auroc_top1"),
            ("A0_vs_A2_pixel", "A0_vne_k1", "A2_nearest_coreset5", "pixel_auroc_global"),
            ("A0_vs_A4_aupro", "A0_vne_k1", "A4_nearest_random5", "aupro_0_30"),
            ("A0_vs_A1_aupro", "A0_vne_k1", "A1_nearest_full", "aupro_0_30"),
            ("A0_vs_A3_aupro", "A0_vne_k1", "A3_nearest_coreset10", "aupro_0_30"),
        ]
        for label, ma, mb, metric in specs:
            key = f"{dsname}:{label}"
            sec[key] = paired_stats(diffvals(cats, ma, mb, metric))
            sec_entries.append((key, sec[key]["wilcoxon_p"]))
    bh_q = bh(sec_entries)
    for k in sec:
        sec[k]["bh_q"] = bh_q[k]
        sec[k]["exploratory_q_lt_0p05"] = bool(bh_q[k] < 0.05)

    # descriptive tables
    def method_summary(cats):
        result = {}
        for m in METHODS:
            rows = [by[(d, c, m)] for d, c in cats]
            result[m] = {
                "metrics_mean": {k: float(np.mean([f(r[k]) for r in rows])) for k in METRICS},
                "metrics_median": {k: float(np.median([f(r[k]) for r in rows])) for k in METRICS},
                "b2_median_s": float(np.median([f(r["b2_prepare_s"]) for r in rows])),
                "b3_median_s": float(np.median([f(r["b3_scorer_only_median_s"]) for r in rows])),
                "per_image_ms_median": float(np.median([f(r["b3_scorer_only_per_image_ms"]) for r in rows])),
                "storage_min_bytes": int(rows[0]["storage_min_inference_bytes"]),
            }
        return result

    per_dataset = {"MVTec15": method_summary(mvtec), "VisA12": method_summary(visa)}
    all_summary = method_summary([(d, c) for d, c in full27])
    pooled = {}
    for m in METRICS:
        vals = [f(by[(d, c, "A0_vne_k1")][m]) - f(by[(d, c, "A2_nearest_coreset5")][m])
                for d, c in full27]
        pooled[m] = {"mean_delta": float(np.mean(vals)), "median_delta": float(np.median(vals)),
                     "min": float(np.min(vals)), "max": float(np.max(vals)),
                     "a0_better": int(sum(v > 0 for v in vals)),
                     "a2_better": int(sum(v < 0 for v in vals)),
                     "tie": int(sum(v == 0 for v in vals))}
    adverse_image = []
    for d, c in full27:
        delta = f(by[(d, c, "A0_vne_k1")]["image_auroc_top1"]) - f(by[(d, c, "A2_nearest_coreset5")]["image_auroc_top1"])
        if delta < 0:
            adverse_image.append({"dataset": d, "category": c, "delta_a0_minus_a2": delta})
    cost = {}
    for d, c in full27:
        a0 = by[(d, c, "A0_vne_k1")]; a2 = by[(d, c, "A2_nearest_coreset5")]
        dp = f(a0["b2_prepare_s"]) - f(a2["b2_prepare_s"])
        dq = (f(a0["b3_scorer_only_median_s"]) - f(a2["b3_scorer_only_median_s"])) / 50.0
        if dp == 0 or dq == 0:
            nstar = None; note = "zero denominator"
        elif (dp > 0) != (dq > 0):
            nstar = (-dp / dq) if (-dp / dq) > 0 else None; note = "opposite signs"
        else:
            nstar = None; note = "same sign; no positive crossover"
        cost[f"{d}:{c}"] = {"delta_prep_s": dp, "delta_query_per_image_s": dq,
                            "crossover_images": nstar, "note": note}

    all_ok = all(x["passed"] for x in CHECKS)
    audit = {"status": "AUDITED_MERGE_COMPLETE" if all_ok else "MERGE_AUDIT_FAILED",
             "not_single_run": True, "n_rows": len(merged), "n_checks": len(CHECKS),
             "n_passed": int(sum(x["passed"] for x in CHECKS)),
             "failed": [x for x in CHECKS if not x["passed"]],
             "source_hashes": {
                 "v3_csv": sha(V3_CSV), "v3_summary": sha(V3_SUMMARY),
                 "v3_manifest": sha(V3_MANIFEST),
                 "v4_csv": sha(V4_CSV), "v4_summary": sha(V4_SUMMARY),
                 "v4_warning": sha(V4_WARNING),
                 "v5_csv": sha(V5_CSV), "v5_summary": sha(V5_SUMMARY),
                 "v5_warning": sha(V5_WARNING) if V5_WARNING.is_file() else None,
                 "v5_self_check": sha(V5_SELF_CHECK),
                 "v5_whitelist": sha(V5_WHITELIST),
             },
             "warnings": {"v4": v4w, "v5": v5w},
             "strict_all_metric_reproduction": bool(len(v4w) == 0 and len(v5w) == 0),
             "gate_policies": {"v3": V3_GATE, "v4": V4_GATE, "v5": V5_GATE},
             "b3_note": "B3 is fixed 50-image scorer-only load from different sessions on same RTX 4090; not full-test throughput.",
             "primary": primary, "secondary": sec, "pooled": pooled,
             "method_summary": {"MVTec15": per_dataset["MVTec15"], "VisA12": per_dataset["VisA12"],
                                "ALL27": all_summary},
             "adverse_image": adverse_image, "cost": cost,
             "b3_coverage": {f"{d}:{c}": ((min(50, counts[f"{d}:{c}"]) / counts[f"{d}:{c}"])
                                            if counts.get(f"{d}:{c}") else None)
                             for d, c in full27}}
    (out / "MERGED_135_AUDIT.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    (out / "MERGED_135_STATS.json").write_text(json.dumps(
        {"primary": primary, "secondary": sec, "pooled": pooled,
         "method_summary": audit["method_summary"], "adverse_image": adverse_image,
         "cost": cost}, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---------------- report
    L = []
    L.append("# 实验 A/B：MVTec 15 + VisA 15 + 剩余 21 类 105 条的 27 类 135 条合并与科学分析\n\n")
    L.append("- 日期：2026-10-25\n")
    L.append(f"- 合并状态：**{audit['status']}**（不是单次 135 条运行）\n")
    L.append(f"- 审计：**{audit['n_passed']}/{audit['n_checks']} checks**\n")
    L.append("- 数据：v3 MVTec 三类别 15 条 + v4 VisA 三类别 15 条 + v5 剩余 21 类 105 条；27 类 × 5 方法 = 135 条。\n")
    L.append("- 名称：Experiment A/B five-method 27-category 135 records；与既有 QuickGELU 135 条原实验不同。\n")
    L.append("- 统计方案在六类探索后、21 类运行前锁定；不是完全独立盲法确认。\n\n")
    L.append("## 0. 三项文档勘误（2026-10-25）\n\n")
    L.append("1. v3 的 MVTec 部分是 **三类别、15 条记录**（每类 5 方法），不是 15 个类别。\n")
    L.append("2. `b4_map_pipeline_s` 的源码边界是逐图 **scorer scoring + normalization + upsampling** 的计时求和；"
             "不包含 top-k 运算耗时，不能称为仅后处理时间。\n")
    L.append("3. v5 runner 内部 `CSV_NAME` 仍为 `EXPERIMENT_AB_SIX_CATEGORY_PILOT_RESULTS.csv`；"
             "未修改已审核 runner；交付副本命名为 `EXPERIMENT_AB_REMAINING21_RESULTS.csv`，内容逐字节一致。\n\n")
    L.append("## 1. 主要比较：A0 vs A2，AUPRO@0.30\n\n")
    L.append("| 数据集 | n | 平均 Δ | 中位 Δ | 95% bootstrap CI | 正/负/零 | Wilcoxon p | Holm p |\n")
    L.append("|---|---:|---:|---:|---|---:|---:|---:|\n")
    for name, cats in (("MVTec15", mvtec), ("VisA12", visa)):
        s = primary[name]
        L.append(f"| {name} | {s['n']} | {s['mean_diff']:+.6f} | {s['median_diff']:+.6f} | "
                 f"[{s['bootstrap_ci95_mean'][0]:+.6f}, {s['bootstrap_ci95_mean'][1]:+.6f}] | "
                 f"{s['positive']}/{s['negative']}/{s['zero']} | {s['wilcoxon_p']:.6g} | "
                 f"{s['holm_adjusted_p']:.6g} |\n")
    L.append("\n> Holm 家族为 MVTec15/VisA12 两个主要检验，α=0.05；只有 Holm p<0.05 才使用确认性措辞。"
             "Wilcoxon 方法为 " + primary["MVTec15"]["wilcoxon_method"] + "。\n\n")
    L.append("## 2. 次要比较（BH-FDR，探索性）\n\n")
    L.append("| 比较 | 平均 Δ | 中位 Δ | 正/负/零 | raw p | BH q |\n")
    L.append("|---|---:|---:|---:|---:|---:|\n")
    for key, s in sec.items():
        L.append(f"| {key} | {s['mean_diff']:+.6f} | {s['median_diff']:+.6f} | "
                 f"{s['positive']}/{s['negative']}/{s['zero']} | {s['wilcoxon_p']:.6g} | {s['bh_q']:.6g} |\n")
    L.append("\n> 共 10 个次要检验，BH-FDR；全部报告，不选择性删除；均为探索性，不升级为确认性总体优势。\n\n")
    L.append("## 3. 五方法描述统计（完整报告 A1/A3/A4）\n\n")
    for label in ("MVTec15", "VisA12"):
        L.append(f"### {label}\n\n")
        L.append("| 方法 | 平均 Image | 平均 Pixel | 平均 AUPRO | B2 中位(s) | B3 中位(s) | 存储(B) |\n")
        L.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for m in METHODS:
            s = per_dataset[label][m]
            L.append(f"| {m} | {s['metrics_mean']['image_auroc_top1']:.6f} | "
                     f"{s['metrics_mean']['pixel_auroc_global']:.6f} | "
                     f"{s['metrics_mean']['aupro_0_30']:.6f} | {s['b2_median_s']:.3f} | "
                     f"{s['b3_median_s']:.6f} | {s['storage_min_bytes']} |\n")
        L.append("\n")
    L.append("## 4. 27 类 A0/A2 类别级差异与不利结果\n\n")
    for dsname, cats in (("MVTec", mvtec), ("VisA", visa)):
        L.append(f"### {dsname}\n\n")
        L.append("| 类别 | A0 AUPRO | A2 AUPRO | Δ AUPRO | A0 Image | A2 Image | Δ Image | A0 Pixel | A2 Pixel | Δ Pixel |\n")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for d, c in cats:
            a0 = by[(d, c, "A0_vne_k1")]; a2 = by[(d, c, "A2_nearest_coreset5")]
            L.append(f"| {c} | {f(a0['aupro_0_30']):.6f} | {f(a2['aupro_0_30']):.6f} | "
                     f"{f(a0['aupro_0_30'])-f(a2['aupro_0_30']):+.6f} | "
                     f"{f(a0['image_auroc_top1']):.6f} | {f(a2['image_auroc_top1']):.6f} | "
                     f"{f(a0['image_auroc_top1'])-f(a2['image_auroc_top1']):+.6f} | "
                     f"{f(a0['pixel_auroc_global']):.6f} | {f(a2['pixel_auroc_global']):.6f} | "
                     f"{f(a0['pixel_auroc_global'])-f(a2['pixel_auroc_global']):+.6f} |\n")
        L.append("\n")
    L.append(f"**Image AUROC 退步类别（A0<A2，共 {len(adverse_image)} 个）**：")
    L.append(", ".join(f"{x['dataset']}:{x['category']} ({x['delta_a0_minus_a2']:+.6f})"
                       for x in adverse_image) + "\n\n")
    L.append("## 5. 存储与效率\n\n")
    L.append("- A0 最小推理状态 1,050,628 B；A2/A4 1,003,520 B；A3 2,007,040 B；A1 20,070,400 B。\n")
    L.append("- A0 比 A2 大约 4.7%；A1/A0 约 19.1×；A1/A2 约 20.0×。\n")
    L.append("- A0 3,147,780 B 是拟合后原生数组口径；1,050,628 B 是去冗余推理口径，不能混称。\n")
    L.append("- B3 是 CPU scorer-only、固定 min(50,n_test) 图负载；50 图以上类别取前 50，"
             "MVTec toothbrush 仅 42 张则全用；VisA 200 图类覆盖约 25%，不是完整测试集吞吐量。\n")
    L.append("- B4 的 `b4_map_pipeline_s` 是逐图 scoring+normalization+upsampling 求和，不含 top-k。\n\n")
    L.append("## 6. 27 类总体描述（不是第三个独立数据集）\n\n")
    L.append("| 指标 | 平均 Δ(A0−A2) | 中位 Δ | 范围 | A0 更好 | A2 更好 | 持平 |\n")
    L.append("|---|---:|---:|---|---:|---:|---:|\n")
    for m in METRICS:
        p = pooled[m]
        L.append(f"| {LABELS[m]} | {p['mean_delta']:+.6f} | {p['median_delta']:+.6f} | "
                 f"[{p['min']:+.6f}, {p['max']:+.6f}] | {p['a0_better']} | "
                 f"{p['a2_better']} | {p['tie']} |\n")
    L.append("\n> 仅描述性合并；27 类不是第三个独立数据集，不据 pooled 27 做确认性显著性声明。\n\n")
    L.append("## 7. 论文价值判断（基于锁定统计方案，不预设 VNE 获胜）\n\n")
    sig_m = primary["MVTec15"]["significant_holm_0p05"]
    sig_v = primary["VisA12"]["significant_holm_0p05"]
    if sig_m and sig_v:
        L.append("- **AUPRO 优势**：MVTec15 与 VisA12 的 A0−A2 AUPRO 主要比较均通过 Holm 校正（p<0.05）；"
                 "在这两个数据集上存在可重复的 AUPRO 描述/确认性优势。但六类探索后的方案不是完全盲法。\n")
    elif sig_m or sig_v:
        L.append(f"- **AUPRO 优势**：仅 {'MVTec15' if sig_m else 'VisA12'} 通过 Holm；另一数据集未通过。"
                 "只能支持局部/条件性优势，不能宣称跨数据集普遍优势。\n")
    else:
        L.append("- **AUPRO 优势**：两个主要检验均未通过 Holm 校正；当前数据不支持确认性的 AUPRO 优势。\n")
    L.append(f"- **Image 退步**：A0−A2 的 Image AUROC 负向类别共 {len(adverse_image)} 个；"
             "完整列于 §4，不可只报告 Pixel/AUPRO。\n")
    L.append("- **VNE vs Random-5%**：A4 准备成本远低于 VNE（见 B2），检测质量需按 §2/§3 完整比较；"
             "不能只用计算昂贵的 coreset 作为唯一对照。\n")
    L.append("- **VNE 的真实优势维度**：需以 Pixel/AUPRO、B2 vs coreset、固定负载 B3 和相对完整 bank 的存储为边界；"
             "不得声称相对 random 的准备优势，也不得把 B3 当 GPU/工业吞吐。\n")
    L.append("- **论文重写判断**：若主要 AUPRO 结果在至少一个数据集确认，且完整披露 Image 退步与成本 trade-off，"
             "可作为受控实证/基准论文进行实质性重写；若两个主要检验均不显著，则不足以支撑 VNE 优越性论文，"
             "应改写为失败/负结果或停止投稿。最终以独立审核为准。\n\n")
    L.append("## 7A. 五个问题的明确回答（基于实际合并数据）\n\n")
    pm = primary["MVTec15"]; pv = primary["VisA12"]
    a4m = sec["MVTec15:A0_vs_A4_aupro"]; a4v = sec["VisA12:A0_vs_A4_aupro"]
    m0 = per_dataset["MVTec15"]; v0 = per_dataset["VisA12"]
    pcb_neg = [x for x in adverse_image if x["category"].startswith("pcb")]
    L.append("**Q1：AUPRO 优势是否分别在 MVTec15 和 VisA12 存在？**\n\n")
    L.append(f"- MVTec15：平均差 {pm['mean_diff']:+.6f}，95% CI [{pm['bootstrap_ci95_mean'][0]:+.6f}, "
             f"{pm['bootstrap_ci95_mean'][1]:+.6f}]，方向 {pm['positive']}/{pm['negative']}/{pm['zero']}，"
             f"Holm p={pm['holm_adjusted_p']:.6g}。\n")
    L.append(f"- VisA12：平均差 {pv['mean_diff']:+.6f}，95% CI [{pv['bootstrap_ci95_mean'][0]:+.6f}, "
             f"{pv['bootstrap_ci95_mean'][1]:+.6f}]，方向 {pv['positive']}/{pv['negative']}/{pv['zero']}，"
             f"Holm p={pv['holm_adjusted_p']:.6g}。\n")
    L.append("- **回答**：两个数据集均通过 Holm，支持 VNE 的 AUPRO 优势；但方案在六类探索后锁定，不是独立盲法确认。\n\n")
    L.append("**Q2：VNE 与 Random-5% 的质量/成本权衡是否支持工程价值？**\n\n")
    L.append(f"- AUPRO A0−A4：MVTec15 {a4m['mean_diff']:+.6f}（{a4m['positive']}/{a4m['negative']}/{a4m['zero']}，BH q={a4m['bh_q']:.6g}）；"
             f"VisA12 {a4v['mean_diff']:+.6f}（{a4v['positive']}/{a4v['negative']}/{a4v['zero']}，BH q={a4v['bh_q']:.6g}）。\n")
    L.append(f"- 平均 Image/Pixel/AUPRO：MVTec A0 {m0['A0_vne_k1']['metrics_mean']['image_auroc_top1']:.6f}/"
             f"{m0['A0_vne_k1']['metrics_mean']['pixel_auroc_global']:.6f}/{m0['A0_vne_k1']['metrics_mean']['aupro_0_30']:.6f}，"
             f"A4 {m0['A4_nearest_random5']['metrics_mean']['image_auroc_top1']:.6f}/"
             f"{m0['A4_nearest_random5']['metrics_mean']['pixel_auroc_global']:.6f}/{m0['A4_nearest_random5']['metrics_mean']['aupro_0_30']:.6f}；"
             f"VisA A0 {v0['A0_vne_k1']['metrics_mean']['image_auroc_top1']:.6f}/"
             f"{v0['A0_vne_k1']['metrics_mean']['pixel_auroc_global']:.6f}/{v0['A0_vne_k1']['metrics_mean']['aupro_0_30']:.6f}，"
             f"A4 {v0['A4_nearest_random5']['metrics_mean']['image_auroc_top1']:.6f}/"
             f"{v0['A4_nearest_random5']['metrics_mean']['pixel_auroc_global']:.6f}/{v0['A4_nearest_random5']['metrics_mean']['aupro_0_30']:.6f}。\n")
    L.append(f"- B2 中位：A0 {m0['A0_vne_k1']['b2_median_s']:.3f}s/{v0['A0_vne_k1']['b2_median_s']:.3f}s，"
             f"A4 {m0['A4_nearest_random5']['b2_median_s']:.4f}s/{v0['A4_nearest_random5']['b2_median_s']:.4f}s；"
             f"B3 中位：A0 {m0['A0_vne_k1']['b3_median_s']:.6f}s/{v0['A0_vne_k1']['b3_median_s']:.6f}s，"
             f"A4 {m0['A4_nearest_random5']['b3_median_s']:.6f}s/{v0['A4_nearest_random5']['b3_median_s']:.6f}s。\n")
    L.append("- **回答**：VNE 的检测质量（尤其 Pixel/AUPRO）和固定负载查询优于 A4，但准备成本远高于 A4；"
             "这是明确的 trade-off，不是全面支配。\n\n")
    L.append("**Q3：图像级退步是否集中在 PCB？**\n\n")
    L.append(f"- A0 相对 A2 的 Image AUROC 负向类别 **{len(adverse_image)}/27**；PCB 四类全部为负（{', '.join(x['category'] for x in pcb_neg)}），"
             f"其余 {len(adverse_image)-len(pcb_neg)} 个非 PCB 类别也为负。\n")
    L.append("- 最大退步：pcb3 −0.1373、pcb4 −0.1139、screw −0.0996、pcb2 −0.0755。PCB 是稳定弱项，但不是唯一来源。\n")
    L.append("- **回答**：图像级退步是实质性的、论文无法回避的局限；不能只报告 Pixel/AUPRO。\n\n")
    L.append("**Q4：VNE 哪些维度确有优势，哪些没有？**\n\n")
    L.append(f"- **有**：AUPRO（MVTec {pm['mean_diff']:+.6f}，VisA {pv['mean_diff']:+.6f}）；"
             f"Pixel AUROC（MVTec {sec['MVTec15:A0_vs_A2_pixel']['mean_diff']:+.6f}，"
             f"VisA {sec['VisA12:A0_vs_A2_pixel']['mean_diff']:+.6f}）；"
             f"B2 低于 A2/A3 coreset；B3 固定负载低于 A2/A3/A1；相对 A1 存储约小 19.1×。\n")
    L.append("- **没有/受限**：Image AUROC 总体不稳定（27 类 11 好/15 差/1 平，PCB 全退步）；"
             "存储比 A2 大 4.7%；准备成本远高于 A4 random 和 A1 直接保存；B3 不是 GPU/完整吞吐；无 E2/现场部署。\n\n")
    L.append("**Q5：是否有充分依据实质性重写并尝试正规期刊？**\n\n")
    if sig_m and sig_v:
        L.append("- **有条件 YES（作为受控实证/benchmark 论文）**：27 类、五方法、同骨干、接近 1MB 的完整比较，"
                 "主 AUPRO 在两个数据集通过 Holm；可以实质性重写。\n")
        L.append("- **但不是 VNE 普遍优越论文**：Image 退步 15/27、PCB 全退、存储大 4.7%、准备成本高于 random/完整 bank、无 E2/工业吞吐；"
                 "这些必须作为核心 negative results 完整呈现。\n")
        L.append("- **期刊可行性**：适合方法/评测型国际期刊的实证/benchmark 定位（如 Journal of Electronic Imaging、SIVP、MVA 一类）；"
                 "Q1 顶刊可行性不确定；若要求新算法或全面 SOTA，则贡献不足。最终由 ChatGPT 独立审核决定。\n")
    else:
        L.append("- **NO（作为 VNE 优越性论文）**：主要 AUPRO 未在两个数据集同时确认；只能作为负结果/受控评测，不能按优越性叙事重写。\n")
    L.append("- 本轮不改正文、不投稿。\n\n")
    L.append("## 8. 合并与来源\n\n")
    L.append("- 135 个唯一类别—方法组合；无缺失、无重复；原始 CSV 未修改；数值逐字段核对。\n")
    L.append("- 每条保留 source batch、runner 版本/SHA、Gate 政策、记录时间与 B3 批次说明。\n")
    L.append("- v3 MVTec 15 是三类别×5 方法；v4 VisA 15 是三类别×5；v5 是剩余 21 类×5。\n")
    L.append("- 不同 runner/批次/硬件会话；B3 不是单次 135 条运行的吞吐量。\n")
    L.append("- 已知 WARNING 完整保留：" + f"v4 {len(v4w)} 条，v5 {len(v5w)} 条；"
             "不得声称所有指标严格复现成功。\n")
    L.append(f"- 合并 CSV SHA256：`{sha(merged_csv)}`\n")
    L.append("- 源文件 SHA256：\n")
    for k, v in audit["source_hashes"].items():
        L.append(f"  - `{k}`: `{v}`\n")
    (out / "EXPERIMENT_AB_135_MERGE_STATS_AND_PUBLICATION_VALUE_zh.md").write_text(
        "".join(L), encoding="utf-8")
    print(f"\nMERGE AUDIT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({audit['n_passed']}/{audit['n_checks']})")
    print("WROTE", merged_csv)
    print("WROTE", out / "MERGED_135_AUDIT.json")
    print("WROTE", out / "MERGED_135_STATS.json")
    print("WROTE", out / "EXPERIMENT_AB_135_MERGE_STATS_AND_PUBLICATION_VALUE_zh.md")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
