# -*- coding: utf-8 -*-
"""STAGE3B VisA controlled research extension — runner v2f (QuickGELU architecture correction).

Derived from the audited Stage3B v1f runner (which itself derived from Stage3A v2-final) with:
  - MODEL ARCHITECTURE FIX (ChatGPT-approved, 2026-10-23): create_model_and_transforms
    ("ViT-B-16-quickgelu", pretrained=None) replaces ("ViT-B-16", pretrained=None).
    The frozen OpenAI weights (sha256 4b869929..., official repo
    timm/vit_base_patch16_clip_224.openai) declare quick_gelu=true in open_clip_config.json;
    the OpenAI CLIP ViT was trained with QuickGELU. open_clip 3.3.0's plain "ViT-B-16"
    config has NO quick_gelu field (GELU default) and pretrained-tag checks only warn.
  - v1f's two fixes retained: smoke default category = first frozen manifest category
    (never hardcoded), and strict 64-char hex check for data_signature_v2.
Unchanged: VisA adapter v1 (12 categories, official 1cls split, explicit name-paired
image<->mask loading, no cross-split name indexing), 50-reference sorted-first rule,
reference_minmax, evaluators, completeness standard 12x5=60, resume integrity, SafetyStop,
output isolation, PRELIMINARY_DIAGNOSTIC timing.
New markers/producer/manifest path/CSV names prevent mixing with v1f and legacy results.

Modes: preflight | smoke (tiny) | formal. No network. No credential handling. Read-only imports.
Exit codes: 0 ok / 2 preflight failed / 4 safety-or-wallclock stop / 5 fingerprint-refusal.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, re, shutil, sys, time
from pathlib import Path

import numpy as np

RUNNER_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = RUNNER_DIR / "STAGE3B_V2F_QUICKGELU_PROTOCOL_MANIFEST.json"

FORMAL_MARKER = "PUBLICATION_FORMAL_STAGE3B_VISA_V2F_QUICKGELU"
SMOKE_MARKER = "NON_PUBLICATION_SMOKE_STAGE3B_VISA_V2F_QUICKGELU"

# Canonical OpenAI CLIP architecture: QuickGELU (matches the frozen weights' official config).
MODEL_NAME = "ViT-B-16-quickgelu"

# test-only injection point for the per-category runner; None in production.
CATEGORY_RUNNER_HOOK = None


# ==================================================================================
# Pure functions (unit/regression-tested; no heavy imports)
# ==================================================================================

def classify_break_even(delta_prep, delta_score, ci_delta_prep=None, ci_delta_score=None):
    """CostDifference(N) = DeltaP - N*DeltaS ; six regimes + zero-boundary + indeterminate."""
    r = {"delta_prep": delta_prep, "delta_score": delta_score,
         "regime": None, "n_star": None, "first_strict_N": None, "note": ""}
    if ci_delta_prep is not None and (ci_delta_prep[0] <= 0 <= ci_delta_prep[1]):
        r["regime"] = "INDETERMINATE"; r["note"] = "CI of DeltaP spans zero"; return r
    if ci_delta_score is not None and (ci_delta_score[0] <= 0 <= ci_delta_score[1]):
        r["regime"] = "INDETERMINATE"; r["note"] = "CI of DeltaS spans zero"; return r
    if delta_score == 0 and delta_prep == 0:
        r["regime"] = "IDENTICAL"; r["note"] = "equal cost structure"; return r
    if delta_score == 0:
        r["regime"] = "NEAREST_DOMINATES" if delta_prep > 0 else "VNE_DOMINATES"
        r["note"] = "DeltaS=0 boundary; dominance follows sign of DeltaP; no division"; return r
    if delta_prep == 0:
        r["regime"] = "VNE_DOMINATES" if delta_score > 0 else "NEAREST_DOMINATES"
        r["note"] = "DeltaP=0 boundary; tie at N=0; strict for N>0; no division"; return r
    if delta_prep > 0 and delta_score > 0:
        n_star = delta_prep / delta_score
        r.update({"regime": "CROSSOVER", "n_star": n_star, "first_strict_N": int(np.floor(n_star)) + 1,
                  "note": "Nearest cheaper at low load; VNE cheaper above N*; tie exactly at integer N*"})
        return r
    if delta_prep < 0 and delta_score < 0:
        n_star = delta_prep / delta_score
        r.update({"regime": "CROSSOVER", "n_star": n_star, "first_strict_N": int(np.floor(n_star)) + 1,
                  "note": "VNE cheaper at low load; Nearest cheaper above N*; tie exactly at integer N*"})
        return r
    if delta_prep > 0 and delta_score < 0:
        r["regime"] = "NEAREST_DOMINATES"; r["note"] = "CostDifference(N)>0 for all N>=0"; return r
    r["regime"] = "VNE_DOMINATES"; r["note"] = "CostDifference(N)<0 for all N>=0"; return r


REQUIRED_CHAIN_FIELDS = ["status", "metrics", "ref_raw_range", "degenerate_reference_range",
                         "storage_bytes", "timing_s"]
REQUIRED_METRICS = ["image_auroc_top1", "pixel_auroc_global", "aupro_0_30"]
REQUIRED_CAT_FIELDS = ["category", "marker", "producer", "n_ref_images", "n_ref_patches", "n_test_images",
                       "n_test_normal", "n_test_defect", "data_signature_v2", "timing_s", "chains"]


def _valid_metric(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return np.isfinite(f) and 0.0 <= f <= 1.0


_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def is_valid_sha256_hex(value):
    """Strict SHA256 hex digest format: exactly 64 hexadecimal characters (v1f fix)."""
    return isinstance(value, str) and bool(_SHA256_HEX_RE.match(value))


def select_run_categories(mode, category, categories, frozen_cats):
    """Run selection: subset only affects WHICH categories are computed, never the COMPLETE standard.
    Smoke default = the FIRST frozen manifest category (VisA: candle); never a hardcoded name (v1f fix)."""
    if mode == "smoke":
        return [category or frozen_cats[0]]
    return categories.split(",") if categories else list(frozen_cats)


def check_completeness(records, expected_categories, expected_chain_ids):
    """COMPLETE standard = frozen (12 x 5 = 60) unique valid records. The expected sets are the
    FROZEN manifest sets and must never be relaxed to the actually-run subset."""
    exp_cats = list(expected_categories)
    exp_chains = set(expected_chain_ids)
    rep = {"expected_categories": len(exp_cats), "expected_chains": sorted(exp_chains),
           "expected_keys": len(exp_cats) * len(exp_chains),
           "found_valid_keys": 0, "missing_categories": [], "unexpected_categories": [],
           "missing_chains_in_category": [], "unexpected_chains_in_category": [],
           "duplicate_keys": [], "failure_statuses": [], "invalid_fields": [], "schema_missing": [],
           "all_ok": False}
    seen = set()
    found_cats = set()
    for e in records:
        cat = e.get("category")
        if cat not in exp_cats:
            rep["unexpected_categories"].append(str(cat)); continue
        found_cats.add(cat)
        for f in REQUIRED_CAT_FIELDS:
            if f not in e:
                rep["schema_missing"].append(f"{cat}:missing_category_field:{f}")
        for f in ("n_ref_images", "n_ref_patches", "n_test_images", "n_test_normal", "n_test_defect"):
            v = e.get(f)
            if not isinstance(v, int) or v < 0:
                rep["invalid_fields"].append(f"{cat}:invalid:{f}={v!r}")
        chain_keys = set(e.get("chains", {}).keys())
        missing = sorted(exp_chains - chain_keys)
        unexpected = sorted(chain_keys - exp_chains)
        if missing:
            rep["missing_chains_in_category"].append(f"{cat}:{missing}")
        if unexpected:
            rep["unexpected_chains_in_category"].append(f"{cat}:{unexpected}")
        for cid, c in e.get("chains", {}).items():
            key = (cat, cid)
            if key in seen:
                rep["duplicate_keys"].append(f"{cat}/{cid}")
                continue
            seen.add(key)
            ok = True
            for f in REQUIRED_CHAIN_FIELDS:
                if f not in c:
                    rep["schema_missing"].append(f"{cat}/{cid}:missing_field:{f}"); ok = False
            if c.get("status") != "OK":
                rep["failure_statuses"].append(f"{cat}/{cid}:status={c.get('status')}"); ok = False
            for mk in REQUIRED_METRICS:
                if not _valid_metric(c.get("metrics", {}).get(mk)):
                    rep["invalid_fields"].append(f"{cat}/{cid}:invalid_metric:{mk}={c.get('metrics', {}).get(mk)!r}"); ok = False
            if not isinstance(c.get("storage_bytes"), int) or c.get("storage_bytes", -1) < 0:
                rep["invalid_fields"].append(f"{cat}/{cid}:invalid:storage_bytes"); ok = False
            if not isinstance(c.get("degenerate_reference_range"), bool):
                rep["invalid_fields"].append(f"{cat}/{cid}:invalid:degenerate_reference_range"); ok = False
            if not is_valid_sha256_hex(e.get("data_signature_v2")):
                rep["invalid_fields"].append(f"{cat}:invalid:data_signature_v2"); ok = False
            if ok and cid in exp_chains:
                rep["found_valid_keys"] += 1
    rep["missing_categories"] = [c for c in exp_cats if c not in found_cats]
    rep["all_ok"] = (not rep["missing_categories"] and not rep["unexpected_categories"]
                     and not rep["missing_chains_in_category"] and not rep["unexpected_chains_in_category"]
                     and not rep["duplicate_keys"] and not rep["failure_statuses"]
                     and not rep["invalid_fields"] and not rep["schema_missing"]
                     and rep["found_valid_keys"] == rep["expected_keys"])
    return rep


def file_sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def build_data_signature_v2(ref_paths, good_paths, defect_pairs, data_root):
    """Content-level machine-independent signature.
    ref_paths: normal reference images; good_paths: normal test images;
    defect_pairs: list of (defect_image_path, mask_path) with EXPLICIT pairing.
    Relative POSIX paths + per-file content SHA256; data_root NOT included in the hash."""
    droot = Path(data_root).resolve()

    def rel(p):
        return Path(p).resolve().relative_to(droot).as_posix()

    payload = {
        "signature_version": "v2_content_sha256",
        "refs": sorted([rel(p), file_sha256(p)] for p in ref_paths),
        "test_good": sorted([rel(p), file_sha256(p)] for p in good_paths),
        "defect_pairs": sorted([rel(ip), file_sha256(ip), rel(mp), file_sha256(mp)] for ip, mp in defect_pairs),
        "counts": {"n_ref": len(ref_paths), "n_good": len(good_paths), "n_defect": len(defect_pairs)},
    }
    sig = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return sig, payload


def validate_resume_state(out_root, fingerprint):
    """Pre-model resume gate. Returns None if OK, else an error string (caller exits 5)."""
    percat = out_root / "per_category"
    existing = sorted(percat.glob("*.json")) if percat.is_dir() else []
    fp_file = out_root / "run_fingerprint.json"
    if not existing:
        return None
    if not fp_file.is_file():
        return ("formal records exist but run_fingerprint.json is missing: RESUME REFUSED "
                "(fingerprints are never backfilled for results of unknown origin)")
    old = json.loads(fp_file.read_text(encoding="utf-8"))
    if old != fingerprint:
        return "run fingerprint mismatch (protocol/runner/weights/code changed): RESUME REFUSED"
    return None


REQUIRED_CAT_TIMING_KEYS = ["ref_encode", "test_encode"]
REQUIRED_CHAIN_TIMING_KEYS = ["fit_s", "ref_score_s", "ref_norm_s", "test_scoring_s", "test_normalize_s",
                              "test_upsample_s", "test_map_pipeline_s", "metrics_s", "chain_total_s"]


def validate_resume_category(prev, category, expected_chain_ids, recomputed_sig):
    if prev.get("marker") != FORMAL_MARKER:
        return f"{category}: record marker is not a formal-v2-final record"
    if prev.get("producer") != "stage3b_v2f_quickgelu":
        return f"{category}: record producer is not stage3b_v2f_quickgelu ({prev.get('producer')!r})"
    if prev.get("category") != category:
        return f"{category}: category identity mismatch ({prev.get('category')!r})"
    missing_t = [k for k in REQUIRED_CAT_TIMING_KEYS if k not in prev.get("timing_s", {})]
    if missing_t:
        return f"{category}: category timing_s missing keys {missing_t}"
    if set(prev.get("chains", {}).keys()) != set(expected_chain_ids):
        return f"{category}: chain set mismatch {sorted(prev.get('chains', {}).keys())}"
    for cid, c in prev.get("chains", {}).items():
        if c.get("status") != "OK":
            return f"{category}/{cid}: non-OK status stored ({c.get('status')!r})"
        for mk in REQUIRED_METRICS:
            if not _valid_metric(c.get("metrics", {}).get(mk)):
                return f"{category}/{cid}: invalid metric stored ({mk})"
        missing_ct = [k for k in REQUIRED_CHAIN_TIMING_KEYS if k not in c.get("timing_s", {})]
        if missing_ct:
            return f"{category}/{cid}: chain timing_s missing keys {missing_ct}"
    if prev.get("data_signature_v2") != recomputed_sig:
        return f"{category}: data signature mismatch (files changed or old-format record)"
    return None


# ==================================================================================
# Portability helpers
# ==================================================================================

def default_project_root():
    """Robust default: walk up from this file to the first ancestor containing src/sara/vne.py.
    Depth-independent (the runner may live at different nesting levels). Falls back to cwd with
    a visible warning; users on other machines should pass --project-root explicitly."""
    here = Path(__file__).resolve()
    for anc in [here.parent, *here.parents]:
        if (anc / "src" / "sara" / "vne.py").is_file() and (anc / "experiments").is_dir():
            return anc
    print("WARNING: could not derive project root by marker search; falling back to cwd. "
          "Pass --project-root explicitly.")
    return Path.cwd()


# ==================================================================================
# VisA adapter v1 loader (self-contained; explicit within-split name pairing)
# ==================================================================================

VISA_SUBDIRS = ["train/good", "test/good", "test/anomaly", "ground_truth/anomaly"]


def load_visa_category(root, category):
    """Load one VisA (visa_mvtec_1cls_v1 adapter) category.

    Layout: {root}/{category}/{train/good, test/good, test/anomaly, ground_truth/anomaly}/*.png
    Masks are NAME-PAIRED with defect test images (000.png <-> 000.png); the adapter does NOT
    use MVTec's '_mask' suffix. Cross-split filename reuse exists in 5 categories as DIFFERENT
    images (content-verified); mask pairing is strictly within test/anomaly <-> ground_truth/anomaly.
    """
    cat_dir = Path(root) / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"VisA category dir missing: {cat_dir}")
    missing = [d for d in VISA_SUBDIRS if not (cat_dir / d).is_dir()]
    if missing:
        raise FileNotFoundError(f"{category}: missing subdirs {missing}")
    train_good = sorted((cat_dir / "train" / "good").glob("*.png"))
    test_good = sorted((cat_dir / "test" / "good").glob("*.png"))
    test_images = sorted((cat_dir / "test" / "anomaly").glob("*.png"))
    gt_masks = sorted((cat_dir / "ground_truth" / "anomaly").glob("*.png"))
    img_by_name = {p.name: p for p in test_images}
    msk_by_name = {p.name: p for p in gt_masks}
    if set(img_by_name) != set(msk_by_name):
        only_i = sorted(set(img_by_name) - set(msk_by_name))[:5]
        only_m = sorted(set(msk_by_name) - set(img_by_name))[:5]
        raise ValueError(f"{category}: image/mask name pairing mismatch "
                         f"(images_without_mask={only_i}, masks_without_image={only_m})")
    pairs = [(img_by_name[n], msk_by_name[n]) for n in sorted(img_by_name)]
    if not train_good or not test_good or not pairs:
        raise ValueError(f"{category}: empty train_good/test_good/test_anomaly split")
    return {"train_good": train_good, "test_good": test_good,
            "test_defect": {"anomaly": {"images": [p for p, _ in pairs],
                                        "masks": [m for _, m in pairs]}}}


def check_visa_structure(data_root, manifest):
    """Full 12-category structure/count check for the VisA adapter (preflight; no model load)."""
    rep = {"ok": False, "issues": [], "counts": {}}
    exp = manifest.get("dataset", {}).get("expected_per_category", {})
    for c in manifest["dataset"]["categories"]:
        cd = Path(data_root) / c
        if not cd.is_dir():
            rep["issues"].append(f"{c}: category dir missing")
            continue
        missing = [d for d in VISA_SUBDIRS if not (cd / d).is_dir()]
        if missing:
            rep["issues"].append(f"{c}: missing subdirs {missing}")
            continue
        n_tg = len(list((cd / "train" / "good").glob("*.png")))
        n_sg = len(list((cd / "test" / "good").glob("*.png")))
        ta = sorted((cd / "test" / "anomaly").glob("*.png"))
        gt = sorted((cd / "ground_truth" / "anomaly").glob("*.png"))
        if {p.name for p in ta} != {p.name for p in gt}:
            rep["issues"].append(f"{c}: test/anomaly vs ground_truth/anomaly name sets differ")
        rep["counts"][c] = {"train_good": n_tg, "test_good": n_sg,
                            "test_anomaly": len(ta), "gt_anomaly": len(gt)}
        e = exp.get(c)
        if e and [n_tg, n_sg, len(ta)] != list(e):
            rep["issues"].append(f"{c}: counts {[n_tg, n_sg, len(ta)]} != manifest expected {list(e)}")
    rep["ok"] = not rep["issues"]
    return rep


def resolve_paths(args):
    project_root = Path(args.project_root).resolve() if args.project_root else default_project_root()
    data_root = Path(args.data_root).resolve() if args.data_root else project_root / "data" / "visa_mvtec_1cls_v1"
    weights = (Path(args.weights).resolve() if args.weights else
               project_root / "<INTERNAL_ASSETS_DIR>"
               / "stage2_e0_pilot" / "weights" / "open_clip_model.safetensors")
    output_dir = (Path(args.output_dir).resolve() if args.output_dir else RUNNER_DIR / "runs" / args.mode)
    return {"project_root": project_root, "data_root": data_root, "weights": weights, "output_dir": output_dir}


def avail_gb():
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("inf")


DEP_FILES = ["src/sara/vne.py",
             "experiments/vne_reference_baseline/run_vne_reference_baseline.py",
             "experiments/ablation/ablation_utils.py"]
CODE_HASH_KEYS = ["src/sara/vne.py",
                  "experiments/vne_reference_baseline/run_vne_reference_baseline.py",
                  "experiments/ablation/ablation_utils.py"]


def preflight(paths, manifest, args):
    rep = {"checks": {}, "all_ok": False, "paths": {k: str(v) for k, v in paths.items()}}
    pr = paths["project_root"]
    rep["checks"]["project_root_exists"] = pr.is_dir()
    missing_deps = [f for f in DEP_FILES if not (pr / f).is_file()]
    rep["checks"]["dependency_files"] = {"ok": not missing_deps, "missing": missing_deps}
    cats = manifest["dataset"]["categories"]
    missing_cats = [c for c in cats if not (paths["data_root"] / c).is_dir()]
    rep["checks"]["data_categories"] = {"ok": not missing_cats, "missing": missing_cats}
    rep["checks"]["visa_structure"] = (check_visa_structure(paths["data_root"], manifest)
                                       if not missing_cats else {"ok": False, "issues": ["category dirs missing"]})
    w = paths["weights"]
    if w.is_file():
        h = file_sha256(w)
        rep["checks"]["weights"] = {"ok": h == manifest["model"]["weights_sha256_expected"], "sha256": h}
    else:
        rep["checks"]["weights"] = {"ok": False, "error": "weights file not found (no download attempted)"}
    info = {"python": sys.version.split()[0]}
    try:
        import torch, open_clip, sklearn
        info.update({"torch": torch.__version__, "open_clip": open_clip.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__})
        if args.device.startswith("cuda"):
            info["cuda_available"] = torch.cuda.is_available()
    except Exception as ex:
        info["import_error"] = str(ex)
    rep["checks"]["versions"] = info
    try:
        from open_clip.factory import get_model_config
        mc = get_model_config(MODEL_NAME)
        m_ok = (mc is not None and mc.get("quick_gelu") is True
                and manifest.get("model", {}).get("model_name") == MODEL_NAME
                and manifest.get("model", {}).get("quick_gelu") is True)
        rep["checks"]["model_arch"] = {"ok": bool(m_ok), "model_name": MODEL_NAME,
                                       "config_quick_gelu": None if mc is None else mc.get("quick_gelu"),
                                       "manifest_model_name": manifest.get("model", {}).get("model_name"),
                                       "manifest_quick_gelu": manifest.get("model", {}).get("quick_gelu")}
    except Exception as ex:
        rep["checks"]["model_arch"] = {"ok": False, "error": str(ex)}
    dev_ok = (not args.device.startswith("cuda")) or bool(info.get("cuda_available"))
    rep["checks"]["device"] = {"ok": dev_ok, "requested": args.device}
    out = paths["output_dir"]
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe"; probe.write_text("ok", encoding="utf-8"); probe.unlink()
        rep["checks"]["output_dir"] = {"ok": True, "path": str(out)}
    except OSError as ex:
        rep["checks"]["output_dir"] = {"ok": False, "error": str(ex)}
    try:
        du = shutil.disk_usage(out if out.exists() else out.parent)
        rep["checks"]["disk_free_gb"] = {"ok": du.free / 2 ** 30 >= 1.0, "free_gb": round(du.free / 2 ** 30, 2)}
    except OSError as ex:
        rep["checks"]["disk_free_gb"] = {"ok": False, "error": str(ex)}
    ag = avail_gb()
    rep["checks"]["ram"] = {"ok": ag >= 1.5, "avail_gb": round(ag, 2)}
    rep["checks"]["offline_env"] = {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                                    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE")}
    rep["all_ok"] = (rep["checks"]["project_root_exists"] and rep["checks"]["dependency_files"]["ok"]
                     and rep["checks"]["data_categories"]["ok"] and rep["checks"]["visa_structure"]["ok"]
                     and rep["checks"]["model_arch"]["ok"]
                     and rep["checks"]["weights"]["ok"]
                     and dev_ok and rep["checks"]["output_dir"]["ok"]
                     and rep["checks"]["disk_free_gb"]["ok"] and rep["checks"]["ram"]["ok"]
                     and "import_error" not in info)
    return rep


class SafetyStop(RuntimeError):
    pass


# ==================================================================================
# Runner
# ==================================================================================

def run(args):
    t_start = time.time()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    FROZEN_CATS = list(manifest["dataset"]["categories"])
    FROZEN_CHAINS = [c["id"] for c in manifest["chains"]]
    paths = resolve_paths(args)
    out_root = paths["output_dir"] / "smoke" if args.mode == "smoke" else paths["output_dir"]
    percat_dir = out_root / "per_category"
    log_dir = out_root / "logs"
    percat_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logf = open(log_dir / f"stage3b_v2f_quickgelu_{args.mode}_{int(time.time())}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    marker = SMOKE_MARKER if args.mode == "smoke" else FORMAL_MARKER
    log(f"START mode={args.mode} marker={marker} device={args.device} pid={os.getpid()}")

    pf = preflight(paths, manifest, args)
    (out_root / "preflight_report.json").write_text(json.dumps(pf, indent=1, ensure_ascii=False), encoding="utf-8")
    if not pf["all_ok"]:
        log("PREFLIGHT FAILED"); return 2
    log("preflight PASS")

    fingerprint = {
        "marker": marker, "device": args.device,
        "protocol_manifest_sha256": file_sha256(MANIFEST_PATH),
        "runner_sha256": file_sha256(Path(__file__)),
        "weights_sha256": file_sha256(paths["weights"]),
        "code_file_sha256": {k: file_sha256(paths["project_root"] / k) for k in CODE_HASH_KEYS},
    }

    # ---- resume / overwrite gate (BEFORE any model load) ----
    if args.mode == "formal":
        err = validate_resume_state(out_root, fingerprint)
        if err:
            log("RESUME GATE: " + err); return 5
        existing = sorted(percat_dir.glob("*.json"))
        if existing and not args.resume:
            log("output dir contains formal records; refusing silent overwrite. Use --resume or a NEW --output-dir.")
            return 5
        (out_root / "run_fingerprint.json").write_text(json.dumps(fingerprint, indent=1), encoding="utf-8")
        # per-category resume validation (listings + content hashes only; no model needed)
        resume_ok = {}
        if args.resume and existing:
            pr = str(paths["project_root"])
            if pr not in sys.path:
                sys.path.insert(0, pr)
            for cf in existing:
                cat = cf.stem
                if cat not in FROZEN_CATS:
                    log(f"resume: unknown category file {cf.name}: REFUSED"); return 5
                prev = json.loads(cf.read_text(encoding="utf-8"))
                data = load_visa_category(str(paths["data_root"]), cat)
                refs = list(data["train_good"])[: int(manifest["dataset"]["max_normal_reference"])]
                goods = list(data["test_good"])
                pairs = []
                for dt in sorted(data["test_defect"].keys()):
                    info = data["test_defect"][dt]
                    pairs += list(zip(info["images"], info["masks"]))
                sig, _ = build_data_signature_v2(refs, goods, pairs, paths["data_root"])
                err = validate_resume_category(prev, cat, FROZEN_CHAINS, sig)
                if err:
                    log("RESUME REFUSED: " + err); return 5
                resume_ok[cat] = prev
            log(f"resume validated: {sorted(resume_ok.keys())}")

    # ---- heavy imports / model ----
    import torch, torch.nn.functional as F
    import open_clip
    from PIL import Image
    pr = str(paths["project_root"])
    if pr not in sys.path:
        sys.path.insert(0, pr)
    from src.sara.vne import VisualNormalcyEncoder
    from experiments.vne_reference_baseline.run_vne_reference_baseline import NearestReferenceScorer
    from experiments.ablation.ablation_utils import (
        load_mask, upsample_score_map, topk_mean, safe_roc_auc,
        global_pixel_auroc, aupro_0_30, reference_normalizer, normalize_visual_scores, validate_metric_ranges)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=None, device=args.device)
    model.eval()
    miss, unexp = open_clip.load_checkpoint(model, str(paths["weights"]), strict=True, weights_only=True, device=args.device)
    if len(miss) or len(unexp):
        log(f"FATAL weight load mismatch missing={len(miss)} unexp={len(unexp)}"); return 2
    visual, proj = model.visual, model.visual.proj
    log("model loaded (strict, offline)")

    class PCAResidualScorer:
        def __init__(self, var):
            from sklearn.decomposition import PCA
            self.var = var; self.pca = None; self._PCA = PCA
        def fit(self, X):
            self.pca = self._PCA(n_components=self.var, svd_solver="full").fit(X.astype(np.float64))
        def score_samples(self, X):
            Z = self.pca.transform(X.astype(np.float64))
            R = self.pca.inverse_transform(Z)
            return np.sum((X.astype(np.float64) - R) ** 2, axis=1).astype(np.float32)
        def storage_bytes(self):
            return 0 if self.pca is None else int(self.pca.components_.nbytes) + int(self.pca.mean_.nbytes)

    def vne_storage_bytes(vne):
        seen, total = set(), 0
        for arr in ("weights_", "means_", "covariances_", "precisions_", "precisions_cholesky_"):
            a = getattr(vne.gmm, arr, None)
            if a is None: continue
            key = (id(a), a.shape, a.dtype.str)
            if key in seen: continue
            seen.add(key); total += int(a.nbytes)
        return total

    def encode(p):
        img = Image.open(p).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            e = visual._embeds(x); tr = visual.transformer(e); _, tokens = visual._pool(tr)
            if proj is not None:
                tokens = tokens @ proj
            tokens = F.normalize(tokens, dim=-1)
        return tokens[0].reshape(-1, tokens.shape[-1]).detach().cpu().numpy().astype(np.float32)

    chain_defs = [("vne_k1", lambda: VisualNormalcyEncoder(n_components=1, covariance_type="full")),
                  ("nearest", lambda: NearestReferenceScorer(chunk_size=1024)),
                  ("pca_residual_0p97", lambda: PCAResidualScorer(0.97)),
                  ("pca_residual_0p90", lambda: PCAResidualScorer(0.90)),
                  ("pca_residual_0p99", lambda: PCAResidualScorer(0.99))]

    def run_category(category):
        t_cat = time.time()
        data = load_visa_category(str(paths["data_root"]), category)
        max_ref = args.max_ref if args.mode == "smoke" else int(manifest["dataset"]["max_normal_reference"])
        ref_paths = list(data["train_good"])[:max_ref]
        good_paths = list(data["test_good"])
        defect_pairs = []
        for dt in sorted(data["test_defect"].keys()):
            info = data["test_defect"][dt]
            if len(info["images"]) != len(info["masks"]):
                raise SafetyStop(f"{category}/{dt}: image-mask pairing mismatch "
                                 f"({len(info['images'])} vs {len(info['masks'])})")
            defect_pairs += list(zip(info["images"], info["masks"]))
        if args.mode == "smoke":
            half = max(1, args.max_test // 2)
            good_paths = good_paths[:half]; defect_pairs = defect_pairs[: args.max_test - half]
        if not ref_paths or not (good_paths or defect_pairs):
            raise SafetyStop(f"{category}: empty ref or test set")

        sig, _payload = build_data_signature_v2(ref_paths, good_paths, defect_pairs, paths["data_root"])
        t0 = time.time(); ref_feats = np.stack([encode(str(p)) for p in ref_paths]); t_ref_enc = time.time() - t0
        t0 = time.time(); test_feats = [encode(str(p)) for p in good_paths] + [encode(str(p)) for p, _ in defect_pairs]
        t_test_enc = time.time() - t0
        ref_pool = ref_feats.reshape(-1, 512)
        gt = [np.zeros((224, 224), dtype=np.uint8) for _ in good_paths] + [load_mask(m, (224, 224)) for _, m in defect_pairs]
        labels = [0] * len(good_paths) + [1] * len(defect_pairs)
        assert ref_feats.shape[1:] == (196, 512)

        entry = {"category": category, "marker": marker, "producer": "stage3b_v2f_quickgelu",
                 "n_ref_images": len(ref_paths),
                 "n_ref_patches": int(ref_pool.shape[0]), "n_test_images": len(labels),
                 "n_test_normal": labels.count(0), "n_test_defect": labels.count(1),
                 "data_signature_v2": sig,
                 "timing_s": {"ref_encode": round(t_ref_enc, 3), "test_encode": round(t_test_enc, 3)},
                 "chains": {}}
        for cid, factory in chain_defs:
            t_ch = time.time()
            scorer = factory()
            t0 = time.time(); scorer.fit(ref_pool); t_fit = time.time() - t0
            t0 = time.time(); ref_raw = scorer.score_samples(ref_pool); t_refscore = time.time() - t0
            if not np.all(np.isfinite(ref_raw)):
                raise SafetyStop(f"{category}/{cid}: non-finite REFERENCE raw scores")
            t0 = time.time(); stats = reference_normalizer(ref_raw)
            ref_range = float(stats["max"] - stats["min"]); t_refnorm = time.time() - t0
            if ref_range <= 1e-12:
                raise SafetyStop(f"{category}/{cid}: DEGENERATE reference normalization range={ref_range:.6e} "
                                 f"(min={stats['min']:.6e}, max={stats['max']:.6e})")
            t_score = t_norm = t_up = 0.0
            down = up1 = 0
            img_scores, norm_maps = [], []
            for feat in test_feats:
                a = time.time(); raw = scorer.score_samples(feat); t_score += time.time() - a
                a = time.time(); nm = normalize_visual_scores(raw, stats); t_norm += time.time() - a
                if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(nm)):
                    raise SafetyStop(f"{category}/{cid}: non-finite test scores")
                a = time.time(); m = upsample_score_map(nm.reshape(14, 14).astype(np.float32), (224, 224)); t_up += time.time() - a
                if not np.all(np.isfinite(m)):
                    raise SafetyStop(f"{category}/{cid}: non-finite anomaly map")
                down += int(np.count_nonzero(nm < 0.0)); up1 += int(np.count_nonzero(nm > 1.0))
                norm_maps.append(m); img_scores.append(topk_mean(m, 0.01))
            t_map = t_score + t_norm + t_up
            t0 = time.time()
            metrics = {"image_auroc_top1": safe_roc_auc(labels, img_scores),
                       "pixel_auroc_global": global_pixel_auroc(gt, norm_maps),
                       "aupro_0_30": aupro_0_30(gt, norm_maps)}
            validate_metric_ranges(metrics)
            metrics_s = time.time() - t0
            storage = (vne_storage_bytes(scorer) if cid == "vne_k1" else
                       scorer.reference.nbytes if cid == "nearest" else scorer.storage_bytes())
            entry["chains"][cid] = {
                "status": "OK", "metrics": {k: float(v) for k, v in metrics.items()},
                "pca_n_components": (int(scorer.pca.n_components_) if cid.startswith("pca") else None),
                "ref_raw_range": ref_range, "degenerate_reference_range": False,
                "test_norm_beyond01": {"below0_count": down, "above1_count": up1,
                                        "note": "informational only; outside [0,1] is NOT a failure"},
                "storage_bytes": int(storage),
                "timing_s": {"fit_s": round(t_fit, 3), "ref_score_s": round(t_refscore, 3), "ref_norm_s": round(t_refnorm, 3),
                              "test_scoring_s": round(t_score, 3), "test_normalize_s": round(t_norm, 3),
                              "test_upsample_s": round(t_up, 3), "test_map_pipeline_s": round(t_map, 3),
                              "metrics_s": round(metrics_s, 3), "chain_total_s": round(time.time() - t_ch, 3),
                              "timing_status": "PRELIMINARY_DIAGNOSTIC",
                              "boundary_note": ("test_scoring_s = scorer.score_samples on cached features ONLY; "
                                                "test_map_pipeline_s = scoring+normalize+upsample (NOT scorer-only, "
                                                "NOT comparable to frozen Table 3)")}}
            log(f"{category}/{cid}: pixel={metrics['pixel_auroc_global']:.4f} image={metrics['image_auroc_top1']:.4f} "
                f"aupro={metrics['aupro_0_30']:.4f} storage={storage}B")
        entry["category_wall_s"] = round(time.time() - t_cat, 1)
        entry["avail_ram_gb_after"] = round(avail_gb(), 2)
        return entry

    # ---- run selection: subset only affects WHICH categories are computed, never the COMPLETE standard ----
    run_categories = select_run_categories(args.mode, args.category, args.categories, FROZEN_CATS)
    max_wall_h = float(args.max_wall_hours if args.max_wall_hours else manifest["resource_guard"]["max_wall_hours"])
    results, stop_report = [], None
    runner_fn = CATEGORY_RUNNER_HOOK or run_category
    try:
        for cat in run_categories:
            cat_file = percat_dir / f"{cat}.json"
            if args.mode == "formal" and args.resume and cat in resume_ok:
                results.append(resume_ok[cat]); log(f"{cat}: resumed (validated)"); continue
            elapsed_h = (time.time() - t_start) / 3600.0
            if elapsed_h > max_wall_h:
                raise SafetyStop(f"WALLCLOCK_BUDGET: elapsed {elapsed_h:.2f} h > limit {max_wall_h:.2f} h "
                                 f"(GPU budget remains USER-verified in console; no auto-billing guard)")
            if avail_gb() < 1.5:
                raise SafetyStop(f"RAM_GUARD before {cat}: available < 1.5 GB")
            entry = runner_fn(cat)
            cat_file.write_text(json.dumps(entry, indent=1), encoding="utf-8")
            results.append(entry)
    except SafetyStop as ex:
        stop_report = {"marker": marker, "safety_stop": True, "reason": str(ex),
                       "partial_records": len(results), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "exit_code": 4}
        (out_root / "SAFETY_STOP_REPORT.json").write_text(json.dumps(stop_report, indent=1), encoding="utf-8")
        log(f"SAFETY STOP (exit 4): {ex}")

    # ---- completeness vs FROZEN 12x5 ----
    comp = check_completeness(results, FROZEN_CATS, FROZEN_CHAINS)
    complete = comp["all_ok"] and stop_report is None
    comp["formal_status"] = ("COMPLETE" if (args.mode == "formal" and complete)
                             else "NON_PUBLICATION_SMOKE" if args.mode == "smoke" else "INCOMPLETE")
    (out_root / "completeness_report.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")

    if args.mode == "formal":
        suffix = "" if comp["formal_status"] == "COMPLETE" else "_PARTIAL"
        with open(out_root / f"STAGE3B_VISA_QUICKGELU_PER_CATEGORY_RESULTS{suffix}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "category", "chain", "status", "image_auroc_top1", "pixel_auroc_global",
                        "aupro_0_30", "pca_n_components", "n_ref_images", "n_ref_patches", "n_test_images",
                        "ref_raw_range", "degenerate_reference_range", "storage_bytes"])
            for e in results:
                for cid, c in e["chains"].items():
                    w.writerow(["VisA", e["category"], cid, c["status"],
                                f"{c['metrics']['image_auroc_top1']:.10f}", f"{c['metrics']['pixel_auroc_global']:.10f}",
                                f"{c['metrics']['aupro_0_30']:.10f}",
                                c.get("pca_n_components") if c.get("pca_n_components") is not None else "",
                                e["n_ref_images"], e["n_ref_patches"], e["n_test_images"],
                                f"{c['ref_raw_range']:.6e}", c["degenerate_reference_range"], c["storage_bytes"]])
        with open(out_root / f"STAGE3B_VISA_QUICKGELU_RESOURCE_BREAKDOWN{suffix}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["category", "chain", "device", "ref_encode_s", "test_encode_s", "fit_s", "ref_score_s",
                        "ref_norm_s", "test_scoring_s", "test_normalize_s", "test_upsample_s",
                        "test_map_pipeline_s", "metrics_s", "chain_total_s", "storage_bytes",
                        "timing_status", "boundary_note"])
            for e in results:
                for cid, c in e["chains"].items():
                    t = c["timing_s"]
                    w.writerow([e["category"], cid, args.device, e["timing_s"]["ref_encode"], e["timing_s"]["test_encode"],
                                t["fit_s"], t["ref_score_s"], t["ref_norm_s"], t["test_scoring_s"],
                                t["test_normalize_s"], t["test_upsample_s"], t["test_map_pipeline_s"],
                                t["metrics_s"], t["chain_total_s"], c["storage_bytes"],
                                "PRELIMINARY_DIAGNOSTIC", t.get("boundary_note", "")])

    run_manifest = {"marker": marker, "mode": args.mode, "device": args.device,
                    "formal_status": comp["formal_status"], "n_records": comp["found_valid_keys"],
                    "completeness": comp, "safety_stop": stop_report, "fingerprint": fingerprint,
                    "wall_s": round(time.time() - t_start, 1),
                    "wall_budget_h": max_wall_h,
                    "cost_note": "GPU budget is NOT auto-enforced; user must verify charges in console"}
    (out_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=1), encoding="utf-8")
    log(f"DONE records={comp['found_valid_keys']} status={comp['formal_status']} wall={run_manifest['wall_s']}s")
    if stop_report is not None:
        return 4
    return 0 if comp["formal_status"] in ("COMPLETE", "NON_PUBLICATION_SMOKE") else 1


def main():
    ap = argparse.ArgumentParser(description="Stage3B VisA runner v1 (Stage3A v2-final derived)")
    ap.add_argument("--mode", choices=["preflight", "smoke", "formal"], required=True)
    ap.add_argument("--project-root", dest="project_root", default=None)
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--output-dir", dest="output_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--category", default=None)
    ap.add_argument("--categories", default=None,
                    help="subsets are PARTIAL/DEBUG ONLY; the formal COMPLETE standard is always 12x5=60")
    ap.add_argument("--max-ref", type=int, default=10)
    ap.add_argument("--max-test", type=int, default=10)
    ap.add_argument("--max-wall-hours", dest="max_wall_hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.mode == "preflight":
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        paths = resolve_paths(args)
        rep = preflight(paths, manifest, args)
        out = paths["output_dir"]; out.mkdir(parents=True, exist_ok=True)
        (out / "preflight_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return 0 if rep["all_ok"] else 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
