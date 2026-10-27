# File-level license scope

This file records the file-level license scope released by the authors. It is an audit
record, not a legal opinion. The repository as a whole is not MIT-licensed.

## MIT-covered code (authors' original implementation)

The following 33 Python files are released under MIT. None contains an explicit
third-party copyright or license header, and none contains a marked copy of an upstream code
file. Third-party libraries are imported at runtime, not vendored.

| Directory | Files | Notes |
|---|---|---|
| `code/` | `recompute_stats.py`, `recompute_extended.py`, `make_tables.py`, `make_figures.py`, `normalization_audit.py`, `verify_references.py`, `merge_v3_v4_v5_audit.py` | Statistics, extended 21-class/PCA recomputation, tables, figures, audits |
| `runners/` | `pilot_experiment_ab_v2.py`, `pilot_experiment_ab_v3.py`, `pilot_experiment_ab_v4.py`, `pilot_experiment_ab_v5.py`, `stage3a_mvtec_runner_v3f_quickgelu.py`, `stage3b_visa_runner_v2f_quickgelu.py` | A/B and historical runners |
| `experiments/` | `__init__.py`, `ablation/__init__.py`, `ablation/ablation_utils.py`, `vne_reference_baseline/__init__.py`, `vne_reference_baseline/run_vne_reference_baseline.py` | Metrics, nearest-reference baseline, and the package layout required by the frozen runners |
| `src/` | `__init__.py`, `config.py`, `dataset.py`, `models/__init__.py`, `models/clip_wrapper.py`, `sara/__init__.py`, `sara/fusion.py`, `sara/pipeline.py`, `sara/refinement.py`, `sara/sp.py`, `sara/trainer.py`, `sara/vne.py`, `utils/__init__.py`, `utils/leakage_check.py`, `utils/utils.py` | Numerical components used by the runners |

Implementation notes:

- `experiments/ablation/ablation_utils.py` implements standard pixel AUROC and AUPRO-style metrics.
  The algorithm is described in the cited literature; the file is the authors' implementation.
- `src/sara/vne.py` implements a Gaussian-mixture normalcy encoder using scikit-learn.
- `src/models/clip_wrapper.py` wraps OpenCLIP; OpenCLIP remains under its own license.
- `runners/` and `code/` contain project-specific orchestration, statistics, and plotting code.

## Excluded from the public package (not MIT-covered here)

These files were present in the original 75-file package but are not part of the public
reproduction copy:

| File | Reason |
|---|---|
| `src/metrics/metrics.py` | Contains a simplified AUPRO implementation with a comment pointing to anomalib. Its provenance is unresolved; it is **not** covered by the MIT grant. It is not used by the frozen 135-record evaluator, which uses `experiments/ablation/ablation_utils.aupro_0_30`. |
| `src/metrics/__init__.py` | Empty package init; removed with the directory. |
| `src/anomaly/localization.py` | WinCLIP-style module not used by the frozen runners; removed to keep the public package focused (no known license conflict). |
| `src/anomaly/localization_backup_v1.py` | Internal backup duplicate; removed. |
| `src/anomaly/__init__.py` | Empty package init; removed with the directory. |
| `code/build_docx.py` | Manuscript-generation script; not needed for CPU reproduction; retained in the long-term project archive. |
| `code/build_mva_docx.py` | Manuscript-generation script; not needed for CPU reproduction; retained in the long-term project archive. |
| `code/build_supplementary_docx.py` | Manuscript-generation script; not needed for CPU reproduction; retained in the long-term project archive. |
| `code/prepare_mva_markdown.py` | Historical manuscript-preparation script containing outdated draft text; not needed for CPU reproduction; retained in the long-term project archive. |

The original 75-file package remains in the long-term project directory for audit. It is not
the public release copy.

## Data, figures, tables, and documentation

These are released under CC BY 4.0 (authors' derived results and documentation):

- `data/EXPERIMENT_AB_MERGED_135_RESULTS.csv`
- `data/STAGE3A_MVTEC_QUICKGELU_PER_CATEGORY_RESULTS.csv`
- `data/STAGE3B_VISA_QUICKGELU_PER_CATEGORY_RESULTS.csv`
- `data/table*.csv`
- `tables/*.csv`, `tables/derived_ratios.json`
- `figures/*.png`
- `outputs/recompute/*`
- `README.md`, `datasets.md`, `LIMITATIONS.md`, `CHANGELOG.md`, `CITATION.cff`,
  `.zenodo.json`, `PROVENANCE_MAPPING.md`, `requirements-*.txt`, `LICENSE_CODE_SCOPE.md`

`LICENSE_CODE`, `LICENSE_DATA`, `SENSITIVE_SCAN.txt`, and `SHA256SUMS.txt` are license/metadata
notices; they are not offered as a separate software grant.

## Not covered by any author license

- MVTec AD images: CC BY-NC-SA 4.0; not included.
- VisA dataset: CC BY 4.0; not included. VisA source code is Apache-2.0; not included.
- OpenCLIP / OpenAI CLIP weights: not included; governed by the original model license.
- Third-party Python libraries: not included; governed by their own licenses.
