# Reproducibility package — MVA manuscript (v1.2.6 final submission preparation)

Manuscript: **Localization–Detection Trade-offs in Industrial Anomaly Scoring at Near-Matched Memory Budgets: A Controlled 27-Category Study**

Authors: **Ruipeng Zhi and Hao Li**, Tomsk Polytechnic University, Tomsk, Russia.
Corresponding author: Ruipeng Zhi (cz08@tpu.ru; ORCID 0009-0002-6065-4930).

Status: **release preparation only.** This directory is not yet uploaded to GitHub or Zenodo,
and no public URL or DOI is assigned. Public release requires final approval by both authors.

The package contains intentionally public academic author metadata (names, affiliation,
corresponding e-mail, ORCID). It does not contain private personal data such as home address,
phone number, passwords, or credentials.

## Historical protocol records

The files under `protocols/` are historical snapshots written before the full 27-category
experiment. Status labels such as `PILOT_NOT_RUN`, `NOT_RUN`, or `PENDING` describe the state
at the time of writing and are retained unchanged for auditability. The full 27-category
experiment was subsequently completed; `data/EXPERIMENT_AB_MERGED_135_RESULTS.csv` is the final
frozen record.

## Figures

`figures/figure1_pipeline.png` is a schematic of the experimental protocol, not a data plot.
`figures/figure2_aupro_differences.png` through `figures/figure5_cost_profile.png` were
generated from the frozen result table by `code/make_figures.py`.

## What this package supports

- Recomputing the primary and secondary statistics from the frozen 135-record CSV.
- Recomputing the post hoc 21-class sensitivity, the historical PCA comparisons
  (same-run vs A0; cross-batch vs A2), and descriptive PCA float64 storage.
- Regenerating the main accuracy, storage, and timing tables.
- Regenerating the five manuscript figures.
- Auditing the frozen result record, batch provenance, and AUPRO warning history.

It does **not** include MVTec AD or VisA images, pretrained weights, or a GPU environment.
The GPU runners are included for transparency, but the GPU experiments were not rerun here.

## Verified locally (CPU only)

The following commands were actually run from this directory before release preparation:

```
python code/recompute_stats.py
python code/recompute_extended.py
python code/make_tables.py
python code/make_figures.py
```

- `recompute_stats.py` reproduced the formal primary A0−A2 AUPRO statistics from the frozen
  CSV: MVTec mean +0.060369612, Holm p = 0.0006103516; VisA mean +0.041830593, Holm p =
  0.01220703125. It also reproduced the 15/27 Image AUROC regressions, pooled mean
  −0.010518514, and the storage values.
- `recompute_extended.py` reproduced the post hoc 21-class sensitivity (MVTec
  +0.055858661, Holm 0.0048828125; VisA +0.032750442, Holm 0.07421875), the historical
  PCA-vs-A0 same-run comparisons, the PCA-vs-A2 cross-batch comparisons, and the descriptive
  float64 PCA storage means. It labels every comparison as formal primary, post hoc,
  same-run historical, or cross-batch; none is presented as an equivalence test.
- `make_tables.py` regenerated Tables 1–6 and A1–A3. Seven generated tables are byte-identical
  to the frozen table CSVs; Tables 5 and 6 differ only in shortened header labels introduced
  for typesetting, with identical values.
- `make_figures.py` regenerated the five PNG figures from the frozen CSV.

No GPU run, no dataset download, no public upload, and no third-party reproduction was
performed. The package must not be described as end-to-end GPU-reproduced.

## Layout

```
data/
  EXPERIMENT_AB_MERGED_135_RESULTS.csv          frozen 135-record A/B table
  STAGE3A_MVTEC_QUICKGELU_PER_CATEGORY_RESULTS.csv
  STAGE3B_VISA_QUICKGELU_PER_CATEGORY_RESULTS.csv
  table1..table8, tableA1..A3, tableB1..B3, tableS7, tableS8
code/
  recompute_stats.py, recompute_extended.py, make_tables.py, make_figures.py
  normalization_audit.py, verify_references.py, merge_v3_v4_v5_audit.py
runners/
  pilot_experiment_ab_v2.py .. pilot_experiment_ab_v5.py
  stage3a_mvtec_runner_v3f_quickgelu.py, stage3b_visa_runner_v2f_quickgelu.py
  pilot_reference_targets_full27.json, pilot_reference_targets.json,
  v5_remaining21_whitelist.json
protocols/
  STAGE3A_V3F_QUICKGELU_PROTOCOL_MANIFEST.json,
  STAGE3B_V2F_QUICKGELU_PROTOCOL_MANIFEST.json,
  V5_STATISTICAL_ANALYSIS_PLAN_zh.md,
  EXPERIMENT_AB_V4_AUPRO_WARNING_REVISION_AND_TEST_AUDIT_zh.md,
  EXPERIMENT_A_CORESET_PROTOCOL_AND_TEST_AUDIT_zh.md,
  EXPERIMENT_B_CONTROLLED_TIMING_PROTOCOL_zh.md,
  EXPERIMENT_AB_PILOT_REVIEW_AND_FULL27_GATE_zh.md
src/                 frozen numerical components used by the runners
experiments/         nearest-reference and ablation utilities used by the runners
outputs/recompute/   RECOMPUTED_STATS.json and summary from the actual local run
tables/              regenerated tables from the actual local run
figures/             regenerated figures from the actual local run
```

Manuscript-generation scripts are retained in the long-term project archive and are not part
of this public reproduction package. The package is curated for statistics, tables, figures,
and provenance review.

## Quick start (CPU recomputation)

Run the following commands from the repository root:

```
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-cpu.txt

python code/recompute_stats.py
python code/recompute_extended.py
python code/make_tables.py
python code/make_figures.py
```

Expected outputs:

- `outputs/recompute/RECOMPUTED_STATS.json`
- `outputs/recompute/RECOMPUTED_STATS_SUMMARY.txt`
- `outputs/recompute_extended/EXTENDED_STATS.json`
- `outputs/recompute_extended/EXTENDED_STATS.txt`
- `tables/table1_aggregate_accuracy.csv` .. `tables/tableA3_pixel.csv`
- `figures/figure1_pipeline.png` .. `figures/figure5_cost_profile.png`

To verify the package inventory (Linux/macOS):

```
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell, compare the hashes in `SHA256SUMS.txt` with `Get-FileHash`.

## CPU self-check (v5 runner)

The v5 runner's `--mode` accepts only `self-check` or `pilot`; `--output-dir` is required.
The following CPU self-check was actually run from this package and returned `all_ok=true`:

```
python runners/pilot_experiment_ab_v5.py --mode self-check \
  --project-root . --output-dir outputs/selfcheck_v5
```

It exercises the GMM storage/score path, k-center and random selection, normalization,
AUROC/AUPRO helpers, budget guard, atomic-save behavior, and the AUPRO warning policy.
It does not need datasets or weights. The local self-check used scikit-learn 1.8.0; the frozen
GPU records used 1.9.0. This is a CPU compatibility check, not a GPU reproduction.

## GPU reproduction (not run here)

The A/B runners require Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
open-clip-torch 3.3.0, scikit-learn 1.9.0, NumPy 2.3.2, Pillow, and an NVIDIA GPU.
The recorded runs used an RTX 4090 (24 GB). The v5 pilot command is:

```
python runners/pilot_experiment_ab_v5.py --mode pilot --remaining21 \
  --project-root . \
  --data-root-mvtec <MVTEC_ROOT> \
  --data-root-visa <VISA_ROOT> \
  --weights <WEIGHTS>/open_clip_model.safetensors \
  --output-dir outputs/pilot_v5 --device cuda:0
```

`--project-root` is required in practice because its default is the literal
`<PROJECT_ROOT>`; pass `.` when running from this package. `--remaining21` is required for
v5 pilot mode, and `--allow-cpu-pilot` is only for debugging. The v2–v5 runners share
`self-check`/`pilot`; the two historical stage3a/stage3b runners use
`preflight`/`smoke`/`formal`. Use `python runners/<runner>.py --help` for the copied version. Do not overwrite
the frozen CSV. The A0/A1 reproduction gate runs before B3 timing; AUPRO warnings in
(3e-5, 1e-3] are recorded but do not block. See `protocols/` and `LIMITATIONS.md`.

**GPU end-to-end reproduction was not run in this package.** The CPU self-check and static
import tests do not substitute for a GPU run; do not describe them as GPU validation.

## Datasets and weights

- MVTec AD: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
  (the dataset is released under CC BY-NC-SA 4.0)
- VisA: https://github.com/amazon-science/spot-diff
  (the **VisA dataset** is released under CC BY 4.0, as stated in the project README;
  the project's **source code** is separately Apache-2.0. The dataset license is not the
  source-code license.)
- OpenCLIP weight SHA-256: `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`

Dataset images and the weight file are **not** included. Obtain them from their official
providers and follow the original licenses. Do not redistribute them from this package.

## Licenses

The authors release the package under file-level mixed licensing. The repository as a
whole is **not** licensed under MIT.

- **Code**: MIT — only for the 33 Python files identified as MIT-covered in
  `LICENSE_CODE_SCOPE.md`; see also `LICENSE_CODE`.
- **Derived result tables, figures, and documentation**: CC BY 4.0 — see `LICENSE_DATA`.
- **Raw datasets and pretrained weights**: not included; governed by their original licenses.
  MVTec AD is CC BY-NC-SA 4.0; the VisA dataset is CC BY 4.0; VisA source code is Apache-2.0.

The MIT license covers only the 33 listed Python files and does not automatically cover
every file in the repository. The file-level scope
is recorded in `LICENSE_CODE_SCOPE.md`. Files with unresolved third-party provenance and
internal backups were removed from this public package; the original project copies remain in
the long-term project archive. Third-party libraries imported at runtime (PyTorch, OpenCLIP,
scikit-learn, NumPy, SciPy, pandas, matplotlib, python-docx) remain under their own licenses.
No license is asserted for third-party datasets or weights.

## Provenance

`PROVENANCE_MAPPING.md` maps each public runner to its original frozen runner, records both
SHA-256 values, and explains how the frozen CSV and protocol metadata are affected by path
sanitization.

## Citation

`CITATION.cff` and `.zenodo.json` are release-preparation metadata. `.zenodo.json` is the
hidden file used by the Zenodo GitHub integration; it is intentionally not named `zenodo.json`.
No DOI has been assigned and no publication date is asserted. Do not add a DOI or repository
URL to the manuscript until a real archive has been created and the identifier is confirmed.

The manuscript Data availability statement describes the package as not publicly archived at
the time of submission. Update it to the real repository/DOI status only after the archive
exists and both authors approve; do not present a placeholder as a final public URL or DOI.

## Integrity and limitations

- The frozen 135-record CSV is authoritative. Re-running the GPU pipeline may produce small
  numerical differences; the four documented AUPRO warnings are the residual example.
- B3 is CPU scorer-only latency on a fixed load, not throughput; B5 is a ten-image fixed subset
  with no confidence interval; A3 and A4 were not measured end-to-end.
- A1 uses a near-degenerate self-match normalization and raw A1 patch scores are not retained.
- The historical PCA batch is separate from the 135-record freeze; PCA-versus-A0 is same-run
  historical, PCA-versus-A2 is cross-batch.
- No external native-backbone baselines are included.
- No third-party reproduction is claimed.
