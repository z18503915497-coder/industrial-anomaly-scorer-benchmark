# Changelog

## v1.2.6 — 2026-10-27 (final submission preparation)

- Updated the manuscript title in `README.md`, `CITATION.cff`, and `.zenodo.json` to the v2.0
  title: “Localization–Detection Trade-offs in Industrial Anomaly Scoring at Near-Matched
  Memory Budgets: A Controlled 27-Category Study”.
- Updated the license status from “proposed” to the authors' released file-level scope: 33
  Python files under MIT; authors' derived tables, figures, and documentation under CC BY 4.0;
  third-party materials under their original licenses; the repository as a whole is not MIT.
- No DOI or publication date is asserted. Final publication still requires the authors'
  explicit authorization.
- No result CSV, runner logic, statistical result, table, figure, or protocol threshold was
  changed.

## v1.2.5 — 2026-10-27 (submission-aligned metadata patch)

- Renamed `zenodo.json` to `.zenodo.json` for the Zenodo GitHub integration and set version
  1.2.5; no DOI or publication date is asserted.
- Corrected the two `experiments/ablation_utils.py` references in `LICENSE_CODE_SCOPE.md` to
  the real path `experiments/ablation/ablation_utils.py`.
- Added a README figures note: Figure 1 is a protocol schematic; Figures 2–5 are generated from
  the frozen result table.
- Removed the internal release checklist and the internal open-issues file from the public
  package; scientific limitations are retained in `LIMITATIONS.md`, and the removed internal
  records are kept only in the private project audit archive.
- Kept the file-level mixed-license statement: 33 Python files proposed as MIT, authors'
  derived tables/figures/documentation proposed as CC BY 4.0, repository as a whole not MIT.
- No result CSV, runner logic, statistical result, table, figure, or protocol threshold was
  changed.

## v1.2.4 — 2026-10-27 (release-metadata patch)

- Removed the pre-filled release/publication dates from `CITATION.cff` and `zenodo.json`;
  the record is explicitly unpublished and no publication date is asserted.
- Replaced the top-level MIT declaration with an accurate file-level mixed-license statement:
  33 authors' Python files proposed as MIT; authors' derived tables, figures, and
  documentation proposed as CC BY 4.0; the repository as a whole is not MIT-licensed.
- Made the README Quick start directly executable by instructing the user to run the commands
  from the repository root, without a placeholder `cd` command.
- Added `PROVENANCE_MAPPING.md` mapping original and public runners, their SHA-256 values,
  the frozen CSV linkage, and the whitelist/manifest effects.
- Removed the personal author-approval record from `LICENSE_CODE_SCOPE.md`; it is retained
  only in the private long-term audit directory.
- Added a historical-snapshot notice to the protocol records so `PILOT_NOT_RUN`, `NOT_RUN`,
  and `PENDING` labels are not read as current experiment status.
- No result CSV, runner logic, statistical result, table, figure, or protocol threshold was
  changed.

## v1.2.3 — 2026-10-26 (final-cleanup release preparation)

- Added the missing `python code/recompute_extended.py` step to the README Quick start; the
  command list now matches the four CPU commands that were actually verified.
- Replaced the internal cloud-instance identifier, internal relative directory names, and
  provider-specific budget notes in the protocol records, two historical runners, and the v5
  whitelist metadata with neutral placeholders. Only default path strings were changed; scoring
  logic, thresholds, and frozen results are unchanged.
- Removed internal approval-state narrative from public-facing documentation; release still
  requires final approval by both authors.
- Updated the README version and the release-status wording. No result CSV, runner, statistical
  result, table, figure, or protocol threshold was changed.

## v1.2.2 — 2026-10-26 (public-package technical-integrity fix)

- Restored the `experiments` package layout required by the frozen runners:
  `experiments/vne_reference_baseline/run_vne_reference_baseline.py` and
  `experiments/ablation/ablation_utils.py`, with `__init__.py` files. The frozen runners were
  not modified.
- Fixed the README v5 CLI: `--mode` accepts only `self-check` or `pilot`; `--output-dir` is
  required; `--remaining21` is required for v5 pilot; data roots are `--data-root-mvtec` and
  `--data-root-visa`. The v2–v4 runners have different CLI options.
- Ran the v5 CPU `self-check` from the public package; `all_ok=true`.
- Added `code/recompute_extended.py` and `outputs/recompute_extended/` for the post hoc
  21-class sensitivity, PCA-vs-A0 same-run comparisons, PCA-vs-A2 cross-batch comparisons,
  and descriptive float64 PCA storage. Values match the frozen manuscript.
- MIT-covered file count is now 33: added `recompute_extended.py` and the three
  `experiments` package `__init__.py` files.
- Recorded Hao Li's confirmed no-competing-interests status and conditional public-release
  consent after code-ownership/third-party-license verification. This is not final MIT
  approval and not final manuscript approval.
- No frozen CSV, statistical result, manuscript, or figure was changed.

## v1.2.1 — 2026-10-26 (public-package license correction)

- Corrected the **VisA dataset** license to **CC BY 4.0** (per the official project README) and
  distinguished it from the VisA **source code** license (Apache-2.0).
- Added `LICENSE_CODE_SCOPE.md`: MIT applies only to the 29 files marked MIT-covered; it is not
  a blanket license for every file in the repository.
- Removed from the public copy: `src/metrics/metrics.py` (simplified AUPRO implementation with
  an unresolved anomalib reference), the unused `src/anomaly` directory (including an internal
  backup), and manuscript-generation scripts not needed for CPU reproduction. These remain in
  the long-term project archive.
- Clarified that author names, affiliation, corresponding e-mail, and ORCID are intentionally
  public academic metadata; the package contains no private personal data such as home address,
  phone number, passwords, or credentials.
- Flagged the manuscript v1.2 Data availability “at the time of submission” wording as a
  pre-submission placeholder to be replaced after a real repository/DOI exists.
- No frozen CSV, statistical result, manuscript, or figure was changed.

## v1.2 — 2026-10-26

- Added the two-author metadata: Ruipeng Zhi (first and corresponding author) and Hao Li,
  Tomsk Polytechnic University, Tomsk, Russia; corresponding e-mail and ORCID for Ruipeng Zhi.
- Added the Author Contributions statement based on confirmed contributions: Hao Li wrote the
  experimental code and co-analyzed the data; both authors contributed to formal analysis and
  interpretation; final approval remains pending.
- Added the confirmed no-external-funding statement.
- Marked the competing-interest declaration as pending Hao Li's final confirmation.
- Corrected "four cost quantities" to "five cost quantities" in the main text and cover letter.
- Updated the data-release statement to reflect the authors' agreement to release the package
  after final approval, without asserting a public URL or DOI.
- Prepared the public-release package: fixed relative paths, added CPU requirements and license
  files, added CITATION.cff and Zenodo metadata, and ran the CPU recomputation, table, and figure
  scripts locally.
- No scientific results, statistical tests, methods, or frozen CSV values were changed.

## v1.1 — 2026-10-26

- Corrected Supplementary Table S6 provenance: PCA-versus-A0 is same-run historical;
  PCA-versus-A2 is post hoc cross-batch.
- Fixed Table 4 category wrapping, Figure 1 right-margin clipping, and Table S8 gate-policy
  wrapping.
- Added the Online Resource 1 supplement PDF with author confirmation placeholders.
- Corrected the data-release and AI-disclosure statements to avoid asserting uncompleted author
  verification or an unapproved public repository.

## v1.0 — 2026-10-26

- Initial MVA submission candidate: 27 categories, five scorers, 135 records, numbered
  references, 8 main tables, 5 figures, and supplementary tables S1–S8.
