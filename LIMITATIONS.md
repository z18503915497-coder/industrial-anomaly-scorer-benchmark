# Scientific and reporting limitations

1. **Four AUPRO warnings remain unresolved** for A0: VisA candle −3.01×10^-5, cashew
   +4.47×10^-5, fryum −4.98×10^-5, macaroni2 −4.56×10^-5. Image and Pixel reproduced within
   6.7×10^-8. The warnings are retained; no record was excluded.
2. **A1 raw patch scores are not retained.** A1 uses a near-degenerate self-match normalization;
   its metrics cannot be recomputed under an alternative normalization from the frozen table
   alone.
3. **B3 is not throughput.** It is CPU scorer-only latency on `min(50, n_test)` cached images,
   with normal images listed first.
4. **B5 is a fixed ten-image subset** with no confidence interval; A3 and A4 were not measured
   end-to-end.
5. **A4 uses one seed.** The random 5% bank uses seed 42 only; the reference set is deterministic
   (sorted-first 50).
6. **Historical PCA is a separate batch.** PCA-versus-A0 is same-run historical;
   PCA-versus-A2 is post hoc cross-batch. No PCA preparation, B3, or B5 measurements exist.
7. **Single backbone, resolution, and reference cap.** ViT-B-16-quickgelu, 224×224, D = 512,
   50 references.
8. **No external native-backbone detectors.** PatchCore, PaDiM, DFM, and SubspaceAD are cited but
   not included as controlled baselines.
9. **No factory data and no deployment claim.** MVTec AD and VisA are public benchmarks.
10. **No third-party reproduction.** This package is a local reconstruction of existing files.
11. **CPU verification only.** `recompute_stats.py`, `recompute_extended.py`, `make_tables.py`,
    and `make_figures.py` were run locally without a GPU. The GPU runners were not rerun. Do not
    describe the package as end-to-end GPU-reproduced.

## Package hygiene

- No dataset images, model weights, SSH keys, cloud credentials, or private personal data
  (home address, phone number, passwords) are included. The author names, affiliation,
  corresponding e-mail, and ORCID are intentionally public academic metadata.
- Project absolute paths in packaged text files were replaced with `<PROJECT_ROOT>` or
  `<WEIGHTS_DIR>`; see `SENSITIVE_SCAN.txt`.
- The package has not been uploaded to any public service.
- `SHA256SUMS.txt` covers the package files; verify it before and after any upload.
- No third-party reproduction is claimed.

The former internal release checklist and approval record are kept only in the private
project audit archive and are not part of this public package.
