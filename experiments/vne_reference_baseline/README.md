# Stage 6 VNE Strong Visual-Reference Baseline

This experiment tests whether VNE K=1 distribution modeling is stronger than a
direct nearest-reference visual baseline using the same target `train/good`
patch pool.

Methods:

- `nearest_reference`: anomaly score is
  `1 - max cosine similarity(test_patch, reference_patches)`. This is a
  nearest-reference visual baseline, not PatchCore.
- `vne_k1`: current `src/sara/vne.py::VisualNormalcyEncoder` with
  `n_components=1`.

Fairness:

- each category builds the target `train/good` reference pool once;
- both methods share exactly the same reference image set and patch features;
- reference image limit is fixed by config and is not category-tuned;
- the same patch processing, heatmap upsampling, reference normalization, image
  AUROC, global pixel AUROC, and AUPRO 0-0.30 are used;
- target test images, defect images, and GT masks are not used for modeling or
  tuning.

Protocol:

- MVTec AD 15 categories
- OpenCLIP `ViT-B-16`, `pretrained=openai`
- image size `224x224`
- seed `42`
- protocol name: `LOCO auxiliary-category SP training + target-normal visual reference`
- this is not a no-reference zero-shot setting because target-normal visual
  reference is used

Leather dry-run:

```bash
python experiments/vne_reference_baseline/run_vne_reference_baseline.py --config experiments/vne_reference_baseline/vne_reference_baseline.yaml --categories leather --output-dir results/vne_reference_baseline/stage6_mvtec15_seed42 --dry-run
```

Leather GPU precheck:

```bash
python experiments/vne_reference_baseline/run_vne_reference_baseline.py --config experiments/vne_reference_baseline/vne_reference_baseline.yaml --categories leather --output-dir results/vne_reference_baseline/stage6_mvtec15_seed42 --device cuda:0 --resume
```

15-category GPU run:

```bash
python experiments/vne_reference_baseline/run_vne_reference_baseline.py --config experiments/vne_reference_baseline/vne_reference_baseline.yaml --categories all --output-dir results/vne_reference_baseline/stage6_mvtec15_seed42 --device cuda:0 --resume
```

Outputs:

- `results/vne_reference_baseline/stage6_mvtec15_seed42/per_category_metrics.csv`
- `results/vne_reference_baseline/stage6_mvtec15_seed42/aggregate_metrics.json`
- `results/vne_reference_baseline/stage6_mvtec15_seed42/deltas_vs_vne_k1.csv`
- `results/vne_reference_baseline/stage6_mvtec15_seed42/failures.json`
- `results/vne_reference_baseline/stage6_mvtec15_seed42/run_manifest.json`
- `results/vne_reference_baseline/stage6_mvtec15_seed42/resolved_config.yaml`

Nearest-reference similarity is computed in fixed test-patch chunks to control
memory use. The chunk size is an engineering parameter only and must not be
category-tuned.
