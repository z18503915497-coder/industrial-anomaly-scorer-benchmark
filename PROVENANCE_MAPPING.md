# Original-to-public runner provenance mapping

This file maps each runner in the public package to the original frozen runner and records
both SHA-256 values. It is provided so that the public copy can be audited without access to
the private long-term archive.

No private absolute paths, cloud credentials, or personal data are included in this mapping.

## Runners

| Public file | Original frozen file (project archive) | Original SHA-256 | Public SHA-256 | Change |
|---|---|---|---|---|
| `runners/pilot_experiment_ab_v2.py` | `pilot_runner_v2_20261025/pilot_experiment_ab_v2.py` | `6d26cc758a9982ce5a621d778c6b6322d98d444cd8655a7795f6f5240ea6bfdf` | `20f4e9d47148e9bd14c887b3bb7fee899bd784b247b5cabab3be4467331b5a85` | Default `--project-root` changed from a private absolute path to `<PROJECT_ROOT>` (2 lines). |
| `runners/pilot_experiment_ab_v3.py` | `pilot_runner_v3_20261025/pilot_experiment_ab_v3.py` | `f48e27d7401e81f492c1d0770f713faf72880ed75c2da94bb3ee74e2da7c3596` | `eaa1407164cb1bf632fe9c415d06109c2778df506552ca59fc0c709060faed71` | Default `--project-root` changed to `<PROJECT_ROOT>` (2 lines). |
| `runners/pilot_experiment_ab_v4.py` | `pilot_runner_v4_20261025/pilot_experiment_ab_v4.py` | `89e1d5ebf5c92a00aa3c54dfcb45447af3503928e91e1618f0a613ac16ec6edb` | `1fc41460a5faa86bd3f90bedbf3ac5054b03c6a95abeb414602326365542ab25` | Default `--project-root` changed to `<PROJECT_ROOT>` (2 lines). |
| `runners/pilot_experiment_ab_v5.py` | `pilot_runner_v5_20261025/pilot_experiment_ab_v5.py` | `1b5dd5103601b2c1198c9c8d962fc2eef3279a0563e548a1bfe4c7aa555b7a0e` | `bed8313c4f64e53606943f6f529d0acdd8b19a615326f6d80aafae518fe5d676` | Default `--project-root` changed to `<PROJECT_ROOT>` (2 lines). |
| `runners/stage3a_mvtec_runner_v3f_quickgelu.py` | Historical QuickGELU correction batch (stage3a) | `61b643809fa23deccb98aa9acf9ff645661825cf325e7dba61b0d8cd2b19ad98` | `543f9a37d7a385b4aaaecda1af95ac034a4f8c7c6ba87fa12528b460441e79f8` | Default weights directory replaced by `<INTERNAL_ASSETS_DIR>`; provider-specific budget prompt generalized (6 lines). |
| `runners/stage3b_visa_runner_v2f_quickgelu.py` | Historical QuickGELU correction batch (stage3b) | `f0bca59fd1b81fc988a915141ab61e52e8a77887f9d62e1202338ca9710aab7b` | `c0dd8633728ddb253db2960c75766c87daa89c75f3eabb01192bb7267fa53354` | Default weights directory replaced by `<INTERNAL_ASSETS_DIR>`; provider-specific budget prompt generalized (6 lines). |

In every case the change is limited to default path strings and platform-specific prompt
text. Scoring, thresholds, normalization, statistics, gate policy, and output-field logic are
unchanged.

## Relationship to the frozen 135-record CSV

`data/EXPERIMENT_AB_MERGED_135_RESULTS.csv` records the original runner hashes:

| Batch | Runner version | Frozen CSV `source_runner_sha256` | Original runner file |
|---|---|---|---|
| `v3_mvtec15_20261025` | `v3-tolerance-20261025` | `f48e27d7401e81f492c1d0770f713faf72880ed75c2da94bb3ee74e2da7c3596` | v3 (matches) |
| `visa_three_category_completion` | `v4-aupro-warning-20261025` | `89e1d5ebf5c92a00aa3c54dfcb45447af3503928e91e1618f0a613ac16ec6edb` | v4 (matches) |
| `remaining_21_ab_extension` | `v5-remaining21-20261025` | `1b5dd5103601b2c1198c9c8d962fc2eef3279a0563e548a1bfe4c7aa555b7a0e` | v5 (matches) |

`data/tableS8_batch_provenance.csv` records the matching SHA-256 prefixes
(`f48e27d7401e81f4`, `89e1d5ebf5c92a00`, `1b5dd5103601b2c1`).

The frozen CSV was not modified. Because the public runners differ from the originals in
default path strings, a public GPU run would write a different `source_runner_sha256` into its
own output. That is a provenance difference, not a result change.

## Whitelist and manifest effects

| File | Original SHA-256 | Public SHA-256 | Effect |
|---|---|---|---|
| `runners/v5_remaining21_whitelist.json` | `291907f6f5c37ddf9d80d13ba2f583348fd511215eeb1978ad71b61e6ebb3e45` | `1cfc917b24531e4dfc2cd602928434ca6a92960b9b208f0855f9d3a08c67c5a5` | 27 `targets_source/*/path` prefixes replaced by `<INTERNAL_RESULTS_DIR>`. The v5 runner does not read the `path` field; it reads the category lists and `targets_sha256`. |
| `runners/pilot_reference_targets_full27.json` | `a52c52528156f02108466790983d98305b48f1558ca3a810e90590b051376057` | `a52c52528156f02108466790983d98305b48f1558ca3a810e90590b051376057` | Unchanged. The whitelist `targets_sha256` matches this file. |
| `protocols/STAGE3A_V3F_QUICKGELU_PROTOCOL_MANIFEST.json` | `9e6dacd67914f77180b1285695837361bd7274da1e8ea83ff2deedd5ef96f514` | `2b945b7ca5ff4b0abc83a08fa92be6c8d7ece6f9442fe6311d7a02c6caffb77e` | Weights path placeholder, provider-neutral budget note, `autodl_linux` key renamed to `gpu_linux`, private Windows example path replaced. |
| `protocols/STAGE3B_V2F_QUICKGELU_PROTOCOL_MANIFEST.json` | `d2f6daebc6a5da008a046dc4d3d07634c29503bf77b6fd07a3b077b95f3ce88e` | `824d37a3be3919ddbf204c80d806a1f4410c5e11537309dcc7089a5489644599` | Weights path placeholder and provider-neutral budget note. |
| `data/EXPERIMENT_AB_MERGED_135_RESULTS.csv` | `1eb55b02bcabc8cad7a641e2918971a9b79b0bd2bb139ab254abd5079a97924f` | `1eb55b02bcabc8cad7a641e2918971a9b79b0bd2bb139ab254abd5079a97924f` | Unchanged. |

The public whitelist SHA differs from the original, so a public run would record a different
`whitelist_sha256`. The `targets_sha256` value is unchanged and verifies against the public
targets file.
