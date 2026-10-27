# 实验 A/B v4：AUPRO WARNING 策略与 VisA-only 补全版本

- 日期：2026-10-25
- 版本：`v4-aupro-warning-20261025`
- 状态：**本地修订与 CPU 测试完成；GPU 未运行**
- 性质声明：**这是第二次 AUPRO 复现 Gate 失败之后的透明协议修订，不是最初预注册参数。**
- 1e-3 是实用重大偏差检查界限，**不是经过验证的数值误差上限**。
- 未修改 AUPRO 算法、冻结 targets、135 条 QuickGELU 结果、编码器、权重、数据、评分器或评价函数。

---

## 1. 修订背景

第一次 GPU pilot（v2）在 MVTec:bottle 的 A0 Gate 因 AUPRO 差 `4.333136e-06` 超过 1e-6 停止。
第二次 GPU pilot（v3）在 VisA:candle 的 A0 Gate 因 AUPRO 差 `3.010044e-05` 超过 3e-5 停止：

| 指标 | 本次值 | 冻结目标 | 差值 | v3 容差 | 结果 |
|---|---:|---:|---:|---:|---|
| Image AUROC | 0.902400000000 | 0.902400000000 | +0.000000e+00 | 1e-6 | PASS |
| Pixel AUROC | 0.931818373941 | 0.931818391300 | -1.735943e-08 | 1e-6 | PASS |
| AUPRO@0.30 | 0.869761398156 | 0.869791498600 | **-3.010044e-05** | 3e-5 | FAIL |

Image/Pixel 几乎逐位复现；AUPRO 仅超出 v3 容差约 `1.00044e-07`。
**不宣称已证明差异来自 scipy、numpy、CUDA 或任何特定来源。**

## 2. v4 AUPRO 判据（唯一修改）

| 指标 | 规则 | 行为 |
|---|---|---|
| Image AUROC | `abs(delta) <= 1e-6` | 超过立即停止 |
| Pixel AUROC | `abs(delta) <= 1e-6` | 超过立即停止 |
| AUPRO | `abs(delta) <= 3e-5` | PASS |
| AUPRO | `3e-5 < abs(delta) <= 1e-3` | `AUPRO_REPRO_WARNING`，记录原值/目标/差值/版本，**不因这一项单独停止** |
| AUPRO | `abs(delta) > 1e-3` | 重大偏差，立即停止并保存证据 |

强制规则：

1. 所有指标必须有限且在 `[0,1]`；非法值立即阻断。
2. 每项独立记录：原值、目标值、差值、绝对差值、容差、valid、status、warning、blocking、code。
3. 不允许任何指标平均抵消；Image/Pixel 失败不会被 AUPRO warning 抵消。
4. AUPRO warning 写入 `AUPRO_REPRO_WARNING.json` 与 `pilot_summary.json["aupro_repro_warnings"]`。
5. warning 不阻断 Gate；最终批次仍可为 `COMPLETE_SUBSET`，但元数据记录 warning 数量与策略版本。
6. 架构、权重、数据、方法、评分或评价函数的实际不一致仍必须停止。

## 3. VisA-only 补全批次

固定 VisA 三类：`candle`、`pcb3`、`macaroni1`。

> 说明：任务文本中“固定类别：VisA / VisA / VisA”未写出具体类别名。本实现按已冻结的六类协议使用 `candle/pcb3/macaroni1`。若这不是预期类别，必须在 GPU 前纠正；v4 不会运行 MVTec。

- v4 pilot **必须**传 `--visa-only`；否则 preflight 拒绝，防止重跑已接受的 MVTec 15 条记录。
- `expected_units = 15`（3 类 × 5 方法）。
- 正常结束状态：`COMPLETE_SUBSET`，不是单次六类 `COMPLETE`。
- 元数据：
  - `batch.name = visa_three_category_completion`
  - `batch.not_single_six_category_complete = true`
  - `batch.does_not_include_mvtec = true`
  - `batch.mvtec_source_batch = ARCHIVE_V3_MVTEC15_20261025 (15 accepted records; not rerun)`
  - `batch.expected_units = 15`
- v4 CSV 每条记录新增 provenance 字段：
  - `runner_version`
  - `runner_sha256`
  - `batch_name`
  - `record_time_utc`
  - `reproduction_policy_id`
  - `aupro_repro_warning`
- 原 A0/A1、B3 3+10、B4、B5、存储口径、60 分钟双预算、增量原子保存均不变。
- B3 仍使用预注册 50 图固定负载；VisA 200/201 图只覆盖约 25%，不能当作完整测试集吞吐量。

## 4. 测试与审计结果

完整输出：`V4_TEST_RESULTS_v4_20261025.txt`。

| 检查 | 结果 |
|---|---|
| `py_compile`（runner/tests/mock） | PASS |
| v4 CPU 单元测试 | **81/81 PASS**（`UNIT_EXIT=0`） |
| v4 mock 端到端冒烟 | **45/45 PASS**（`MOCK_EXIT=0`） |
| v4 self-check（本地 sklearn 1.8.0） | **all_ok = true** |
| GMM 存储 | 3,147,780 / 1,050,628 B；float32；去冗余 max abs diff 0.0 |

指定测试覆盖：

| 要求 | 测试 | 结果 |
|---|---|---|
| candle 实际 3.010044e-5 → WARNING | `T12.candle_aupro_3p010e5_warning`；mock `SMOKE.warning.*` | PASS |
| Image >1e-6 阻断 | `T12.image_over_1e6_fails`；mock `SMOKE.image_mismatch.*` | PASS |
| Pixel >1e-6 阻断 | `T12.pixel_over_1e6_fails` | PASS |
| AUPRO >1e-3 阻断 | `T12.aupro_2e3_fails`；mock `SMOKE.aupro_major.*` | PASS |
| AUPRO 4e-5 / 9e-4 WARNING | `T12.aupro_4e5_warning`、`T12.aupro_9e4_warning` | PASS |
| WARNING 不阻断，但不可抵消 Image 失败 | mock `SMOKE.warning.*`、`T12.warning_does_not_offset_image` | PASS |
| 非法指标阻断 | `T12.invalid_metric_blocks` | PASS |
| VisA-only 15/15、COMPLETE_SUBSET、批次元数据 | mock `SMOKE.match.*` | PASS |
| 缺 `--visa-only` 拒绝，防 MVTec 重跑 | mock `SMOKE.no_visa.*` | PASS |
| 显式请求 MVTec 类别仍被拒绝 | mock `SMOKE.mvtec_request.*` | PASS |
| COMPLETE 不得由部分触发 | `T8.incomplete_complete_refused`、`T8.incomplete_complete_subset_refused` | PASS |

mock warning 路径产生 2 条 warning（A0/A1 candle），15/15 行完成，B3 调用 15 次，`AUPRO_REPRO_WARNING.json` 存在且记录完整精度。

## 5. MVTec 15 条永久归档

已创建 `manuscript_rescue_experiments/ARCHIVE_V3_MVTEC15_20261025/`：

- 15 行正式 CSV、完整 summary（含 10 次 B3 原始计时）、运行日志、云 self-check、复现失败/中止证据、部分审计/决策报告。
- v3 runner、targets、wrapper、v3 容差协议副本。
- `V3_MVTEC15_ARCHIVE_MANIFEST.json`（12 个文件 SHA256；manifest SHA256 `9538c2f00cc333c6a31aed55fe39bb2a7061fccb62422a9dfc9615dc44feae5a`）。
- **未覆盖、未修改、未重新生成 MVTec 15 条科研结果。**

## 6. 后续 30 条合并协议（仅 VisA 15 条完成后）

1. 只有 VisA 15 条有效记录完成后，才允许与 MVTec 15 条合并。
2. 每条记录保留：源 runner 版本、SHA256、来源批次、实验时间、Gate 策略。
   - v4 CSV 已含这些字段；v3 MVTec CSV 无逐行字段，合并时从 v3 summary/永久归档补充 provenance，**不得修改源 CSV 数值**。
3. 核验 30 个唯一类别—方法组合，无缺失、无重复。
4. 保留两批不同 Gate 规则的真实记录（v3 MVTec：3e-5 硬规则；v4 VisA：3e-5 WARNING / 1e-3 硬限）。
5. 综合状态标记为 **`AUDITED_MERGE_COMPLETE`**，不得冒称单次六类 `COMPLETE`。
6. 明确注明 B3 来自同一硬件上的不同实验批次；不得混称为单次运行吞吐量。
7. 不利结果完整披露，不选择性删除；不扩展 27 类；不改论文结论直到独立审核。

## 7. 文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `pilot_experiment_ab_v4.py` | `89e1d5ebf5c92a00aa3c54dfcb45447af3503928e91e1618f0a613ac16ec6edb` |
| `tests_pilot_unit_v4.py` | `c1883f005fa0174ee06bab2c4bab32fda131f776c19f7574e7634cea8c99aaab` |
| `mock_pilot_smoke_v4.py` | `fed8eccea1e6ee0afa68add168ff89531f16f5b21209a5e8f9c1e27c3a24c00b` |
| `pilot_reference_targets.json` | `c72e1a6bd5c6a61c5ef1c45a445ab8f65d37b19c290c11230047aa44174f6e72` |
| `run_with_wall_timeout.sh` | `a8079d9785b60f168a1ea4cde95122e95b842ebd9beb73898f6e9d9f2104707e` |
| `V4_TEST_RESULTS_v4_20261025.txt` | `5b6cd8c7eb3d76819fd3e25a5a23663e47e1c0e894b8a840b5bd758c05029817` |
| `self_check_report.json`（本地 v4） | `78f60b34490624f0ca8bce56657aff1b868391562deae0d254c4dc504116c087` |

## 8. 结论

- v4 仅修改 AUPRO Gate 的阻断策略；AUPRO 函数、targets、135 条冻结结果、编码器、权重、数据、方法与评价函数未变。
- 本地语法、81/81 单元、45/45 mock、self-check 全部通过。
- **GPU 未运行，MVTec 15 条未重跑，27 类未扩展，论文正文未修改。**
- 正式运行前仍需在云端 v4 `--mode self-check` 得到 `all_ok=true`，然后仅执行 VisA 三类。

**READY_FOR_VISA_GPU = YES（仅本地 v4 验证通过；云端 v4 self-check 仍是强制前置）。**
