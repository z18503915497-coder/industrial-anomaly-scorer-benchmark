# 实验 A：压缩参考库公平性比较 — 协议与测试审计

> **Historical snapshot notice (added 2026-10-27).** This file records the state before the full 27-category experiment. Labels such as `PILOT_NOT_RUN`, `NOT_RUN`, and `PENDING` are retained unchanged as historical evidence and are not current status. The full 27-category experiment was subsequently completed; the final record is `data/EXPERIMENT_AB_MERGED_135_RESULTS.csv`.


- 日期：2026-10-25
- 状态：**协议冻结 + 本地测试完成；试点未运行（PILOT_NOT_RUN — GPU 实例离线）**
- 范围：仅本地准备与 CPU 测试；未启动 GPU；未修改任何冻结 runner/结果。
- 执行件：`pilot_experiment_ab.py`（sha256 见 §7）；测试件：`tests_pilot_unit.py`；复现目标：`pilot_reference_targets.json`（从冻结 135 条 CSV 提取，未重算特征）。

---

## 1. 科学问题

**VNE-K1 相对于保留完整正常 patch 库的 Nearest，是否在公平、接近相同的评分器存储预算下，仍有实际的准确率或计算成本优势？**

公平性定义（本协议）：两种评分器都在（i）相同冻结特征、（ii）相同参考预算（每类 sorted-first 50 张正常图 → 9800 patches）、（iii）相同测试 split/GT/evaluator、（iv）各自**原生数值精度**、（v）各自**去除冗余后**的状态口径下比较；任何压缩/精度转换都必须单独标注为候选并经过验证。

## 2. 对照设计（A0–A4）

| 编号 | 评分器 | 参考/状态 | 大小（float32, 每类） | 说明 |
|---|---|---|---|---|
| A0 | VNE-K1（GaussianMixture, n_components=1, full covariance） | 完整 9800-patch 参考池拟合；原评分定义（-log p） | 记录口径 3,147,780 B；**去冗余最小推理 1,050,628 B** | 不针对测试结果调参；实现为冻结 `src/sara/vne.py` |
| A1 | Nearest-Full（1-NN cosine） | 全部 9800 patches | 20,070,400 B | 冻结 `NearestReferenceScorer`（chunk=1024）逐字复用 |
| A2 | Nearest-Coreset-5% | k-center-greedy 选择 490 patches | 1,003,520 B | 选择仅用参考特征 |
| A3 | Nearest-Coreset-10% | k-center-greedy 选择 980 patches | 2,007,040 B | 同上 |
| A4 | Nearest-Random-5% | 均匀随机 490 patches（seed=42） | 1,003,520 B | 区分"覆盖式选样"与"单纯缩小 bank" |

- **算法**：采用标准 k-center-greedy（覆盖式选择；与 PatchCore 的 coreset 子采样同类方法）：随机种子固定首点（seed=42），随后每一步选择"到已选集合最近距离最大"的参考点，直到目标数量。**未使用随机投影加速**——因此无投影维度/误差参数需要冻结；选择过程确定性可复现。
- **不冒充 PatchCore**：A2–A4 只是"在相同特征与评分定义下压缩参考库"，不是完整 PatchCore（后者使用不同的特征提取、coreset 实现、局部邻域与加权评分），论文中只能作为"coreset-Nearest 文献做法在本协议下的对照"，PatchCore 仍为外部参考方法（anomalib 2.6.0 配置，旧有数字需与新协议分开陈述）。
- **泄漏禁令（结构性保证）**：`kcenter_greedy(ref_features, n_select, seed)` 与 `random_subset(ref_features, n_select, seed)` 的签名只接受参考特征矩阵，无法接触测试图像、测试标签、GT mask 或任何 AUROC 值；单元测试 T4 断言该签名不变。试点中 coreset/比例/超参数在任何测试评分之前确定并写入运行摘要。
- **A1/A0 复现校验**：pilot 同时校验 A0（对冻结 `vne_k1`）与 A1（对冻结 `nearest`）在六个试点类别上的三项指标；容差 1e-6。任一超差 → 立即 `REPRODUCTION_MISMATCH`（exit 3）、保存证据、停止。这保证试点流程与 135 条冻结记录在数值路径上同构。

## 3. reference_minmax 稳定性检查（协议要求）

- 冻结归一化定义保持不变：`reference_normalizer(ref_raw)` 的 min/max 应用于测试分数；`ref_range <= 1e-12` 视为退化并触发 SafetyStop（与冻结 runner 相同）。
- pilot 额外记录每类每方法的：`ref_raw_range`（参考池自评分的分数范围）、测试归一化越界计数（<0 / >1，仅信息性，不裁剪）。
- **风险已知项**：Nearest 对自身参考池评分时范围极小（~1e-6 量级，冻结环境实测），float32 舍入可能敏感；coreset 缩小 bank 后该范围会变化。本协议**不改变归一化定义**；任何替代校准（例如改用较大参考集估计 stats）都只能作为预先标记的独立敏感性实验，且不得在看到测试结果后选择。若试点中出现退化范围触发 SafetyStop，保留证据并停止该类。
- 单元测试 T5 验证归一化的 min-max 恒等性与零范围行为；pilot 运行时的实际范围将写入结果 CSV。

## 4. 存储口径（公平报告）

pilot 对每个方法同时输出：

| 口径 | 定义 | 数值来源 |
|---|---|---|
| A. 实际拟合后保留数组 | 当前实现持有的全部数组 | VNE：5 数组（3,147,780 B，f32，实测构成精确复现）；Nearest：bank 全量 |
| B. 经验证的去冗余推理状态 | score_samples 实际使用的数组 | VNE：weights+means+precisions_cholesky = **1,050,628 B**（本地 sklearn 1.8.0 实测：删除 covariances_/precisions_ 后分数逐位一致，max|diff|=0.0；**pilot 将在云端 sklearn 1.9.0 复跑同一 self-check 并记录**） |
| C. coreset bank 状态 | 选样后的参考矩阵 | A2/A3/A4 由实际 bank bytes 输出 |
| D. 额外索引/元数据/运行时内存 | 选择索引、模型元数据 | 本实现无额外索引结构（index 在拟合时即被丢弃，不参与推理）；报告 0 并在 notes 说明；峰值 RAM 不做受控声明 |
| E. 完整共享骨干 + 每类评分器 | backbone 598,516,980 B + 每类状态 | 结果 CSV 中 `full_deployment_bytes_shared_backbone` 列 |

- 对称性要求：不对 VNE 用冗余口径、对 Nearest 用压缩口径，或反之。主比较同时给出 A（实际当前实现）与 B/C（可实现的去冗余/压缩方案）两列；论文正文必须区分二者。
- **PCA0.97 不参与本实验**（实验 A 仅为 VNE vs Nearest 的公平性；PCA 的 27 类结果仅作辅助参照，且不得因结果改选主方法——见 Gate 报告）。

## 5. 试点类别（预先固定，承认选择偏差）

MVTec：`bottle`（普通对象）、`grid`（纹理）、`screw`（已知图像级困难对象）。
VisA：`candle`（历史复现锚点）、`pcb3`（已知 image 级失效类别）、`macaroni1`（另一种结构类别）。

这些类别依据已观察到的历史数据选择，属**诊断样本**；六类试点结果不能作为无选择偏差的正式验证，也不得出"VNE 胜出/落败"的总体结论。

## 6. 本地测试审计（全部执行，17/17 PASS）

| 测试 | 结果 |
|---|---|
| T1 coreset 大小/确定性/唯一性/种子敏感性 | PASS（seed=42 复现；seed=43 不同） |
| T2 k-center 覆盖优于均匀随机 | PASS（合成数据上平均最近选中距离更小） |
| T3 random 子集大小/边界/确定性 | PASS |
| T4 泄漏防护（选择函数签名仅接受参考特征） | PASS（`ref_features, n_select, seed`） |
| T5 归一化 min-max/退化范围/完美 AUROC | PASS |
| T6 冻结 Nearest 评分器复用一致性 + 暴力核对 | PASS（类与函数路径 float32 双重归一化末位差 ≤1e-6，属既有实现特性） |
| T7 VNE 存储口径（5 数组和 = 3,147,780；最小推理 = 1,050,628） | PASS |
| self-check（本地 sklearn 1.8.0）：GMM dtype/字节构成、去冗余分数一致（0.0）、coreset 确定性/覆盖、normalizer、AUROC | **all_ok=true** |

- `pilot_experiment_ab.py` 通过 `py_compile`。
- 云端 sklearn **1.9.0** 的核对步骤已内置于 runner 的 `--mode self-check`（试点运行前必须先在云端执行并保存 `self_check_report.json`）；在核验通过前不得开始 GPU 试点。

## 7. 运行与输出

- 运行入口（GPU 试点）：`pilot_experiment_ab.py --mode pilot --device cuda:0 --project-root <PROJECT_ROOT> --data-root-mvtec .../data/mvtec_ad --data-root-visa .../data/visa_mvtec_1cls_v1 --weights .../weights/open_clip_model.safetensors --output-dir <全新隔离目录>`
- 方法执行顺序：每类用固定 seed 打乱（记录于摘要），避免固定顺序优势。
- 输出：`EXPERIMENT_AB_SIX_CATEGORY_PILOT_RESULTS.csv`、`pilot_summary.json`（指纹/环境/原始计时/存储/复现结果）、`pilot_run.log`；任何 SafetyStop 或复现失败均保留证据。
- 预算：GPU 目标 ≤60 分钟（特征提取 ~9 min + 准备 <2 min + 主评估 ~20 min + B3 ~12 min + 端到端小样本 ~3 min + 余量）。

**当前状态：PILOT_NOT_RUN**（实例 `<GPU_INSTANCE>` 于 2026-10-25 检查时为离线）。恢复条件见 Gate 报告：用户以正常 GPU 模式开机并确认价格后，先上传 `pilot_experiment_ab.py` + `pilot_reference_targets.json`，在云端执行 `--mode self-check`（sklearn 1.9.0），通过后再运行 pilot。
