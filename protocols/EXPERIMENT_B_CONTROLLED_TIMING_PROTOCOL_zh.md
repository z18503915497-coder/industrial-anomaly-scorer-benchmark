# 实验 B：统一条件下的生命周期效率测量 — 协议

> **Historical snapshot notice (added 2026-10-27).** This file records the state before the full 27-category experiment. Labels such as `PILOT_NOT_RUN`, `NOT_RUN`, and `PENDING` are retained unchanged as historical evidence and are not current status. The full 27-category experiment was subsequently completed; the final record is `data/EXPERIMENT_AB_MERGED_135_RESULTS.csv`.


- 日期：2026-10-25
- 状态：**协议冻结；未执行（PILOT_NOT_RUN — GPU 实例离线）**
- 目的：在与实验 A 完全相同的条件下重新测量 scorer latency、preparation cost 与存储权衡；**不继承旧 GELU 环境下的任何速度倍数**。
- 实现：同一 `pilot_experiment_ab.py`（与实验 A 共享特征与评分器实例）。

---

## 1. 统一条件（同一运行内强制一致）

| 维度 | 冻结值 |
|---|---|
| 机器 | GPU RTX 4090 24GB 实例（同一实例、同一会话） |
| CPU/GPU 环境 | torch 2.8.0+cu128 / open_clip 3.3.0 / sklearn 1.9.0 / numpy 2.3.2（venv_oc330） |
| 编码器 | ViT-B-16-quickgelu（运行时断言 QuickGELU）+ safetensors sha256 `4b869929…` |
| 参考协议 | 每类 sorted-first ≤50 张 train/good；相同 test split、masks、evaluator |
| 特征 | 224×224 → 196×512 patch features，proj + L2-normalize（冻结 encode 路径） |
| 测试特征 | 每类只提取一次，全方法共享同一内存副本（缓存控制） |
| batch/chunk | Nearest chunk = 1024（冻结值）；scorer-only 批量矩阵 = 该类全部测试 patch 一次性拼接（真实特征） |
| 线程 | 运行前设置 `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`（记录于摘要）；不中途改变 |
| 方法顺序 | 每类用固定 seed 随机打乱（seed=42+1），顺序写入摘要 |
| 计时器 | `time.perf_counter()`；GPU 路径在 encode 内 `.cpu().numpy()` 处自然同步（记录边界） |

## 2. 分项定义（严格边界，不得混称）

| 编号 | 内容 | 计时边界 | 路径 |
|---|---|---|---|
| **B1** | 特征提取（共同成本） | 每类：50 参考图 encode + 全部测试图 encode；分 ref/test 两段 | GPU（含同步） |
| **B2** | 一次性准备 | VNE fit；k-center coreset 选择（A2/A3）；随机子集（A4）；bank 构建（A1 的 fit）；不含特征提取 | CPU |
| **B3** | scorer-only 测试 | 在**已提取的真实测试特征**拼接矩阵上调用 `score_samples`：3 次预热 + 10 次正式计时；保存每次原始耗时、median、IQR；另给出"per-image 等效 ms = median / 测试图数"（明确标注 scorer-only 等效值） | CPU（numpy/sklearn）；GPU 不参与 |
| **B4** | 后处理 | normalize + reshape/upsample + top-k 的图像级流水线累计时间（单次 pass 累计，与冻结 runner 的 `test_map_pipeline_s` 同边界） | CPU |
| **B5** | 端到端（信息性） | preprocess + GPU encode + score + normalize + upsample 的单图总时间；每类取前 5 正常 + 5 缺陷测试图，3 轮；仅对 A0/A1/A2 执行 | 混合（明确标注） |

- **B2 中的 coreset 选择时间必须计入**：A2/A3 的 `b2_coreset_selection_s` 单独列出，且 `b2_prepare_s` 包含它。不得只报告压缩后的查询加速而忽略选样代价。
- **B3 只测评分器计算**：不包含特征提取（B1）、后处理（B4）或 Python 图像循环开销；不得将 B3 表述为完整推理速度或产线吞吐。
- **B5 是混合路径**：不能与 B3 数值比较；不得把 B5 描述为"纯 GPU 推理"。

## 3. 统计与报告

- B3 每次：3 预热 + 10 计时（不足 10 次时必须在结果中说明原因，不得伪造）；报告 median、四分位距（Q3−Q1）与全部原始重复值（`pilot_summary.json` 中 `b3_reps_raw_s`）。
- 单类结果与六类汇总同时保留；不做跨数据集的方法排名宣告，仅报告实测分布。
- 每次运行保存：runner sha256、环境版本、方法顺序、coreset seed、线程配置、`b3_n_test_patches`（计时负载规模）——可供独立复算。

## 4. 历史数值处置（关键约束）

- 原论文/旧稿中的 **20.9–26.1×** 与 **36.08 → 1.72 ms/image** 来自**旧 GELU 环境的历史 scorer-only 测量**；本次修改后：
  1. 在实验 B 完成前，**不得将这些数字写入新的正式结果、摘要或结论**；
  2. 新测量若与旧值不一致，以新测量为准并在修订方案中标注差异原因（架构纠正 + 环境版本 + 计时配置）；
  3. 旧值可作为"历史环境记录"仅出现在修订说明/回复审稿意见中，且必须注明为旧 GELU 环境。
- 任何"scorer-level latency"与"完整推理/端到端/产线吞吐"之间的换算声明一律禁止（除非 B5 数据支持且明确标注混合路径边界）。

## 5. 预算与停止条件

- GPU 试点目标 ≤60 分钟；B3 的重复次数在极端情况下可减少（记录原因），但不得低于 3 预热 + 10 计时，除非资源异常导致无法完成——届时安全停止并保留已完成方法的结果。
- SafetyStop（非有限评分、退化归一化范围）或复现不一致（A0/A1 对冻结值 > 1e-6）→ 立即停止（exit 3/4），保留证据。
- 若 B3 的单次调用耗时导致预计超预算（例如 Nearest-Full 在 200 图类上的大批量评分），允许把 B3 的计时负载限制为**预注册的前 K 张测试图**（K=50，先于任何测试结果固定），并在结果中注明负载规模；不得看到指标后选择负载。

## 6. 输出

- `EXPERIMENT_AB_SIX_CATEGORY_PILOT_RESULTS.csv` 中的 `b1_*`、`b2_*`、`b3_*`、`b4_*`、`b5_*` 列；
- `pilot_summary.json` 中的原始重复计时与运行指纹。

**当前状态：PILOT_NOT_RUN**（GPU 离线）。在执行前，云端必须先用 `--mode self-check` 在 sklearn 1.9.0 下完成 VNE 去冗余核验与 coreset/归一化测试（输出 `self_check_report.json`），通过后方可启动 pilot；线程设置须在启动命令环境中显式导出。
