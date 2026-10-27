# 实验 A/B 六类试点审核与 27 类扩展 Gate

> **Historical snapshot notice (added 2026-10-27).** This file records the state before the full 27-category experiment. Labels such as `PILOT_NOT_RUN`, `NOT_RUN`, and `PENDING` are retained unchanged as historical evidence and are not current status. The full 27-category experiment was subsequently completed; the final record is `data/EXPERIMENT_AB_MERGED_135_RESULTS.csv`.


- 日期：2026-10-25
- 状态：**PILOT_NOT_RUN** —— GPU 实例 `<GPU_INSTANCE>` 在检查时离线（连接被关闭）；按授权未擅自启动/创建收费实例，未执行任何 GPU 计算。
- 本轮已完成：协议冻结（实验 A §2、实验 B §1-2）、独立 runner 编写与编译、17 项 CPU 单元测试全部 PASS、本地 self-check（sklearn 1.8.0）all_ok=true、六类复现目标 JSON 生成、原稿读取与修订方案（另文）。
- 本报告不包含任何伪造的试点数值；六问中凡依赖试点数据的条目均标注 NOT_RUN，并给出预注册的判断框架。

---

## 1. 试点执行前置条件（当前未满足）

| 条件 | 状态 |
|---|---|
| 1. pilot runner 编译 + 单元测试（coreset 大小/随机性/泄漏、归一化、指标复现、评分器复用） | ✅ 17/17 PASS |
| 2. 本地 self-check（GMM 存储口径与去冗余分数一致性、coreset 覆盖、normalizer） | ✅ all_ok=true（sklearn 1.8.0） |
| 3. 云端 sklearn **1.9.0** 下重跑 self-check | ⏳ PENDING（需开机后执行） |
| 4. GPU 正常模式在线 + 用户确认实际价格 | ❌ 实例离线 |
| 5. 六类试点运行（≤60 min，全新隔离输出目录） | NOT_RUN |
| 6. A0/A1 对冻结 135 条记录复现（容差 1e-6） | NOT_RUN（运行时自动校验） |

## 2. 科研决策 Gate（A–F）——PILOT_NOT_RUN 状态下的回答

**A. 在约 1 MB 的接近等状态存储预算下，VNE 与 5% coreset Nearest 的准确率差异是什么？**
**NOT_RUN（无法回答）。** 预注册的判定口径：A0 最小推理状态 1,050,628 B；A2 coreset-5% bank 1,003,520 B（相差 4.5%）。将报告六类上三项指标的配对差值（逐类 + 汇总）；若差异方向在六类内不一致或量级与冻结 27 类的 VNE−Nearest 差值（image −0.0150 / pixel +0.0094 / AUPRO +0.0207，池化 27 类）相同量级，则"接近等预算下的差异"将无法支持 VNE 的单向优越性。
**同时说明结构性事实（基于复杂度分析，非测量结论）**：1-NN 在 M=490 时的每 patch 查询成本约为 490×512 ≈ 2.5×10^5 次乘加，与单分量 512 维高斯评分的 512² ≈ 2.6×10^5 次乘加**同量级**——因此"压缩到 VNE 存储预算后，Nearest 的查询成本仍远高于 VNE"这一假设在理论上不成立，必须由实验 B 实测确认。

**B. Nearest 压缩后，VNE 的 scorer-only 时间优势是否仍然存在？准备成本又如何变化？**
**NOT_RUN。** 设计要点已冻结：B3（3 预热 + 10 计时、真实测试特征、CPU 路径、原始重复保存）；B2 包含 coreset 选择时间（k-center 在 9800×512 上的贪心选择为一次性成本，预计不低但需实测）。无论结果如何，报告将分别给出"全量 bank"与"5%/10% coreset"的时间，不混称。

**C. 是否存在至少一种准确率、存储、准备时间与查询负载组合下，VNE 仍有可解释的实际价值？**
**NOT_RUN。** 预注册的价值条件：若在等存储预算（≈1MB）下，VNE 在 pixel/AUPRO 上的优势超过其 image 劣势的可接受范围，且 scorer-only 时间不劣于 coreset-Nearest、准备时间可接受——则可陈述为"低存储参数化表示在像素级定位上的可用权衡"。反之（coreset 全面接近或更优），VNE 的工程主张必须收缩。

**D. 如果 5% coreset 在关键指标和效率上优于 VNE，原论文的工程主张是否需要撤回或改写？**
**若发生，则必须改写。** 预先声明改写方式：不得通过"VNE 仍是参数化模型"之类的形式化理由维持优势；论文的 trade-off 叙事将改为"在等存储预算下，压缩的显式参考库是 VNE 的强竞争者；VNE 保留的差异化仅在（如实测成立的）具体维度"；摘要与结论中的效率主张同步降级。**不会**在看到结果后修改评测或选样定义。

**E. 六类试点是否提供足够理由扩展至 27 类？**
**NOT_RUN。** 预注册扩展标准：① 六类内 A0/A1 复现冻结值（1e-6）；② 所有方法无 SafetyStop/退化；③ 结果在六类上显示可解释的、方向一致的差异（尤其 coreset-5% 相对 A0/A1 的行为）；④ 预算可持续（27 类特征与计时预计为试点的数倍，需单独授权）。扩展时必须沿用同一 seed/算法/口径，全 27 类一次性预注册，结果全量报告（含不利类别），不得选择性报告。
**选择偏差声明**：六类（bottle/grid/screw/candle/pcb3/macaroni1）依据历史观察选定，属诊断样本，不是无偏验证集；这一点将写入论文任何相关表述。

**F. 现有 PCA0.97 的 27 类结果可以提供什么辅助参照？**
仅作**辅助参照**（其方法学独立性亦有限：见 SubspaceAD）：PCA0.97 的 27 类结果可用于三角验证"参数化评分器的存储-精度权衡"形状（如 PCA0.97 的 pixel/AUPRO 相对 Nearest 的 BH 显著优势、image 劣势），但**不得**因试点结果不利而事后将其改选为主方法；PCA 不进入实验 A 的主比较。该约束已写入实验 A 协议。

## 3. 本轮可确立的事实（不依赖试点数据）

1. **存储口径修正生效**：VNE-K1 最小推理状态 1,050,628 B（已验证去冗余分数逐位一致，sklearn 1.8.0；1.9.0 待复验）；原 3,147,780 B 为"含 2 个冗余矩阵"的记录口径。论文中 **6.38×** 存储倍数（20.07 MB / 3.15 MB）必须先声明口径，若采用去冗余口径则为 **19.1×**（20.07 MB / 1.05 MB）——两者都需与"是否已部署"的表述严格分开。
2. **QuickGELU 27 类的冻结描述统计**（供修订稿引用，非新实验）：池化 27 类 VNE−Nearest 配对均值差 = image **−0.014971**、pixel **+0.009354**、AUPRO **+0.020736**；分数据集见统计审计报告。这些数字将替换原稿中的旧 GELU 数字（原稿旧值 image 低 0.0215、pixel/AUPRO 高约 0.006）。
3. **公平性风险已被量化**：任何"VNE 更省存储"的陈述必须同时给出 A/B 两种口径，并注明 Nearest 为**未压缩**完整池（PatchCore coreset 文献先例已存在）。

## 4. 继续条件（解除 PILOT_NOT_RUN 的步骤）

1. 用户在 GPU 提供方控制台以**正常 RTX 4090 GPU 模式**开机，并确认当前实际价格；
2. 上传 `pilot_experiment_ab.py`、`pilot_reference_targets.json` 至云端独立目录（不改原 runner/结果）；
3. 云端执行 `--mode self-check`（sklearn 1.9.0），通过后保存 `self_check_report.json`；
4. 执行 `--mode pilot`（六类，≤60 分钟，`OMP_NUM_THREADS=4`，全新输出目录）；
5. 下载结果，保存指纹；本 Gate 报告由 ChatGPT 审核后再裁定是否扩展 27 类与启动全文修订。

**最终状态：EXPERIMENT_AB_PILOT = NOT_RUN（准备完成，等待用户开机与价格确认）；FULL_27_EXTENSION = PENDING_PILOT_REVIEW；MANUSCRIPT_REVISION = AFTER_SCIENTIFIC_GATE。**
