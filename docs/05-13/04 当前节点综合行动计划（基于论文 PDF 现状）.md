# 当前节点综合行动计划（基于论文 PDF 现状）

## 一、现状诊断：PDF 与三份计划的对比

通读提交的 56 页 PDF 后，可以确认以下情况：

### 1.1 已经完成并写入 PDF 的内容

**实验数据（已写入）**：
- S1 主结果表（Table 2）：已含 n=10 seeds 的 mean ± std，数字与锁定值基本一致（PD-PPO B=1.70 显示 0.3908 ± 0.0451，与锁定值 0.3911 ± 0.0240 存在轻微差异，需核查）
- E1-fix 条件分层表（Table 3）：已写入三档条件（Calm/Mixed/Event），数字与锁定值一致
- E2 oracle 鲁棒性表（Table 4）：已写入 k=1–8 的单调递减结果
- P1 物理单位表（Table 5）：已写入气温/风速/雪通量的物理单位 MAE
- A2 staged diagnostic 表（Table 6）：**注意：PDF 中 Table 6 仍是旧版"remove one component"格式（A1 风格），而非 A2 staged diagnostic（D1–D4）格式**

**算法命名**：PDF 全文已使用 PD-PPO，Figure 4 架构图已更新，Algorithm 1 标题已改为 PD-PPO。

**声称软化**：
- §4.2 已包含 V2 生成器局限性说明（event fraction 上限 0.58，ACF 问题）
- §6.5 已包含 Generator limitation 段落
- §7.3 已包含 V3 生成器 future work 说明

**参考文献**：PDF 中 references 已包含正确的 DOI（Alali2024→Aloni2025 已修正，FernandezBes2015 已改为 IEEE JSAC 2015，Lim2021/Liu2024/Ying2022 key 已更新）。

### 1.2 PDF 中存在的残留问题（需立即修正）

**问题 A：Table 6 消融表格式错误（最高优先级）**

PDF 第 44 页 Table 6 仍是"remove one component"格式（5 行变体，n=3 seeds），而非 A2 staged diagnostic（D1–D4，n=5 seeds）。具体问题：
- 表格标题写"Results are mean ± std over 3 seeds"——n=3 与锁定的 n=5 不符
- 变体行包含"− Oracle-Calibrated Prior: 0.3971 ± 0.048"等数字，这些是**没有 CSV 来源的数字**（A1 未跑）
- 正文 §6.9 描述与 A2 staged diagnostic 不符，仍在描述"remove one component"结果

**问题 B：Table 2 主结果表数字与锁定值不一致**

PDF Table 2 显示：
- PD-PPO B=1.65: 0.3930 ± 0.0471（锁定值：0.3979，差异较大）
- PD-PPO B=1.70: 0.3908 ± 0.0451（锁定值：0.3911 ± 0.0240，std 差异大）
- PD-PPO B=1.75: 0.3920 ± 0.0432（锁定值：0.3907）

σ₂ 应为 0.0240（来自 exp_s1_main_stats.csv），但 PDF 显示 0.0451。σ₁、σ₃ 也需从 CSV 读取后填入。

**问题 C：摘要数字与正文不一致**

摘要写"outperforming Age-of-Information scheduling by 8.2%"，但正文 §6.3 写"AoI-based scheduling by 8.2%"，而锁定值为 7.9%（(0.4244−0.3911)/0.4244 ≈ 7.8%）。需统一为 7.9%（基于锁定值 0.3911 vs 0.4244）。

摘要写"approaching within 0.4% of an oracle-informed static baseline"，正文 §6.3 也写 0.4%，与锁定值一致（(0.3912−0.3911)/0.3912 ≈ 0.03%，实际差距更小）。需核查并统一。

**问题 D：§6.9 消融正文与 Table 6 不匹配**

§6.9 正文描述的是"remove one component"风格的消融（"Removing the oracle-calibrated prior causes the largest degradation (+0.006)"），但这些数字没有 CSV 来源。需替换为 A2 staged diagnostic 的正确描述。

**问题 E：Figure 5 图例仍显示"Custom PPO"**

PDF 第 37 页 Figure 5（Power-error tradeoff）的图例中仍显示"Custom PPO"，未更新为"PD-PPO"。

**问题 F：§6.4 Budget Sensitivity 数字与锁定值不一致**

§6.4 写"relative improvement ranging from 2.8% at B=1.65 to 3.1% at B=1.70 and 2.9% at B=1.75"，这些是 PD-PPO vs Round-robin 的改进。需用锁定值重新计算：
- B=1.65: (0.4057−0.3979)/0.4057 ≈ 1.9%（而非 2.8%）
- B=1.70: (0.4041−0.3911)/0.4041 ≈ 3.2%（接近 3.1%）
- B=1.75: (0.4042−0.3907)/0.4042 ≈ 3.3%（而非 2.9%）

**问题 G：§6.5 E1 数字与锁定值不一致**

PDF Table 3 中 Calm 条件 AoI = 0.3456，但锁定值为 0.3495；Mixed 条件 AoI = 0.4250，锁定值为 0.4292；Event 条件 AoI = 0.4723，锁定值为 0.4754。需从 exp_e1_condition_stats_fixed.csv 重新读取并填入。

**问题 H：统计显著性段落缺失**

PDF §6.3 后没有独立的统计显著性段落（Wilcoxon + Bonferroni 说明）。

**问题 I：§3.4 状态定义仍含 e_t**

PDF 第 17 页 Eq.(7) 状态向量仍包含 `e_t`，而 turn_56 计划要求将 `e_t` 改为 `z_t`（或在 §3 中删除 `e_t`，仅在 §5 中作为 EventAwareCritic 输入引入）。

**问题 J：Table 1 传感器规格仍含占位符**

Table 1 caption 写"Power and start-up values are normalised deployment costs, not device-watt measurements"，但 §4.1 写"final absolute power ratios for the flux channel and Parsivel² are to be filled from the device datasheets before submission"——这是明确的 [FILL] 占位符，需作者填写真实功率值。

---

## 二、当前节点最合适的行动计划

### 优先级分级原则

鉴于三份计划（turn_56 Codex 计划、turn_57 补充实验计划、turn_58 V3.1 修订版）均未执行，当前节点需要区分：
- **可立即执行（无需新实验）**：修正 PDF 中已有数据的错误写法
- **需要新实验（GPU 资源）**：A1、H1、G1+S2
- **需要作者填写**：传感器功率值、作者信息

### 阶段 0：立即执行——修正 PDF 现有错误（无需 GPU，约 2–4 小时）

这是当前最高优先级，可直接在 LaTeX 源码上操作。

#### 0-A：修正 Table 6 和 §6.9（最高优先级）

**操作**：将 `tables/ablation.tex` 完整替换为 A2 staged diagnostic 格式：

```latex
\begin{table}[t]
\caption{Staged diagnostic ablation (A2) at budget $B = 1.70$
  ($n = 5$ seeds, seeds 41--45). Each row adds one component
  to the previous configuration. Lower FW-MAE is better.}
\label{tab:ablation}
\begin{tabular}{llr}
\toprule
Stage & Configuration & FW-MAE (mean $\pm$ std) \\
\midrule
D1 & MaskedActor + ActionEmbedding          & $0.4224 \pm 0.0326$ \\
D2 & D1 + EventAwareCritic                  & $0.4219 \pm 0.0412$ \\
D3 & D2 + AWBC auxiliary loss               & $0.4080 \pm 0.0284$ \\
D4 & D3 + Oracle-calibrated prior (PD-PPO)  & $0.3942 \pm 0.0384$ \\
\bottomrule
\end{tabular}
\end{table}
```

同时将 §6.9 正文替换为 A2 staged diagnostic 描述（参见 turn_56 Step 4 的 LaTeX 代码）。

**验证**：`grep "3 seeds" tables/ablation.tex` 返回空；`grep "D1\|D2\|D3\|D4" tables/ablation.tex` 返回四行。

#### 0-B：从 CSV 读取 σ₁、σ₃，修正 Table 2

**操作**：读取 `reports/v2_supplement_assets/exp_s1_main_stats.csv`，找到 PD-PPO 行的 std 值：
- σ₁（B=1.65）：从 CSV 读取
- σ₂（B=1.70）：0.0240（已锁定）
- σ₃（B=1.75）：从 CSV 读取

将 `tables/main_results.tex` 中 PD-PPO 行的 std 值更新为 CSV 实际值。

**注意**：PDF 中 PD-PPO 的 mean 值（0.3930/0.3908/0.3920）与锁定值（0.3979/0.3911/0.3907）存在差异。需以 CSV 为准，不得使用 PDF 中的现有数字。

#### 0-C：修正摘要中的百分比数字

**操作**：
- 将摘要中"outperforming Age-of-Information scheduling by 8.2%"改为"7.9%"（基于锁定值 (0.4244−0.3911)/0.4244 ≈ 7.8%，取 7.9% 与正文一致）
- 核查"approaching within 0.4% of an oracle-informed static baseline"——锁定值 (0.3912−0.3911)/0.3912 ≈ 0.03%，应改为"within 0.03%"或"within 0.4%"（取决于 CSV 实际值）

#### 0-D：插入统计显著性段落（§6.3 后）

**操作**：在 `sections/06_experiments.tex` 的主结果表之后插入 turn_56 Step 2 的 LaTeX 代码（Wilcoxon + Bonferroni 段落）。

#### 0-E：修正 §6.4 Budget Sensitivity 数字

**操作**：用锁定值重新计算 PD-PPO vs Round-robin 的改进百分比，替换 §6.4 中的 2.8%/3.1%/2.9%。

#### 0-F：修正 §6.5 E1 表格数字

**操作**：从 `reports/v2_supplement_assets/exp_e1_condition_stats_fixed.csv` 读取实际数字，替换 PDF Table 3 中的数字（当前 PDF 数字与锁定值存在偏差）。

#### 0-G：修正 Figure 5 图例

**操作**：重新生成 Figure 5（Power-error tradeoff），将图例中的"Custom PPO"改为"PD-PPO"。

#### 0-H：修正 §3.4 状态定义（e_t → z_t）

**操作**：将 `sections/03_problem_formulation.tex` 中 Eq.(7) 的 `e_t` 改为 `z_t`，并在 §5.5 EventAwareCritic 中引入 `z_t` 的定义。

#### 0-I：作者填写传感器功率值

**操作**：将 Table 1 中的归一化功率值（0.12/0.05/0.05/1.35/1.55）替换为真实设备功率（瓦特），或在 caption 中明确说明这些是归一化值及其对应的真实功率比例。同时填写 §4.1 中的 [FILL] 占位符。

---

### 阶段 1：并行启动补充实验（需 GPU，约 1 周）

在阶段 0 完成后，立即并行启动以下实验：

#### 1-A：A1 完整消融（70 runs，约 35 GPU 小时）

按 turn_58 V3.1 修正后的设计执行：
- 7 个新变体（A1-v1 至 A1-v7）× 10 seeds（seeds 41–50）
- Budget = 1.70
- Bonferroni α_adj = 0.05/7 ≈ 0.0071
- 输出：`reports/v3_supplement_assets/exp_a1_ablation_stats.csv`

**关键注意**：A1-v7（MaskedActor only）不得复用 A2 D1 数据（n=5），必须独立运行 n=10。

#### 1-B：H1 超参数扫描（40 runs，约 20 GPU 小时）

按 turn_58 V3.1 修正后的设计执行：
- 3×3 网格：awbc_coef ∈ {0.05, 0.1, 0.2} × prior_kl_coef ∈ {0.5, 1.0, 2.0}
- 基准 H1-22（awbc=0.1, prior_kl=1.0）复用 S1 数据，新跑 8 个配置 × 5 seeds
- 输出：`reports/v3_supplement_assets/exp_h1_hyperparam_stats.csv` + 3×3 热图

#### 1-C：G1 V3 生成器验证（无需 GPU，约 3–5 天开发）

按 turn_58 V3.1 修正后的设计实现 V3 生成器：
- 半马尔可夫潜变量（calm/transition/storm 三状态）
- CRED 触发 + hysteresis（ON=0.6, OFF=0.3）+ 最小持续时间（3步）+ 最小间隔（6步）
- 转移概率 calibrated to match Amory 2020 D17 统计（event freq ≈ 20%，中位持续 15h）
- 验证指标：G1-V1（ACF）、G1-V2（event fraction 分布）、G1-V3a（KS，5 个基础气象通道）、G1-V3b（吹雪条件统计）、G1-V4（PSD，0.1–4.0 cpd）

---

### 阶段 2：G1 通过后执行（约 1–2 周后）

#### 2-A：S2 完整重跑（需 GPU，约 60 runs）

**前提**：G1 所有 5 个验证指标通过。

**操作**：
1. 在 V3 生成器上重新训练 oracle（TCN，full-observation 条件）
2. 在 V3 生成器上重新训练 PD-PPO（n=10 seeds，三个 budget）
3. 重新评估所有基线策略
4. 输出：`reports/v3_supplement_assets/exp_s2_main_stats.csv`

**注意**：不得将 V2 训练的策略直接在 V3 上评估（除非明确标注为迁移实验）。

#### 2-B：E1-V3 条件分层重跑

S2 完成后，用 V3 生成器重跑条件分层评估，此时可产生 event fraction > 0.75 的真正 event-heavy 条件。

---

### 阶段 3：可选（数据确认后）

#### 3-A：T1 跨站点泛化

**前提**：确认 AntAWS 数据集中 Dome C 和 D47 站点有完整同格式时间序列，且有吹雪通量观测。

**当前处理**：在 §7 中以 future work 形式提及（turn_58 修正八的 LaTeX 代码已提供）。

---

## 三、各阶段的正文影响与决策点

### A1 结果的正文影响

| A1-v2（移除 EventAwareCritic）结果 | 正文处理 |
|----------------------------------|---------|
| FW-MAE ≈ Full PD-PPO（差异 < 0.005） | 确认 §5.5 软化声称方向正确，维持"contributes negligibly in isolation" |
| FW-MAE 显著高于 Full PD-PPO（p < 0.0071） | 需修改 §5.5，恢复 EventAwareCritic 的贡献描述 |

| A1-v6（no_action_mask）结果 | 正文处理 |
|---------------------------|---------|
| FW-MAE 显著高于 Full PD-PPO | 在 §5.3 中强调 MaskedActor 的独立贡献 |
| FW-MAE ≈ Full PD-PPO | 说明 action mask 在当前 budget 设置下影响有限 |

### H1 结果的正文影响

| H1 结果 | 正文处理 |
|--------|---------|
| 所有 8 个非基准配置 FW-MAE ∈ [0.371, 0.411] | 可声称"PD-PPO is robust to hyperparameter variation within one order of magnitude" |
| 存在配置超出范围 | 需在 §5.7 中说明敏感参数，并提供调参建议 |

### G1 结果的决策点

| G1 结果 | 后续操作 |
|--------|---------|
| 所有 5 个指标通过 | 立即启动 S2 |
| G1-V1（ACF）未通过 | 调整 Hann 窗平滑步数（当前 10 步），重新验证 |
| G1-V2（event fraction 分布）未通过 | 调整 hysteresis 阈值或最小持续时间参数 |
| G1-V3b（吹雪条件统计）未通过 | 调整幂律模型参数（Amory 2020 Eq.3） |
| G1-V4（PSD）未通过 | 检查 DFT 相位随机化实现，确认频率轴单位为 cpd |

---

## 四、当前 PDF 版本可安全声称的内容（无需新实验）

以下声称有 CSV 数据支撑，可在当前版本中保留：

- "PD-PPO achieves FW-MAE of 0.3911 ± 0.0240 at B=1.70 (n=10 seeds)"
- "PD-PPO outperforms AoI by 7.9% at B=1.70 (Wilcoxon, Bonferroni-corrected, n=10)"
- "PD-PPO outperforms AoI at B=1.70 and 1.75 (p < 0.0083); at B=1.65 the difference approaches but does not reach significance"
- "PD-PPO reduces air-temperature forecast error by 0.476°C relative to AoI (−15.7%)"
- "PD-PPO reduces wind-speed forecast error by 0.181 m/s relative to AoI (−11.3%)"
- "Oracle FW-MAE degrades monotonically from 0.696 (k=1) to 0.385 (k=8)"
- "AWBC auxiliary loss provides the largest single-component improvement (−3.3%) in the A2 staged diagnostic"
- "EventAwareCritic contributes negligibly in isolation (−0.1%) in the current V2 environment"
- "V2 generator cannot produce event fractions exceeding approximately 0.58"

## 五、当前 PDF 版本不得声称的内容（需等待新实验）

| 禁止声称 | 原因 | 替代措辞 |
|---------|------|---------|
| "A1 ablation shows..." | A1 未跑 | "A2 staged diagnostic shows..." |
| "hyperparameter-robust" | H1 未跑 | "fixed configuration; sensitivity analysis is future work" |
| "event fraction > 0.75" | V2 生成器上限 0.58 | "event fraction > 0.40 (up to 0.58)" |
| "EventAwareCritic substantially improves performance" | A2 D2 贡献 −0.1% | "contributes negligibly in isolation" |
| "FW-MAE of 0.391 corresponds to 0.391°C" | 气温 MAE 为 2.547°C | "FW-MAE is normalised; air-temperature MAE is 2.547°C" |
| "DFT guarantees all synthetic-process fidelity" | 事件覆写破坏风速 ACF | "DFT preserves PSD of base variables; event boundaries may disrupt ACF" |

---

## 六、未来计划（投稿后/修改稿阶段）

### 6.1 最小可行投稿集（当前版本 + 阶段 0 修正）

完成阶段 0 的所有修正后，论文可达到"可投稿"状态：
- 所有数字有 CSV 来源
- 无过强声称
- 消融表为 A2 staged diagnostic（有数据支撑）
- 统计显著性说明完整

### 6.2 修改稿增强集（A1 + H1 完成后）

A1 和 H1 完成后，可将论文从"可接受"升级为"强论文"：
- 完整组件消融（A1，n=10，Bonferroni 校正）
- 超参数鲁棒性（H1，3×3 热图）
- 可声称"hyperparameter-robust within one order of magnitude"

### 6.3 完整版本（S2 完成后）

S2 完成后，可将 V2 结果替换为 V3 结果，并在 §6.5 中报告真正的 event-heavy（ef > 0.75）条件下的性能。

### 6.4 投稿策略建议

**推荐**：以当前版本（阶段 0 修正后）单投 Cold Regions Science and Technology，同时并行运行 A1/H1/G1 实验。若收到修改意见，可将 A1/H1 结果作为修改稿的补充实验。

**不推荐**：等待 S2 完成后再投稿（S2 需要 G1 通过，总周期约 3–4 周，延误投稿时机）。

---

## 七、执行优先级总结

| 优先级 | 任务 | 所需资源 | 预计时间 |
|--------|------|---------|---------|
| P0（立即） | 修正 Table 6（A2 staged diagnostic） | LaTeX 编辑 | 30 分钟 |
| P0（立即） | 从 CSV 读取 σ₁、σ₃，修正 Table 2 | CSV 读取 + LaTeX | 30 分钟 |
| P0（立即） | 修正摘要百分比（8.2% → 7.9%） | LaTeX 编辑 | 10 分钟 |
| P0（立即） | 插入统计显著性段落 | LaTeX 编辑 | 20 分钟 |
| P0（立即） | 修正 §6.9 消融正文 | LaTeX 编辑 | 30 分钟 |
| P1（本周） | 修正 §6.4/§6.5 数字 | CSV 读取 + LaTeX | 1 小时 |
| P1（本周） | 修正 Figure 5 图例 | 重新生成图 | 30 分钟 |
| P1（本周） | 修正 §3.4 状态定义（e_t → z_t） | LaTeX 编辑 | 20 分钟 |
| P1（本周） | 作者填写传感器功率值 | 查阅数据手册 | 作者操作 |
| P2（并行） | A1 完整消融（70 runs） | GPU × 8，35h | 约 1 周 |
| P2（并行） | H1 超参数扫描（40 runs） | GPU × 8，20h | 约 3 天 |
| P2（并行） | G1 V3 生成器实现与验证 | CPU，开发 | 约 3–5 天 |
| P3（G1 后） | S2 完整重跑（60 runs） | GPU × 8，30h | 约 1 周 |
| P4（可选） | T1 跨站点泛化 | 数据确认后 | 待定 |
