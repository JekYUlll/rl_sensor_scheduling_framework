# V3.1 完整重跑（S2）：决策分析与执行指南

## 总体判断

**推荐立即启动 S2，但采用"先跑完再决策是否替换主线"的保守策略。**

Pilot 结果（5 seeds，B=1.70）已经满足最小成功标准的核心条件：full-open 是最优上界，PD-PPO 优于所有可部署基线，与 static projection 差距约 1.05%。这个信号足够强，值得投入完整 S2 的计算资源。但 pilot 的 n=5 不足以支撑统计检验，且只覆盖单一 budget，因此在完整 S2 结果出来之前，论文主线不应切换。

---

## 第一部分：S2 启动前的必要准备

### 1.1 冻结当前可投稿稿（Stage S2.0）

在启动任何 S2 计算之前，必须先确认当前论文处于可回退状态。具体操作：

```bash
# 确认当前编译状态
cd paper && xelatex paper.tex && bibtex paper && xelatex paper.tex && xelatex paper.tex
# 生成 fallback PDF
cp paper.pdf paper_v2_fallback_$(date +%Y%m%d).pdf
# 提交 git snapshot（如果使用版本控制）
git add -A && git commit -m "v2-fallback: pre-S2 submission draft"
```

**验收条件**：`paper_v2_fallback_YYYYMMDD.pdf` 存在且可读，包含完整的 V2 主结果。

### 1.2 turn_61 三处精确替换的确认

在启动 S2 之前，应先确认 turn_61 的三处修正已经写入论文，因为这些修正与 S2 无关，且是投稿前必须完成的。需要检查：

- `paper/tables/sensor_specs.tex` 脚注：`"calibrated against bench measurements"` → `"informed by datasheet electrical characteristics"`
- `paper/sections/03_problem_formulation.tex` §3.2 最后一句：功率比例描述
- `paper/sections/04_simulation_environment.tex` §4.1：功率比例段落

如果这三处尚未修正，应在启动 S2 之前完成，避免 S2 过程中论文处于不一致状态。

---

## 第二部分：S2 脚本设计建议

### 2.1 新建 `scripts/42_v31_s2_full.py`

建议新建独立脚本而非扩展 pilot 脚本，原因是 pilot（seeds 41–45）和 full S2（seeds 41–50）的输出目录需要严格区分，避免混淆。

脚本的核心设计要点：

**断点续跑支持**：每个 (budget, seed) 组合在完成后写入一个 `.done` 标记文件，脚本启动时先扫描已完成的组合，跳过已有结果。这对于服务器供电不稳定的情况至关重要。

```python
done_marker = f"reports/v31_s2_main/done/{budget}_{seed}.done"
if os.path.exists(done_marker):
    continue
# ... run experiment ...
Path(done_marker).touch()
```

**输出目录结构**：

```
reports/v31_s2_main/
  raw/
    budget_1p65_seed_41/evaluation/v2_eval_overall.csv
    budget_1p65_seed_42/evaluation/v2_eval_overall.csv
    ...
  done/
    1.65_41.done
    ...
  v31_s2_overall_long.csv      # 收集后生成
  v31_s2_main_stats.csv        # 统计后生成
  v31_s2_condition_long.csv    # condition 分层后生成
```

**并行策略**：建议使用 3–4 个并行 worker，每个 worker 处理一个 (budget, seed) 组合。不建议超过 4 个 worker，以避免内存竞争和 GPU 争用导致结果不稳定。

### 2.2 Collector 脚本 `scripts/43_v31_s2_collect.py`

收集脚本应独立于训练脚本，支持在 S2 运行过程中随时调用，以便监控进度。收集逻辑：

1. 扫描 `reports/v31_s2_main/raw/` 下所有已完成的 `v2_eval_overall.csv`；
2. 合并为 `v31_s2_overall_long.csv`（long format，列：budget, seed, policy, fw_mae, power_mean, event_fw_mae, non_event_fw_mae）；
3. 按 (budget, policy) 分组计算 mean ± std；
4. 运行 Wilcoxon + Bonferroni 检验（仅在每个 budget 的 n=10 全部完成后）；
5. 输出 `v31_s2_main_stats.csv`。

---

## 第三部分：S2 结果的判断标准

### 3.1 最小成功标准（必须全部满足才能替换主线）

以下五个条件必须同时满足：

**条件 1**：在所有三个 budgets（1.65, 1.70, 1.75）上，`full_open_unconstrained` 的 FW-MAE 均低于所有其他策略。这是物理合理性的基本检验，如果不满足说明实验设置有问题。

**条件 2**：在所有三个 budgets 上，PD-PPO 的 FW-MAE 均低于 `round_robin`、`AoI`、`random`。

**条件 3**：在至少两个 budgets 上，PD-PPO vs AoI 的 Wilcoxon 检验在 Bonferroni 校正后显著（α_adj = 0.0083，n=10）。

**条件 4**：PD-PPO 与 `feasible_static_projected` 的差距不超过 3%（绝对 FW-MAE 差值）。超过 3% 说明 PD-PPO 相对于 oracle-informed 静态分配的优势不足，需要诊断。

**条件 5**：event-heavy stratum（event fraction > 0.75）中，PD-PPO 优于 `round_robin`、`AoI`、`random`。这是 V3.1 相对于 V2 的核心新增验证。

### 3.2 降级使用标准

如果最小成功标准只部分满足：

- 满足条件 1–4 但不满足条件 5（event-heavy stratum 不显著）：V3.1 可作为主结果，但 condition-stratified 部分需保守表述，不能声称"在极端风暴下验证有效"。
- 满足条件 1–3 但不满足条件 4（PD-PPO 与 static projection 差距 > 3%）：需要诊断 oracle 质量或 reward 设计，不建议替换主线，保留 V2 为主结果。
- 不满足条件 2（PD-PPO 在某个 budget 下不优于 round-robin）：暂停主线替换，回到 reward/oracle calibration 诊断，V2 保持主结果。

### 3.3 Pilot 结果的参考价值

Pilot（B=1.70，n=5）的结果已经满足条件 1、2、4 的单 budget 版本：

- full-open FW-MAE 0.4130 < PD-PPO 0.4398（条件 1 ✓）
- PD-PPO 0.4398 < round-robin 0.4501 < AoI 0.4820 < random 0.4827（条件 2 ✓）
- PD-PPO 与 static projection 差距：(0.4398 − 0.4352) / 0.4352 = 1.06%（条件 4 ✓）

但 n=5 不足以做 Wilcoxon 检验（条件 3 需要 n=10），且 event-heavy stratum 数据尚未报告（条件 5 未知）。

---

## 第四部分：论文改造的具体操作

### 4.1 如果 S2 成功：主线切换的改造范围

以下是需要修改的文件和内容，按修改量从小到大排序：

**`paper/tables/main_results.tex`**：替换为 V3.1 S2 数字。这是最核心的改动，所有其他改动都依赖于此。建议新建 `paper/tables/main_results_v31.tex`，在确认 S2 结果后再替换 `\input{tables/main_results}` 为 `\input{tables/main_results_v31}`。

**`paper/sections/06_experiments.tex`**：
- §6.1 Main Results：替换所有 FW-MAE 数字和百分比改进数字；
- §6.4 Statistical Significance：替换 p 值和显著性结论；
- §6.5 Condition-Stratified：替换为 V3.1 的三档分层结果（含真正的 event-heavy stratum）；
- §6.6 Physical Units：替换 P1 对应数字（需重新运行 P1 on V3.1）。

**`paper/sections/07_discussion.tex`**：
- 将 V2 event cap 从"当前主结果局限"降级为"开发历史中识别并修复的问题"；
- 删除"S2 remains future work"的表述；
- future work 改为"跨站点真实数据验证、硬件闭环部署、长期季节性供电约束"。

**`paper/sections/08_conclusion.tex`**：替换所有结果数字。

**`paper/sections/01_introduction.tex`**：替换 abstract 和 introduction 中的结果数字。

**`paper/sections/04_simulation_environment.tex`**：
- G1 验证段落从"V3.1 generator validated for future use"改为"V3.1 is the simulation environment used in all experiments"；
- 删除关于 V2 event cap 的 caveat（或移入 appendix）。

### 4.2 如果 S2 失败：保持 V2 主线的最小修改

如果 S2 不满足最小成功标准，论文应保持 V2 为主结果，但需要在 §7 Discussion 中诚实地写出 V2 的局限性，并将 G1 + pilot 作为"已识别问题并开始修复"的证据。具体措辞建议：

> The V2 simulation environment has two known limitations: the event fraction in any 512-step window is capped at approximately 0.58, and wind-speed autocorrelation is partially disrupted by event injection. The V3.1 generator addresses both limitations (G1 validation: ACF deviation 0.048, P(ef>0.75) = 0.065), and a pilot rerun (B=1.70, n=5) confirms that PD-PPO retains its advantage over deployable baselines in the improved environment. Full retraining across all budgets and seeds (S2) is left as future work.

---

## 第五部分：S2 过程中的监控建议

### 5.1 每完成一个 budget 的中间检查

建议在每个 budget 的 10 个 seeds 全部完成后，立即运行 collector 脚本并检查：

1. PD-PPO 的 mean FW-MAE 是否低于 round-robin、AoI、random；
2. full-open 是否仍是最优上界；
3. PD-PPO 与 static projection 的差距是否在 3% 以内。

如果 B=1.65 的结果不满足条件，应在继续 B=1.70 之前诊断原因，而不是等到全部 30 个 runs 完成后再发现问题。

### 5.2 服务器稳定性措施

鉴于服务器供电不稳定，建议：

```bash
# 使用 tmux 保持会话
tmux new-session -s s2_main
# 每完成一个 budget 立即同步
rsync -avz reports/v31_s2_main/ user@local:/backup/v31_s2_main/
# 设置每小时自动同步
crontab -e
# 添加：0 * * * * rsync -avz /path/to/reports/v31_s2_main/ user@local:/backup/v31_s2_main/
```

---

## 第六部分：当前论文的遗留问题处理

无论 S2 结果如何，以下问题应在 S2 运行期间并行处理（不需要等待 S2 结果）：

### 6.1 A1 与 A2 消融路径差异的正文解释（第一优先级）

需要在 §6.9（或 §6 的消融小节）中加入以下内容：

> Two complementary ablation protocols are reported. The staged diagnostic (A2, Table~\ref{tab:ablation}) adds components sequentially to a minimal base, measuring the marginal contribution of each addition. The full ablation (A1, Table~\ref{tab:ablation_full}) removes one component at a time from the complete PD-PPO, measuring the independent contribution of each component in the presence of all others. The two protocols are not equivalent when components interact: EventAwareCritic shows negligible marginal gain in A2 (−0.1\% when added to MaskedActor+ActionEmbedding) but a larger effect in A1 (+2.38\% when removed from full PD-PPO), consistent with a positive interaction between EventAwareCritic and the oracle prior. The joint removal of AWBC and oracle prior (A1: +13.97\%) substantially exceeds the sum of their individual effects (+3.08\% + +5.35\% = +8.43\%), confirming a synergistic interaction between these two components.

### 6.2 H1 措辞修正（第二优先级）

将 §6 中所有"robust"或"insensitive"的表述替换为：

> All nine configurations remain within 5\% of the default setting (maximum deviation: $+4.70\%$ at $\lambda_{\text{awbc}}=0.10$, $\lambda_{\text{kl}}=0.5$), suggesting local stability around the chosen hyperparameters.

### 6.3 GillGMX500/SensecaLPS10/ApogeeSI111 三条 bib 条目的确认

需要在 `paper.tex` 中搜索这三个 cite key 是否有对应的 `\citet{}` 或 `\citep{}` 命令。如果没有，应从 `references.bib` 中删除这三条条目，避免 bibtex 警告和审稿人质疑。

---

## 第七部分：时间线建议

基于 pilot 速度（5 seeds，B=1.70，约 1–2 小时）和完整 S2 规模（30 runs），预计：

- S2 main rerun（30 runs，3–4 worker 并行）：约 6–10 小时
- S2 condition evaluation（基于已有 truth，无需重训）：约 1–2 小时
- 中间检查和诊断：约 1 小时
- 表格/图生成：约 1–2 小时
- 论文改造（如果 S2 成功）：约 2–3 小时

**建议时间线**：
- 今天：完成 turn_61 三处修正确认 + 冻结 fallback PDF + 启动 S2 main rerun
- 明天上午：检查 B=1.65 结果，决定是否继续
- 明天下午：S2 全部完成后运行 collector + 统计检验
- 后天：根据 S2 结果决定是否替换主线，完成论文改造

这个时间线与 turn_60 的"一周内完成可投稿初稿"评估完全兼容。

---

## 总结

当前状态是：V2 主结果已经支撑一篇可投稿的论文，S2 是锦上添花而非必要条件。推荐的执行顺序是：

1. 先完成 turn_61 三处修正（不依赖 S2，今天可完成）；
2. 冻结 fallback PDF；
3. 启动 S2 main rerun，使用断点续跑脚本；
4. S2 运行期间并行处理 A1/A2 消融路径差异的正文解释和 H1 措辞修正；
5. S2 完成后按最小成功标准判断是否替换主线；
6. 无论 S2 结果如何，论文均可在本周内提交。
