# TODO: V3.1 必须实验补充与论文修改执行清单

Goal: 完成现阶段所有必须的实验补充和论文修改，使论文进入“数字可追溯、声称不过界、图表正文一致”的可投稿初稿状态；同时为 V3.1 主结果切换保留清晰、可恢复的执行入口。

Last updated: 2026-05-15

Active planning entry point:

- `task_plan.md`: current phase-based plan for paper closure and V3.1 evidence alignment.
- `findings.md`: durable summary of experimental findings, constraints, and decisions.
- `progress.md`: chronological execution log.

This document is now preserved as the historical TODO ledger. Use the three root-level planning files above for day-to-day continuation and context recovery.

Primary references:

- `docs/05-13/02_V31_rerun_report_and_paper_plan.md`: V3.1 重跑报告与论文改造总计划。
- `docs/05-14-V3-1-suggestion.md`: S2 启动前检查、S2 脚本拆分、断点续跑和判定标准建议。
- `docs/05-14_phase1_a1_h1_g1_report.md`: Phase 1 A1/H1/G1 当前结果报告。
- `reports/v31_pilot/v31_pilot_summary.csv`: V3.1 pilot 汇总结果。

Current state snapshot:

- [x] Phase 0 paper/table/figure consistency pass completed locally.
- [x] `paper.tex` 已能用 XeLaTeX 编译，且此前检查未发现 undefined references/citations。
- [x] Phase 1 A1/H1 completed on the GPU server and synced locally under `reports/v3_supplement_assets/`.
- [x] G1 V3.1 generator validation completed locally and passed current validation criteria.
- [x] G1 V2/V3.1 diagnostic comparison completed by `scripts/40_v2_compare_v31_generator.py`.
- [x] V3.1 pilot completed: `B=1.70`, seeds 41--45, V3.1 semi-Markov truth, per-seed TCN oracle retraining, PD-PPO retraining, baseline evaluation.
- [x] Full V3.1 S2 runner added: `scripts/42_v31_s2_full.py`.
- [x] Full V3.1 S2 collector added: `scripts/43_v31_s2_collect.py`.
- [x] Full V3.1 S2 main experiment completed on server tmux `v31_s2_main` and synced locally.
- [x] V3.1 S2 results integrated into the manuscript as the main result line.
- [x] V3.1 S2 completion report added: `docs/05-13/03_V31_s2_completion_report.md`.

## 0. 不可越界原则

- [x] 不使用和 CSV 不一致的 PDF 手抄数字。
- [x] 不把 V2 的 event-heavy condition 说成极端风暴验证；V2 event fraction 上限约为 0.58。
- [x] 不在 S2 完成前声称 V3.1 policy performance。
- [x] 不把 `feasible_static_projected` 写成在线可部署策略；它是强可行参考，不是部署主算法。
- [x] 不把 `full_open_unconstrained` 写成满足预算约束的策略；它是直观上界。
- [x] 正文正式名称统一为 PD-PPO；`custom_ppo` 只允许出现在脚本、CSV、复现实验说明中。
- [x] `raw.tex` 不是主论文源；正式论文修改以 `paper/paper.tex` 和 `paper/sections/*.tex` 为准。

## 1. Phase 0 投稿 fallback 检查清单

### 1.1 数字与表格

- [x] Main Results Table 2 使用 S1 CSV 数字。
- [x] A2 staged diagnostic table 替换旧 A1-style ablation。
- [x] E1 fixed condition evaluation 使用固定 CSV 数字。
- [x] E2 oracle robustness 表述为 oracle-capacity sanity check。
- [x] P1 physical-unit MAE 使用物理单位，不与 normalized FW-MAE 混写。
- [x] 重新跑一次文本检查，确认没有残留旧数字和旧说法。

Text checks:

```bash
grep -R "Custom PPO" paper/sections paper/tables paper/algorithms
grep -R "\\[FILL\\]\\|FILL" paper
grep -R "3 seeds\\|remove one component" paper/tables paper/sections
```

### 1.2 统计与叙事

- [x] Wilcoxon signed-rank test + Bonferroni correction 写入实验部分。
- [x] 不声称 “PD-PPO significantly beats all baselines at all budgets”。
- [x] EventAwareCritic 在 V2 中写为“当前环境下不显著/贡献不稳定”，不写成确认驱动因素。
- [x] AWBC 与 oracle-calibrated prior 写为主要稳定化来源。
- [x] 检查 A1/A2 同时存在时的口径：A2 是 staged diagnostic，A1 是 matched-seed remove-one-component ablation。

### 1.3 传感器硬件口径

`docs/05-14-V3-1-suggestion.md` 要求启动 S2 前确认 turn_61 三处口径。检查重点如下：

- [x] `paper/tables/sensor_specs.tex`: power values 写成 normalised deployment cost units informed by datasheet electrical characteristics，不写成严格 bench-calibrated absolute watt values。
- [x] `paper/sections/03_problem_formulation.tex`: GMX500、Parsivel2、FC4 的真实电气参数只用于 motivate/inform normalized cost model，不直接等同仿真 budget。
- [x] `paper/sections/04_simulation_environment.tex`: 说明仿真使用 normalized relative costs；真实 watt ratios 只是设计依据。
- [x] 删除或补足 `GillGMX500`、`SensecaLPS10`、`ApogeeSI111` 等 bib 条目的正文引用，避免无用文献堆叠。
- [x] 正文无 `to be filled`、`[FILL]` 等投稿面向占位符。

### 1.4 fallback 冻结

- [x] 编译当前论文。
- [x] 保存 fallback PDF：`paper/paper_v2_fallback_20260514.pdf`。
- [x] 记录 fallback 对应 git diff 或当前修改状态。
- [x] S2 已完成且满足主线切换标准；主结果已切换为 V3.1。

## 2. Phase 1 已完成/待确认清单

### 2.1 A1 Full Component Ablation

Status:

- [x] A1/H1 已由 Phase 1 runner/collector 完成并同步。
- [x] 结果位于 `reports/v3_supplement_assets/`。
- [x] 若正文加入 A1 remove-one-component 表格，需再次确认 matched-seed、n、Bonferroni 口径。

Acceptance reminder:

- [x] EventAwareCritic 仍不得写成显著性能驱动。
- [x] `no_awbc_oracle` 解释为 AWBC/prior interaction，而不是简单线性加和。

### 2.2 H1 Hyperparameter Sensitivity

Status:

- [x] H1 已完成。
- [x] 正文只允许写“tested cells remain within/around default-scale variation”之类保守表述。
- [x] 不写 broad robustness / hyperparameter insensitive，除非 CSV 明确支持。
- [x] H1 default 说明为 diagnostic training batch 默认设置在 diagnostic seeds 上的局部检查。

### 2.3 G1 V3.1 Generator Validation

Status:

- [x] `scripts/39_v2_validate_v31_generator.py` 已完成 V3.1 generator validation。
- [x] `scripts/40_v2_compare_v31_generator.py` 已完成 V2/V3.1 diagnostic comparison。
- [x] `reports/v3_supplement_assets/exp_g1_v2_v31_generator_comparison.csv` 已生成。

Key matched diagnostics:

- V2 clustered: ACF delta `0.174999`, `P(event_fraction>0.75)=0.0`, max event fraction `0.488281`, max KS `0.224590`, flux-wind slope `6.27148`。
- V3.1 semi-Markov: ACF delta `0.048140`, `P(event_fraction>0.75)=0.064600`, max event fraction `0.810547`, max KS `0.014625`, flux-wind slope `2.96576`。

Acceptance reminder:

- [x] V3.1 是 generator version，不是算法版本。
- [x] Semi-Markov 参数写为 Amory-inspired/calibrated statistics，不写成直接从 Amory transition matrix 测得。
- [x] Blowing-snow variables 使用 conditional statistics，不对 AntAWS 做不合理 KS。

## 3. V3.1 Pilot 结论

Pilot scope:

- Budget: `B=1.70`
- Seeds: `41--45`
- Truth: V3.1 semi-Markov generator
- Oracle: per-seed TCN oracle retraining
- Policy: per-seed PD-PPO retraining
- Outputs: `reports/v31_pilot/`

Pilot mean FW-MAE:

| Policy | Mean FW-MAE | Std | Mean power |
|---|---:|---:|---:|
| full_open_unconstrained | 0.4130 | 0.0333 | 4.620 |
| feasible_static_projected | 0.4352 | 0.0419 | 1.460 |
| PD-PPO / custom_ppo | 0.4398 | 0.0425 | 1.511 |
| round_robin | 0.4501 | 0.0430 | 1.555 |
| AoI | 0.4819 | 0.0484 | 1.623 |
| random | 0.4827 | 0.0445 | 1.604 |

Pilot decision:

- [x] PD-PPO beats round_robin, AoI, and random in pilot.
- [x] Full-open returns to the best intuitive upper bound.
- [x] PD-PPO remains close to feasible_static_projected.
- [x] Proceed to full S2 is justified.

## 4. Full S2 执行清单

### 4.1 本地准备

- [x] Add `scripts/42_v31_s2_full.py`.
- [x] Add `scripts/43_v31_s2_collect.py`.
- [x] `42_v31_s2_full.py` 支持 budgets x seeds：默认 budgets `[1.65, 1.70, 1.75]`，seeds `41--50`。
- [x] `42_v31_s2_full.py` 每个 `(budget, seed)` 成功后写 `.done` 文件。
- [x] `43_v31_s2_collect.py` 可在运行中随时汇总已有结果。
- [x] 本地 dry-run 检查 42 的命令构造。
- [x] 本地用 pilot 目录检查 43 的收集与统计逻辑。

### 4.2 推荐输出结构

```text
reports/v31_s2_main/
  raw/
    budget1p65_seed41/
    budget1p65_seed42/
    ...
  done/
    budget1p65_seed41.done
    budget1p65_seed42.done
    ...
  logs/
    budget1p65_seed41.log
    ...
  v31_s2_overall_long.csv
  v31_s2_main_stats.csv
  v31_s2_significance.csv
  v31_s2_condition_long.csv
  v31_s2_condition_stats.csv
  v31_s2_budget_check.csv
```

### 4.3 服务器启动

- [x] 同步本地代码到服务器：`remote-gpu:~/_code/microclimate_demo/rl_sensor_scheduling_framework`。
- [x] 检查服务器 tmux、GPU、已有 `reports/v31_s2_main/`。
- [x] 在 tmux 中启动 S2，不直接前台跑长任务。
- [x] 使用 3 workers；当前启动在空闲 GPU `1,4,5` 上运行，避开忙卡 `0,2,3`。
- [x] 按 budget 顺序运行并收集：`1.65 -> 1.70 -> 1.75`。

### 4.4 每个 budget 完成后的中间检查

- [x] `full_open_unconstrained` 是三个 budgets 中 mean FW-MAE 最低的策略。
- [x] PD-PPO mean FW-MAE 在三个 budgets 中均低于 round_robin。
- [x] PD-PPO mean FW-MAE 在三个 budgets 中均低于 AoI。
- [x] PD-PPO mean FW-MAE 在三个 budgets 中均低于 random。
- [x] Near-pass accepted with conservative wording: PD-PPO 相对 feasible_static_projected 的 gap 接近 3%：`2.24%`, `1.47%`, `3.09%`；正文写为 “within about 3.1% across budgets”。
- [x] `B=1.65` 满足主趋势，因此完整 S2 已跑完。

## 5. S2 成功/降级判定

Full success criteria:

- [x] full_open_unconstrained 在三个 budgets 上都是最佳上界。
- [x] PD-PPO 在三个 budgets 上均优于 round_robin、AoI、random。
- [x] PD-PPO vs AoI 至少两个 budgets 在 Wilcoxon signed-rank test + Bonferroni 后显著。
- [x] 若 family 仍按 6 个 pairwise comparisons，使用 `alpha_adj = 0.05/6 ~= 0.0083`；正文采用 Bonferroni-corrected pairwise wording，不声称 round-robin 全显著。
- [x] Near-pass: PD-PPO vs feasible_static_projected gap 在三个 budgets 上接近但不完全低于 3%；`B=1.75` 为 `3.09%`，正文已保守写为 “within about 3.1%”。
- [x] Event condition 中 PD-PPO 优于朴素基线。
- [x] 独立 event-heavy stratum 已由 `scripts/44_v31_event_heavy_collect.py` 输出；正文使用 512-step window event fraction `>0.75` 口径。

Degraded but usable（本轮未采用，因为 S2 已满足主线切换条件）:

- [x] 未触发：PD-PPO 在三个 budgets 上均优于所有朴素可部署基线。
- [x] 未触发：PD-PPO 未出现不稳定到需要降级为 supplement 的情况。
- [x] 未触发：full_open_unconstrained 是三个 budgets 上的最佳上界。

## 6. 论文切换清单

Only after S2 passes:

- [x] 用 S2 表格替换主结果表。
- [x] 用 V3.1 condition-stratified 表替换 V2 condition 表，或明确并列说明。
- [x] 用 V3.1 generator validation 图/表支撑仿真环境可信度。
- [x] Introduction 的贡献从 “V2 fallback results” 切换为 “V3.1 main results”。
- [x] Discussion 中保留 V2 -> V3.1 的修正脉络：V2 发现问题，V3.1 修正事件结构与物理耦合。
- [x] 删除或降级所有 S2 不支持的旧强声明。

If S2 does not pass（未采用分支）:

- [x] 未触发：V2 不再保持主结果。
- [x] 未触发：V3.1 已作为主结果而非仅 generator correction/pilot/future work。
- [x] 未触发：S2 通过后无需将 V3.1 降级为 future-work-only。

## 7. 当前下一步

Immediate execution order:

1. [x] 完成 turn_61 三处硬件口径检查。
2. [x] 冻结当前 fallback PDF。
3. [x] 新增并验证 `scripts/42_v31_s2_full.py`。
4. [x] 新增并验证 `scripts/43_v31_s2_collect.py`。
5. [x] 同步服务器。
6. [x] tmux 启动 full S2。
7. [x] 每完成一个 budget 拉回/汇总一次结果。
8. [x] 根据 `v31_s2_budget_check.csv` 完成初步判定：V3.1 S2 可作为主结果候选，但需保守处理 static gap 和 round-robin 显著性。
9. [x] 生成 V3.1 论文候选资产：`main_results_v31.tex`、`condition_results_v31.tex`、`figure6_power_error_tradeoff_v31.{png,svg}`。
10. [x] 决定是否将 `paper/sections/06_experiments.tex` 主入口切换到 V3.1 资产。
11. [x] 若切换主线，更新 Introduction / Simulation / Experiments / Discussion / Conclusion 中所有 V2 数字和 “future S2” 表述。

## 8. 2026-05-15 TODO 复查结论

现阶段“必须补齐”的实验已经完成：

- Main S2：V3.1 semi-Markov generator，3 budgets x 10 seeds，已完成并作为论文主结果。
- Event-heavy：已补充 512-step window event-fraction 分层，`event_heavy > 0.75` 已进入正文表格。
- A1 remove-one-component ablation：已完成，8 个配置，full PD-PPO + 7 个移除变体，每个 n=10 seeds。
- A2 staged diagnostic ablation：已完成并作为组件路径诊断保留。
- H1 hyperparameter sensitivity：已完成，`awbc_coef x prior_kl_coef` 的 3x3 局部敏感性检查。
- G1 generator validation：已完成，V3.1 修正 V2 event-heavy 不可达与风速时序统计扰动问题。

当前已升级为执行项的实验：

- V3.1 上重新跑完整 A1/A2/H1 消融：用户于 2026-05-15 明确要求补齐，使所有消融和最终主生成器完全一致。执行计划见第 9 节。

当前仍不建议作为“必须项”继续跑的实验：

- 更大 seeds 的 S2：可增强统计功效，尤其是 PD-PPO vs round-robin 的显著性，但当前 n=10 已支持主表均值优势和 AoI/random 显著优势。
- 跨站点/真实 AntAWS 验证、硬件闭环、长期平均能耗约束：应作为下一阶段或 future work，而不是当前投稿前必须补齐项。

## 9. V3.1-aligned A1/A2/H1 消融重跑（已完成，可并入论文）

Goal:

- 在最终 V3.1 semi-Markov truth generator、最终 S2 truth 序列、最终 TCN oracle/PD-PPO 训练参数下，重新运行完整 A1/A2/H1。
- 让消融表从 “V2/development diagnostics” 升级为 “V3.1 mainline-aligned diagnostics”。
- 输出可直接进入论文 supplement 或正文消融小节的 CSV、统计表和图。

Current decision:

- [x] 2026-05-15 服务器断电时该部分曾无法验收。
- [x] 服务器恢复后，已在 tmux `v31_ablation_aligned` 中断点续跑。
- [x] 续跑完成后，已同步 `reports/v31_ablation_aligned/` 到本地。
- [x] completion check 通过：A1 `80/80`、A2 `40/40`、H1 `45/45`。
- [x] A1/A2/H1 已可从 V2/development diagnostics 升级为 V3.1-aligned diagnostics。
- [x] 论文中 A1/A2/H1 的表格、H1 heatmap 和正文解释已切换到 V3.1-aligned 结果。
- [x] 叙事保持保守：AWBC + oracle prior 的联合稳定化贡献显著；ActionEmbedding、EventAwareCritic、action mask 的单独移除在当前 V3.1-aligned batch 中不显著。

Execution design:

- [x] 新增 runner：`scripts/45_v31_aligned_ablation.py`。
- [x] 新增 collector：`scripts/46_v31_aligned_ablation_collect.py`。
- [x] Truth 来源锁定为 `reports/v31_s2_main/raw/budget1p70_seed*/truth_v31.csv`。
- [x] 训练参数锁定为 S2 参数：`truth_steps=30000`、`freq_s=3600`、`event_coverage=0.28`、semi-Markov event model、TCN oracle、`total_timesteps=100000`、`eval_rollouts=6`。
- [x] A2-D4 full PD-PPO 作为 A1/H1 的 matched full reference，避免再跑一套重复 full-reference。
- [x] 同步脚本和必要 v2 代码到服务器。
- [x] 在服务器 tmux `v31_ablation_aligned` 中运行。
- [x] 跑完后拉回 `reports/v31_ablation_aligned/`。
- [x] 检查 completion：A1 `80/80` rows、A2 `40/40` rows、H1 `45/45` rows。
- [x] 如果通过，更新论文：A1/A2/H1 不再称为 V2 development diagnostics，而改为 V3.1-aligned diagnostics。

Final V3.1-aligned summary:

- A2 staged diagnostic (`n=10`): D1 `0.1821±0.0147`, D2 `0.1939±0.0179`, D3 `0.1788±0.0176`, D4/full `0.1629±0.0137`。
- A1 remove-one (`n=10`): full `0.1629±0.0137`; no AWBC/prior `0.1853±0.0209`, significant under Bonferroni; masked-only `0.1828±0.0156`, significant under Bonferroni。
- H1 sensitivity (`n=5` per cell): default `(AWBC=0.1, KL=1.0)` is `0.1616±0.0138`; all tested cells remain within 2.5% of default; lowest mean is `(0.1, 0.5)` at `0.1599±0.0123`。

Run scope:

- A1 remove-one-component: full reference + 7 个移除变体，`B=1.70`，seeds `41--50`。
- A2 staged diagnostic: D1/D2/D3/D4，`B=1.70`，seeds `41--50`。
- H1 hyperparameter sensitivity: `awbc_coef in {0.05,0.10,0.20}` x `prior_kl_coef in {0.5,1.0,2.0}`，默认 cell 由 A2-D4 提供，seeds `41--45`。

Expected output:

```text
reports/v31_ablation_aligned/
  raw/
  done/
  logs/
  v31_aligned_a1_raw.csv
  v31_aligned_a1_stats.csv
  v31_aligned_a2_raw.csv
  v31_aligned_a2_stats.csv
  v31_aligned_h1_raw.csv
  v31_aligned_h1_stats.csv
  v31_aligned_completion_check.csv
  figures/
```
