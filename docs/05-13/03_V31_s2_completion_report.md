# V3.1 S2 完成报告与论文推进建议

Last updated: 2026-05-15

本报告承接 `docs/05-13/02_V31_rerun_report_and_paper_plan.md` 和
`docs/05-13/00_TODO_goal_required_experiments_and_paper_fixes.md`。完整 S2 已在
服务器 `remote-gpu` 的 tmux 任务 `v31_s2_main` 中跑完，并已同步回本地：

- 结果目录：`reports/v31_s2_main/`
- 完成标记：`reports/v31_s2_main/done/*.done`
- 完成数量：`30/30`，即 3 个预算 x 10 个 seeds。
- 汇总表：
  - `reports/v31_s2_main/v31_s2_overall_long.csv`
  - `reports/v31_s2_main/v31_s2_main_stats.csv`
  - `reports/v31_s2_main/v31_s2_condition_stats.csv`
  - `reports/v31_s2_main/v31_s2_significance.csv`
  - `reports/v31_s2_main/v31_s2_budget_check.csv`

## 1. 主结果概览

| Budget | Full obs. | Static proj. | PD-PPO | Round-robin | AoI | Random |
|---:|---:|---:|---:|---:|---:|---:|
| 1.65 | 0.1487 ± 0.0118 | 0.1593 ± 0.0116 | 0.1628 ± 0.0140 | 0.1671 ± 0.0138 | 0.1700 ± 0.0139 | 0.1803 ± 0.0142 |
| 1.70 | 0.1493 ± 0.0123 | 0.1597 ± 0.0119 | 0.1620 ± 0.0142 | 0.1674 ± 0.0141 | 0.1844 ± 0.0191 | 0.1862 ± 0.0173 |
| 1.75 | 0.1502 ± 0.0123 | 0.1612 ± 0.0115 | 0.1661 ± 0.0145 | 0.1687 ± 0.0137 | 0.1933 ± 0.0184 | 0.1914 ± 0.0163 |

直接观察：

- Full-observation upper bound 在三个 budgets 上均为最低误差，直觉上界恢复正常。
- PD-PPO 在三个 budgets 上均优于 round-robin、AoI 和 random。
- PD-PPO 接近但弱于 oracle-informed feasible static projection。
- PD-PPO 相对 static projection 的差距分别为 `2.24%`、`1.47%`、`3.09%`。

## 2. 统计检验解释

`v31_s2_significance.csv` 同时给出原始双侧检验 p 值和 Bonferroni 校正后的 p 值。
需要注意：

- `p_value_bonferroni` 是已经乘以 family size 的校正 p 值，应与 `0.05` 比较。
- 若使用 `alpha_adj = 0.05 / 6 = 0.0083`，则应比较原始 `p_value_two_sided`。

因此，PD-PPO 对 AoI 和 random 的优势在主要预算上是显著的：

- B=1.65: PD-PPO vs AoI 校正 p=0.0352，通过；vs random 校正 p=0.0117，通过。
- B=1.70: PD-PPO vs AoI 校正 p=0.0117，通过；vs random 校正 p=0.0117，通过。
- B=1.75: PD-PPO vs AoI 校正 p=0.0117，通过；vs random 校正 p=0.0117，通过。

PD-PPO 对 round-robin 的均值优势成立，但统计显著性较弱：

- B=1.65: 校正 p=0.0586。
- B=1.70: 校正 p=0.0586。
- B=1.75: 校正 p=0.7852。

论文中应写成：

> PD-PPO consistently improves mean FW-MAE over round-robin, AoI-based, and random
> feasible scheduling across all tested budgets. The improvements over AoI-based and
> random scheduling are Bonferroni-significant, while the margin over round-robin is
> smaller and not uniformly significant.

不应写成：

> PD-PPO significantly beats every baseline at every budget.

## 3. Condition-stratified 结果

以主预算 `B=1.70` 为例：

| Condition | Full obs. | Static proj. | PD-PPO | Round-robin | AoI | Random |
|---|---:|---:|---:|---:|---:|---:|
| Event | 0.1560 | 0.1664 | 0.1692 | 0.1747 | 0.1928 | 0.1940 |
| Non-event | 0.1123 | 0.1214 | 0.1212 | 0.1259 | 0.1352 | 0.1407 |
| Low temp. | 0.1215 | 0.1325 | 0.1319 | 0.1364 | 0.1423 | 0.1461 |
| Normal temp. | 0.1230 | 0.1317 | 0.1312 | 0.1360 | 0.1507 | 0.1602 |

解读：

- PD-PPO 在 event、non-event、low-temperature、normal-temperature 四个切片中均优于朴素基线。
- PD-PPO 在部分切片中接近 static projection，尤其是 non-event 和 normal-temperature。
- 已在后续补充 `scripts/44_v31_event_heavy_collect.py`，按 512 步窗口的
  event fraction 重新切分为 `calm < 0.25`、`mixed 0.25--0.75`、
  `event_heavy > 0.75`。因此论文可以使用独立 event-heavy table，但应明确
  它是窗口级 event-fraction 分层，而不是单独重训的极端风暴专用实验。

## 4. 成功标准判定

| Criterion | Status | Comment |
|---|---|---|
| Full-open is best upper bound at all budgets | Pass | 三个 budgets 均成立。 |
| PD-PPO beats round-robin, AoI, random in mean FW-MAE | Pass | 三个 budgets 均成立。 |
| PD-PPO vs AoI significant in at least two budgets | Pass | 校正 p 值均小于 0.05。 |
| PD-PPO close to static projection | Near-pass | B=1.65 和 B=1.70 小于 3%；B=1.75 为 3.09%。 |
| Event-condition PD-PPO beats naive baselines | Pass | B=1.70 event slice 成立；其他 budgets 也保持同样趋势。 |
| Separate event-heavy table | Pass | 已补充 `v31_s2_event_fraction_stats.csv` 与论文表格。 |

结论：S2 不再只是 pilot，已经足以作为 V3.1 主结果候选。唯一需要论文中保守处理的是：

- 不写 “within 3% of static at all budgets”，改写为 “within about 3.1%” 或
  “within 1.5--3.1% across budgets”。
- 不写 “significantly beats round-robin”，改写为 “consistently improves mean FW-MAE over round-robin”。
- 可写 “event-heavy windows”，但需说明分层方式为 512-step window event fraction
  `>0.75`。

## 5. 已生成论文候选资产

本轮已补齐 V3.1 专用资产，并已开始覆盖当前正文入口：

- `paper/tables/main_results_v31.tex`
- `paper/tables/condition_results_v31.tex`
- `paper/figures/figure6_power_error_tradeoff_v31.png`
- `paper/figures/figure6_power_error_tradeoff_v31.svg`

已执行的论文替换要点：

1. 已将 `paper/sections/04_simulation_environment.tex` 中旧式“V3.1 仍是后续验证门槛”
   的说法改为 “V3.1 is the final generator used for S2 main results”。
2. 已将 `paper/sections/06_experiments.tex` 中主结果入口从 V2 表切换到 V3.1 表。
3. 已把 main result 文本数字替换为 V3.1：
   - Primary budget B=1.70: PD-PPO `0.1620 ± 0.0142`。
   - Static projection: `0.1597 ± 0.0119`。
   - Full-open upper bound: `0.1493 ± 0.0123`。
   - Round-robin: `0.1674 ± 0.0141`。
   - AoI: `0.1844 ± 0.0191`。
   - Random: `0.1862 ± 0.0173`。
4. Discussion 中保留 V2 的历史位置，但已把它降级为开发诊断，而不是主结果。
5. 已补充 event-heavy collector，正文可切换为 V3.1 event-fraction-stratified
   condition table。

## 6. 2026-05-15 更新：V3.1 主线切换状态

已完成：

- 备份论文当前状态：`paper` 子仓库 commit
  `94f18a1 backup paper before V3.1 mainline switch`。
- 服务器 S2 与 event-heavy 汇总均已完成，当前无相关训练或 tmux 进程。
- 新增并运行 `scripts/44_v31_event_heavy_collect.py`。
- 同步回本地：
  - `reports/v31_s2_main/v31_s2_event_fraction_long.csv`
  - `reports/v31_s2_main/v31_s2_event_fraction_stats.csv`
  - `reports/v31_s2_main/v31_s2_event_fraction_check.csv`
  - `reports/v31_s2_main/v31_event_heavy_tmux.log`
- 论文已切换主结果入口：
  - `paper/tables/main_results_v31.tex` 使用 label `tab:main_results`。
  - `paper/tables/condition_results_v31.tex` 使用 event-fraction strata 并 label
    `tab:e1_condition`。
  - `paper/sections/06_experiments.tex` 主结果、预算敏感性、条件分层均使用
    V3.1 S2 数字。
  - 旧 V2 的 E2/A1/H1/DQN 结果降级为 development diagnostics，不再作为主结果。

event-heavy 主预算 `B=1.70` 的核心结果：

| Stratum | Full obs. | Static proj. | PD-PPO | Round-robin | AoI | Random |
|---|---:|---:|---:|---:|---:|---:|
| Calm (`<0.25`) | 0.1145 | 0.1247 | 0.1240 | 0.1289 | 0.1485 | 0.1502 |
| Mixed (`0.25--0.75`) | 0.1640 | 0.1735 | 0.1776 | 0.1832 | 0.1988 | 0.2002 |
| Event-heavy (`>0.75`) | 0.1822 | 0.1979 | 0.2008 | 0.2058 | 0.2222 | 0.2269 |

仍需收尾：

- 编译 `paper.tex`，检查是否存在引用、表格和图宽问题。
- 再次全文搜索旧 V2 数字，确认只保留在明确标注为 diagnostic 的段落或未引用旧表中。
- 如编译成功，可考虑提交一次 `paper` 子仓库的新 commit，作为 V3.1 主线切换点。

## 7. 2026-05-16 更新：V3.1-aligned A1/A2/H1 消融完成

服务器恢复后，先前断电中断的 V3.1-aligned 消融已完成并同步回本地：

- 结果目录：`reports/v31_ablation_aligned/`
- 完成标记：`reports/v31_ablation_aligned/done/*.done`
- 完成数量：`150/150` 个 eval 目录。
- collector completion：
  - A1 `80/80`
  - A2 `40/40`
  - H1 `45/45`

核心结果：

| Experiment | Key result |
|---|---|
| A2 staged diagnostic | Full PD-PPO (D4) reaches `0.1629 ± 0.0137`, better than D1/D2/D3. |
| A1 remove-one | Removing AWBC and oracle prior jointly worsens FW-MAE by `13.8%` and is Bonferroni-significant. |
| A1 masked-only | MaskedActor-only worsens FW-MAE by `12.2%` and is Bonferroni-significant. |
| A1 single removals | Removing ActionEmbedding, EventAwareCritic, or action mask alone is not significant in this batch. |
| H1 sensitivity | All tested AWBC/KL cells remain within `2.5%` of default; best mean is `(0.1, 0.5)` rather than the default `(0.1, 1.0)`. |

论文处理：

- `paper/tables/ablation.tex` 已切换为 V3.1-aligned A2 数字。
- `paper/tables/ablation_full.tex` 已切换为 V3.1-aligned A1 数字。
- `paper/figures/figure_h1_heatmap.png` 已切换为 V3.1-aligned H1 heatmap。
- `paper/sections/06_experiments.tex` 已将 A1/A2/H1 从 V2 development diagnostics 改为 V3.1-aligned diagnostics。
- 解释口径保持保守：AWBC + oracle prior 是有统计支持的主要稳定化组合；EventAwareCritic/ActionEmbedding/action mask 作为结构组件保留，但不声称其单独显著提升性能。
