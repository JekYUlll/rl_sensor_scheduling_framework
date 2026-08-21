# V3.1 重跑与论文改造：报告和执行计划

Last updated: 2026-05-14

本文档基于 `docs/05-13/00_TODO_goal_required_experiments_and_paper_fixes.md`、`docs/05-14-V3-1-suggestion.md`、Phase 1 A1/H1/G1 结果，以及刚完成的 V3.1 pilot 结果，给出下一阶段是否切换到 V3.1 主结果、如何重跑、以及论文叙事如何改造的综合计划。

`docs/05-14-V3-1-suggestion.md` 的关键建议已经合并到本文档：S2 启动前冻结可投稿稿、确认 turn_61 三处硬件口径修正、为完整 S2 新建独立脚本、加入 `.done` 断点续跑、独立 collector、按 budget 分阶段检查，以及用明确的成功/降级标准决定是否替换论文主线。

2026-05-15 update: 完整 V3.1 S2 已完成并同步回本地。详细结果、成功标准判定和论文切换建议见
`docs/05-13/03_V31_s2_completion_report.md`。简要结论是：V3.1 S2 已足以作为主结果候选；
full-open 在三个 budgets 上均恢复为最佳上界，PD-PPO 在均值上优于 round-robin、AoI 和 random，
且对 AoI/random 的优势通过 Bonferroni 校正；但 PD-PPO 相对 feasible static projection 的最大差距为
`3.09%`，论文中应写成 “within about 3.1%” 而不是 “within 3% at all budgets”。

## 1. 当前论文状态

当前论文已经达到“可投稿初稿”的基本状态，但主结果仍基于 V2 生成器。

已完成：

- Phase 0：主表、统计检验、图表引用、PD-PPO 命名、符号和硬件参数口径已经完成一轮一致性修正。
- A1：完整组件消融已完成，`n=10` seeds，证明 ActionEmbedding、AWBC、oracle prior、action mask 均有显著贡献。
- H1：AWBC 与 oracle-prior KL 权重敏感性已完成，所有测试格点在默认值 +5% 内。
- G1：V3.1 生成器统计验证已完成，说明 semi-Markov storm-regime 生成器能够修复 V2 的关键生成器缺陷。
- S2 pilot：V3.1 最小闭环重训已完成，`B=1.70`、`n=5` seeds，PD-PPO 仍优于所有朴素可部署基线。
- `paper.tex` 已可用 XeLaTeX 编译，无 undefined reference/citation。

当前主论文的主结果仍是：

- V2 生成器；
- 3 budgets：`B=1.65, 1.70, 1.75`；
- `n=10` seeds；
- PD-PPO 优于 round-robin、AoI、random；
- PD-PPO 接近 oracle-informed static projection；
- full-open unconstrained 是不可部署上界。

## 2. 当前实验的主要局限

### 2.1 V2 生成器的事件结构过弱

V2 生成器通过 clustered / injected intervals 构造吹雪事件。这个机制可控，但带来两个明显局限：

- 512-step 窗口中的 event fraction 最高约 `0.58`，无法产生真正 event-heavy 的 `>0.75` 窗口。
- wind-speed 序列在事件注入后会偏离 AntAWS anchor 的 ACF / PSD 结构。

G1 对比结果已经量化了这个问题：

| Generator | ACF delta | P(ef>0.75) | Max 512-step event fraction | Max KS |
|---|---:|---:|---:|---:|
| V2 clustered diagnostic | 0.175 | 0.000 | 0.488 | 0.2246 |
| V3.1 semi-Markov diagnostic | 0.048 | 0.065 | 0.811 | 0.0146 |

这意味着：V2 主结果虽然能支持“在当前仿真环境中 PD-PPO 有效”，但不足以强力支持“在持续风暴和高吹雪事件占比下 PD-PPO 的动态 event-aware 优势”。

### 2.2 EventAwareCritic 在 V2 中难以显著

A2 staged diagnostic 中，EventAwareCritic 的增益约 `-0.1%`；A1 full ablation 中，移除 EventAwareCritic 造成 `+2.38%` FW-MAE，但 Bonferroni 校正后不显著。

这并不意味着 EventAwareCritic 设计错误，更可能说明 V2 环境没有提供足够强的事件上下文变化。

如果继续使用 V2 作为唯一主结果，论文中必须保守表述：

- EventAwareCritic 是机制上合理的设计；
- 但其独立统计显著性在 V2 中尚未充分体现；
- 更强 event-heavy 生成器是后续验证方向。

### 2.3 Condition-stratified 结果仍不是极端风暴验证

当前 E1 condition evaluation 的 event stratum 实际上是 `event fraction in [0.40, 0.58]`，不是原本想要的 `>0.75` event-heavy storm。

因此，当前论文不能写成：

> PD-PPO 在极端吹雪风暴中已经验证有效。

只能写成：

> PD-PPO 在 V2 可达到的 higher-event-fraction windows 中保持优势；更强 event-heavy validation 需要 V3.1 全量重跑。

## 3. V3.1 的定义

V3.1 是新的合成数据生成器版本，不是新的调度算法版本。

V3.1 的核心变化：

- `blowing_snow_event_model="semi_markov"`；
- 使用 semi-Markov storm-regime latent process 生成持续风暴状态；
- 加入 CRED-style hysteresis；
- 加入 event minimum duration / minimum gap；
- 质量通量使用 wind-speed power-law coupling；
- 保留 AntAWS-anchored DFT base meteorological sequence；
- 对 base meteorological variables 使用 KS / ACF / PSD 验证；
- 对 blowing-snow variables 使用 conditional event statistics 验证。

当前代码入口：

- truth 生成支持：`scripts/20_build_public_weather_truth.py`
- PD-PPO 训练支持：`scripts/25_v2_train_custom_ppo.py`
- G1 统计验证：`scripts/39_v2_validate_v31_generator.py`
- V2/V3.1 生成器对比：`scripts/40_v2_compare_v31_generator.py`
- V3.1 pilot 重训：`scripts/41_v31_pilot.py`

## 4. V3.1 pilot 结果

数据源：

- `reports/v31_pilot/v31_pilot_overall_long.csv`
- `reports/v31_pilot/v31_pilot_summary.csv`

设计：

- `B=1.70`
- seeds `41--45`
- 每个 seed 独立生成 V3.1 truth
- 每个 seed 重新训练 TCN oracle
- 每个 seed 重新训练 PD-PPO
- 同场评估 full-open、static projection、round-robin、AoI、random

结果：

| Policy | FW-MAE mean | FW-MAE std | Event FW-MAE | Non-event FW-MAE | Power mean | n |
|---|---:|---:|---:|---:|---:|---:|
| full_open_unconstrained | 0.4130 | 0.0333 | 0.4413 | 0.1881 | 4.6200 | 5 |
| feasible_static_projected | 0.4352 | 0.0419 | 0.4634 | 0.2121 | 1.4600 | 5 |
| PD-PPO (`custom_ppo`) | 0.4398 | 0.0425 | 0.4683 | 0.2148 | 1.5112 | 5 |
| round_robin | 0.4501 | 0.0430 | 0.4791 | 0.2223 | 1.5550 | 5 |
| AoI | 0.4820 | 0.0484 | 0.5120 | 0.2450 | 1.6225 | 5 |
| random | 0.4827 | 0.0445 | 0.5123 | 0.2495 | 1.6040 | 5 |

Pilot 判断：

- full-open 仍是最优不可部署上界，符合直觉。
- PD-PPO 仍优于所有朴素可部署基线。
- PD-PPO 略弱于 oracle-informed static projection，差距约 `1.05%`。
- AoI 与 random 在 V3.1 下明显落后，说明预测驱动调度目标仍有区分度。
- pilot 支持继续做完整 S2。

## 5. 完整 V3.1 重跑可能带来的改进

### 5.1 更强的仿真环境可信度

V3.1 可以把论文从“V2 有已知 event-heavy 局限，但我们做了诊断修复”推进到：

> 主实验本身就在一个统计验证通过、能够产生持续吹雪事件和 event-heavy 窗口的生成器上完成。

这会显著降低审稿人对仿真环境的质疑。

### 5.2 更自然的 condition-stratified evaluation

V3.1 支持真正的 event-heavy stratum：

- calm：event fraction `<0.25`
- mixed：event fraction `[0.25, 0.75]` 或更细分
- event-heavy：event fraction `>0.75`

这会让论文中的“南极风吹雪场景”更有辨识度，而不是只在中等事件占比区间验证。

### 5.3 更公平地验证 EventAwareCritic

如果 V3.1 完整重跑后 EventAwareCritic 在 A1/S2 中更明显，论文可以把它从“机制上合理但 V2 中不显著”提升为：

> 在持续事件上下文下，event-conditioned value function improves policy evaluation and warm-up timing.

如果 V3.1 后仍不显著，也仍然是有价值的结论：

> 在本任务中，主要收益来自 MaskedActor、ActionEmbedding、AWBC 和 oracle prior；EventAwareCritic 是次要组件。

### 5.4 论文叙事更简洁

当前论文需要解释：

- 主结果是 V2；
- G1 是 V3.1 生成器验证；
- V3.1 不是主结果；
- S2 留作 future work。

如果完整 S2 成功，可以改成更直观的结构：

1. V3.1 是最终仿真环境；
2. G1 是该环境的统计验证；
3. S2 是该环境下的主调度结果；
4. V2 只作为开发历史或附录诊断，不再承担主结果。

这会减少“为什么主结果不是最新生成器”的解释成本。

## 6. 完整 V3.1 重跑的风险

### 6.1 主结果可能变弱

Pilot 是 5 seeds / 1 budget，信号很好，但完整 S2 可能出现：

- 某个预算下 PD-PPO 和 round-robin 差距变小；
- static projection 过强，使 PD-PPO 显得更像学习到 near-static allocation；
- event-heavy stratum 中方差很大，显著性不足。

### 6.2 论文需要较大范围替换

如果 V3.1 成为主结果，需要替换或重写：

- main results table；
- power-error figure；
- condition-stratified table；
- statistical significance paragraph；
- discussion 中关于 V2 局限的段落；
- conclusion 中的未来工作表述；
- abstract / introduction 中所有结果数字。

这不是“补一张表”，而是主结果切换。

### 6.3 计算时间和稳定性风险

完整 S2 需要重新生成 truth、重训 oracle、重训 PD-PPO 并评估所有基线。服务器近期有供电/功耗不稳定问题，因此必须：

- 分阶段运行；
- 使用 tmux；
- 每个预算/seed 可断点续跑；
- 及时 rsync 结果回本地。

## 7. 推荐决策

建议进入完整 S2，但采用“先不替换论文主线，跑完再决策”的策略。

决策门槛：

- 若 V3.1 S2 中，PD-PPO 在 3 个 budgets 上均优于 round-robin、AoI、random，并且 full-open 仍是最优上界，则将 V3.1 提升为主结果。
- 若 V3.1 S2 只在部分 budget 成立，保留 V2 为主结果，V3.1 作为补充实验或 discussion 中的 future benchmark evidence。
- 若 V3.1 S2 中 PD-PPO 不稳定，则不替换主结果，转为诊断 oracle / reward / event-heavy training 的后续工作。

## 8. S2 启动前检查

### Stage S2.0：冻结当前可投稿稿

目标：

- 保留当前 V2 主结果版本，避免 S2 过程中破坏已可编译论文。

操作：

- 确认 `paper.tex` 当前可编译。
- 记录当前 PDF。
- 不把 V3.1 S2 未完成结果写入正文。
- 生成 fallback PDF，例如 `paper/paper_v2_fallback_YYYYMMDD.pdf`。
- 若准备提交 git，应在 S2 前做一个清晰 snapshot；若当前 worktree 混有大量未提交文件，至少保存 PDF 与当前报告，避免 S2 修改后无法回退。

验收：

- 当前论文仍可作为 fallback submission draft。
- fallback PDF 存在且可读。
- S2 启动前不出现“主结果已切换 V3.1”之类未完成声称。

### Stage S2.0b：确认 turn_61 三处硬件口径修正

`docs/05-14-V3-1-suggestion.md` 提醒：以下三处与 S2 无关，但属于投稿前必须稳定的论文口径，应在启动 S2 或等待 S2 时确认。

需要检查：

- `paper/tables/sensor_specs.tex` 脚注中，功耗来源应写为 datasheet-informed / normalised deployment cost，而不是暗示完全来自 bench measurement。
- `paper/sections/03_problem_formulation.tex` 的硬件原型段落中，GMX500、Parsivel2、FC4 的电气参数与 normalised cost model 的关系应表述为“motivate / inform”，不是一一精确标定。
- `paper/sections/04_simulation_environment.tex` 中，normalised deployment cost 与实际 watt 值之间应保持“相对量级参考”而不是“绝对功率比就是实验成本”的口径。

验收：

- 论文没有把 normalized deployment cost 写成真实绝对瓦特预算。
- 不出现 `to be filled`、`[FILL]` 或与设备手册冲突的功耗声称。
- `GillGMX500`、`SensecaLPS10`、`ApogeeSI111` 等 bib 条目若未在正文实际引用，应删除或补充合理引用，避免无用文献堆叠。

## 9. 完整 S2 执行计划

### Stage S2.1：完整 V3.1 主结果重跑

设计：

- budgets：`1.65, 1.70, 1.75`
- seeds：`41--50`
- generator：V3.1 semi-Markov
- 每个 budget/seed 重新生成 truth
- 每个 budget/seed 重新训练 TCN oracle
- 每个 budget/seed 重新训练 PD-PPO
- 同场评估：
  - full_open_unconstrained
  - feasible_static_projected
  - round_robin
  - AoI
  - random

建议输出目录：

- `reports/v31_s2_main/`

建议新增脚本：

- 新建 `scripts/42_v31_s2_full.py`，不要直接扩展 pilot 脚本作为主入口。这样可以避免 pilot 的 `reports/v31_pilot/` 与 full S2 的 `reports/v31_s2_main/` 混淆。
- 新建 `scripts/43_v31_s2_collect.py`，独立负责结果收集、统计检验和中间进度检查。

建议目录结构：

```text
reports/v31_s2_main/
  raw/
    budget1p65_seed41/
      evaluation/v2_eval_overall.csv
    budget1p65_seed42/
      evaluation/v2_eval_overall.csv
  done/
    budget1p65_seed41.done
  logs/
  v31_s2_overall_long.csv
  v31_s2_main_stats.csv
  v31_s2_significance.csv
```

断点续跑要求：

- 每个 `(budget, seed)` 完成后写入 `.done` 标记。
- 启动时先扫描 `done/` 和 `evaluation/v2_eval_overall.csv`，跳过已完成组合。
- 如果服务器断电，重启后可直接恢复未完成组合。

并行建议：

- 默认 `3--4` workers，优先使用空闲 GPU。
- 不建议盲目开满 6 workers，因为当前 custom PPO 仍偏 CPU rollout，过多 worker 可能造成 CPU/IO 竞争。
- 每个 budget 完成后先 collect 一次，再决定是否继续下一档 budget。

验收：

- 30 个 budget/seed runs 全部有 `evaluation/v2_eval_overall.csv`。
- 生成 `v31_s2_overall_long.csv`。
- 生成 `v31_s2_main_stats.csv`。
- 生成 Wilcoxon + Bonferroni 统计表。
- 每个完成组合都有 `.done` 标记或等价可恢复记录。

### Stage S2.1b：按 budget 中间检查

每完成一个 budget 的 10 个 seeds 后，立即运行 collector 并检查：

- full-open 是否仍为最低 FW-MAE 上界；
- PD-PPO 是否优于 round-robin、AoI、random；
- PD-PPO 与 feasible static projection 的差距是否在 `3%` 以内；
- 是否存在某个 seed 的异常反转需要单独查看日志。

如果 `B=1.65` 已经不满足基本条件，不建议盲目继续跑完整三档；应先诊断 oracle quality、event distribution、candidate prior 或 power constraint 设置。

### Stage S2.2：V3.1 condition-stratified evaluation

目标：

- 真正验证 event-heavy windows，而不是 V2 的 capped event stratum。

设计建议：

- 基于 V3.1 生成器抽取 3 个 strata：
  - calm：`event_fraction < 0.25`
  - mixed：`0.25 <= event_fraction <= 0.75`
  - event-heavy：`event_fraction > 0.75`
- 对每个 budget 至少保留主预算 `B=1.70` 的分层结果。
- 若计算量允许，三个 budgets 都做分层。

建议输出：

- `reports/v31_s2_main/v31_s2_condition_long.csv`
- `reports/v31_s2_main/v31_s2_condition_stats.csv`

验收：

- event-heavy stratum 有足够窗口数；
- PD-PPO 在 event-heavy stratum 中至少优于 round-robin、AoI、random；
- 如果 static projection 仍略优，正文应解释 static projection 是 oracle-informed non-deployable reference。

### Stage S2.3：图表生成

需要生成：

- V3.1 main results table；
- V3.1 power-error tradeoff；
- V3.1 condition-stratified table/figure；
- 可选：V3.1 learning curves；
- 可选：V3.1 representative sensor timeline；
- 可选：V2 vs V3.1 generator comparison figure 放入 supplement。

建议输出：

- `paper/tables/main_results_v31.tex`
- `paper/tables/condition_results_v31.tex`
- `paper/figures/figure6_power_error_tradeoff_v31.png`
- `paper/figures/figure_condition_v31.png`

### Stage S2.4：论文改造

如果 V3.1 S2 成功，论文应改为：

- Section 4：V3.1 是最终 simulation environment，G1 是环境验证，不再是 future-work readiness。
- Section 6：Main Results 使用 V3.1 S2 表格。
- Section 6：Condition-stratified evaluation 使用 V3.1 event-heavy strata。
- Section 7：V2 event cap 从“当前主结果局限”降级为“开发历史中识别并修复的问题”。
- Section 8：future work 不再写“完成 V3.1 重跑”，而写“跨站点真实数据验证、硬件闭环部署、长期季节性供电约束”。
- Abstract / Introduction：全部结果数字替换为 V3.1 S2 数字。

如果 V3.1 S2 不成功，论文应保持：

- V2 为主结果；
- G1 + pilot 作为补充诊断；
- 不在 abstract 中提 V3.1 performance。

## 10. S2 结果判定标准

完整 S2 被认为“足以替换论文主结果”的条件：

- 条件 1：三个 budgets 上，`full_open_unconstrained` 的 FW-MAE 均低于所有其他策略。
- 条件 2：三个 budgets 上，PD-PPO 的 FW-MAE 均低于 `round_robin`、`AoI`、`random`。
- 条件 3：至少两个 budgets 上，PD-PPO vs AoI 的 Wilcoxon signed-rank test 在 Bonferroni 校正后显著。若主论文仍采用 6 个 pairwise comparisons，则 $\alpha_{\mathrm{adj}}=0.05/6\approx0.0083$。
- 条件 4：PD-PPO 与 `feasible_static_projected` 的相对差距不超过 `3%`。
- 条件 5：event-heavy stratum 中，PD-PPO 仍优于 `round_robin`、`AoI`、`random`。
- 条件 6：统计检验不出现与主要结论相冲突的结果。

降级使用标准：

- 满足条件 1--4 但不满足条件 5：V3.1 可以作为主结果，但 condition-stratified 部分必须保守，不声称极端风暴下已充分验证。
- 满足条件 1--3 但不满足条件 4：不建议替换主线，应诊断 oracle quality 或 reward 设计。
- 不满足条件 2：停止主线替换，V2 保持主结果，V3.1 作为诊断和 future-work readiness。

Pilot 对照：

- Pilot 已满足条件 1、2、4 的单 budget 版本。
- Pilot 未满足论文级证据要求，因为只有 `B=1.70`、`n=5`，不足以做完整显著性检验，也尚未报告真正 event-heavy stratum。

## 11. S2 运行监控与同步

服务器近期有供电/功耗不稳定风险，因此 S2 运行中必须：

- 使用 tmux；
- 每完成一个 budget 立即 rsync 到本地；
- 保留 driver log 和每个 seed 的单独 log；
- 定期检查 GPU/CPU 负载，但不因 GPU 利用率低而误判卡死，因为 custom PPO 的 rollout 主要受 CPU/environment loop 限制；
- collector 可随时运行，不能依赖训练全部结束后才知道前面是否失败。

建议关键文件：

- `reports/v31_s2_main/driver.log`
- `reports/v31_s2_main/logs/*.log`
- `reports/v31_s2_main/v31_s2_overall_long.csv`
- `reports/v31_s2_main/v31_s2_main_stats.csv`
- `reports/v31_s2_main/v31_s2_significance.csv`

## 12. 预计时间

基于 pilot 与 Phase 1 速度粗估：

- 完整 S2 main rerun：约 `6--10 小时`，取决于 GPU/CPU 负载和是否 3/4 worker 并行。
- S2 condition evaluation：约 `1--3 小时`。
- 表格/图生成与论文替换：约 `1--2 小时`。
- 若加入新的 A1 on V3.1：额外 `8--14 小时`，不建议作为当前必须项。

建议先完成 S2 main + condition，不要立即在 V3.1 上重做所有消融。

建议时间线：

- Day 0：确认 turn_61 三处修正，冻结 fallback PDF，启动 S2 main rerun。
- Day 1：每完成一个 budget 做中间检查，必要时暂停诊断。
- Day 1--2：完成 S2 main collector 和显著性检验。
- Day 2：若满足最小成功标准，生成 V3.1 图表并改造论文主线。

## 13. S2 期间可并行处理的论文问题

以下任务不依赖 S2 结果，可在服务器运行时并行完成：

- A1/A2 消融路径差异的正文解释已经写入，但应再检查最终 PDF 中是否清晰。
- H1 不应使用 broad robustness / insensitive 等过强措辞，应使用 “within 5% of default” 或 “local stability”。
- 检查 `GillGMX500`、`SensecaLPS10`、`ApogeeSI111` 是否真实被正文引用；若只是 bib 中残留，应删除。
- 检查 `raw.tex` 只是中文直译/草稿，不应作为英文论文主源。
- 检查 `paper/tables/sensor_specs.tex`、`paper/sections/03_problem_formulation.tex`、`paper/sections/04_simulation_environment.tex` 的 normalised cost 口径一致。

## 14. 下一步

推荐下一步操作：

1. 确认 turn_61 三处硬件口径修正和 fallback PDF。
2. 新增 `scripts/42_v31_s2_full.py`，支持 budgets x seeds 的完整 V3.1 S2，并实现 `.done` 断点续跑。
3. 新增 `scripts/43_v31_s2_collect.py`，支持运行中随时收集结果和做 Wilcoxon + Bonferroni 检验。
4. 在服务器 tmux 中启动 S2 main rerun。
5. 每完成一个 budget 即同步一次结果并运行 collector。
6. 若主表满足最小成功标准，再做 V3.1 condition-stratified evaluation。
7. 最后决定是否将论文主结果从 V2 切换为 V3.1。
