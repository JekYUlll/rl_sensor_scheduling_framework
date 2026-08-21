# 论文叙述重设报告：当前实验、诊断与可支持结论

日期：2026-05-24  
目的：把当前已有实验结果、机制诊断、失败路径和可支持 claim 统一整理，作为重写论文叙述脉络的依据。本文不是新实验报告，而是写作控制文档。

---

## 1. 总体判断

当前结果已经不适合继续沿用早期的强叙事：

> PD-PPO 学会了事件触发的高成本传感器 warm-up，并因此稳定优于 AoI、round-robin 和静态策略。

更准确的叙述应改为：

> 在固定瞬时预算下，PD-PPO 学到的是一个低切换、低 warm-up abort 的紧凑可行调度策略。它稳定优于 round-robin、AoI 和 random 等动态启发式，但接近强静态参考，说明该 regime 下存在明显的静态最优结构。随后引入 calibrated energy-account 后，动态调度机会在 storm-window 中变得真实可见；curriculum PD-PPO 能稳定优于 static projection、round-robin 和 random，并与 AoI 竞争，但还不能声称稳健支配 AoI 或学到 clean event-triggered laser gating。

因此，论文主线应从“RL 明显学会动态事件触发控制”调整为：

1. 预测驱动调度框架：把传感器调度目标从即时状态误差转为下游预测质量。
2. 固定预算结果：PD-PPO 在固定预算下优于动态启发式，接近强静态参考。
3. 机制诊断：固定预算 regime 下静态/近静态是合理结果，不应被包装成事件触发调度。
4. Energy-account 扩展：长期可持续性约束区分“瞬时可行”和“长期可持续”，在 storm-window 中创造真实动态优势。
5. 学习结果边界：curriculum PD-PPO 能利用部分机会，但 AoI 仍是强基线，clean laser gating 尚未建立。

---

## 2. 证据主线 A：固定预算 V3.1 S2

### 2.1 实验定位

固定预算 V3.1 S2 是当前论文主表 Table 3 的来源。它使用固定瞬时预算 `B=1.65/1.70/1.75`，对每个策略执行共同的 feasibility projector 和 warm-up FSM。主指标是 forecast-weighted MAE，越低越好。

该设置回答的问题是：

> 在固定瞬时功率约束下，预测驱动 PPO 是否能学到比常见动态启发式更好的可行调度？

它不能回答的问题是：

> PPO 是否学到了事件触发的高成本传感器动态 warm-up？

### 2.2 主结果

来源：

- `paper/tables/main_results_v31.tex`
- `reports/v31_s2_main/v31_s2_main_stats.csv`

固定预算主表：

| Policy | B=1.65 | B=1.70 | B=1.75 |
|---|---:|---:|---:|
| Full observation | 0.1487 ± 0.0118 | 0.1493 ± 0.0123 | 0.1502 ± 0.0123 |
| Feasible static projection | 0.1593 ± 0.0116 | 0.1597 ± 0.0119 | 0.1612 ± 0.0115 |
| PD-PPO | 0.1628 ± 0.0140 | 0.1620 ± 0.0142 | 0.1661 ± 0.0145 |
| Round-robin | 0.1671 ± 0.0138 | 0.1674 ± 0.0141 | 0.1687 ± 0.0137 |
| AoI | 0.1700 ± 0.0139 | 0.1844 ± 0.0191 | 0.1933 ± 0.0184 |
| Random | 0.1803 ± 0.0142 | 0.1862 ± 0.0173 | 0.1914 ± 0.0163 |

Primary budget `B=1.70`：

- PD-PPO vs round-robin：`0.1620` vs `0.1674`，约 `3.2%` 改善。
- PD-PPO vs AoI：`0.1620` vs `0.1844`，约 `12.1%` 改善。
- PD-PPO vs feasible static projection：`0.1620` vs `0.1597`，PD-PPO 略差约 `1.5%`。

### 2.3 行为诊断

来源：

- `reports/v31_s2_main/behavior_diagnostics/diagnostic_summary.md`
- `reports/v31_s2_main/behavior_diagnostics/policy_behavior_summary.csv`

`B=1.70` 下：

| Policy | near-constant sensors | switches/step | warmup abort rate | mean power |
|---|---:|---:|---:|---:|
| PD-PPO | 5.4 | 0.337 | 0.0168 | 1.559 |
| Feasible static projection | 8.0 | 0.0005 | 0.0000 | 1.460 |
| Round-robin | 0.0 | 2.000 | 0.2490 | 1.555 |
| AoI | 1.0 | 3.647 | 0.4189 | 1.623 |
| Random | 0.0 | 3.288 | 0.3189 | 1.605 |

关键解释：

- PD-PPO 的优势主要来自低切换、低 abort、紧凑传感器子集。
- Round-robin 和 AoI 虽然更“动态”，但大量动态切换没有转化为有效观测，尤其 AoI 在 `B=1.70` 的 abort rate 很高。
- Feasible static projection 是强参考，不应被弱化成 strawman。
- 固定预算下的 Figure 8 应被解读为机制诊断图，而不是“动态事件触发调度图”。

### 2.4 AoI 为什么在 Table 3 中弱于 round-robin

在固定预算 V3.1 S2 中，AoI 的 freshness 目标与 downstream forecast loss 不完全对齐。AoI 优先刷新最陈旧通道，但在异质功耗、warm-up 和 projector 下，最陈旧不等于预测收益最高。`B=1.70` 时 AoI warmup abort rate 达 `0.4189`，显著高于 round-robin 的 `0.2490`，导致观测收益不足。

这与 energy-account 结果不矛盾。在 calibrated energy-account 中，AoI 强于 round-robin，因为长期能量账户使 blind cycling 极其昂贵，round-robin 的 abort 数达到 `768`，而 AoI 更稳定。

### 2.5 固定预算 V3.1 S2 支持的 claim

可写：

- PD-PPO 在固定预算 V3.1 S2 中优于 round-robin、AoI 和 random。
- PD-PPO 接近 feasible static projection，说明它能学习稳定、可行、预测有效的调度。
- 固定预算下动态启发式的高 switching 不等于有效动态调度。
- AoI 在固定预算下不是弱实现错误，而是 freshness proxy 与 forecast objective 存在 regime-specific mismatch。

不可写：

- 固定预算 V3.1 S2 证明了事件触发 laser gating。
- PD-PPO 的性能来自强事件条件高成本传感器 warm-up。
- Feasible static projection 是可忽略或不公平的弱基线。

---

## 3. 证据主线 B：消融实验

### 3.1 Staged ablation

来源：

- `paper/tables/ablation.tex`

| Stage | Configuration | FW-MAE |
|---|---|---:|
| D1 | MaskedActor + ActionEmbedding | 0.1821 ± 0.0147 |
| D2 | D1 + EventAwareCritic | 0.1939 ± 0.0179 |
| D3 | D2 + AWBC auxiliary loss | 0.1788 ± 0.0176 |
| D4 | D3 + oracle-calibrated prior (PD-PPO) | 0.1629 ± 0.0137 |

解释：

- 组合结构整体有效，尤其 oracle-calibrated prior / AWBC 相关组件对稳定训练重要。
- EventAwareCritic 在 staged diagnostic 中没有单独改善，不能被叙述成主要贡献证据。

### 3.2 Leave-one-out / full ablation

来源：

- `paper/tables/ablation_full.tex`

关键结果：

| Variant | FW-MAE | p vs full |
|---|---:|---:|
| Full PD-PPO | 0.1629 ± 0.0137 | --- |
| No ActionEmbedding | 0.1614 ± 0.0130 | 0.695 |
| No EventAwareCritic | 0.1630 ± 0.0138 | 0.846 |
| No AWBC auxiliary loss | 0.1654 ± 0.0152 | 0.037 |
| No oracle-calibrated prior | 0.1736 ± 0.0205 | 0.064 |
| No AWBC/prior | 0.1853 ± 0.0209 | 0.002 |
| No action mask | 0.1633 ± 0.0133 | 0.770 |
| MaskedActor only | 0.1828 ± 0.0156 | 0.002 |

支持的写法：

- PD-PPO 的有效性来自完整训练稳定化设计，而不是单个模块的孤立效果。
- AWBC/prior 组合对避免退化明显重要。
- EventAwareCritic 应弱化为架构组件，而不是核心实证贡献。

---

## 4. 证据主线 C：Figure 8 暴露的固定预算机制

Figure 8 的当前文件：

- `paper/figures/figure5_sensor_timeline.png`
- `reports/v31_s2_main/figure_assets/figure5_sensor_timeline.png`

当前 Figure 8 的价值不在于展示“PPO 动态性”，而在于展示固定预算 regime 的行为结构：

1. Feasible static projection 基本完全固定。
2. PD-PPO 是 quasi-static / compact-subset 策略。
3. Round-robin、AoI、random 有大量切换。
4. 大量切换常伴随 warm-up abort 或无效 warming，不必然带来预测收益。

因此 Figure 8 应重绘为“行为诊断图”，包含：

- event/storm indicator；
- 每个 policy 的 mode heatmap；
- 每个 panel 的 FW-MAE / abort / switches 摘要；
- caption 明确说明固定预算下 PD-PPO 不是 clean event-triggered laser gating。

建议不要把 power trace 加回图中，以免分散主信息。

---

## 5. 证据主线 D：Energy-account oracle 诊断

### 5.1 为什么需要 energy-account

固定瞬时预算只能判断某个传感器组合是否在当前步功率可行，不能表达长期可持续性。例如：

- static laser 子集可能满足 `power < B`；
- 但如果 `power > harvest`，它在长窗口中会持续消耗 SOC，最终不可持续。

Energy-account 的价值在于区分：

- 瞬时可行：`power < B`
- 长期可持续：`average power <= harvest`

这正是固定预算 regime 无法创造真实动态张力的原因。

### 5.2 标定链

来源：

- `docs/05-22-judge.md`
- `docs/05-24-claim-audit.md`

关键参数：

- `B = 1.20`
- `harvest = 0.92`
- `capacity = 180`
- `reserve = 20`

核心诊断：

- 全分布事件率约 `27%` 时，static snow core 仍最优，动态策略的事件收益被非事件损失稀释。
- storm-window 事件率约 `42%--70%` 时，动态 snow-core -> event-laser-fc4 策略超过 static snow core。

### 5.3 Oracle 结果

Claim audit 记录：

- storm-window dynamic `snow_core -> event_laser_fc4`：`0.4169`
- static snow-core：`0.4248`
- 事件段 loss：dynamic `0.3190` vs static `0.3517`
- dynamic guard drops：`0`
- dynamic warm-up aborts：`0`
- static laser 被 energy guard 裁剪，说明 laser 常开长期不可持续。

支持的 claim：

- Energy-account 在 storm-window 中建立了真实动态机会。
- 动态机会来自长期可持续性约束，而不是固定瞬时预算。
- 该结果是机制诊断，证明“在什么条件下动态调度有价值”。

不可写：

- Energy-account oracle 证明 learned PPO 已经学会该 oracle 策略。
- 全分布下动态调度稳定优于静态。
- event labels 可以直接用于真实部署策略。

---

## 6. 证据主线 E：Energy-account learned PD-PPO / curriculum

### 6.1 锁定结果

来源：

- `reports/energy_account_convergence_20260524/energy_account_main_summary.csv`
- `paper/tables/energy_account_curriculum_results.tex`
- `docs/05-24-claim-audit.md`

锁定配置：

- 100k storm-window curriculum；
- seeds `41--45`；
- storm-window 和 full-distribution 两种评估；
- 结果指标为 frozen-oracle loss。

### 6.2 Storm-window 结果

| Policy | Storm loss | PD-PPO wins | Storm aborts |
|---|---:|---:|---:|
| Full obs. | 0.3961 ± 0.0021 | 0/5 | 0.0 |
| Static projection | 0.4742 ± 0.0236 | 5/5 | 0.0 |
| PD-PPO | 0.4153 ± 0.0051 | --- | 25.8 |
| AoI | 0.4176 ± 0.0105 | 3/5 | 10.0 |
| Round-robin | 0.4451 ± 0.0167 | 5/5 | 768.0 |
| Random | 0.4565 ± 0.0140 | 5/5 | 2031.6 |

解释：

- PD-PPO 稳定优于 static projection、round-robin、random。
- PD-PPO 与 AoI 竞争，均值略优，但只有 `3/5` seeds 优于 AoI。
- AoI 在此 setting 是强基线，不应被描述为明显弱基线。

### 6.3 Full-distribution no-retrain 结果

| Policy | Full loss | PD-PPO wins | Full aborts |
|---|---:|---:|---:|
| Full obs. | 0.2994 ± 0.0088 | 0/5 | 0.0 |
| Static projection | 0.3318 ± 0.0062 | 4/5 | 0.0 |
| PD-PPO | 0.3155 ± 0.0133 | --- | 10.2 |
| AoI | 0.3168 ± 0.0135 | 4/5 | 6.0 |
| Round-robin | 0.3375 ± 0.0195 | 5/5 | 768.0 |
| Random | 0.3431 ± 0.0188 | 5/5 | 2067.6 |

解释：

- Curriculum PPO 不是简单 storm-window 过拟合；它在 full-distribution 上仍有小平均优势。
- 但 margins 很窄，且 per-seed 不统一。
- 不能声称 robust AoI dominance，也不能声称 full-distribution 下稳定优于 static projection。

### 6.4 机制诊断：PPO 赢 AoI 的方式

来源：

- `docs/05-23-curriculum-ppo-mechanism-diagnosis.md`

三 seed 机制诊断显示：

- laser event/non-event selected ratio 约 `1.03x`，不是强事件触发。
- laser abort 多发生在非事件期：event `10`，non-event `31`。
- PPO 相对 AoI 的优势主要来自 non-event loss 更低，而不是 event loss 更低。

三 seed event/non-event oracle loss：

| Policy | Event loss | Non-event loss |
|---|---:|---:|
| PPO | 0.3274 | 0.5256 |
| AoI | 0.3243 | 0.5431 |

结论：

> PPO 学到的是 conservative storm-window allocation / energy management，而不是 laser-on-event / laser-off-calm 的 clean gating 策略。

---

## 7. 已尝试但未解决的问题

来源：

- `reports/energy_account_convergence_20260524/energy_account_probe_summary.csv`
- `docs/05-23-curriculum-ppo-mechanism-diagnosis.md`

### 7.1 延长训练到 300k

结果：

- storm loss 从 `0.4106` 改善到 `0.4053`；
- laser event/non-event ratio 提高；
- 但 abort 从 `5` 增加到 `66`；
- full-distribution seed-41 略输 AoI：PPO `0.3122` vs AoI `0.3118`。

解释：

延长训练能强化事件期激活，但没有同步学会长时 SOC 储备，导致全分布退化。

### 7.2 Event-gated actor

结果：

- storm：PPO `0.4105` vs AoI `0.4128`，仍略优；
- full：PPO `0.3128` vs AoI `0.3117`，输；
- abort `38`。

解释：

输出头条件化没有解决长时 credit assignment 问题。

### 7.3 SOC auxiliary

结果：

- storm：PPO `0.4105` vs AoI `0.4144`；
- full：PPO `0.3138` vs AoI `0.3127`；
- abort `16`。

解释：

SOC 表征辅助能降低部分 abort，但不能稳定改善 full-distribution AoI 对比。

### 7.4 SOC soft penalty

单 seed probe：

- storm PPO `0.4069` vs AoI `0.4128`；
- abort `9`；
- mean power 降低。

解释：

这是有希望的 probe，但不是锁定主结果。它可以作为 future work / follow-up，而不应进入主 claim。

### 7.5 Event reward multiplier

单 seed probe：

- storm PPO `0.4088` vs AoI `0.4118`；
- event loss 改善；
- abort `7`；
- laser event/non-event ratio 仍不强。

解释：

事件加权改善 event loss，但把 laser 推成高占空比 storm-context sensor，而不是 event-triggered sensor。

---

## 8. 当前论文应采用的叙述结构

### 8.1 推荐主标题式叙事

可考虑将论文主线重设为：

> Prediction-driven sensor scheduling under power and sustainability constraints for Antarctic microclimate digital twins.

重点从“RL beats all baselines”转为：

> Forecast-oriented scheduling reveals when adaptive control is useful, when static policies are sufficient, and how long-term energy sustainability changes the optimal scheduling regime.

### 8.2 推荐 Section 组织

#### Introduction

强调问题：

- 南极微气候数字孪生需要在有限电力下选择传感器观测。
- 传统调度代理，如 freshness / AoI，不直接优化下游预测质量。
- 固定功率预算和长期能量可持续性是两个不同问题。

贡献建议写为：

1. 提出 forecast-oracle-driven PD-PPO 调度框架，把调度 reward 对齐到未来预测质量。
2. 构建 warm-up-aware 多传感器调度环境，包含 OFF/WARMING/READY 状态。
3. 在 V3.1 S2 固定预算实验中，证明 PD-PPO 优于动态启发式并接近强静态参考。
4. 通过 behavior diagnostics 揭示固定预算下的静态最优结构，避免把高切换误判为有效动态性。
5. 引入 calibrated energy-account 诊断，说明长期可持续性约束能在 storm-window 中创造真实动态调度机会；curriculum PD-PPO 在该 setting 中稳定优于 static projection、round-robin、random，并与 AoI 竞争。

#### Method

保留：

- warm-up FSM；
- forecast-weighted reward；
- masked actor / action embedding / AWBC / oracle prior；
- feasibility projector。

需要降调：

- EventAwareCritic 作为架构元素，而非已证明的核心收益来源。
- 不要暗示 event flag 可在真实部署中直接可用，除非限定为 simulator/diagnostic feature。

#### Experiments

建议分成三层：

1. **Fixed-budget main benchmark**：Table 3，V3.1 S2。
2. **Behavior diagnostics**：Figure 8，解释 quasi-static behavior 与 heuristic warm-up churn。
3. **Energy-account diagnostic and curriculum result**：作为补充但重要的机制章节，展示动态调度何时真的有价值。

#### Discussion

核心讨论：

- 固定预算下 PD-PPO 接近 static projection 不是失败，而是 regime 诊断。
- AoI 是强 freshness baseline，在不同 regime 排序会变化。
- Dynamic scheduling value 需要长期能量账户、事件密度、成本异质性、warm-up 时间尺度共同满足。
- 当前 PPO 的限制是 long-horizon SOC credit assignment 和 clean event-triggered gating。

---

## 9. 建议修改的主要图表

### Table 3

当前 Table 3 数据应保留，但 caption 和正文需要确保：

- 它是 fixed-budget V3.1 S2；
- feasible static projection 是 fixed-priority projected baseline；
- AoI 在该表中弱于 round-robin 是固定预算 regime 的结果，不应泛化到 energy-account。

### Figure 8

建议重绘。

新目标：

- 展示 fixed-budget behavior diagnostic；
- 强调 PD-PPO low-switch / low-abort / compact-subset；
- 不暗示事件触发 laser gating。

图内建议添加：

- event indicator row；
- per-policy metric badge：FW-MAE, abort, switches；
- clear row separators；
- OFF/WARMING/ACTIVE discrete legend。

### Energy-account table

可以作为新表或 appendix 表：

- storm/full loss；
- wins；
- aborts；
- power；
- 明确 n=5。

当前 `paper/tables/energy_account_curriculum_results.tex` 已具备基本内容。

### Energy-account mechanism figure

如果篇幅允许，建议新增一张简洁机制图：

- fixed budget：static laser instantaneously feasible；
- energy account：static laser drains SOC；
- dynamic snow-core/event-laser fits storm windows；
- 对比 static snow core vs dynamic event-laser 的 event loss。

---

## 10. 可支持 claim 与禁止 claim

### 10.1 可支持 claim

强支持：

- PD-PPO 在 fixed-budget V3.1 S2 中优于 round-robin、AoI、random。
- PD-PPO 接近 feasible static projection。
- 固定预算下 PD-PPO 的 rollout 是 quasi-static / compact-subset，而不是强事件触发。
- Energy-account storm-window oracle 证明动态策略在长期可持续性约束下可优于静态子集。
- 100k curriculum PD-PPO 在 energy-account storm-window 中稳定优于 static projection、round-robin、random。

中等支持：

- Curriculum PD-PPO 与 AoI 竞争，并在均值上略优。
- Curriculum PD-PPO 的 full-distribution no-retrain 泛化有小平均优势。
- SOC soft penalty 和 event reward multiplier 是有希望的后续优化方向。

### 10.2 禁止或必须降级的 claim

不能写：

- PD-PPO 稳健支配 AoI。
- PD-PPO 稳健学到 clean event-triggered laser gating。
- Fixed-budget V3.1 证明动态调度本身有价值。
- Energy-account 结果证明 full-distribution 动态调度优于静态。
- Event labels 在真实部署中直接可用。
- 当前 energy-account 是完整物理电池/发电/除冰 heater 模型。

---

## 11. 推荐摘要级表述

下面是当前证据最匹配的摘要级叙述：

> We study prediction-driven sensor scheduling for power-constrained Antarctic microclimate monitoring. A warm-up-aware PD-PPO scheduler is trained against frozen forecast oracles so that scheduling decisions are evaluated by downstream forecast quality rather than instantaneous estimation error. In a fixed-budget V3.1 benchmark, PD-PPO reduces forecast-weighted MAE relative to round-robin, AoI, and random scheduling, while remaining close to a strong feasible static reference. Behavior diagnostics show that this fixed-budget regime favors compact, low-abort allocations rather than high-frequency switching. To test when dynamic scheduling becomes genuinely useful, we introduce a calibrated energy-account diagnostic that separates instantaneous feasibility from long-term sustainability. In event-rich storm windows, the energy-account setting creates a real dynamic opportunity: curriculum PD-PPO consistently outperforms static projection, round-robin, and random scheduling over five seeds, while remaining competitive with AoI. These results support prediction-driven adaptive scheduling under binding energy constraints, while also identifying long-horizon SOC credit assignment and robust event-triggered high-latency activation as open limitations.

---

## 12. 推荐下一步

### 12.1 立即做

1. 重写 Introduction 和 Discussion 的 claim 层级。
2. 重绘 Figure 8，使其变成 fixed-budget behavior diagnostic。
3. 检查所有 “event-triggered warm-up”、“selectively warms event-sensitive instruments” 等措辞，改成更准确表述。
4. 在 Experiments 中分开 fixed-budget 和 energy-account 两个 regime。
5. 明确 AoI 在两个 regime 下角色不同：fixed-budget 中弱于 round-robin，energy-account 中强于 round-robin。

### 12.2 暂不做

1. 不继续追求 full-distribution robust AoI dominance，边际收益低。
2. 不把 300k / event-gated actor / SOC auxiliary probe 写成主结果。
3. 不把 SOC soft penalty 单 seed probe 提升为 locked result，除非后续做多 seed。

### 12.3 可作为 future work

1. 分层策略：高层 storm/event mode，低层 SOC-aware actuator。
2. 显式事件预测器：由 always-on low-power channels 预测 high-cost sensor activation。
3. 更真实的 energy harvesting / battery / heater 功耗模型。
4. 多季节 forcing 和更长时间尺度泛化。

---

## 13. 主要资产索引

固定预算主结果：

- `rl_sensor_scheduling_framework/paper/tables/main_results_v31.tex`
- `rl_sensor_scheduling_framework/reports/v31_s2_main/v31_s2_main_stats.csv`
- `rl_sensor_scheduling_framework/reports/v31_s2_main/v31_s2_significance.csv`

固定预算行为诊断：

- `rl_sensor_scheduling_framework/reports/v31_s2_main/behavior_diagnostics/diagnostic_summary.md`
- `rl_sensor_scheduling_framework/reports/v31_s2_main/behavior_diagnostics/policy_behavior_summary.csv`
- `rl_sensor_scheduling_framework/reports/v31_s2_main/behavior_diagnostics/sensor_behavior_summary.csv`
- `rl_sensor_scheduling_framework/reports/v31_s2_main/figure_assets/figure5_sensor_timeline.png`

消融：

- `rl_sensor_scheduling_framework/paper/tables/ablation.tex`
- `rl_sensor_scheduling_framework/paper/tables/ablation_full.tex`

Energy-account：

- `rl_sensor_scheduling_framework/docs/05-22-judge.md`
- `rl_sensor_scheduling_framework/docs/05-23-curriculum-ppo-mechanism-diagnosis.md`
- `rl_sensor_scheduling_framework/docs/05-24-claim-audit.md`
- `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_main_summary.csv`
- `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_probe_summary.csv`
- `rl_sensor_scheduling_framework/paper/tables/energy_account_curriculum_results.tex`

写作控制：

- `findings.md`
- `progress.md`
- `task_plan.md`

