# 05-13-02 审阅意见补充实验回应

日期：2026-05-13

## 执行概况

本轮针对 `docs/05-13-02.md` 的最高优先级意见补齐了三项实验/表格：

1. E1-fix：修复条件分组采样器，重新评估 calm / mixed / event 条件。
2. E2：评估 frozen TCN oracle 在部分观测下的鲁棒性。
3. P1：提取物理单位的逐变量 forecast MAE，避免把 FW-MAE 误读为摄氏度误差。

新增或修改脚本：

- `scripts/32_v2_condition_eval.py`：加入显式事件占比分桶、候选池诊断、严格模式和阈值参数。
- `scripts/33_v2_run_supplement_parallel.py`：支持 fixed-E1 子目录和条件阈值传参。
- `scripts/34_v2_oracle_robustness_eval.py`：新增 E2 部分观测 oracle 鲁棒性评估。
- `scripts/35_v2_physical_unit_mae_table.py`：新增 P1 物理单位 MAE 汇总表。
- `scripts/31_v2_build_supplement_assets.py`：支持从 `E1_condition_eval_fixed` 汇总 fixed-E1 图表和统计。

主要输出：

- `reports/v2_supplement_assets/exp_e1_condition_stats_fixed.csv`
- `reports/v2_supplement_assets/figure_e1_condition_eval_fixed.png`
- `reports/v2_supplement_assets/exp_e2_oracle_robustness_stats.csv`
- `reports/v2_supplement_assets/exp_e2_oracle_reference_stats.csv`
- `reports/v2_supplement_assets/exp_p1_physical_unit_mae.csv`
- `reports/v2_supplement_assets/exp_p1_physical_unit_mae_budget1p70.csv`

## E1-fix：条件分组评估

审阅意见建议使用：

- calm: event fraction `< 0.20`
- mixed: event fraction `0.35-0.65`
- event-heavy: event fraction `> 0.75`

但检查当前 V2 truth 后发现，1024 步窗口中 event fraction 最高约为 `0.50`，不存在 `>0.75` 的 event-heavy 候选窗口。因此本轮没有伪造不可达条件，而是采用当前 V2 可实现且互不重叠的 fixed-E1 条件：

- calm: event fraction `< 0.25`
- mixed: event fraction `0.28-0.36`
- event: event fraction `> 0.40`
- eval steps: `512`
- rollouts: `6`
- budgets: `1.65, 1.70, 1.75`
- seeds: `41-50`

采样诊断：

| condition | mean event fraction | min | max | candidate warning |
|---|---:|---:|---:|---:|
| calm | 0.172 | 0.072 | 0.246 | 0 |
| mixed | 0.321 | 0.295 | 0.346 | 0 |
| event | 0.450 | 0.400 | 0.584 | 0 |

fixed-E1 结果：

| condition | full observation | static projection | PD-PPO | round-robin | AoI | random |
|---|---:|---:|---:|---:|---:|---:|
| calm | 0.3031 | 0.3222 | 0.3233 | 0.3413 | 0.3495 | 0.3631 |
| mixed | 0.3828 | 0.3999 | 0.4018 | 0.4177 | 0.4292 | 0.4431 |
| event | 0.4164 | 0.4343 | 0.4406 | 0.4544 | 0.4754 | 0.4876 |

结论：

- 原 E1 的 mixed/event 重叠问题已修复。
- PD-PPO 在三个条件下均优于 round-robin、AoI、random。
- PD-PPO 与 static projection 非常接近，但略弱于 static projection；这与 S1 主表一致，说明当前 V2 场景仍有较强静态最优结构。
- V2 不能支持真正的 `event-heavy >0.75` 结论；如果论文需要 event-heavy，应进入 V3 生成器修改。

## E2：Oracle 部分观测鲁棒性

设置：

- budget: `1.70`
- seeds: `41-45`
- subset sizes: `k=1,2,3,4,5`
- 每个 k 随机采样最多 `16` 个固定传感器子集；`k=1` 全枚举。
- 同时评估 full observation (`k=8`)。

结果：

| active sensors k | FW-MAE mean | std | n |
|---:|---:|---:|---:|
| 1 | 0.6960 | 0.1180 | 40 |
| 2 | 0.6211 | 0.1374 | 80 |
| 3 | 0.5598 | 0.1263 | 80 |
| 4 | 0.5155 | 0.1173 | 80 |
| 5 | 0.4723 | 0.0878 | 80 |
| 8 full | 0.3848 | 0.0384 | 5 |

同一批 seed 的参考策略：

| policy | FW-MAE mean | std | n |
|---|---:|---:|---:|
| full observation | 0.3629 | 0.0286 | 5 |
| static projection | 0.3889 | 0.0333 | 5 |
| PD-PPO | 0.3901 | 0.0340 | 5 |
| random | 0.4302 | 0.0327 | 5 |

结论：

- Oracle 的误差随活跃传感器数量增加而平滑下降，未出现近常数奖励或完全塌缩。
- 随机策略与 full observation 仍有明显差距，说明 reward landscape 对调度策略是有区分度的。
- PD-PPO 与 static projection 接近，支持“当前 V2 中 RL 学到了接近强静态策略的调度，而不是无意义节电”的叙事。

## P1：物理单位误差表

budget `B=1.70` 的关键变量 raw forecast MAE：

| variable | unit | full observation | static projection | PD-PPO | round-robin | AoI | random |
|---|---|---:|---:|---:|---:|---:|---:|
| air_temperature_c | degC | 2.4544 | 2.5586 | 2.5471 | 2.7725 | 3.0229 | 3.0444 |
| wind_speed_ms | m s^-1 | 1.3352 | 1.4090 | 1.4143 | 1.4443 | 1.5951 | 1.6200 |
| snow_mass_flux_kg_m2_s | kg m^-2 s^-1 | 9.33e-05 | 9.74e-05 | 9.69e-05 | 9.81e-05 | 1.00e-04 | 1.01e-04 |

结论：

- FW-MAE 不应直接解释为摄氏度。
- 论文中如需物理含义，应引用 P1 表格中的逐变量 raw MAE。

## 对仿真数据生成的后续修改建议

本轮最关键的新发现不是算法失败，而是 V2 生成器无法产生真正 event-heavy 条件：

- 即使缩短到 512 步窗口，当前 event fraction 最大也只到约 `0.58`。
- 这与 `docs/05-13-02.md` 中指出的“事件覆写而非持久风速 regime 驱动”一致。

建议 V3 生成器按以下方向修改：

1. 保留 AntAWS + DFT phase randomization 作为基础气象序列。
2. 用平滑 storm regime latent 生成持久风速异常，而不是独立事件区间硬覆写。
3. 用 CRED 风格概率从持久风速触发吹雪事件。
4. 将 flux 设为 wind-driven power-law + lognormal AR noise。
5. 增加事件持续时间、事件间隔、event/non-event 风速分布、flux-wind log-log slope 等验证图。

当前论文若仍使用 V2，应把 event-heavy 表述改成“higher-event-fraction windows”，并明确当前 V2 无法支持 `>0.75` event-heavy claim。
