# 2026-05-14 Phase 1 补充实验报告：A1 / H1 / G1

## 结论概览

本轮 Phase 1 已完成三项增强实验：

- A1 完整组件消融：8 个配置，含 full PD-PPO 与 7 个 remove-one-component 变体；每个配置 n=10 seeds。
- H1 超参数敏感性：`awbc_coef` x `prior_kl_coef` 的 3x3 网格；默认配置复用 S1，其他 8 个格点每格 n=5 seeds。
- G1 V3.1 生成器验证：新增 semi-Markov storm-regime + CRED hysteresis + minimum duration/gap 的诊断生成器，并通过统计验证。

远端任务完成状态：

- A1/H1 训练与评估产物：110/110 个 evaluation CSV 完成。
- 失败日志：0。
- 本地已同步：`reports/v3_supplement_assets/` 下的 raw/stats CSV、PNG/PDF 图、manifest 与 driver log。

## A1：完整组件消融

数据源：

- `reports/v3_supplement_assets/exp_a1_ablation_raw.csv`
- `reports/v3_supplement_assets/exp_a1_ablation_stats.csv`

主要结果：

| Variant | FW-MAE mean | Std | Delta vs full | Delta % | p vs full | Significant |
|---|---:|---:|---:|---:|---:|---|
| Full PD-PPO | 0.3911 | 0.0240 | 0.0000 | 0.00% | - | reference |
| No ActionEmbedding | 0.3998 | 0.0282 | +0.0087 | +2.23% | 0.00195 | yes |
| No EventAwareCritic | 0.4004 | 0.0306 | +0.0093 | +2.38% | 0.01953 | no |
| No AWBC | 0.4031 | 0.0288 | +0.0121 | +3.08% | 0.00195 | yes |
| No oracle prior | 0.4120 | 0.0259 | +0.0209 | +5.35% | 0.00195 | yes |
| No AWBC/prior | 0.4457 | 0.0597 | +0.0546 | +13.97% | 0.00195 | yes |
| No action mask | 0.4006 | 0.0303 | +0.0095 | +2.42% | 0.00391 | yes |
| MaskedActor only | 0.4376 | 0.0300 | +0.0465 | +11.90% | 0.00195 | yes |

解释：

- Full PD-PPO 是所有 A1 配置中最优。
- AWBC 和 oracle prior 是贡献最大的稳定化组件，二者同时移除会造成约 14% 的 FW-MAE 恶化。
- `No AWBC/prior` 的退化幅度大于任一单独移除，说明 AWBC 与 oracle prior 存在正向交互：前者给出暖机感知的行为目标，后者稳定早期探索。
- action mask 与 ActionEmbedding 都有独立显著贡献。
- EventAwareCritic 的均值方向是正向的，但在 Bonferroni 校正后不显著。这与此前 A2 staged diagnostic 的结论一致：当前 V2 事件结构还不足以充分放大 event-conditioned critic 的价值。
- A2 是逐步添加组件的 path diagnostic，A1 是从 full PD-PPO 移除组件的 matched-seed ablation。二者路径不等价，尤其在组件存在交互时不能把 delta 数字逐项对齐比较。

论文处理：

- 已新增 `paper/tables/ablation_full.tex`。
- 已在 `paper/sections/06_experiments.tex` 增加 “Full Component Ablation (A1)” 小节。

## H1：超参数敏感性

数据源：

- `reports/v3_supplement_assets/exp_h1_hyperparam_raw.csv`
- `reports/v3_supplement_assets/exp_h1_hyperparam_stats.csv`
- `reports/v3_supplement_assets/figure_h1_heatmap.png`

网格：

- `awbc_coef`: 0.05, 0.10, 0.20
- `prior_kl_coef`: 0.5, 1.0, 2.0
- 默认格点：0.10 / 1.0

主要结果：

| awbc_coef | prior_kl_coef | Mean FW-MAE | Std | Delta % vs default | Within 5% |
|---:|---:|---:|---:|---:|---|
| 0.05 | 0.5 | 0.4019 | 0.0434 | +3.03% | yes |
| 0.05 | 1.0 | 0.3924 | 0.0329 | +0.58% | yes |
| 0.05 | 2.0 | 0.3927 | 0.0358 | +0.66% | yes |
| 0.10 | 0.5 | 0.4084 | 0.0470 | +4.70% | yes |
| 0.10 | 1.0 | 0.3901 | 0.0340 | baseline | yes |
| 0.10 | 2.0 | 0.3970 | 0.0381 | +1.78% | yes |
| 0.20 | 0.5 | 0.3978 | 0.0352 | +1.97% | yes |
| 0.20 | 1.0 | 0.3941 | 0.0392 | +1.03% | yes |
| 0.20 | 2.0 | 0.3936 | 0.0352 | +0.91% | yes |

解释：

- 所有非默认格点都在默认配置 +5% 范围内。
- 可在论文中谨慎声称：PD-PPO 在默认值附近的 AWBC 和 oracle-prior KL 权重上不脆弱。
- 不应声称“完全不敏感”，因为 (0.10, 0.5) 已接近 +5% 上限。
- 默认格点是 S1 同一配置在诊断 seeds 41--45 上的子集结果；S1 主表使用 n=10，因此默认均值 `0.3901` 与主表 `0.3911` 不需要完全一致。
- H1 仅作为描述性敏感性检查，不做格点间显著性检验。

论文处理：

- 已将 H1 heatmap 同步到 `paper/figures/figure_h1_heatmap.png`。
- 已在 `paper/sections/06_experiments.tex` 增加 “Hyperparameter Sensitivity (H1)” 小节。

## G1：V3.1 生成器验证

数据源：

- `reports/v3_supplement_assets/exp_g1_generator_validation.csv`
- `reports/v3_supplement_assets/exp_g1_v2_v31_generator_comparison.csv`
- `reports/v3_supplement_assets/g1_v31_synthetic_truth.csv`
- `reports/v3_supplement_assets/g1_v31_synthetic_metadata.json`
- `reports/v3_supplement_assets/figure_g1_*.png`

实现要点：

- 在 `src/data_sources/public_weather_synthesis.py` 中新增可选 `blowing_snow_event_model="semi_markov"`。
- V2 默认 `clustered` 路径保留，避免破坏既有实验。
- V3.1 路径加入：
  - long storm-regime latent process；
  - event pulses with minimum duration/gap；
  - CRED-style hysteresis；
  - wind-speed power-law mass-flux generation；
  - AntAWS scalar base variables 的 KS/ACF/PSD 统计验证。

验证结果：

| Criterion | Value | Rule | Passed |
|---|---:|---|---|
| Wind-speed ACF max deviation, lags 1-12 h | 0.048 | < 0.05 | yes |
| P(event fraction > 0.75) in 512-step windows | 0.065, 95% Wilson CI [0.062, 0.067], n=29489 windows | > 0.05 | yes |
| P(event fraction < 0.25) in 512-step windows | 0.539 | > 0.30 | yes |
| Max KS statistic for AntAWS scalar variables | 0.0146 | < 0.05 | yes |
| Median event duration | 18.0 h | 12-20 h | yes |
| Flux-wind log-log slope | 2.966 | 2.5-3.5 | yes |
| Diameter-wind Spearman rho | -0.544 | < -0.30 | yes |
| Wind-speed PSD log-MSE, 0.1-4.0 cpd | 0.0419 | < 0.10 | yes |

V2/V3.1 同配置诊断对比：

| Generator | ACF delta | P(ef>0.75) | Max 512-step event fraction | Max KS |
|---|---:|---:|---:|---:|
| V2 clustered diagnostic | 0.175 | 0.000 | 0.488 | 0.2246 |
| V3.1 semi-Markov diagnostic | 0.048 | 0.065 | 0.811 | 0.0146 |

解释：

- G1 表明 V3.1 生成器可以解决 V2 的两个关键限制：event-heavy 窗口不可达、风速时序统计被事件硬覆写破坏。
- 但 G1 只是生成器验证，不等同于已经完成 V3 主实验。
- 只有在 V3.1 上重新训练 oracle、重新训练 PD-PPO，并重新评估所有基线后，才能把 V3.1 结果替换为论文主结果。
- 论文中对 V3.1 的所有声称应限制为“生成器统计质量通过验证”，不应暗示 V3.1 策略性能已优于 V2。

论文处理：

- 已新增 `paper/tables/g1_generator_validation.tex`。
- 已在 `paper/sections/04_simulation_environment.tex` 增加 G1 验证段落和表格。
- 已在 `paper/sections/07_discussion.tex` 将 “V3 planned” 改为 “V3.1 diagnostic generator validated; full S2 rerun remains future/next step”。

## 下一步建议

1. 编译论文并检查新增表格/图是否溢出或浮动位置异常。
2. 若论文长度允许，保留 A1/H1/G1；若篇幅过长，G1 表可转为附录或补充材料。
3. 如果还要继续实验，下一步是 S2：在 V3.1 生成器上重新训练 oracle、PD-PPO 和所有基线。不能直接把 V2 策略拿到 V3.1 上评估后称为 V3 主结果。

## S2 pilot：V3.1 最小闭环重训

目的：

- 快速判断 V3.1 semi-Markov 生成器下，PD-PPO 是否仍优于朴素可部署基线。
- pilot 不替代完整 S2 主实验；若结果稳定，再扩展到 3 budgets x 10 seeds。

设计：

- 预算：`B=1.70`。
- seeds：`41 42 43 44 45`。
- 每个 seed 独立生成 V3.1 truth：`blowing_snow_event_model=semi_markov`。
- 生成器参数：`event_coverage=0.28`，`min_duration=12`，`max_duration=24`，`min_gap=4`，`lead_steps=6`，`wind_margin=1.2`，`flux_wind_exponent=3.0`。
- 每个 seed 重新训练 TCN oracle，并重新训练 PD-PPO。
- 同一 run 内评估 `full_open_unconstrained`、`feasible_static_projected`、`round_robin`、`aoi`、`random` 等默认基线。

运行状态：

- 已新增 pilot runner：`scripts/41_v31_pilot.py`。
- 已同步到服务器 `remote-gpu`。
- 已在服务器 tmux 会话 `v31_pilot` 中启动。
- 使用 GPU：`1,4,5`，避开当时已满载的 `0,2,3`。
- 输出目录：`reports/v31_pilot/`。
- driver 日志：`reports/v31_pilot/driver.log`。
- seed 日志：`reports/v31_pilot/logs/v31_pilot_budget1p70_seed*.log`。

## 收尾检查

- `scripts/20_build_public_weather_truth.py`、`scripts/23_v2_train_ppo.py`、`scripts/25_v2_train_custom_ppo.py` 已支持 V3.1 事件生成参数透传。
- `scripts/40_v2_compare_v31_generator.py` 已新增，用于复现 V2 clustered 与 V3.1 semi-Markov 的生成器诊断对比。
- `scripts/41_v31_pilot.py` 已新增，用于运行 V3.1 最小闭环重训 pilot。
- V3.1 truth 生成 smoke test 已通过。
- `tests/test_public_weather_synthesis.py` 已通过。
- `paper.tex` 已在本轮口径修正后通过 XeLaTeX 编译，输出 `paper/paper.pdf`，无 undefined reference/citation；仍保留若干普通 overfull hbox 与 6 条 bib empty-pages 警告。
