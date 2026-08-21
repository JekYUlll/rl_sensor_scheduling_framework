# v1 DQN 实现与历史结果复盘

日期：2026-05-09

## 1. 结论先行

v1 阶段的 DQN/CMDP-DQN 确实出现过“相对较优”的结果，但这个判断需要加限定：

1. 较优主要发生在**暖机机制正式引入之前**，或暖机尚未成为环境级延迟动作后果的时候。
2. v1 DQN 并不是稳定碾压所有朴素基线；在不少 run 中，`periodic` / `round_robin` 仍然接近甚至优于 DQN。
3. 最强的历史证据来自两个 run：
   - `full_mainline_20260329`：`cmdp_dqn` 在聚合 RMSE 上排名第 1，明显优于 `full_open`、`periodic`、`round_robin`。
   - `routeA_ctx_20260408_0335`：`dqn` 与 `cmdp_dqn` 并列第 1，优于 `periodic` / `round_robin`。
4. 暖机三态机制引入后，DQN 的优势明显下降；这不是简单“参数没调好”，而是任务结构从即时 subset 选择变成了带延迟收益、暖机中断损失和长期承诺的序列决策。

因此，v1 DQN 可以作为论文中的一个有价值 ablation：**在无暖机或弱延迟场景下，subset-conditioned DQN 已能学到有效调度；引入真实仪器暖机后，问题变成更困难的 delayed-action CMDP，这也是转向 PPO/更强策略优化主线的动机。**

## 2. 代码边界：v1 与暖机后的区别

关键提交：

| commit | 作用 | 判断 |
|---|---|---|
| `761cdee` | `Refine windblown RL rewards and subset-conditioned Q learning` | v1 DQN 强化点；引入/完善 subset-conditioned DQN 与 CMDP-DQN |
| `d62b43b` | `Implement three-state sensor warmup pipeline` | 正式把传感器变成 OFF/WARMING/READY 三态，并让环境级采样受暖机影响 |

在 `d62b43b` 之前，`SensorSpec` 里已有 `startup_delay` / `startup_peak_power` 字段，但 `AbstractSensor.can_sample()` 基本只做了：

```python
if t < self.spec.startup_delay:
    return False
return (t - self._last_sample_time) >= self.spec.refresh_interval
```

也就是说，旧逻辑更像“全局初始延迟 + 刷新间隔”，不是现在这种每次关机后再开都要经历的三态暖机。

`d62b43b` 后才新增了：

```python
_powered
_mode: off / warming / ready
_time_since_power_on
_warm_remaining_steps
begin_step(selected, t)
is_warming()
is_ready()
```

这会带来本质变化：选择一个传感器不再意味着本步或短期立即获得有效观测；策略必须提前开机、保持开机直到 ready，并避免中途放弃暖机。

## 3. v1 DQN 实现特征

v1 主要有两条 DQN 路径。

### 3.1 传统离散动作 DQN

文件：`src/scheduling/rl/dqn_agent.py`

特点：

- 固定离散动作表，动作是预枚举 sensor subset。
- Q 网络输出每个 action id 的 Q 值。
- replay buffer + target network + SmoothL1Loss。
- epsilon-greedy 探索。
- 适合 action space 较小、可枚举的早期实验。

局限：

- 对较多传感器的 subset 组合不友好。
- 动作可行性和上一步动作相关时，固定 action id 表达不够自然。

### 3.2 subset-conditioned DQN

文件：`src/scheduling/rl/score_dqn_agent.py`

特点：

- Q 网络不是输出固定 action id，而是输入 `(state, subset_mask)` 后给出该 subset 的分数。
- 候选动作由 `OnlineSubsetProjector` 在线生成可行 subset。
- `act()` 在可行 subset 中 epsilon-greedy 选择。
- `cmdp_dqn` 版本额外维护 `cost_q` 和拉格朗日乘子 `lambda_cost`，以平均功耗/能量约束修正动作评分。

这正是 v1 DQN 比旧离散 DQN 更合理的地方：它不再依赖完整枚举动作空间，而是能在在线约束下评分候选 sensor subset。

但 v1 仍然主要解决的是“当前选哪个 subset 最合适”。三态暖机后，真正困难的是“现在是否值得为了未来收益预热某个传感器，以及是否要持续保持它直到 ready”，这是更强的时序 credit assignment。

## 4. 历史结果摘要

以下结果来自本地 `reports/aggregate/metrics_forecast_all_<run_tag>_scheduler_summary.csv`。指标以 `rmse_mean` 排名，越低越好；`save%` 是相对 `full_open` 的功耗节省。

### 4.1 暖机前/弱暖机 v1：DQN 曾经较优

| run tag | 最好策略 | DQN/CMDP-DQN 表现 | 解读 |
|---|---:|---:|---|
| `full_mainline_20260327` | `full_open` RMSE 3.017 | `dqn` 第 2，RMSE 3.041，save 46.5% | DQN 接近 full_open，略优于 periodic/round_robin |
| `full_mainline_20260329` | `cmdp_dqn` RMSE 3.262 | `cmdp_dqn` 第 1，`dqn` 第 3 | v1 中最强证据；CMDP-DQN 明显优于规则法 |
| `routeA_ctx_20260408_0335` | `dqn/cmdp_dqn` RMSE 0.4285 | 二者并列第 1，save 46.5% | context-aware 后 DQN/CMDP-DQN 明显优于 periodic/round_robin |
| `full_schemeA_v1` | `dqn` raw RMSE 1.985e-05 | `dqn` raw RMSE 低于 full_open | 早期 Scheme A 中 DQN 确有最佳 raw RMSE；但该表的 dRMSE 字段与 raw 排名不完全一致，需谨慎引用 |

代表性明细：

`full_mainline_20260329`

| rank | scheduler | RMSE | dRMSE vs full_open | save% |
|---:|---|---:|---:|---:|
| 1 | `cmdp_dqn` | 3.262 | -45.62% | 65.47% |
| 2 | `random` | 4.255 | -28.77% | 63.40% |
| 3 | `dqn` | 5.750 | -0.94% | 44.86% |
| 4 | `full_open` | 6.002 | 0.00% | 0.00% |
| 6 | `periodic` | 6.069 | +1.35% | 48.40% |
| 7 | `round_robin` | 6.069 | +1.36% | 48.40% |

`routeA_ctx_20260408_0335`

| rank | scheduler | RMSE | dRMSE vs full_open | save% |
|---:|---|---:|---:|---:|
| 1 | `cmdp_dqn` | 0.4285 | -5.05% | 46.47% |
| 2 | `dqn` | 0.4285 | -5.05% | 46.47% |
| 3 | `round_robin` | 0.4355 | -3.73% | 48.40% |
| 4 | `periodic` | 0.4355 | -3.72% | 48.40% |
| 5 | `full_open` | 0.4549 | 0.00% | 0.00% |

### 4.2 暖机前也有不少 DQN 不占优的 run

| run tag | DQN 排名/表现 | 说明 |
|---|---:|---|
| `full_refactor_v1` | `dqn` 第 2，略差于 `periodic` | DQN 接近最优，但不是第一 |
| `full_refactor_online_20260324` | `dqn` 第 5 | periodic/full_open/info_priority/round_robin 都更好 |
| `full_subsetq_primary_20260325` | `dqn` 第 4 | periodic 与 full_open 更强 |
| `routeA_4090_20260404_153124` | `cmdp_dqn` 第 3，`dqn` 第 5 | periodic/round_robin 优于 DQN |

这说明 v1 DQN 的强结果真实存在，但不稳定。它依赖当时的 reward oracle、目标集、预训练覆盖、数据分布和 action projector 设定。

### 4.3 暖机引入后：DQN 明显变弱

| run tag | 现象 | 解读 |
|---|---|---|
| `warmup_ctx_complex_dqnonly_20260414_0049` | `cmdp_dqn` 第 2，`dqn` 第 3，但都远差于 full_open | 暖机复杂场景中 RL 优于多数规则法，但 forecast loss 放大严重 |
| `staleness_medium_20260426_172341` | `dqn` 第 5，periodic/round_robin 更好 | 规则轮换仍很强，DQN 没学出明显上下文优势 |
| `cmdp_active_medium_20260428_004154` | `cmdp_dqn` 第 2，明显优于规则法；`dqn` 接近 periodic | CMDP-DQN 有恢复迹象，但仍未达到 full_open |
| `routeA_warmup_full_20260426_005425` | `info_priority` 第 2，DQN/CMDP-DQN 大幅变差 | 暖机 reward/action 机制仍有结构性难点 |

代表性明细：

`cmdp_active_medium_20260428_004154`

| rank | scheduler | RMSE | dRMSE vs full_open | save% |
|---:|---|---:|---:|---:|
| 1 | `full_open` | 5.952 | 0.00% | 0.00% |
| 2 | `cmdp_dqn` | 6.782 | +13.95% | 76.95% |
| 3 | `round_robin` | 7.418 | +24.65% | 70.36% |
| 4 | `periodic` | 7.425 | +24.76% | 70.36% |
| 5 | `dqn` | 7.427 | +24.80% | 69.23% |

这说明暖机后 CMDP-DQN 不是完全无效；但它的优势不稳定，且普通 DQN 更容易退化到与规则轮换相近。

## 5. 为什么 v1 DQN 看起来更好

### 原因 1：动作后果更即时

无三态暖机时，开启传感器通常很快带来观测收益；DQN 的一步 TD 更新更容易把“选这个 subset”与“预测误差变好”关联起来。

三态暖机后，动作有延迟：

- 本步选择传感器，可能只是在 warming。
- warming 中途断电会浪费已投入的功率。
- 真正收益可能在若干步后才出现。
- 同一个动作的价值强烈依赖上一步是否已经在暖机、还剩几步 ready、未来目标是否会需要它。

这会显著增加 DQN 的 credit assignment 难度。

### 原因 2：规则 baseline 在简单结构里天然很强

v1 许多场景仍然接近“低成本基础传感器常开 + 高功耗传感器轮换”。这种结构对 `periodic` / `round_robin` 非常友好。DQN 能赢的 run 往往是 reward/target/context 让静态轮换不再那么接近最优。

### 原因 3：部分早期 full_open 不是干净的科学上界

早期曾出现 full_open 噪声注入/分布偏置问题，所以某些 run 中 `periodic`、`round_robin`、`cmdp_dqn` 优于 full_open 并不能直接解释为“少观测物理上更好”。它更可能说明旧 pipeline 里 full_open 输入分布或滤波处理存在偏置。后续 v2 修正方向已经让 full_open 重新接近上界。

### 原因 4：v1 CMDP-DQN 的“好”可能来自合适的功耗约束形状

`cmdp_dqn` 在 `full_mainline_20260329` 表现非常强，但同一时期也有 `cmdp_dqn` 崩坏的 run，例如 `full_mainline_20260328_rerun`、`cmdp_full_20260324`。这说明它对 dual、budget、oracle、训练稳定性敏感，不能简单当成稳定最优算法。

## 6. 与当前 v2 DQN 补充实验的关系

当前 v2 DQN 补充实验 `reports/v2_dqn_supplement_20260506/diagnosis.md` 显示，在 `budget=1.70`、3 个 seed 下：

| variant | full_open | static projected | round_robin | dqn | random |
|---|---:|---:|---:|---:|---:|
| `D1_warm2_500k_nstep3` | 0.3685 | 0.3889 | 0.4024 | 0.4044 | 0.4327 |
| `D2_full_500k_nstep8` | 0.3681 | 0.3888 | 0.4033 | 0.4133 | 0.4334 |
| `D4_full_500k_nstep8_prefill3k_lh8` | 0.3692 | 0.3892 | 0.4028 | 0.4223 | 0.4338 |

v2 DQN 已经优于 random，但没有稳定超过 round_robin。这和 v1 的历史复盘并不矛盾：

- v1 的成功来自较简单、即时反馈的 subset 选择。
- v2 的难点是 warmup-aware、forecast-driven、delayed-action scheduling。
- 如果论文主线强调真实仪器暖机，PPO/策略梯度路线比强行复活 DQN 更自然。

## 7. 建议

1. 保留 v1 DQN 作为历史/消融证据：说明在无暖机或弱延迟场景中，value-based subset scheduling 可以工作。
2. 不建议把 v1 DQN 直接作为当前 warmup 主线的理论依据；两者 MDP 动力学不同。
3. 如果继续保留 DQN baseline，应定位为“value-based baseline”，不是主算法。
4. 若要让 DQN 在暖机场景继续变强，应该做结构性改造，而不是只调参：
   - 把 warm remaining / ready state / previous action 显式纳入状态，并验证归一化。
   - 使用 n-step return 或 TD(lambda)，缩短暖机延迟造成的 credit gap。
   - 对“中途放弃暖机”加入明确 transition penalty 或 action mask 约束。
   - 让候选动作包含 commitment-aware subsets，例如保持当前 warming sensor 至 ready 的动作优先候选。
   - 若仍用 CMDP-DQN，dual 更新应只约束平均功耗/能量，不应把省电当作额外主目标。
5. 论文主线更建议表述为：
   - 无暖机场景：DQN/CMDP-DQN 已能达到接近或优于规则法的调度。
   - 有暖机场景：问题变为 delayed-action constrained scheduling，DQN 的稳定性下降，PPO 更适合作为主线策略优化方法。

## 8. 证据文件

关键实现：

- `src/scheduling/rl/dqn_agent.py`
- `src/scheduling/rl/score_dqn_agent.py`
- `src/scheduling/online_projector.py`
- `src/sensors/base_sensor.py`
- `src/sensors/dataset_sensor.py`

关键结果表：

- `reports/aggregate/metrics_forecast_all_full_mainline_20260329_scheduler_summary.csv`
- `reports/aggregate/metrics_forecast_all_routeA_ctx_20260408_0335_scheduler_summary.csv`
- `reports/aggregate/metrics_forecast_all_routeA_4090_20260404_153124_scheduler_summary.csv`
- `reports/aggregate/metrics_forecast_all_warmup_ctx_complex_dqnonly_20260414_0049_scheduler_summary.csv`
- `reports/aggregate/metrics_forecast_all_cmdp_active_medium_20260428_004154_scheduler_summary.csv`
- `reports/v2_dqn_supplement_20260506/diagnosis.md`

