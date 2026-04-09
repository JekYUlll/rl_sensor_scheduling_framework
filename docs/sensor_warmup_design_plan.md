# 传感器暖机三态机制设计文档

## 1. 背景与目标

当前 Route A 主线已经具备：

- 功率约束下的多传感器调度；
- 共享真值回放环境；
- 估计器闭环更新；
- frozen forecast reward；
- 调度后统一 forecasting 评估。

但当前传感器层仍然默认：

- 只要某一步被选中，就立即尝试产生观测；
- 代价主要表现为稳态功耗和启动峰值功耗；
- `startup_delay` 仅表现为 episode 初期不可采样，而不是“每次重新上电都需要预热”。

这会导致一个现实系统中很重要的工程因素缺失：

**高价值传感器往往不是“现在打开、现在就有高质量观测”，而是需要连续上电预热若干步后才真正可用。**

本设计文档的目标是把“暖机”提升为调度问题的核心结构，使其成为：

- 复杂场景中的关键工程约束；
- DQN / CMDP-DQN / PPO 与简单周期策略真正拉开差距的重要机制；
- 论文中可解释、可复现、可对比的主要创新点之一。

---

## 2. 设计原则

本次暖机机制设计遵循以下原则。

### 2.1 先做最小但完整的机制

第一版不追求所有物理细节，而追求：

- 语义明确；
- 能进入主线；
- 能改变调度问题本质；
- 能稳定训练与评估。

### 2.2 不引入“为了让 RL 赢”的不自然设定

暖机机制必须能对应真实仪器特性，例如：

- 激光类、红外类、粒子谱类传感器需要预热；
- 低温环境下电子与光学部件稳定时间更长；
- 上电后前几步数据质量不足，不能视作可直接用于控制或预测的正常观测。

### 2.3 保持问题的 Markov 性

如果暖机状态会影响未来可观测性，RL 状态中必须显式包含暖机进度。否则：

- 当前状态不足以决定未来转移；
- 策略学习会退化为部分可观测问题；
- DQN / PPO 的训练稳定性会下降。

### 2.4 第一版不把所有复杂性一起引入

第一版只引入：

- 三态状态机；
- 暖机期间耗电；
- 暖机期间无有效观测；
- 关闭后重新暖机。

以下内容放在后续扩展版：

- 暖机期间高噪声而非完全无观测；
- 暖机期间与 ready 期不同的稳态功耗；
- 待机态 / 热保持态；
- 冷却时间；
- 状态相关暖机长度。

---

## 3. 新问题定义

加入暖机后，当前问题不再只是“每一步选择一个满足功率约束的传感器子集”，而变成：

**在功率与峰值约束下，对具有延迟收益的异质传感器进行状态依赖的承诺式调度。**

换句话说：

- 打开某个高价值传感器不再立刻带来信息；
- 如果预计未来一段时间内该传感器会重要，策略必须提前打开；
- 如果打开后过早关闭，则前面的预热成本可能全部浪费；
- 简单的 step-by-step 轮换策略会因为频繁重启而变差；
- 能进行前瞻与持有决策的策略会更有优势。

这正是暖机机制对方法层面的核心贡献。

---

## 4. 三态状态机定义

### 4.1 状态集合

每个传感器在任意时刻处于以下三种状态之一：

1. `OFF`
2. `WARMING`
3. `READY`

### 4.2 第一版语义

#### `OFF`

- 当前未上电；
- 不提供观测；
- 不消耗稳态功率；
- 如果本步首次被选中，则进入 `WARMING`。

#### `WARMING`

- 当前已上电但尚未达到可用状态；
- 消耗稳态功率；
- 本步不提供有效观测；
- 记录剩余暖机步数 `warm_remaining_steps`；
- 如果连续保持开启并倒计时归零，则转入 `READY`。

#### `READY`

- 当前已完成预热；
- 消耗稳态功率；
- 在满足刷新间隔的情况下提供正常观测；
- 若本步被关闭，则回到 `OFF`。

### 4.3 关闭与重新启动

第一版默认：

- 只要从 `WARMING` 或 `READY` 切换到未选中，就立即回到 `OFF`；
- 下一次再被选中时，必须从头暖机；
- 不保留“半热状态”或“待机保温状态”。

这一定义最简单，也最容易解释。

### 4.4 时间语义

设某传感器 `warmup_steps = 3`，且在时刻 `t` 首次从 `OFF` 被选中，则：

- `t`：进入 `WARMING`，无有效观测；
- `t+1`：仍为 `WARMING`，无有效观测；
- `t+2`：仍为 `WARMING`，无有效观测；
- `t+3`：首次进入 `READY`，从这一刻开始允许产生正常观测。

也就是说，`warmup_steps` 表示“从首次上电到首次可正常采样之间需要连续保持开启的步数”。

---

## 5. 配置设计

建议在传感器 yaml 中新增以下字段。

### 5.1 第一版必需字段

- `warmup_steps: int`
  含义：从 `OFF` 到 `READY` 需要连续保持选中的步数。

### 5.2 第一版保留但不一定启用的字段

- `warmup_observation_mode: none | degraded`
  第一版默认 `none`。
- `warmup_noise_scale: float`
  仅在 `degraded` 模式下使用。

### 5.3 现有字段继续保留

- `refresh_interval`
- `power_cost`
- `startup_peak_power`
- `noise_std`

### 5.4 建议的第一版传感器分层

建议简单分层如下：

- backbone sensors：`warmup_steps = 0`
- medium sensors：`warmup_steps = 1 ~ 2`
- heavy snow sensors：`warmup_steps = 3 ~ 6`

这样既能体现真实差异，也不会一开始就把问题做得过难。

---

## 6. 对当前调度框架的影响

## 6.1 传感器层

当前 [base_sensor.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/sensors/base_sensor.py) 和 [dataset_sensor.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/sensors/dataset_sensor.py) 只支持：

- 被选中时尝试采样；
- 基于 `refresh_interval` 判断可否采样；
- 生成观测并返回。

要支持暖机，需要把传感器从“无内部模式的函数对象”改成“有内部状态机的对象”。

建议新增内部状态：

- `mode`
- `warm_remaining_steps`
- `time_since_power_on`

建议新增接口语义：

- `begin_step(selected: bool, t: int)`：先更新内部模式；
- `observe(...)`：在状态更新完成后决定是否给出有效观测；
- `reset()`：回到 `OFF`。

第一版不强制改接口名称，但必须做到语义上把“状态推进”和“观测生成”区分开。

### 关键改动点

- [base_sensor.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/sensors/base_sensor.py)
- [dataset_sensor.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/sensors/dataset_sensor.py)
- [windblown_sensor.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/sensors/windblown_sensor.py)

---

## 6.2 环境层

当前 [truth_replay_env.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/envs/truth_replay_env.py) 在 `step()` 里：

- 对被选中的传感器调用 `observe()`；
- 只有 `available=True` 时，才把该传感器加入观测列表；
- 同时只在 `available=True` 时累加功率。

加入暖机后，这个逻辑必须修改，因为：

- `WARMING` 期间传感器没有有效观测；
- 但仍然应该消耗功率；
- 否则暖机在资源意义上是“免费的”，会把机制做假。

因此环境层需要区分：

- `powered selected sensors`
- `available observed sensors`

第一版建议：

- 只要传感器本步处于 `WARMING` 或 `READY` 且被选中，就计入功率；
- 只有 `READY` 且满足刷新间隔时，才计入 `available_observations`。

这意味着环境返回值里最好新增：

- `powered_sensor_ids`
- `warming_sensor_ids`
- `ready_sensor_ids`

---

## 6.3 估计器与 RL state

当前 RL state 主要来自 [kalman_filter.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/estimators/kalman_filter.py) 和 [state_summary.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/estimators/state_summary.py)，包含：

- `x_hat`
- `diag_P`
- `trace_P`
- `freshness`
- `coverage_ratio`
- `budget_ratio`
- `previous_action`
- `event`
- 昼夜相位

这对暖机问题还不够，因为策略需要知道：

- 哪些传感器虽然当前没有观测，但已经在预热；
- 还要等多久才会变成 `READY`；
- 哪些传感器刚关掉，下一次开启要重新付出预热成本。

### 第一版建议新增 RL state 字段

- `sensor_mode`
  每个传感器 one-hot 或数值编码：
  - `0 = off`
  - `1 = warming`
  - `2 = ready`
- `warm_remaining_norm`
  每个传感器剩余暖机步数的归一化值。

### 第一版可以不加的字段

- `time_since_power_on`
- `cooldown_remaining`
- `warm_quality_level`

只加 `sensor_mode + warm_remaining_norm`，通常已经足以恢复 Markov 性。

---

## 6.4 动作层与可行子集投影器

当前 [online_projector.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/scheduling/online_projector.py) 已经支持：

- 稳态功率约束；
- 启动峰值约束；
- 安全裕度；
- 最大激活数。

### 第一版是否必须改 projector

**不一定。**

如果第一版采用：

- 暖机期间 steady power 与 ready 相同；
- warm-up 只影响“是否有有效观测”；
- `OFF -> WARMING` 仍然沿用现有 startup peak；

那么 projector 的核心可保持不变，因为它只需要知道：

- 本步选了哪些传感器；
- 上一步选了哪些传感器；
- 新启动的传感器要支付 startup peak。

### 何时需要改 projector

如果后续引入：

- 暖机期间特殊功耗；
- 已在 warming 的传感器与 ready 传感器有不同 steady cost；
- 热保持/待机态；

则 projector 需要显式接收当前 per-sensor mode。

因此本设计建议：

- **第一版不改 projector 的接口**
- 只改环境与传感器层的可观测性语义

这样最稳。

---

## 6.5 reward 与训练目标

当前 reward 由 [mainline_reward.py](/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/src/reward/mainline_reward.py) 计算，主要依赖：

- `forecast_loss`
- `switch_penalty`
- `coverage_penalty`
- `violation_penalty`

加入暖机后，reward 的意义会发生两个变化。

### 变化 1：切换代价的工程含义更强

当前切换惩罚更多是“减少频繁抖动”的 regularizer。  
加入暖机后，频繁切换会真实导致：

- 持续耗电；
- 多次重新预热；
- 信息长期拿不到。

因此 `lambda_switch` 的解释会更自然。

### 变化 2：reward 必须看得到延迟收益

如果某个 heavy sensor 需要 `warmup_steps = 5`，但 reward oracle 只关注 `horizon = 3`，那么：

- 策略现在打开它；
- 未来 3 步内仍拿不到信息；
- reward 中看不到任何收益；
- 模型会倾向于永远不预热该传感器。

因此，只要引入暖机，就必须同步检查：

- frozen reward oracle 的 `lookback`
- `horizon`
- `horizon_weights`

### 第一版建议

- 若引入 warmup，重审 reward oracle 的 `horizon`
- 保证 `horizon >= max_warmup_steps + 1` 至少在主要 heavy sensors 上成立

否则暖机机制在训练目标里会天然吃亏。

---

## 6.6 reward predictor / frozen oracle 预训练

当前 reward predictor 预训练会混合多种 scheduler rollout。  
如果主线加入暖机，那么 reward predictor 预训练的 rollout 也必须覆盖：

- `off -> warming -> ready` 的状态转移；
- warming 期间“有功耗、无有效观测”的样本；
- 重启与持续开启的不同模式。

否则 reward oracle 学到的输入分布仍然过于接近“瞬时可观测”旧世界。

第一版建议：

- 使用新的暖机语义重建 reward pretrain rollouts；
- 保留 rule-based 与 random subset 切换数据；
- 增加更长 hold-time 的 rollout，避免数据集中全是快速切换。

---

## 6.7 数据集导出与绘图

当前 `03_build_forecast_dataset.py` 导出的主要字段包括：

- `input_series`
- `target_series`
- `observed_mask`
- `power`
- `trace_p`
- `time_index`

加入暖机后，仅靠 `observed_mask` 已经不够，因为它无法区分：

- 传感器关闭
- 传感器 warming
- 传感器 ready 但本步未被选中

### 第一版建议新增导出字段

- `powered_mask`
- `warming_mask`
- `ready_mask`
- `warm_remaining`

### 对后处理的影响

当前传感器时间线图只有 on/off 两态。  
加入暖机后，建议改成三种颜色：

- `off`
- `warming`
- `ready`

这样才能直观看出：

- 策略是否提前开机；
- 是否经常在快 ready 时又关掉；
- DQN 是否学会持续保持 heavy sensor。

---

## 7. 对不同调度方法的影响

## 7.1 对 `periodic` / `round_robin`

暖机机制会显著削弱简单轮换策略的效果，因为：

- 轮换意味着频繁切换；
- 高频切换意味着重复预热；
- 重复预热意味着高能耗但低有效观测。

这正是暖机机制能帮助复杂场景拉开差距的原因之一。

## 7.2 对 `dqn`

`dqn` 会从“即时选择收益更高的子集”转向：

- 是否该提前打开某个未来会有价值的传感器；
- 是否该继续保持 warming/ready 状态；
- 是否该为未来事件期牺牲当前一步的局部最优。

这更接近真正的序列决策问题。

## 7.3 对 `cmdp_dqn`

`cmdp_dqn` 的价值会更明显，因为暖机会自然放大：

- 平均功率约束；
- 总能量约束；
- 启动峰值与持续保持之间的 tradeoff。

即使预测效果相近，`cmdp_dqn` 也更有机会在长期资源约束上优于普通 `dqn`。

## 7.4 对 `ppo`

`ppo` 理论上也能受益，但由于当前实现仍偏 CPU-bound，且 PPO 在此类离散/组合/延迟收益任务里不一定天然占优，因此：

- 可以继续保留为对照；
- 但不建议把暖机场景的主要结论建立在 PPO 是否成功上。

---

## 8. 对 baseline 公平性的影响

为避免“人为打压 baseline”的质疑，暖机引入后不应只保留原始 `periodic` 与 `round_robin`。

建议至少新增一个**暖机友好规则基线**，例如：

- `sticky_periodic`
  思路：一旦打开某类高价值传感器，至少保持 `warmup_steps + hold_margin` 步；
- 或 `warmup_aware_round_robin`
  思路：只在 ready 后持有一段时间，再轮换。

这样论文叙事会更公平：

- 不是“拿一个明显不适应暖机机制的 baseline 去陪跑”；
- 而是“即使给了规则法暖机友好的改造，学习法在复杂上下文下仍可能更优”。

---

## 9. 建议的实施阶段

## 阶段 1：最小可运行版本

目标：在不重写主线的前提下，把暖机真正接入。

实施内容：

1. 传感器三态状态机；
2. warming 期间耗电但无有效观测；
3. RL state 加入 `sensor_mode` 与 `warm_remaining_norm`；
4. dataset 导出 warming 相关字段；
5. timeline 图支持三态；
6. reward oracle `horizon` 与 warmup 长度匹配。

不做的内容：

- degraded observation
- warmup power 单独建模
- standby / cooldown

## 阶段 2：暖机友好 baseline

目标：让规则法对照更公平。

实施内容：

- 新增 `sticky_periodic` 或 `warmup_aware_round_robin`

## 阶段 3：更真实的物理细节

可选扩展：

- warming 期间高噪声而不是完全无观测；
- warmup power 与 ready power 区分；
- standby / hot-start；
- 事件驱动预热策略。

---

## 10. 建议的验收标准

暖机机制合入主线前，至少满足以下验收标准。

1. 状态机语义明确且可测试：
   - `OFF -> WARMING -> READY`
   - 关闭后重新暖机
2. warming 期间会耗电，但 `observed_mask` 不会误记为已观测
3. RL state 中包含足够暖机信息，不再依赖隐式历史推断
4. dataset 能区分 `off / warming / ready`
5. 时间线图可以直接看出“提前预热”行为
6. reward oracle 的 horizon 不会系统性短于主要 heavy sensor 的 warmup
7. complex regime 下，规则轮换的优势进一步减弱，DQN/CMDP-DQN 有机会体现 anticipatory 行为

---

## 11. 最终推荐

如果把暖机机制作为本项目的核心创新点之一，推荐的落地顺序是：

1. **先做第一版三态暖机机制**
2. **同步把 warm_remaining 接入 RL state**
3. **调整 reward oracle horizon**
4. **补一个暖机友好规则基线**
5. **在 complex regime 上重跑对照实验**

这条路线的好处是：

- 改动足够集中；
- 机制本身有明确工程解释；
- 能显著改变调度问题难度；
- 对论文叙事非常友好。
