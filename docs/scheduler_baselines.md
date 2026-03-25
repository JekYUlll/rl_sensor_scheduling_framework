# 调度算法基线说明

## 1. 文档目的

本文件说明 `rl_sensor_scheduling_framework` 里各类调度算法的实际实现、适用范围和论文中的比较意义。当前框架已经分成两类调度问题：

- `linear_gaussian`：保留离散 `action_id` 形式，适合作为 toy benchmark；
- `windblown`：使用 **online projector + sensor scoring**，在每一步在线生成满足功率约束的传感器子集，不再依赖静态动作编号。

如果不区分这两类问题，就很容易误解 `periodic`、`round_robin`、`dqn`、`cmdp_dqn` 当前到底在选择什么。

## 2. Windblown 场景下的统一约束

当前主实验是 `windblown`，对应：

- `configs/env/windblown_case.yaml`
- `configs/sensors/windblown_sensors.yaml`

这里所有传感器都可调度。约束分成两层：

### 2.1 硬约束

由 `src/scheduling/online_projector.py` 负责在线投影：

- 瞬时稳态功率不超过 `per_step_budget`
- 启动/加热峰值功率不超过 `startup_peak_budget`
- 保留 `power_safety_margin`
- 同时开启传感器数不超过 `max_active`

这部分不是 reward 惩罚，而是动作可行域本身的定义。

### 2.2 长期约束

只对 `cmdp_dqn` 生效：

- 平均功耗约束 `average_power_budget`
- 单 episode 总能量约束 `episode_energy_budget`

这部分通过 dual variable 更新处理，不通过硬裁剪处理。

## 3. 动作是如何生成的

### 3.1 `linear_gaussian` 的旧机制

在 toy `linear_gaussian` 任务中，动作仍然是：

- `DiscreteActionSpace` 预枚举所有满足约束的传感器组合
- 调度器直接选择一个 `action_id`

这套机制仍然存在，但主要用于小规模离散动作的基准实验。

### 3.2 `windblown` 的现机制

在 `windblown` 中，主机制已经改成：

- 调度器先输出 **传感器分数 / 排序**；
- `OnlineSubsetProjector` 根据当前状态和上一步动作，在线求解可行子集；
- 对于传感器数较少的情况，projector 在运行时对所有候选子集做精确搜索；
- 传感器数变大时，再退化到 greedy 投影。

所以在当前主实验中：

- `periodic` 不是轮换预枚举 `action_id`
- `round_robin` 也不是先构造固定离散动作表
- `dqn/cmdp_dqn` 学的也不是静态动作编号

而是统一变成了：

> 输出每个传感器的优先级，再由在线约束投影器决定当前步真正开启哪些传感器。

## 4. 各调度算法说明

### 4.1 `full_open`

配置：

- `configs/scheduler/full_open.yaml`

实现：

- `src/scheduling/baselines/full_open_scheduler.py`

逻辑：

- 每一步都尝试开启全部传感器；
- 它作为 oracle baseline 保留，不走受约束的 projector；
- 作用是提供“信息最充分”参考上限，而不是部署可行策略。

解释口径：

- `full_open` 应视为 oracle 上限；
- 预算受限策略的目标是逼近它，而不是在理论上优于它；
- 如果局部模型上出现 `round_robin > full_open`，更应解释为输入表达或模型利用不足，而不是信息量真的更高。

### 4.2 `random`

配置：

- `configs/scheduler/random.yaml`

实现：

- `src/scheduling/baselines/random_scheduler.py`

逻辑：

- 对每个传感器先给随机分数；
- 再由 projector 在线投影为可行子集。

作用：

- 无结构随机基线；
- 用来说明“只满足功率约束”本身不足以保留有预测价值的信息。

### 4.3 `periodic`

配置：

- `configs/scheduler/periodic.yaml`

实现：

- `src/scheduling/baselines/periodic_scheduler.py`

逻辑：

- 先按固定节拍生成一个周期性传感器排序；
- 再交给 projector 求出当前时刻可行子集。

需要避免的误解：

- 在旧的离散动作空间里，`periodic` 的“周期”指 action id 轮换；
- 在当前 `windblown` 主实验里，`periodic` 的“周期”是对传感器优先级排序做周期切换；
- 它已经不再等价于“遍历预枚举动作列表”。

作用：

- 表示“预定义调度节奏”的工程基线；
- 用于和真正状态自适应策略区分。

### 4.4 `round_robin`

配置：

- `configs/scheduler/round_robin.yaml`

实现：

- `src/scheduling/baselines/round_robin_scheduler.py`

逻辑：

- 以传感器顺序做轮询；
- 每一步输出一个传感器级排序；
- 再经 projector 映射到当前可行子集。

与 `periodic` 的关键区别：

- `periodic`：按固定时间节奏给排序打分；
- `round_robin`：按传感器覆盖公平性推进指针；
- 因此 `round_robin` 更强调长期覆盖结构，而不是固定时间节拍。

额外说明：

- 此前 online projector 版本里，`round_robin` 曾有一个真实 bug：指针步长与排序长度耦合，导致指针不前进；
- 该问题已修复，当前实现确实会轮换高功耗传感器，而不是固定输出同一子集。

### 4.5 `info_priority`

配置：

- `configs/scheduler/info_priority.yaml`

实现：

- `src/scheduling/baselines/info_priority_scheduler.py`

逻辑：

- 为每个传感器计算启发式 score；
- score 结合：
  - uncertainty
  - freshness
  - event signal
  - coverage deficit
  - switch penalty
- 再由 projector 在约束下选出可行子集。

作用：

- 任务感知的强规则基线；
- 在变量关系较明确、启发式设计合理时，本来就可能非常强。

当前解释要点：

- `info_priority` 不是弱基线；
- 若 learned scheduler 不能超过它，通常说明 reward、状态表示或输入表达还不够对齐。

### 4.6 `dqn`

配置：

- `configs/scheduler/dqn.yaml`

实现：

- `src/scheduling/rl/score_dqn_agent.py`
- `src/scheduling/rl/q_network.py`

逻辑：

- 在 `windblown` 场景中，`dqn` 采用 factorized / score-based value learning；
- 网络不直接输出一个动作编号，而是输出每个传感器的 on/off 价值；
- 再由 `OnlineSubsetProjector` 在硬约束下选出当前可执行子集。

这与旧版静态 action-id DQN 的区别必须明确：

- 旧版：学 `Q(s, a_id)`；
- 现版 windblown：学每个传感器的局部开关价值，再在线组合。

作用：

- 用于验证学习型策略是否能在同等约束下超过强规则基线；
- 当前它已经摆脱了早期“极端省电 + 预测崩坏”的明显实现错误，但仍未稳定超过 `periodic / round_robin / info_priority`。

### 4.7 `cmdp_dqn`

配置：

- `configs/scheduler/cmdp_dqn.yaml`

实现：

- `src/scheduling/rl/constrained_dqn_agent.py`
- `src/scheduling/rl/score_dqn_agent.py`
- `src/scheduling/online_projector.py`

逻辑：

- 继承同样的 score-based DQN 主体；
- 瞬时稳态功率、启动峰值、供电安全裕度由 projector 做硬约束；
- 平均功耗和总能量用 CMDP dual variable 做长期约束；
- 因此它优化的是“在约束内尽量保住预测价值”，而不是“把功耗直接压进 reward 主目标”。

作用：

- 表示更符合 AWS 供电场景的 constrained RL 方案；
- 重点不是一定比 `dqn` 更准，而是要在不违反供电约束的前提下获得更合理的功率—精度折中。

### 4.8 `ppo`

配置：

- `configs/scheduler/ppo.yaml`

实现：

- `src/scheduling/rl/sb3_ppo.py`
- `src/scheduling/online_projector.py`

逻辑：

- 使用 Stable-Baselines3 的 `PPO` 作为现成开源 baseline；
- policy 不直接输出传感器子集，而是输出每个传感器的连续 score；
- `OnlineSubsetProjector` 再把 score 排序投影成满足硬功率约束的可执行子集；
- 因此 `ppo` 与当前 `windblown` 架构兼容，同时避免回到静态 `action-id` 设计。

作用：

- 提供一个来自成熟开源 RL 框架的对照基线；
- 用于和本地自定义的 `dqn / cmdp_dqn` 做实现层面对照；
- 若后续在 `cmdp_dqn` 上做创新，`ppo` 可以作为“通用现成 RL baseline”保留。

## 5. 当前论文里应如何分层使用

建议分三层：

### 第一层：oracle 上限

- `full_open`

### 第二层：规则型基线

- `random`
- `periodic`
- `round_robin`
- `info_priority`

### 第三层：学习型方法

- `dqn`
- `cmdp_dqn`
- `ppo`

当前最关键的比较对象不是 `random`，而是：

- `periodic`
- `round_robin`
- `info_priority`

如果 learned scheduler 不能接近这三者，就不能说明 RL 方案已经成熟。

## 6. 当前解释建议

当前 `windblown` 结果若出现：

- `round_robin`、`periodic` 很强；
- `dqn/cmdp_dqn` 只达到“可用但不占优”；

更合理的解释是：

- 当前任务的结构性强基线本来就非常接近工程合理策略；
- online projector 虽然修复了旧版动作空间失真问题，但 learned scheduler 仍受限于 reward 对齐、输入表达和状态摘要；
- 因此规则基线仍然是需要严肃对待的主要比较对象，而不是陪衬。
