# 调度算法基线说明

## 1. 文档目的

本文件用于统一说明 `rl_sensor_scheduling_framework` 中各个调度算法的角色、实现逻辑与论文中的比较意义。

当前实验的基本设定是：

- 在共享的高频 truth 环境上进行多传感器调度；
- 传感器受到 `per_step_budget` 与 `max_active` 约束；
- 调度器输出的是当前时刻要开启的传感器子集；
- 不同调度器生成不同的估计状态序列，再交给统一的下游预测模型训练与评估。

因此，调度算法的比较目标不是“谁更省电”这一件事，而是：

> 在预算受限条件下，谁能保留对目标预测最有价值的信息。

## 2. 统一约束与比较口径

所有调度器都使用同一个离散动作空间：

- 传感器子集由 `DiscreteActionSpace` 预先枚举；
- 每个动作必须满足总功耗不超过 `per_step_budget`；
- 同时开启的传感器数不超过 `max_active`；
- 在当前 windblown 设定下，基础气象类传感器功耗较低，风吹雪专用传感器功耗较高。

论文中应统一使用以下两类指标：

- 估计层指标：
  - `trace_P_mean`
  - `power_mean`
  - `coverage_mean`
- 预测层指标：
  - `rmse`
  - `mae`
  - 相对 `full_open` 的误差增幅

其中 `full_open` 应视为信息最充分的 oracle 上限；预算受限算法的目标是尽量逼近它，而不是理论上超越它。

## 3. 各调度算法说明

### 3.1 `full_open`

配置：

- `configs/scheduler/full_open.yaml`

实现：

- `src/scheduling/baselines/full_open_scheduler.py`

逻辑：

- 每个时间步都尝试开启所有传感器；
- 实际动作仍需通过动作空间约束，因此它代表的是“预算允许条件下尽量全开”的 oracle baseline。

作用：

- 作为信息最充分的参考上限；
- 用于比较预算受限策略的误差保持能力；
- 所有 scheduler 的预测误差通常都要与它做差分比较。

注意：

- `full_open` 不是直接输入真值，而是“全开传感器 + 观测噪声 + estimator”后的结果；
- 因此它在单次实验、个别模型上偶尔不占优，并不等价于理论上信息更少的策略真的优于它。

### 3.2 `random`

配置：

- `configs/scheduler/random.yaml`

实现：

- `src/scheduling/baselines/random_scheduler.py`

逻辑：

- 每一步从可行动作空间中均匀随机采样一个动作；
- 不考虑不确定性、事件状态、覆盖率或历史动作。

作用：

- 作为最低限度的无结构基线；
- 用来验证“只要满足预算约束并随机开关传感器”通常不足以保留稳定预测性能。

预期特点：

- 节电通常较高；
- 误差波动大；
- 对不同随机种子较敏感。

### 3.3 `periodic`

配置：

- `configs/scheduler/periodic.yaml`

实现：

- `src/scheduling/baselines/periodic_scheduler.py`

逻辑：

- 按固定周期在离散动作空间中循环切换；
- 不依赖当前状态，只依赖时间步和内部指针。

作用：

- 代表“预先设定采样轮换计划”的工程基线；
- 用于和真正状态自适应策略区分。

预期特点：

- 可解释性强；
- 行为稳定；
- 但无法根据事件段或不确定性动态调整资源分配。

### 3.4 `round_robin`

配置：

- `configs/scheduler/round_robin.yaml`

实现：

- `src/scheduling/baselines/round_robin_scheduler.py`

逻辑：

- 以传感器列表顺序轮询；
- 每次优先选择一组传感器；
- 通过 `nearest_feasible(...)` 映射到满足预算约束的最近可行动作；
- `min_on_steps` 用于控制动作保持时间，避免过快切换。

作用：

- 代表“公平覆盖优先”的强结构化基线；
- 特别适合当前“低功耗基础气象 + 高功耗专用风吹雪传感器”的结构。

为什么它通常很强：

- 基础气象信息可以相对稳定保留；
- 高功耗风吹雪传感器按结构化方式轮换；
- 往往不会像随机策略那样丢失长期覆盖；
- 也不会像极端目标驱动策略那样塌缩到少量传感器。

当前实验中，如果 `round_robin` 在部分模型上接近甚至略优于 `full_open`，更应解释为：

- 输入表示还没有充分利用 `full_open` 的额外信息；
- 或 `round_robin` 带来了轻度正则化/去噪效应；
- 而不是它在信息论意义上优于 `full_open`。

### 3.5 `info_priority`

配置：

- `configs/scheduler/info_priority.yaml`

实现：

- `src/scheduling/baselines/info_priority_scheduler.py`

逻辑：

- 对每个传感器计算启发式评分；
- 评分由以下部分加权组成：
  - 不确定性
  - freshness
  - event 指示
  - coverage deficit
  - switch penalty
- 选择得分最高的若干传感器，再映射到最近的可行动作。

作用：

- 代表“基于任务相关启发式的自适应调度”；
- 是当前实验中非常关键的强基线。

特点：

- 比 `random` 和 `periodic` 更有针对性；
- 仍然不需要 RL 训练；
- 在目标变量较明确、驱动变量结构比较清晰时，通常表现非常强。

在当前 windblown 任务中，`info_priority` 强是合理现象，因为：

- 目标是 `snow_mass_flux_kg_m2_s`；
- 风速、辐射、雪面温度、粒径/粒子速度等驱动具有明显物理相关性；
- 一个设计合理的启发式评分，本来就可能接近最优手工策略。

### 3.6 `dqn`

配置：

- `configs/scheduler/dqn.yaml`

实现：

- `src/scheduling/rl/dqn_agent.py`
- `src/scheduling/rl/q_network.py`
- `src/scheduling/action_space.py`

逻辑：

- 使用离散动作 DQN；
- 通过 replay buffer、target network 和 epsilon-greedy 进行 value-based 学习；
- 学到的是 `Q(s, a)`，不是显式策略分布。

状态包含：

- 当前 Kalman 估计 `x_hat`
- 协方差对角 `diag_P`
- 总不确定性 `trace_P`
- freshness
- coverage ratio
- previous action
- event indicator

作用：

- 代表“可学习的自适应调度”；
- 目标是验证：在相同预算约束下，学习型策略是否能超过强规则基线。

当前解释要点：

- 旧版本 DQN 曾明显受到 reward 错配影响；
- 后续重构已将 reward 更强地对齐到预测目标；
- 但目前 DQN 是否最终优于 `round_robin` / `info_priority`，仍需依赖实验结果，而不能先验假定。

## 4. 论文中建议的基线分层

建议将调度基线分成三层解释：

### 第一层：oracle 上限

- `full_open`

含义：

- 信息最充分；
- 用作误差保持能力的比较上限。

### 第二层：无学习规则基线

- `random`
- `periodic`
- `round_robin`

含义：

- 分别对应无结构、固定周期、结构化公平覆盖三类典型调度方式。

### 第三层：任务感知与学习型方法

- `info_priority`
- `dqn`

含义：

- `info_priority`：强启发式任务感知基线
- `dqn`：学习型任务感知调度

论文中若要证明 RL 的价值，应优先证明它在稳定性或平均性能上超过 `info_priority`，而不是只超过 `random` 或 `periodic`。

## 5. 当前使用建议

在当前框架下，建议默认保留以下比较组：

- `full_open`
- `random`
- `periodic`
- `round_robin`
- `info_priority`
- `dqn`

如果篇幅受限，最不建议删除的是：

- `full_open`
- `round_robin`
- `info_priority`
- `dqn`

因为这四个算法基本覆盖了：

- 理论上限
- 强规则结构基线
- 强启发式自适应基线
- 学习型方法

这也是当前论文叙述最完整的一组比较对象。
