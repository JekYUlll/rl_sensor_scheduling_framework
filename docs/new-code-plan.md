# v2主线代码框架与论文创新点计划

## 一、v2主线目标

v2主线的目标是从当前历史代码中提炼出一条更清晰、可复现、可写成论文的方法链：

> 公开气象数据锚定的物理合成环境 + 传感器暖机状态机 + 硬功率约束投影 + 冻结预测奖励Oracle + projected-score PPO主算法 + 固定功率预算下的预测误差评估。

这条主线只服务一个核心问题：

> 在每步功率不超过额定预算的条件下，调度哪些异构传感器，才能让后续多变量时序预测尽可能准确？

v2不再把“省电”作为独立优化目标。功率是约束，不是奖励目标。评价时可以报告节能率，但主表格应以固定功率约束下的预测误差为主。

v2也不再把OTT Parsivel²低温强制OFF作为核心创新点或必须实现机制。低温可用性可以作为未来扩展或数据标记保留，但当前主线只实现通用的传感器暖机、可用性标记和功率约束。

---

## 二、总体模块结构

v2按单向数据流划分为六个模块：

1. `PublicDataSynthesis`
2. `WarmupSchedulingEnv`
3. `PowerProjector`
4. `FrozenForecastOracle`
5. `SchedulingPolicies`
6. `EvaluationSuite`

数据流如下：

```text
AntAWS / ERA5 / blowing-snow priors
        -> PublicDataSynthesis
        -> truth table
        -> WarmupSchedulingEnv
        -> FrozenForecastOracle reward
        -> PPO / DQN / baselines training
        -> scheduler rollouts
        -> EvaluationSuite
```

---

## 三、模块一：PublicDataSynthesis

### 3.1 职责

该模块负责生成与当前调度环境兼容的多变量truth table。输出必须是数值稳定、无NaN的训练主表，同时可以附加事件和可用性标记列。

核心输出变量：

- `air_temperature_c`
- `wind_speed_ms`
- `wind_direction_deg`
- `wind_dir_sin`
- `wind_dir_cos`
- `relative_humidity`
- `air_pressure_pa`
- `solar_radiation_wm2`
- `snow_surface_temperature_c`
- `snow_particle_mean_diameter_mm`
- `snow_particle_mean_velocity_ms`
- `snow_mass_flux_kg_m2_s`
- `event_flag`
- 可选标记：`blowing_snow_active`、`parsivel_available`

### 3.2 生成方法

气象基础变量使用公开观测锚定的DFT/AAFT-like合成：

- 从本地 `data/AntAWS/3_hourly/` 读取AntAWS站点CSV。
- 优先使用 `Panda100`、`Panda200`、`Taishan` 作为第一版气象基信号。
- DFT相位随机化用于保留频谱和自相关结构。
- 经验分布重映射用于保留边际分布，避免普通相位随机化无法通过KS检验的问题。

吹雪变量采用条件参数化生成：

- 当风速低于阈值时，`snow_mass_flux_kg_m2_s = 0`。
- 当风速高于阈值时，用幂律关系生成质量通量：

```text
snow_mass_flux = a * max(wind_speed - threshold, 0)^alpha * lognormal_noise
```

- 粒径和粒子速度根据风速、吹雪事件和随机扰动条件生成。
- 暂不在truth主表中写入NaN。不可观测性使用标记列表达，避免破坏Kalman、Oracle和窗口化训练链路。

### 3.3 当前实现入口

已新增第一版入口：

- `scripts/20_build_public_weather_truth.py`
- `src/data_sources/public_weather_synthesis.py`
- `tests/test_public_weather_synthesis.py`

该入口会生成：

- truth CSV：可被后续环境读取。
- `synthetic_validation.csv`：记录KS、PSD MSE、ACF差异等统计保真度指标。
- `synthetic_metadata.json`：记录站点、步长、随机种子、合成参数。

### 3.4 后续增强

ERA5暂不作为第一版阻塞项。后续可加入ERA5用于：

- `solar_radiation_wm2` 年周期模板。
- 温度和风速背景场校验。
- AntAWS站点外推的稳健性分析。

PANGAEA粒径数据暂不作为第一版阻塞项。后续可用于校准粒径分布形状。

---

## 四、模块二：WarmupSchedulingEnv

### 4.1 职责

该模块是v2主线的Gym兼容环境，负责：

- 读取truth table。
- 维护传感器状态机。
- 接收调度动作。
- 生成受调度影响的观测序列。
- 调用冻结Oracle计算预测驱动奖励。
- 返回PPO/DQN可消费的状态、奖励、done和info。

### 4.2 传感器状态机

每个传感器具有三种状态：

```text
OFF -> WARMING_UP -> ACTIVE
```

状态含义：

- `OFF`：未供电，不产生观测，不消耗稳态工作功率。
- `WARMING_UP`：已供电但尚未稳定，消耗功率，可配置是否产生低质量观测。
- `ACTIVE`：已稳定，可产生正常观测。

每个传感器配置：

- `power_cost`
- `startup_peak_power`
- `warmup_steps`
- `observed_variables`
- `noise_std`
- `sampling_interval`

暖机机制是v2的重要工程创新之一。它让调度问题从“每步选哪些传感器”变成真正的序列决策：提前开启、持续保持、避免中途放弃暖机都会影响未来预测质量。

### 4.3 状态空间

v2状态应尽量简洁，避免历史版本中大量冗余特征：

- 最近 `lookback` 步的观测/估计矩阵。
- 每个传感器的状态：OFF/WARMING/ACTIVE。
- 每个传感器的剩余暖机步数。
- 每个传感器的freshness/AoI。
- 上一步动作mask。
- 当前功率占用比例。
- 时间特征：`time_of_day_sin`、`time_of_day_cos`。
- 可选事件标记：`event_flag`。

原则：只保留能影响调度决策的信息。`delta_*` 这类历史遗留特征默认不进入v2主线，除非消融实验证明确有价值。

### 4.4 动作空间

PPO主线采用 projected-score action：

- Actor输出每个传感器的连续score。
- 按score排序传感器。
- `PowerProjector` 根据硬约束选择可行子集。
- 环境执行投影后的子集。

这个设计比固定离散动作表更适合扩展到更多传感器，也和当前已有PPO包装思路更接近。

---

## 五、模块三：PowerProjector

### 5.1 职责

`PowerProjector` 是硬约束执行层。它不学习，不调参，只根据当前传感器状态和候选动作输出可行子集。

功率约束是v2的核心设定：

> 只允许执行不超过额定功率预算的动作；算法目标是在这个可行集合中最小化未来预测误差。

### 5.2 约束类型

第一版保留三类约束：

- `max_active`：同时处于供电/激活状态的传感器数量上限。
- `per_step_budget`：当前步稳态功耗上限。
- `startup_peak_budget`：当前步启动峰值功耗上限。

长期平均功率约束作为可选扩展：

- PPO主线默认不优化长期省电目标。
- CMDP-DQN可以使用 `average_power_budget` 做对照实验。

### 5.3 设计原则

- 不可行动作直接被投影或屏蔽，不通过奖励惩罚解决。
- 约束违反率应接近0。
- `full_open` 可以作为直观上界，但需标注为“不满足功率约束的理想参考”。

---

## 六、模块四：FrozenForecastOracle

### 6.1 职责

Oracle是两阶段训练的核心：

1. Phase 1：用完整truth和多种调度扰动预训练多步预测器。
2. Phase 2：冻结Oracle，用未来预测误差作为调度奖励。

Oracle不在线更新，不和PPO端到端联合训练。

### 6.2 预测任务

默认配置：

- `lookback = 20`
- `reward_horizon = 8` 或 `10`
- 输入：受调度影响的历史观测/估计序列。
- 输出：未来多步目标变量。

默认预测目标：

- `air_temperature_c`
- `snow_surface_temperature_c`
- `wind_speed_ms`
- `wind_dir_sin`
- `wind_dir_cos`
- `solar_radiation_wm2`
- `snow_mass_flux_kg_m2_s`
- `snow_particle_mean_diameter_mm`
- `snow_particle_mean_velocity_ms`

如果某些变量在当前数据生成版本中可预测性不足，应通过数据生成机制修复，而不是在奖励中临时删除。

### 6.3 奖励定义

主奖励只关注预测误差：

```text
reward_t = - weighted_MAE(predicted_future, true_future)
```

可选的小项：

- 暖机中途放弃惩罚：用于减少无效暖机。
- 频繁切换惩罚：用于减少高频抖动。

不应加入“越省电越好”的长期奖励项。功率通过约束控制，而不是通过奖励鼓励省电。

### 6.4 Oracle模型

第一版建议使用轻量、稳定的TCN作为Oracle，先保证奖励方向正确。

后续可扩展为：

- TCN
- LSTM
- Transformer
- ensemble

论文中如果主算法是PPO，不必同时把Oracle模型做得过于复杂。Oracle的角色是稳定奖励提供者，不是论文的第二个主创新点。

DTW先作为评估指标使用，不作为第一版在线reward：

- TCN Oracle负责提供非线性、多步预测损失。
- DTW用于检查曲线形状和相位偏移，防止线性或逐点误差指标掩盖时序形态问题。
- 若PPO在TCN reward下已经稳定，不额外引入DTW reward；若后续仍出现形状错位，再以很小权重加入近似DTW/Soft-DTW消融。

---

## 七、模块五：SchedulingPolicies

### 7.1 Baselines

保留三类基线：

- `random`：从可行子集中随机采样。
- `periodic / round_robin`：固定轮换或周期策略。
- `AoI / freshness`：优先选择最陈旧或最久未观测的传感器。

这些基线用于证明：

- 简单规则在简单场景中可能很强。
- 引入暖机、异构功耗和事件相关变量后，固定规则难以适应状态依赖的边际价值变化。

### 7.2 PPO主算法

PPO是v2论文主算法，建议实现为 projected-score PPO：

- Actor输出每个传感器score。
- Projector将score排序转换为可行动作。
- Critic估计投影后策略的状态价值。
- 使用并行环境提升采样速度。

实现优先级：

1. 先基于Stable-Baselines3包装跑通。
2. 如果速度仍不足，再迁移到CleanRL/TorchRL式向量化实现。
3. 不要一开始就追求复杂GPU加速；先保证环境step和奖励计算足够轻。

### 7.3 DQN对照

DQN作为value-based对照保留：

- 可使用动态可行集mask。
- 可作为证明PPO稳定性优势的对照。
- 不作为论文主算法。

### 7.4 CMDP-DQN对照

CMDP-DQN不再承担主线最优算法定位。它的作用是：

- 展示长期平均功率约束可以用Primal-Dual方式处理。
- 与PPO的projected hard constraint形成对照。
- 用于消融“长期平均预算是否必要”。

如果第一版实验中CMDP-DQN不稳定，可以放入附录或删去，不影响v2主线成立。

---

## 八、模块六：EvaluationSuite

### 8.1 主表格

主表格使用固定功率约束下的预测误差：

- MAE
- RMSE
- sMAPE
- Pearson
- DTW
- 平均功率
- 约束违反率
- 暖机放弃率

排序以预测误差为主，不以节电率为主。

理想结果叙事：

```text
full_open: 不满足功率约束的理想上界，预测最好。
PPO: 满足功率约束，在可行策略中预测最好或接近最好。
DQN/CMDP-DQN: 可行RL对照。
periodic/round_robin/AoI/random: 规则或非学习基线。
```

### 8.2 图表

核心图表：

- 预测曲线：true vs full_open vs PPO vs best rule baseline。
- 传感器状态Gantt图：OFF/WARMING/ACTIVE。
- 功率曲线：每步功率与预算线。
- Pareto图：预测误差 vs 平均功率。
- 消融图：无暖机、无projector、无forecast reward等。

### 8.3 Sim-to-Real定位

第一版不要把Sim-to-Real写成完整真实部署验证。AntAWS主要作为真实气象背景锚定来源。

可选实验：

- 用AntAWS真实气象变量驱动部分观测子集。
- 只在可观测变量上报告预测误差。
- 将完整五传感器吹雪系统仍定位为物理约束合成环境。

---

## 九、论文创新点定位

### 创新点一：预测驱动的传感器调度目标

v2用冻结多步预测Oracle的未来误差作为奖励，替代AoI、覆盖率、KF后验协方差等代理目标。

核心论点：

- AoI只描述新鲜度，不描述变量对未来预测的边际贡献。
- KF协方差依赖线性高斯假设，不适合重尾、阈值触发的吹雪变量。
- 预测驱动奖励直接对齐最终任务：在功率受限下保持数字孪生预测质量。

### 创新点二：暖机感知的硬约束调度环境

v2显式建模传感器暖机状态、启动峰值功耗和稳态功耗预算。

核心论点：

- 暖机让调度具有历史依赖和提前规划需求。
- 规则轮换容易产生无效暖机或错过事件窗口。
- Projector把物理约束作为动作空间硬约束，而不是奖励惩罚。

### 创新点三：两阶段解耦训练范式

v2先训练预测Oracle，再冻结Oracle训练调度策略。

核心论点：

- 避免端到端训练中预测器和策略相互漂移。
- Oracle可用离线数据充分训练。
- PPO可在稳定奖励面上探索调度策略。

---

## 十、建议实现顺序

### Stage 0：冻结当前历史主线

保留当前代码，不再继续在旧链路上大幅叠补丁。旧链路作为结果对照和代码素材库。

### Stage 1：数据与环境最小闭环

- 完善 `PublicDataSynthesis`。
- 新建或封装 `WarmupSchedulingEnv`。
- 实现简洁状态空间。
- 实现 `PowerProjector`。
- 用随机策略跑通环境rollout。

当前进度：

- 已新增独立v2路径 `src/v2/`，不依赖旧 `truth_pipeline.py`。
- 已实现 `SensorSpecV2`、`SensorRuntime`、`PowerProjector`、`WarmupSchedulingEnv`。
- 已新增 `scripts/21_v2_smoke_rollout.py`，可用public weather truth CSV和复杂传感器配置跑随机projected-score rollout。
- 已新增 `tests/v2/test_warmup_env.py`，覆盖投影约束、暖机中断和环境step。
- 当前v2环境已支持冻结预测Oracle奖励；未传入Oracle时才回退到单步观测误差占位。

### Stage 2：Oracle最小闭环

- 用TCN训练冻结Oracle。
- 验证full_open预测误差最低。
- 验证不同调度产生不同预测误差。

当前进度：

- 已新增轻量 `LinearFrozenForecastOracle`，用于v2本地/烟测闭环，不依赖PyTorch。
- 已新增 `TCNFrozenForecastOracle`，用于v2主线冻结预测奖励；训练脚本可通过 `--oracle-type tcn` 启用。
- `WarmupSchedulingEnv` 已支持传入冻结Oracle，并用未来预测误差替换单步观测误差作为reward。
- 已新增 `scripts/22_v2_run_pipeline.py`，可运行 `truth -> oracle训练 -> 多策略评估 -> metrics/rollout/图表`。
- 线性ridge版本保留为快速回归测试和消融对照；论文实验默认使用TCN Oracle。

### Stage 3：PPO最小闭环

- 使用projected-score PPO训练。
- 与random、round_robin、AoI、DQN比较。
- 重点检查是否存在无效暖机、只开低功耗传感器、reward尺度异常等问题。

当前缺口：

- 已新增 Stable-Baselines3 版 projected-score PPO 适配层 `src/v2/sb3_ppo.py`。
- 已新增训练入口 `scripts/23_v2_train_ppo.py`，完整执行 `truth -> oracle -> PPO训练 -> PPO/规则策略评估 -> metrics/model artifact`。
- PPO动作是每个传感器的连续score，环境通过 `PowerProjector` 投影为满足硬功率约束的可执行子集。
- 远端GPU训练建议使用 `--device cuda --n-envs 8/16/32 --vec-type subproc`。GPU负责PPO网络前向/反向更新，CPU并行环境负责rollout采样。
- `scripts/23_v2_train_ppo.py` 已保存PPO和所有规则基线的 `rollout_*.npz`，方便训练后复盘。
- `scripts/23_v2_train_ppo.py` 已支持 `--diagnostic-freq`，定期输出 `ppo_training_diagnostics.csv`，记录训练中 deterministic rollout 的reward、oracle loss、DTW、功率、暖机中断和score分布。
- 当前已发现并修复一个关键混淆：v2 reward Oracle 不应预测全部 `STATE_COLUMNS`。环境现在区分 `state_columns` 与 `reward_target_columns`，默认reward target为9个forecast目标，避免 `air_pressure_pa` 等非目标变量支配奖励。
- 当前仍需补充Figure 4训练曲线绘图、PPO动作分布长训诊断和更系统的预算/seed扫描。

### Stage 3.5：EvaluationSuite最小闭环

- 将每个策略的rollout转换为论文可用表格。
- 输出总体指标、逐变量指标、事件/非事件指标和传感器使用率。
- 将DTW作为评估指标保留，不作为默认在线reward。

当前进度：

- 已新增 `src/v2/evaluation.py`，支持读取 `rollout_*.npz` 并计算 `MAE/RMSE/sMAPE/Pearson/DTW/power/constraint violation/warmup abort/action score`。
- 已新增 `scripts/24_v2_evaluate_rollouts.py`，输出 `v2_eval_overall.csv`、`v2_eval_by_variable.csv`、`v2_eval_by_event.csv`、`v2_eval_sensor_usage.csv`、`v2_eval_action_scores.csv`、主摘要图和传感器诊断图。
- 已新增 `tests/v2/test_evaluation_suite.py`。
- 已在远端用 `reports/v2_eval_suite_smoke` 验证 `PPO + full_open_projected + round_robin + random + AoI` 的完整评估闭环。

### Stage 4：论文实验闭环

- 多seed。
- 预算扫描。
- 暖机消融。
- 预测目标消融。
- 图表与表格生成。

---

## 十一、当前实现注意事项

- 不要把低温强制OFF作为当前必须实现功能。
- 不要把节电率写进奖励主项。
- 不要默认 `delta_*` 特征进入v2。
- 不要让 `full_open` 参与可行策略排序；它是理想参考，不是满足约束的baseline。
- 不要把CMDP-DQN写成和PPO并列的主算法。
- 不要在truth主表中直接写入NaN，除非后续所有Oracle、Kalman和窗口化代码都支持mask。
