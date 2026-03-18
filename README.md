# RL Sensor Scheduling Framework

`rl_sensor_scheduling_framework` 是一个面向“有限功耗约束下多传感器在线调度”的独立实验框架。它关注的问题不是传统的“先造缺失、再插值、再预测”，而是把传感器开关本身视为一个连续决策问题：在每个时刻只能开启部分传感器、且总功耗受到限制的前提下，如何让系统的长期状态估计误差尽可能小，并进一步考察这种调度方式对下游预测任务的影响。

当前仓库已经实现了一个可运行的基础版本，其核心闭环是：

1. 先生成一份高频、全量可观测的真值序列，作为假定真实环境。
2. 调度器在这份相同的真值序列上决定每个时刻要激活哪些传感器。
3. 只有被激活且满足刷新条件的传感器返回观测，并产生对应功耗。
4. 估计器根据有限观测恢复系统状态，并把不确定性反馈给调度器。
5. 不同调度策略分别生成自己的策略数据集，再交给同一组时序预测模型训练与评估。

因此，当前仓库的实验定义已经从“固定数据集比较预测模型”重构为更严格的闭环问题：

- 固定同一份高频真值环境数据；
- 改变调度策略；
- 比较不同调度策略在相同功率约束下对下游预测精度的影响；
- 再与 `full_open`（所有传感器全开、不调度）基线比较精度下降幅度与节电比例。

## 1. 研究背景与目标

在风吹雪、微气候、边缘设备监测等场景中，观测系统通常同时面对以下约束：

- 不能让所有传感器持续全开，否则总功耗过高。
- 同一时刻可激活的设备数有限，存在带宽或采样通道限制。
- 不同传感器的刷新频率、噪声水平和启动代价不同。
- 真正关心的不是单次采样误差，而是长期状态感知质量以及下游预测性能。

基于这些约束，本项目的目标可以写成一个受约束优化问题。设 `a_t` 表示时刻 `t` 激活的传感器集合，`x_t` 表示潜在真实状态，`P_t` 表示估计器后验协方差，那么我们希望学习一个调度策略 `pi(a_t | s_t)`，使得长期总代价最小：

`min_pi E[sum_t C_t]`

其中单步代价 `C_t` 由以下几部分组成：

- 估计误差代价：当前默认使用 `trace(P_t)`。
- 功耗代价：当前时刻被激活传感器的总功耗。
- 切换代价：频繁切换传感器组合带来的额外开销。
- 覆盖率惩罚：某些传感器长期不被观测时产生的惩罚。

同时满足：

- `|a_t| <= K`，即每个时刻最多开启 `K` 个传感器。
- `sum_{j in a_t} c_j <= B_step`，即单步功耗不能超过预算。

这实际上对应一个带资源约束的序列决策问题。当前实现从工程上采用“显式代价加权”的方式处理约束，而不是更复杂的 CMDP/Lagrangian 训练。

## 2. 整体实验原理

### 2.1 环境层：高频真值环境如何构建

当前默认实验不再把 `linear_gaussian` 作为主训练环境，而是先通过 `windblown` 业务环境生成一份高频真值序列，再把这份序列作为后续调度和预测实验共享的“世界”。

`windblown` 环境会模拟：

- 风速与风向的缓慢变化；
- 温度和湿度的日变化与随机扰动；
- 风暴状态的二态切换；
- 风速超过阈值后雪质量通量与雪粒子数通量增加。

生成得到的高频真值数据包含：

- 多个连续物理变量；
- `timestamp` / `time_idx`；
- `storm_flag` 与 `event_flag`；
- 频率默认由 `base_freq_s` 控制，当前默认是 `1s`。

在调度实验阶段，这份真值序列不再被当成普通离线数据表，而是被封装成一个可回放环境：调度器每一步只能看到自己实际采到的观测，不能直接访问全量真值。这样做的好处是：

- 所有调度策略面对的是完全相同的底层世界；
- 预测差异可以真正归因于“观测流的不同”，而不是数据源不同；
- 可以和 `full_open` 基线做直接对照，分析节电与精度损失的折中。

### 2.2 传感器层：为什么会产生调度问题

每个传感器都有一组静态规格：

- `sensor_id`
- 可观测变量 `variables`
- 刷新周期 `refresh_interval`
- 功耗 `power_cost`
- 启动延迟 `startup_delay`
- 噪声水平 `noise_std`

因此，并不是所有传感器在每个时刻都能以相同成本提供同等质量的信息。调度器的任务就是在预算内挑选最值得开启的设备。

### 2.3 动作空间：调度器究竟在选什么

当前 Phase 1 使用离散动作空间。系统会根据配置中的：

- `max_active`
- `per_step_budget`
- 每个传感器的 `power_cost`

预先枚举一批可行动作，每个动作就是一个满足约束的“传感器子集”。DQN 并不是直接输出连续功率，而是在这些可行动作中选一个索引。

这种设计简单、稳定，但它也带来一个直接局限：当传感器数量增大时，动作空间会快速膨胀。

### 2.4 估计器层：为什么用 Kalman Filter

在当前默认实验中，调度器并不直接观察真值序列，而是只能依赖估计器的内部摘要。默认估计器仍然是 Kalman Filter，但它不再服务于一个独立在线生成的线性系统，而是服务于“高频真值回放环境上的部分观测恢复”任务。

当前实现采用一个一阶随机游走近似作为状态模型：

1. `predict`：使用 `A = I` 的持久性假设把先验状态推进到下一时刻。
2. `Q`：由真值序列在训练段上的一阶差分协方差估计得到，用来描述自然波动强度。
2. `update`：只使用当前激活传感器返回的观测，对状态进行后验修正。

Kalman Filter 的核心价值在于它不仅给出状态估计 `x_hat_t`，还给出不确定性 `P_t`。本项目把 `trace(P_t)` 看作调度任务中最重要的“估计不确定性代理量”，因为它直接反映当前观测配置是否足够支撑状态感知。

更具体地说，`src/estimators/kalman_filter.py` 中的实现承担了三个角色。

第一，它是环境与调度器之间的“信息压缩器”。环境内部的真实状态 `x_t` 对调度器不可见，调度器只能看到 Kalman Filter 输出的后验估计 `x_hat_t` 和不确定性 `P_t`。因此，RL 实际上不是在真值状态上做决策，而是在“估计后的信念状态”上做决策。

第二，它是 reward 的主要来源。当前代价函数里最核心的项是 `trace(P_t)`，也就是协方差矩阵对角元素之和。直观上，`trace(P_t)` 越大，说明系统对当前状态越不确定；`trace(P_t)` 越小，说明当前被激活的传感器组合越有效。因此，调度器的学习目标可以理解为：在预算允许的前提下，让 `trace(P_t)` 长期尽可能低。

第三，它还负责为 RL 策略构造可观测状态摘要。除了 `x_hat_t` 和 `diag(P_t)`，实现里还维护了：

- `freshness`：每个传感器距离上次成功采样过了多久；
- `coverage_ratio`：每个传感器到当前为止被采到的比例；
- `previous_action`：上一步哪些传感器被激活；
- `budget_ratio`：当前动作占用了多少预算。

这些量不是经典 Kalman Filter 公式的一部分，但它们和调度问题直接相关，因此被并入 RL 状态向量。

如果按当前代码的执行顺序来看，每个时间步的 Kalman Filter 使用方式是：

1. 调度器先根据上一时刻的估计摘要选择动作；
2. 真值回放环境只让被选中的传感器返回观测；
3. 估计器执行一次 `predict`，用随机游走模型把 `x_hat` 和 `P` 推到下一时刻；
4. 估计器对当前收到的每条观测顺序执行 `update`；
5. 用更新后的 `x_hat`、`P`、freshness、coverage 等量构造下一步 RL 状态；
6. 用更新后的 `trace(P)` 参与计算 reward。

在数学上，当前实现对应标准线性 Kalman Filter 的顺序观测更新形式。对每条有效观测 `(y, C, R)`，代码会计算：

- 创新协方差：`S = C P C^T + R`
- Kalman 增益：`K = P C^T S^{-1}`
- 状态更新：`x_hat <- x_hat + K (y - C x_hat)`
- 协方差更新：`P <- (I - K C) P`

这里之所以逐条观测顺序更新，而不是先把所有传感器观测拼成一个大矩阵再一次性更新，是因为当前调度问题天然就是“每一步只会收到一个稀疏、可变长度的观测集合”。顺序更新在这种场景下更直接，也更符合代码结构。

还需要注意两点实现层面的事实。

第一，当前 `DatasetSensor` 会把每个传感器对应的观测选择矩阵 `C` 和观测噪声协方差 `R` 一起返回给估计器，因此 Kalman Filter 不需要预先写死“哪个传感器观测哪个维度”，而是可以按当前到达的观测动态更新。

第二，当前实现使用的是标准线性 KF，而不是 EKF/UKF。仓库里的 `ekf.py` 目前仍是预留骨架，尚未进入默认实验链路。

### 2.5 RL 状态：策略网络看到了什么

当前 RL 策略使用的不是环境真值，而是“估计器可提供的摘要状态”，主要包括：

- 当前状态估计 `x_hat_t`
- 协方差对角线 `diag(P_t)` 或相关不确定性摘要
- `trace(P_t)`
- 各传感器 freshness，即距离上次成功观测已经过去多久
- 各传感器 coverage ratio，即最近一段时间的覆盖比例
- 当前时间步 `t`

这种设计满足一个关键原则：训练和部署使用的是同一类信息，避免直接用真实隐变量做策略输入而产生信息泄露。

### 2.6 奖励设计：为什么不是直接优化预测误差

当前 RL 训练的单步 reward 本质上是负代价：

`reward_t = - C_t`

代价由 `src/evaluation/cost_metrics.py` 计算，默认最核心的部分是：

- `alpha_estimation * trace(P_t)`
- `lambda_power * power_cost`
- `eta_switch * switch_cost`
- `rho_coverage * coverage_penalty`

当前默认配置里，预测误差项 `beta_prediction` 仍为 `0.0`，也就是说当前强化学习阶段主要优化的是“状态估计质量与资源消耗”的平衡，而不是直接端到端地优化预测器指标。这样做的原因是：

- 估计误差信号更稳定；
- 训练成本更低；
- 更容易解释调度策略为什么有效。

后续如果希望更贴近最终业务目标，可以继续把预测模型损失并入 reward，形成联合优化。

### 2.7 为什么还保留下游预测实验

如果只报告 `trace(P_t)` 或 `reward`，结论仍然停留在“估计层面”。但很多业务最终关心的是预测性能，因此当前仓库还保留了一个预测实验链路，用于回答：

- 更好的调度是否能给出更有价值的状态序列？
- 这些状态序列喂给预测器后，RMSE/MAE/MAPE 是否真的改善？

需要注意的是，当前实现中的“预测数据集构建”还没有完全与某个具体调度器绑定，详见后面的“真实数据流”部分。

## 3. 当前实现的实验分层

为了准确理解当前版本，可以把实验拆成三个连续阶段。

### 3.1 Phase A：高频真值环境生成

这一步先生成一份高频、全量可观测的风吹雪真值序列，作为后续所有调度策略共享的环境数据。默认脚本是：

- `scripts/00_generate_business_data.py`

输出文件默认是：

- `data/generated/windblown_truth.csv`

这份文件包含真实状态变量、时间索引以及事件标记。后续所有 scheduler 都在同一份 truth dataset 上进行采样，因此实验具有严格的可比性。

### 3.2 Phase B：预算约束下的调度训练与评估

这一层的目标是：在相同功率预算下，比较不同调度策略对观测质量和估计质量的影响。当前支持的策略包括：

- `full_open`
- `random`
- `periodic`
- `round_robin`
- `info_priority`
- `dqn`

其中 `dqn` 是学习策略，其余是规则或基线策略。对应脚本：

- `scripts/01_train_rl_scheduler.py`
- `scripts/02_evaluate_scheduler.py`

训练和评估都不再直接依赖在线生成的线性环境，而是依赖一份 truth replay environment。调度器面对的是同一条真实时间序列，只是每次看到的观测子集不同。

### 3.3 Phase C：策略数据集构建与预测评估

这一层的目标不再是“固定一个数据集，比较不同预测模型”，而是：

- 固定同一份真值环境；
- 让每个调度策略分别生成自己的观测流与估计序列；
- 用这些策略数据集分别训练多个常见时序预测模型；
- 比较新调度算法在不同模型上的预测精度保持能力。

对应脚本：

- `scripts/03_build_forecast_dataset.py`
- `scripts/04_train_predictors.py`
- `scripts/05_evaluate_forecasts.py`
- `scripts/06_posthoc_analysis.py`

因此，这一层的结论口径应该表述为：

- “在多个常见预测模型上，我们的调度算法都能较好保持预测准确率”
- “相对某些规则调度算法，预测误差更低”
- “相对 full-open 基线，虽然使用了更低功耗，但精度仅下降了有限比例”

## 4. 数据流向与执行步骤

下面按默认一键脚本 `scripts/run_full_experiment_tmux.sh` 的真实顺序说明数据如何流动。

### Step 0：读取配置

核心配置包括：

- `configs/base.yaml`：随机种子、episode 数、预算约束、数据切分比例。
- `configs/env/windblown_case.yaml`：高频真值环境参数、状态列、事件定义。
- `configs/sensors/windblown_sensors.yaml`：传感器变量、刷新频率、功耗、噪声。
- `configs/estimator/kalman.yaml`：Kalman 初始协方差设置。
- `configs/scheduler/*.yaml`：调度策略配置。
- `configs/predictor/*.yaml`：预测模型配置。

### Step 1：生成高频真值数据

脚本：

- `scripts/00_generate_business_data.py`

流向：

1. 读取 `windblown` 环境配置和传感器配置；
2. 推进业务环境，生成高频真值序列；
3. 输出 `timestamp`、`time_idx`、物理变量、`storm_flag`、`event_flag`；
4. 保存为 `data/generated/windblown_truth.csv`。

这一步产出的是唯一的“世界真值”。后续所有 scheduler 的观测流都从它采样。

### Step 2：在真值序列上训练调度器

脚本：

- `scripts/01_train_rl_scheduler.py`

流向：

1. 读取 `windblown_truth.csv`；
2. 按时间切分 train/val/test；
3. 用 train 段的一阶差分估计过程噪声协方差 `Q`，构造随机游走 Kalman 模型；
4. 构造真值回放环境、传感器对象和离散动作空间；
5. 在 train 段上训练 DQN，或在同一段上评估规则策略；
6. 输出调度性能指标与训练日志。

这里的关键点是：调度器看到的不是全量真值，而是从真值序列中“按策略采样后”得到的观测。

### Step 3：在测试段评估调度器

脚本：

- `scripts/02_evaluate_scheduler.py`

流向：

1. 在 test 段上回放同一份真值数据；
2. 若是 DQN，则加载 `scheduler_dqn.pt` 并用 greedy 策略选择动作；
3. 输出 `reward`、`trace(P)`、`power`、`coverage` 等指标。

这一步回答的是：
“在未见过的真值片段上，该调度策略是否依旧能稳定工作？”

### Step 4：为每个调度策略生成策略数据集

脚本：

- `scripts/03_build_forecast_dataset.py`

流向：

1. 固定同一份 `windblown_truth.csv`；
2. 使用指定 scheduler 在整条真值序列上 rollout；
3. 记录每一步的：
   - 真实状态 `target_series`
   - 估计状态 `input_series`
   - 观测 mask
   - 功耗序列
   - `trace(P)`
   - 事件标记
4. 保存为 `data/processed/<run_id>.npz`
5. 同时保存 `dataset_stats.csv` 与 sidecar metadata。

这一阶段是整个重构的关键。因为从这里开始，预测实验真正变成了“同一环境，不同调度策略，不同策略数据集”的设计。

### Step 5：在每个策略数据集上训练预测模型

脚本：

- `scripts/04_train_predictors.py`

流向：

1. 读取某个 scheduler 对应的 `.npz`；
2. 用 `input_series` 作为输入，用 `target_series` 作为预测目标；
3. 构造滑动窗口样本；
4. 按时间顺序切分 train/val/test；
5. 用 train 段统计量对输入和目标做标准化；
6. 训练预测模型；
7. 反标准化后在原尺度上评估预测误差；
8. 输出 `metrics_forecast.csv`。

因此，现在每个预测结果都带有明确的来源：

- 哪个 scheduler 生成了这份数据集；
- 该 scheduler 平均消耗了多少功率；
- 相对 full-open 节省了多少能耗。

### Step 6：聚合并与 full-open 基线比较

脚本：

- `scripts/05_evaluate_forecasts.py`

流向：

1. 收集当前 `RUN_TAG` 下所有 `metrics_forecast.csv`；
2. 汇总得到 `metrics_forecast_all_<RUN_TAG>.csv`；
3. 用 `full_open` 作为 oracle baseline；
4. 计算：
   - `rmse_increase_pct_vs_full_open`
   - `mae_increase_pct_vs_full_open`
   - `power_saving_pct_vs_full_open`
   - `total_energy_saving_pct_vs_full_open`
5. 输出对比表和 scheduler summary。

这一层开始，实验结果已经可以直接支撑如下表述：

- “在节省 xx% 功耗的前提下，RMSE 仅增加 xx%”
- “相对于 full-open，我们的调度策略保持了较高的预测精度”

### Step 7：后处理分析与可视化

脚本：

- `scripts/06_posthoc_analysis.py`

流向：

1. 输出 scheduler-model RMSE 热图；
2. 输出相对 full-open 的误差增幅热图；
3. 输出“节电比例 vs 误差增幅”散点图；
4. 输出按 scheduler 聚合的 summary 表。

这一阶段的结果适合直接进入论文图表或汇报材料，因为它已经把“预测精度”和“资源节约”放在同一张图里了。

## 5. 模块之间的关系

从工程结构上看，当前项目可以概括为五层：

- `envs/`
  负责状态演化和观测生成，是问题的物理/业务模拟器。
- `sensors/`
  负责把传感器规格具体化，决定哪些变量可测、何时可测、测量噪声多大、每次采样耗多少电。
- `estimators/`
  负责把不完整观测融合成状态估计，是调度器和环境之间的“信息中介”。
- `scheduling/`
  负责根据当前估计摘要选择动作，包括规则策略和 DQN。
- `forecasting/`
  负责把生成的状态序列转成监督学习窗口，并训练下游预测模型。

这套分层的好处是每一层都可以被替换：

- 可以换环境，但保留估计器和调度器。
- 可以换估计器，但保留动作空间和 reward 定义。
- 可以换 RL 算法，但保留同一条 rollout/评估链路。
- 可以换预测模型，而不动调度模块。

## 6. 默认实验命令

### 6.1 训练单个调度器

```bash
cd /home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate darts

python scripts/01_train_rl_scheduler.py \
  --truth_csv data/generated/windblown_truth.csv \
  --env_cfg configs/env/windblown_case.yaml \
  --sensor_cfg configs/sensors/windblown_sensors.yaml \
  --estimator_cfg configs/estimator/kalman.yaml \
  --scheduler_cfg configs/scheduler/dqn.yaml \
  --run_id phase1_debug
```

### 6.2 运行完整流程

```bash
cd /home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate darts

bash scripts/run_full_experiment_tmux.sh --conda-env darts --run-tag full_debug
```

如果要放在 `tmux` 中运行，请在外部创建 session，脚本本身只负责执行实验，不负责创建或管理 `tmux`。

## 7. 输出文件说明

### 7.1 调度阶段

目录：

- `reports/runs/<run_id>/`

常见文件：

- `config_used.yaml`：本次运行使用的环境、传感器、估计器、调度器配置路径。
- `metrics_estimation.csv`：汇总后的调度性能指标。
- `metrics_estimation_eval.csv`：评估阶段逐 episode 指标。
- `training_log.csv`：DQN 训练过程日志。
- `scheduler_dqn.pt`：训练好的 DQN 权重。
- `fig_training_curves.png`：训练 reward/loss 曲线。
- `fig_trace_power.png`：估计误差与功耗走势。
- `fig_action_hist*.png`：动作分布统计。

### 7.2 预测阶段

目录：

- `data/processed/<run_id>.npz`
- `reports/runs/<run_id>/metrics_forecast.csv`

内容：

- `.npz` 中包含：
  - `input_series`：该 scheduler 产生的估计状态序列；
  - `target_series`：同一时刻对应的真值序列；
  - `observed_mask`：每个时间步哪些变量被实际观测到；
  - `power`、`trace_p`、`event_flags` 等辅助分析量。
- `metrics_forecast.csv` 除了 RMSE、MAE、MAPE，还会记录该数据集来自哪个 scheduler，以及该 scheduler 的平均功耗和覆盖率。

### 7.3 聚合与后处理阶段

目录：

- `reports/aggregate/metrics_forecast_all_<RUN_TAG>.csv`
- `reports/aggregate/posthoc_<RUN_TAG>/`

内容：

- 汇总后的预测指标表。
- 相对 `full_open` 的误差增幅与节电对比表。
- scheduler-model 热图以及“节电比例 vs 误差增幅”散点图。

## 8. 解释结果时应注意的事实

当前项目已经可运行，但在解释实验结论时必须注意以下几点。

第一，当前默认实验已经切换到“同一份高频风吹雪真值序列下的调度对比”，因此现在可以把结果解释为：不同 scheduler 在同一真实环境数据上的观测策略差异。

第二，当前预测数据集已经由具体 scheduler rollout 生成，因此预测结果现在可以被解释为“该调度策略对下游预测任务的实际影响”。

第三，`full_open` 是一个不受预算约束的 oracle baseline，用来衡量“完全不调度时的上限精度”。因此，相对 `full_open` 的误差增幅应被解释为“为了节电而付出的精度代价”，而不是简单的输赢关系。

第四，聚合预测指标时如果不给 `--run_tag`，仍可能扫到历史结果；默认完整脚本已经显式传入 `run_tag` 进行过滤。

这些边界条件决定了当前实验最适合支撑的结论是：

- 哪种调度策略在预算约束下更能保持预测精度；
- 相对 `full_open`，精度下降了多少、节省了多少能耗；
- 这种结论在多个常见预测模型上是否一致。

## 9. 当前框架的价值与下一步扩展

当前版本已经从“原型验证”进入“可比较实验框架”阶段，它的价值主要体现在：

- 问题形式化已经明确：多传感器调度是一个带资源约束的长期决策问题。
- 真值环境已经统一：所有策略共享同一份高频 truth dataset。
- 策略数据集已经打通：每个 scheduler 都会产生自己的观测流和估计序列。
- 下游预测评估已经可闭环：可以直接比较“节电比例”和“误差增幅”。

后续最重要的工程与研究扩展方向有三个：

- 在多个预算水平下重复整套实验，绘制完整的 accuracy-power Pareto frontier。
- 引入更强的事件段分析，例如风暴段、输运段、峰值段误差比较。
- 在 RL 层引入更标准的 CMDP、Lagrangian 或 `gymnasium` 兼容封装，以支持更复杂的算法对比。
