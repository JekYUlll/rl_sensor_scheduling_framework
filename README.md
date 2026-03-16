# RL Sensor Scheduling Framework（强化学习传感器调度框架）

本项目是一个面向“有限功耗下多传感器在线调度”的独立实验框架。核心思想是：  
在每个时刻只激活一部分传感器，在满足预算约束的前提下，最小化长期状态估计误差，并进一步验证下游预测性能。

---

## 1. 实验背景

在微气候/风雪输运等连续观测场景中，传感器网络通常面临以下现实约束：

- 总功耗受限：不能让所有传感器持续全开。
- 通道与带宽受限：同一时刻最多激活有限个设备。
- 采样异步与刷新限制：不同传感器刷新频率不同。
- 任务目标长期化：我们关注的是“长期预测质量”，而非单次采样误差。

传统离线流程通常是“先人为造缺失，再插值，再训练预测模型”。  
本框架转向闭环决策：调度策略直接决定观测流，观测流再决定估计和预测质量。

---

## 2. 实验目的

本实验的研究目标是：

1. 将传感器调度问题形式化为受约束序列决策问题（MDP/CMDP 视角）。  
2. 在相同功耗预算下比较规则策略与学习策略（DQN）的长期性能。  
3. 将“调度器产生的数据流”用于预测任务，评估调度对下游模型的真实影响。  
4. 产出可复用的工程框架，支持后续接入真实业务场景。

---

## 3. 问题建模与方法原理

### 3.1 动态系统与观测模型

Phase 1 默认使用线性高斯系统：

- 状态演化：`x_{t+1} = A x_t + w_t`, `w_t ~ N(0, Q)`
- 传感器观测：`y_{j,t} = C_j x_t + v_{j,t}`, `v_{j,t} ~ N(0, R_j)`

每个传感器有独立规格：观测维度、刷新周期、噪声、功耗、可用性。

### 3.2 动作空间（调度空间）

调度动作为“传感器子集选择”。  
框架在 `src/scheduling/action_space.py` 中预先枚举可行动作，满足：

- 激活数约束：`|a_t| <= K`
- 单步功耗约束：`sum_j c_j * 1(j in a_t) <= B_step`

该离散动作空间是 Phase 1 使用 DQN 的基础。

### 3.3 估计器（Kalman Filter）

默认估计器是线性 Kalman Filter（`src/estimators/kalman_filter.py`）：

1. `predict`：根据系统模型推进先验估计。  
2. `update`：仅利用当前被激活传感器的观测进行后验更新。  

估计不确定性以 `trace(P_t)`、`diag(P_t)` 等量化，既用于分析也用于 RL 状态输入。

### 3.4 RL 状态设计

策略网络不直接看真值状态，而看“可观测信息摘要”：

- 估计均值 `x_hat_t`
- 不确定性 `diag(P_t)` 与 `trace(P_t)`
- 每个传感器 freshness（距上次观测时间）
- 每个传感器 coverage ratio（最近覆盖率）
- 当前预算占用比
- 上一步动作编码

这保证了训练与部署条件一致，避免信息泄露。

### 3.5 奖励/代价设计

单步代价形式：

`C_t = alpha * EstimationLoss + lambda * PowerCost + eta * SwitchCost + rho * CoveragePenalty`

默认 Phase 1 重点是估计侧稳定性：

- `EstimationLoss = trace(P_t)`
- `Reward = -C_t`

这样做的原因是：估计误差信号更稳定、可解释性更强，适合作为 RL 起始阶段目标。

### 3.6 为什么仍然做预测评估

仅优化估计误差不等于最终业务收益。  
因此本框架保留预测评估层：在固定调度策略下滚动生成数据，再训练预测器，比较：

- RMSE / MAE / MAPE
- 事件段 vs 非事件段误差
- 性能-功耗折中关系

---

## 4. 当前实现状态（重要）

### 已实现（可运行）

- Phase 1 核心闭环：线性高斯环境 + KF + 基线调度 + DQN 训练/评估
- 规则基线：random / periodic / round_robin / max_uncertainty / info_priority
- 预测链路基础：dataset builder + naive/mlp/lstm/transformer/informer(tiny)/tcn
- 汇总与后处理基础：指标聚合、rank correlation、热图输出
- 测试集：7 个单元测试（环境、动作空间、KF、调度器、DQN、数据集构建）

### 已提供骨架（可扩展）

- Windblown 业务适配接口（adapter/sensor specs/event definitions）
- EKF / UKF 预留文件
- 更复杂约束优化与大规模动作空间策略（待扩展）

---

## 5. 实验流程（脚本级）

`scripts/` 下核心流程：

1. `00_generate_business_data.py`：生成业务场景数据（当前示例为 windblown）。  
2. `01_train_rl_scheduler.py`：训练调度器（DQN 或规则策略）。  
3. `02_evaluate_scheduler.py`：评估调度器估计侧指标。  
4. `03_build_forecast_dataset.py`：按调度结果构建预测数据集。  
5. `04_train_predictors.py`：训练预测模型。  
6. `05_evaluate_forecasts.py`：汇总预测指标。  
7. `06_posthoc_analysis.py`：后处理（相关性/热图等）。  

一键脚本：

- `scripts/run_full_experiment_tmux.sh`  
说明：脚本本身不包含 tmux 控制逻辑，只执行实验流程；tmux 由外部命令管理。

---

## 6. 快速开始（conda / darts）

```bash
cd /home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate darts

# 单次训练（DQN）
python scripts/01_train_rl_scheduler.py \
  --env_cfg configs/env/linear_gaussian_case.yaml \
  --sensor_cfg configs/sensors/linear_gaussian_sensors.yaml \
  --estimator_cfg configs/estimator/kalman.yaml \
  --scheduler_cfg configs/scheduler/dqn.yaml \
  --run_id phase1_debug
```

在 tmux 中跑完整流程（推荐）：

```bash
tmux new -s rl_full 'cd /home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework && bash scripts/run_full_experiment_tmux.sh --conda-env darts'
```

---

## 7. 输出产物说明

每个 run 默认写入：`reports/runs/<run_id>/`

- `config_used.yaml`：本次运行配置快照
- `metrics_estimation.csv`：估计侧关键指标
- `training_log.csv`：训练过程日志（DQN）
- `scheduler_dqn.pt`：策略网络参数（DQN）
- `fig_*.png`：训练/动作/估计曲线图

汇总输出：`reports/aggregate/`

- `metrics_forecast_all_*.csv`
- `rank_correlation.csv`
- `rank_correlation.png`

---

## 8. 预期结果（你在报告中可直接使用）

在合理预算约束下，理论上应观察到以下趋势：

1. DQN 相比 random 在 `trace(P)` 和长期 reward 上更优。  
2. 当预算从低到高增加时，估计误差先明显下降后趋于平台（边际收益递减）。  
3. 规则策略中，考虑不确定性/覆盖率的策略通常优于纯周期策略。  
4. 在下游预测任务中，调度策略差异会转化为可测的 RMSE/MAE 差异。  
5. 事件段误差通常高于非事件段，且更依赖“关键传感器是否被及时激活”。

---

## 9. 局限与后续计划

- 当前 DQN 使用离散子集动作，传感器规模很大时会出现动作空间爆炸。  
- 当前 Informer 为简化实现，后续可替换为完整版本。  
- EKF/UKF 尚未实装；非线性系统仍以适配器近似处理。  
- 未来重点：CMDP/Lagrangian 约束训练、连续功率控制、真实传感器在线接入。

---

## 10. 与旧项目关系

- 本项目与 `experiments_scheduling_suite` 独立。  
- 旧项目作为 benchmark/历史对照保留。  
- 若复用历史逻辑，需复制并在本仓库内适配，避免跨项目硬依赖。
