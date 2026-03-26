# 组会汇报提纲：功率约束下的传感器调度与微气候预测

## 1. 这项工作在解决什么问题

这套框架想回答一个很具体的问题：

- 在南极 AWS 供电受限的条件下，传感器不能一直全开；
- 但实验室又需要提前知道未来的微气候状态，才能做 1:1 的环境调控；
- 所以我们要在**有限功率约束**下，决定每一时刻开哪些传感器，同时尽量保住后续预测效果。

一句话概括：

> 这是一个“功率受限的多传感器调度 + 状态估计 + 下游预测”的联合问题。

---

## 2. 整体框架是什么

先看整体框架图：

![framework](assets/windblown_framework_architecture.svg)

这张图里有几个关键点：

- 我们先生成一条共享的 truth 序列，所有调度策略都在同一条 truth 上比较；
- 调度器不能随便选传感器，而是要经过 `OnlineSubsetProjector`，满足瞬时功率和启动峰值约束；
- 传感器观测不是直接拿去预测，而是先经过 Kalman estimator；
- 每个 scheduler 都会生成自己的一份数据集，然后再训练同样的 forecasting model；
- 最后用 `full_open` 作为信息最充分的 oracle 上限做比较。

这里不是简单地比较“哪个预测模型更强”，而是在比较：

> 不同调度策略改变了可用信息之后，下游预测还能保留多少性能。

---

## 3. 这次实验具体用了哪些方法

### 3.1 调度算法

这次主要对比了几类 scheduler：

- `full_open`：全开，作为 oracle 上限；
- `random`：随机调度；
- `periodic`：周期调度；
- `round_robin`：轮询调度；
- `info_priority`：按信息优先级启发式调度；
- `ppo`：基于 Stable-Baselines3 的 PPO baseline。

这里要强调一点：

- `ppo`、`dqn`、`cmdp_dqn` 目前都只是 baseline；
- 后续真正要做创新的对象，是 `cmdp_dqn` 这一条线。

### 3.2 预测模型

下游预测模型用了一个 model zoo：

- `naive`
- `MLP`
- `LSTM`
- `Transformer`
- `Informer`
- `TCN`
- `PINN`
- `SERT-like`
- `S4M-like`

### 3.3 评价指标

现在不再只看 `dRMSE`。

我们同时看：

- `RMSE`
- `MAE`
- `sMAPE`
- `Pearson`
- `DTW`

这里 `DTW` 很重要，因为时序预测里有些结果幅值接近，但会有轻微相位偏差，只看 RMSE 不够。

---

## 4. 这次最重要的一个工程修复：PPO 之前是坏的

先看修复后的 PPO 传感器时间线图：

![ppo_timeline](../reports/aggregate/posthoc_full_with_ppo_fix_20260326/sensor_timelines/ppo/air_temperature_c_0_300_activation.png)

这张图想说明一件事：

- 旧版 PPO 基本塌成了近固定策略；
- 修复后，PPO 已经不是“几乎不调度”，而是在两个高功耗传感器之间真实切换。

修复后的统计结果也支持这个判断：

- `laser_disdrometer` duty 约 `0.678`
- `fc4_flux` duty 约 `0.322`
- 两个高功耗传感器都发生了大量切换，而不是某一个几乎全程固定。

这一步很关键，因为如果 scheduler 自己就是坏的，后面的预测结果都没有解释意义。

---

## 5. 当前结果怎么看

先看当前 run 的总体汇总图：

![rmse_tradeoff](../reports/aggregate/posthoc_full_with_ppo_fix_20260326/power_saving_vs_rmse_increase.png)

这张图的核心信息很直观：

- `random` 省电最多，但误差涨得也最多；
- `round_robin`、`periodic`、`info_priority` 形成了比较强的规则基线；
- 修复后的 `ppo` 已经回到合理范围，但还没有超过最强规则基线。

从汇总表看，当前相对 `full_open`：

- `round_robin`：`RMSE +2.63%`
- `info_priority`：`RMSE +2.71%`
- `periodic`：`RMSE +4.83%`
- `ppo`：`RMSE +8.52%`
- `random`：`RMSE +31.58%`

所以现在的结论不是“PPO 最强”，而是：

> PPO 已经从坏结果修到可用结果，但目前最强的还是结构化规则策略。

---

## 6. 再看一个具体的预测曲线例子

例如 `Informer` 在 `air_temperature_c, H=1` 上的对比图：

![air_temp_curve](../reports/aggregate/posthoc_full_with_ppo_fix_20260326/prediction_curves/informer/air_temperature_c_H1_overlay.png)

这张图适合现场讲两点：

- `full_open` 仍然是最稳定的上限参考；
- `ppo` 修复后不再出现之前那种明显离谱的形状崩坏，但和最优规则策略相比还有差距。

这比只给一个表格更容易理解，因为组里的人一眼就能看到：

- 曲线有没有整体偏移；
- 峰值和谷值有没有跟上；
- 学习型策略现在至少已经“像样”了。

---

## 7. 这轮实验最应该怎么总结

### 结论一：框架已经能稳定跑通

这个阶段最重要的不是宣称 RL 已经赢了，而是：

- 功率约束、在线投影、状态估计、预测训练、aggregate、posthoc 这些链路已经完整打通；
- PPO 这条外部 baseline 也已经真正接入并跑通，不再是无效实现。

### 结论二：规则策略仍然很强

当前最稳的 baseline 仍然是：

- `round_robin`
- `info_priority`
- `periodic`

这说明在现在这个传感器和约束结构下，规则方法并不弱，后续做 RL 创新不能建立在“规则方法很差”的假设上。

### 结论三：RL 还能继续做，但方向要更明确

接下来更有意义的不是继续无休止加模型，而是：

- 以 `cmdp_dqn` 为主线继续做 constrained RL；
- 让约束处理更合理；
- 让 learned policy 在不违反功率约束的前提下，逼近或超过最强规则基线。

---

## 8. 下一步计划

我建议下一阶段集中在三件事：

1. 固化当前 baseline 体系：规则方法 + `dqn/cmdp_dqn` + `ppo`。  
2. 以 `cmdp_dqn` 为创新对象，继续做约束 RL 改进。  
3. 继续用主任务口径汇报，也就是以微气候状态预测为主，而不是只盯单一雪通量。  

一句话说，下一步不是再补工程漏洞，而是：

> 在一个已经能稳定工作的框架上，开始做真正有研究价值的 constrained RL 改进。
