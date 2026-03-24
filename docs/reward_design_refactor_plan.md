# RL 调度奖励重构记录与实施计划

## 1. 问题背景与原始思路

当前实验的核心目标不是单纯降低状态估计误差，而是：

- 在总功耗受限的条件下进行多传感器调度；
- 让调度策略生成的数据尽可能保留对下游时序预测有用的信息；
- 最终在多个常见预测模型上，保持较好的预测精度，并与 `full_open`、`random`、`periodic`、`round_robin`、`info_priority` 等基线比较。

当前 windblown 任务中的观测结构已经扩展为：

- 基础气象站：`wind_speed_ms`、`wind_direction_deg`、`air_temperature_c`、`relative_humidity`、`air_pressure_pa`
- 总辐射：`solar_radiation_wm2`
- 红外雪面温度：`snow_surface_temperature_c`
- 激光雨滴谱仪：`snow_particle_mean_diameter_mm`、`snow_particle_mean_velocity_ms`
- FC4 风吹雪传感器：`snow_mass_flux_kg_m2_s`

其中两个风吹雪专用传感器（激光谱仪与 FC4）功耗较高，其余基础气象类传感器功耗较低。这个新设计使调度问题更接近真实业务设定：低功耗气象背景信息可长期保留，而高功耗风吹雪信息需要在预算约束下有选择地获取。

最初的直觉是：

> 直接把 forecasting model 的预测误差作为 RL 的 reward。

这个思路在目标层面是正确的，但直接落地会遇到一个闭环问题：

- scheduler 生成观测流；
- predictor 依赖观测流训练与预测；
- predictor 的误差又要反过来作为 scheduler 的 reward；
- 如果 predictor 尚未训练好，reward 不存在；
- 如果 predictor 与 scheduler 同时更新，reward 分布会持续漂移，DQN 训练会明显不稳定。

因此，调度训练与预测评估不能简单地端到端混在一起。

## 2. 已发现的五个关键问题（必须保留）

### 问题 1：传感器噪声配置不合理

旧版本里，某些传感器对多个量纲差异巨大的变量共用一个噪声标量。例如 `snow_flux` 同时观测：

- `snow_mass_flux_kg_m2_s`
- `snow_number_flux_m2_s`

但两者数量级相差很大，共用单一 `noise_std` 会导致：

- 对某些变量而言观测几乎无效；
- 对另一些变量而言观测几乎完美；
- RL 被错误地诱导去偏好这类“表面收益极高”的传感器。

### 问题 2：RL reward 没有真正对齐预测目标

旧版本 DQN 的 reward 本质上由估计导向项构成，主要优化的是：

- `trace(P)`
- 功耗
- 切换代价
- 覆盖率惩罚

这意味着调度器学到的是“如何降低估计不确定性”，而不是“如何保住下游预测精度”。

这是本轮重构中最重要的问题。

### 问题 3：原始 `trace(P)` 直接跨量纲求和

状态向量中包含：

- 风速
- 风向
- 温度
- 湿度
- 气压
- 雪质量通量
- 雪数通量

这些变量的量纲和方差尺度不同。旧版本直接使用原始 `trace(P)`，会导致 reward 被大尺度变量主导，从而削弱对目标变量的针对性。

### 问题 4：策略塌缩，调度器长期只开启少数传感器

由于 coverage 约束过弱、reward 设计偏向局部收益，旧版本 DQN 很容易退化为长期固定开启少量局部收益较高的传感器组合，而忽略那些对目标预测同样关键、但即时 reward 较弱的驱动观测。

结果是：

- 风速、温湿度等关键驱动量长期不可观测；
- 生成的数据集结构性偏移严重；
- 下游预测模型即使结构合理，也无法恢复缺失的信息。

### 问题 5：被保留的信息与目标变量预测需求错位

对 `snow_mass_flux_kg_m2_s` 而言，决定预测性能的关键驱动并不只是同类雪输运变量，还包括：

- `wind_speed_ms`
- `relative_humidity`
- `solar_radiation_wm2`
- `snow_surface_temperature_c`
- `snow_particle_mean_velocity_ms`
- 其它与事件形成相关的气象量

如果调度策略只保留对自身 reward 有利的变量，而忽略真正与目标预测强相关的驱动项，就会出现：

- 估计指标看似尚可；
- 最终 forecast 曲线整体漂移或明显失真。

## 3. 已完成的结构性修复

目前已完成以下基础修复：

1. 传感器噪声改为按变量配置，而不是统一标量；
2. 用标准化后的不确定性指标（如 `weighted_trace_P_norm`）替代原始 `trace(P)`；
3. 增加目标变量相关性权重，使不确定性项更贴近目标任务；
4. 将 forecasting task 对齐到目标列 `snow_mass_flux_kg_m2_s`，而不再混合统计全部状态变量；
5. 拟合线性高斯动力学 `A, b, Q`，避免始终使用 `A = I` 的过度简化模型；
6. 强化 coverage penalty，缓解策略塌缩。

这些修复解决的是“实验明显不合理”的问题，但还没有彻底解决：

> RL 如何与实际预测误差建立更可信的联系。

## 4. 研究取舍：为什么不采用端到端联合优化

对于当前这篇小论文，端到端 joint training（scheduler 与 predictor 同时训练）不是合适方案，原因如下：

1. predictor 尚未训练时，forecast-MSE reward 不可计算；
2. 若 predictor 与 scheduler 同时更新，reward 非平稳，DQN 很容易失稳；
3. 为多个 predictor 同时构建 reward oracle 会带来过高的计算和叙事复杂度；
4. 这会把问题升级为 bilevel / alternating optimization，工作量超出当前可控范围。

因此，本项目不采用“完全端到端”主线，而采用：

- **主训练阶段：使用可即时计算的代理 reward；**
- **扩展方案 A：引入一个冻结的单一 predictor，作为辅助 forecast reward；**
- **最终评估阶段：对多个常见预测模型分别训练、分别测试、分别作图，而不是把多个模型的误差平均后塞回 reward。**

这个取舍更稳，也更适合论文写作与结果解释。

## 5. 方案 A：冻结单一 predictor 的辅助 forecast reward

### 5.1 设计原则

方案 A 不是把 forecasting loss 作为唯一 reward，而是：

- 先在一段独立数据上训练一个单一 predictor；
- 将该 predictor 完全冻结；
- RL 训练时保留当前代理 reward 主体；
- 额外加入一个由冻结 predictor 产生的辅助 forecast loss 项。

这样可以同时满足：

- reward 可计算；
- 训练稳定性可接受；
- 不需要把 predictor 与 scheduler 联合训练；
- 论文中可以清楚解释“为何引入 forecast-aware 辅助项”。

### 5.2 数据划分原则

为了避免 reward predictor 与 RL 使用同一段数据导致闭环偏置，需要把 truth 数据划分为至少四段：

1. `reward_pretrain`：仅用于训练并冻结 reward predictor；
2. `rl_train`：仅用于 DQN 训练；
3. `rl_val`：用于 DQN 选择和调参（可选，但建议保留）；
4. `rl_test`：仅用于最终 scheduler 评估与下游 forecasting pipeline。

核心原则：

- `reward_pretrain` 不参与 RL 训练；
- `rl_test` 不参与 reward predictor 训练；
- 下游 forecast 比较使用 scheduler 在统一 truth 上生成的数据集，但评价时必须保持严格测试隔离。

### 5.3 reward 形式

建议新的训练 reward 为混合型：

\[
r_t = -\Big(
\alpha U_t +
\beta E^{state}_t +
\gamma E^{forecast}_t +
\lambda C^{power}_t +
\eta C^{switch}_t +
\rho C^{coverage}_t
\Big)
\]

其中：

- \(U_t\)：标准化后的不确定性，例如 `weighted_trace_P_norm`；
- \(E^{state}_t\)：目标变量即时标准化状态误差；
- \(E^{forecast}_t\)：冻结 predictor 计算得到的短期 forecast loss；
- \(C^{power}_t\)：功耗；
- \(C^{switch}_t\)：切换代价；
- \(C^{coverage}_t\)：覆盖率惩罚。

其中 `E^{forecast}_t` 只作为辅助项，不应在第一版中压过代理 reward 主项。

### 5.4 forecast reward 的具体实现建议

- 使用单一 predictor，例如 `lstm`；
- 输入为最近 `lookback` 步估计状态序列；
- 输出为未来 `H` 步目标变量预测；
- 只对 `snow_mass_flux_kg_m2_s` 计算 forecast loss；
- loss 建议使用 `Huber` 或 `MSE`；
- horizon 建议先从 `H=1` 或短 horizon 开始，避免 credit assignment 过长。

## 6. 小论文可落地的实施步骤（TODO）

### TODO 1：增加 reward predictor 独立数据划分

目标：让 reward predictor 与 RL 训练数据严格隔离。

实施：

- 在 truth pipeline 中新增 `reward_pretrain` split；
- 配置独立比例；
- 确保 `01_train_rl_scheduler.py` 不会在 `reward_pretrain` 段上训练 DQN。

交付：

- 配置字段；
- split 说明；
- 日志中打印各 split 范围。

### TODO 2：实现 reward predictor 训练与冻结

目标：在 RL 之前训练一个单一 predictor 并冻结保存。

实施：

- 新增 reward predictor 训练模块；
- 复用现有 predictor factory；
- 输入使用 full-observation truth 序列；
- 仅对目标列训练；
- 输出 checkpoint 与归一化统计量。

交付：

- `reward_predictor.pt`；
- `reward_predictor_meta.yaml/json`；
- 可复现实验脚本入口。

### TODO 3：在 RL 环境中加入 forecast-aware 辅助 reward

目标：在每个 step 中计算冻结 predictor 的 forecast loss。

实施：

- 维护最近 `lookback` 步估计状态缓存；
- 用冻结 predictor 预测未来 `H` 步目标；
- 从 truth replay 环境读取未来真值；
- 计算 `E^{forecast}_t`；
- 把它作为辅助项加入 reward。

交付：

- 新 reward 模块；
- 配置开关；
- 训练日志输出 `forecast_reward_mean`。

### TODO 4：保留现有代理 reward，并做权重对比

目标：防止 DQN 训练被 noisy forecast term 拖垮。

实施：

- 保留 uncertainty / state error / power / switch / coverage 主体；
- 设置较小 `forecast_reward_weight`；
- 至少比较：
  - 无 forecast reward；
  - 小权重 forecast reward；
  - 中等权重 forecast reward。

交付：

- 一组消融实验；
- 训练稳定性对比图；
- 最终选择理由。

### TODO 5：下游评估仍保持多模型分开比较

目标：保证论文结果可信，不把多个模型平均后掩盖差异。

实施：

- 对 `naive / mlp / lstm / transformer / informer / tcn` 分别训练；
- 分别报告 RMSE / MAE；
- 分别画 scheduler 对比曲线图；
- 不在训练 reward 中对多个模型求平均。

交付：

- 每个 predictor 单独的对比表和图；
- 总结图仅用于展示，不进入 reward 设计。

## 7. 当前版本的研究主线建议

本文最稳妥、最可信的主线应写成：

1. 先构建一个功耗约束下的传感器调度环境；
2. 使用 estimator-oriented proxy reward 训练调度器；
3. 引入单一冻结 predictor 作为可控的 forecast-aware 辅助项；
4. 最终对多个下游预测模型分别评估调度策略的有效性。

这样的叙述既保留了 RL 调度创新点，也避免夸大为尚未完成的端到端联合优化。

## 8. 本轮重构的边界

本轮只做符合当前工作量的小论文级别改造，不做以下内容：

- 不做 predictor ensemble 作为正式 reward；
- 不做 bilevel alternating optimization；
- 不做 actor-critic / PPO / SAC 全面切换；
- 不做复杂的联合训练闭环。

如果后续结果证明方案 A 有效，再考虑更强的方案 C。


## 9. 当前版本的经验诊断（基于现有设计与结果）

当前实验已经完成了从“估计导向”到“预测导向”的第一轮重构，并且新风吹雪场景已经采用了更合理的传感器设计。但是，当前结果也暴露出一个新的、更加关键的问题：

> 在理论上，`full_open` 代表信息最充分的观测条件，因此它应该是下游预测的上限基线；如果某些简单模型在个别实验里出现 `round_robin` 或其它预算受限策略优于 `full_open` 的现象，更合理的解释通常不是“调度创造了更多信息”，而是“现有输入表达尚未充分利用全开观测带来的额外信息”。

这意味着当前阶段的主要瓶颈已经从“调度完全错位”转移为“输入表示与特征工程不足”。从现有结果看，这种现象尤其容易出现在：

- `Naive`
- `MLP`
- 部分短 horizon 的轻量模型

其原因更可能是：

1. `full_open` 虽然信息更全，但 estimator 与 predictor 并没有显式编码“哪些观测是新鲜的、哪些观测是直接测得的、哪些观测是通过状态估计推断的”；
2. 对于平滑、短时强自相关的目标变量，低复杂度模型会天然偏好更平滑、更低噪声的输入；
3. 因此，一些结构化低功耗策略会表现出“轻度正则化/去噪”的效果，从而在个别模型上偶然超过 `full_open`。

这个现象不应被直接写成“轮询优于全开”的论文结论，而应被明确写成：

- 现有表示下，`full_open` 的额外观测优势尚未被完全发挥；
- 后续优化重点应放在输入表示、观测时效建模和物理派生特征，而不是简单把这一现象解释为调度本身更优。

## 10. 下一轮最优先的改进方向（小论文工作量内可完成）

在当前基础上，最值得优先推进的不是继续堆更复杂的 RL 算法，而是让 `full_open` 的信息优势能够真正被下游模型利用。优先级建议如下：

### TODO 6：加入观测时效与掩码特征

目标：显式告诉预测模型“哪些变量刚刚被观测到、哪些变量已经 stale”。

建议新增：

- `is_observed_<var>`
- `time_since_observed_<var>`
- 必要时加入按传感器汇总的 freshness 特征

这是让 `full_open` 优势重新显现的第一优先级改造。

### TODO 7：加入关键物理派生特征

目标：把高功耗传感器带来的额外观测转化为模型更容易利用的输入表示。

建议优先新增：

- `wind_dir_sin`
- `wind_dir_cos`
- `wind_u`
- `wind_v`
- `surface_air_temp_gap`
- `particle_kinetic_proxy`
- `size_velocity_interaction`
- `transport_exceedance`

这些特征足以保持当前方法简洁，同时又能更贴近风吹雪任务的物理结构。

### TODO 8：重新验证 `full_open` 是否恢复为上限基线

目标：在改进表示后重新检查是否仍出现预算受限策略系统性优于 `full_open` 的现象。

判据应为：

- `full_open` 在大多数模型、尤其是中高容量模型上重新成为性能上限；
- 预算受限策略不再依赖“偶然的平滑/去噪效应”取得表面优势；
- 学习型调度器的贡献被重新解释为“在预算受限下逼近 `full_open`”，而不是“在个别模型上碰巧超过 `full_open`”。
