# 预测器冻结奖励主线方案（唯一主路线）

## 1. 目标纠偏（必须统一）

本项目后续的唯一优化目标是：

- 在功率约束下，提升下游多步时序预测效果；
- 关注 TCN / LSTM / Transformer 等模型在未来若干步的误差与序列形状一致性。

因此，调度训练中的主奖励必须直接对应“未来预测误差”。

当前默认奖励中的 `latent truth vs estimator` 误差，本质是状态跟踪误差，不等于下游 forecasting loss。该项不再作为主优化目标，仅允许作为辅助稳定项或完全关闭。

---

## 2. 新主线（固定流程）

后续实验统一采用以下单一路线：

1. **先训练并冻结预测器**（仅使用独立的预测器训练数据段）；
2. **再训练调度策略**（每一步奖励由冻结预测器给出的多步预测误差构成）；
3. **最后在独立验证/测试段评估调度器**（不参与预测器训练与调度训练），输出完整图表与 aggregate 结果。

---

## 3. 数据切分与防泄漏设计

必须把时间轴拆为 4 段，且严格不交叉：

- **S0: predictor_pretrain**：训练/选择预测器；
- **S1: rl_train**：训练调度器；
- **S2: rl_val**：调度训练的早停与模型选择；
- **S3: final_test**：最终报告，仅用于结果展示。

约束：

- 冻结预测器只能用 S0；
- 调度训练只能用 S1（可在 S2 做验证）；
- 最终指标和图只能用 S3；
- S3 任何结果不得回流训练。

---

## 4. 预测器冻结层设计

### 4.1 冻结对象

默认冻结三类预测器并行作为 reward oracle：

- TCN
- LSTM
- Transformer

每个预测器都输出多步预测（例如 H=1,2,3 或更长）。

### 4.2 奖励计算输入

在调度第 t 步，输入给冻结预测器的是“当前调度导致的可见特征历史”，包括：

- 估计状态序列；
- 观测掩码/新鲜度特征；
- 已定义的物理增强特征（如风向分解、交互项）。

### 4.3 预测损失与聚合

对每个预测器计算多步损失：

- `L_model(t) = Σ_h w_h * loss(y_{t+h}, yhat_{t+h|t})`

再做模型聚合：

- `L_forecast(t) = Σ_m alpha_m * L_model_m(t)`

其中：

- `w_h` 为不同预测步权重；
- `alpha_m` 为不同预测器权重（默认均匀）。

---

## 5. 调度奖励函数（新定义）

调度训练每步奖励统一定义为：

- `reward_t = - L_forecast(t) - lambda_switch * switch_cost - lambda_cov * coverage_penalty - lambda_violation * constraint_violation`

说明：

- 主项是 `L_forecast(t)`；
- 功率通过硬约束优先控制（可行域 + 峰值限制 + 安全裕度）；
- 功率软惩罚只保留低权重兜底，不再主导奖励；
- 旧的状态跟踪误差项默认设为 0（或极小稳定项）。

---

## 6. 约束体系（保持不变但前置）

动作层继续执行硬约束筛选：

1. 瞬时稳态功率上限；
2. 启动峰值功率上限；
3. 安全裕度；
4. 可选最小开机时间/冷却时间（后续可加）。

长期约束继续保留：

- 平均功率预算；
- 回合能量预算；
- 违约率统计。

CMDP 的价值在于“保证约束可控”，主目标改为“在约束内优化 forecasting”。

---

## 7. 配置改造（必须落地）

### 7.1 奖励主线开关

- 把冻结预测奖励设为默认开启；
- 把旧状态误差主项权重降为 0（或极小）；
- 保留切换与覆盖惩罚。

### 7.2 新增/统一配置键

建议在配置中统一增加：

- `forecast_reward.enabled: true`（默认 true）
- `forecast_reward.models: [tcn, lstm, transformer]`
- `forecast_reward.model_weights`
- `forecast_reward.horizon_weights`
- `reward.lambda_forecast`（主权重）
- `reward.lambda_switch`
- `reward.lambda_coverage`
- `reward.lambda_violation`
- `reward.lambda_state_tracking`（默认 0）

### 7.3 训练段切分配置

- `splits.predictor_pretrain`
- `splits.rl_train`
- `splits.rl_val`
- `splits.final_test`

并在脚本入口强校验：区间不能重叠。

---

## 8. 训练与评估脚本流程（目标状态）

### 8.1 训练流程

1. 生成 truth 数据；
2. 用 S0 训练并冻结 TCN/LSTM/Transformer；
3. 在 S1 训练调度器（DQN/CMDP-DQN/PPO）；
4. 在 S2 做早停和选模；
5. 固化最优调度器。

### 8.2 最终验证流程

1. 在 S3 回放各调度器生成观测序列；
2. 使用同一组冻结预测器评估 forecasting 指标；
3. 额外可做“scheduler-specific retrain”作为补充分析，不作为主结论来源；
4. 输出 aggregate、task-focus、曲线图、激活时间线。

---

## 9. 指标与图表（与现有风格对齐）

必须输出：

- RMSE / MAE / sMAPE；
- DTW / Pearson；
- 相对 full_open 的误差增幅；
- 节电率、平均功率、峰值违约率；
- 预测曲线 overlay + zoom；
- 传感器开关时间线；
- RMSE increase 箱线图；
- DTW increase 箱线图。

主结论口径：

- 在同等约束下，谁的 forecasting 综合表现更好；
- 在同等误差下，谁更省电；
- 是否逼近 full_open 上限。

---

## 10. 验收标准（必须满足）

本主线合入前，至少满足：

1. 奖励日志中 forecast loss 占主导，且随训练下降；
2. 调度动作存在真实时变，不出现长期全开/全关伪策略；
3. S3 上 RL 至少优于 random / periodic 中的一类，且不出现数量级崩坏误差；
4. CMDP 约束违约率显著低于无约束策略；
5. 图表与 csv 一致，可复现实验结论。

---

## 11. 实施顺序（MVP -> v2）

### MVP（先做）

- 完成 4 段切分与防泄漏检查；
- 接入冻结三模型 forecasting reward；
- 让 DQN 与 CMDP-DQN 在新奖励下可稳定收敛；
- 产出一轮完整 aggregate + 图表。

### v2（扩展）

- 调整模型权重与步长权重；
- 加入事件段加权奖励；
- 加入不确定性加权（高不确定时段提高预测损失权重）；
- 扩展到更多 horizon 与多目标联合评分。
