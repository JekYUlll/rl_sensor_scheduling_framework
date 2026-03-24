# 时序预测模型基线说明

## 1. 文档目的

本文件说明 `rl_sensor_scheduling_framework` 中各类下游预测模型的输入、目标和比较意义。当前框架的核心问题不是“哪个预测模型最好”，而是：

> 某种调度策略生成的估计状态序列，是否还能支撑稳定的微气候状态预测。

因此，预测模型是调度评估器，不是调度方法本身。

## 2. 当前输入与目标定义

### 2.1 预测器输入

下游 predictor 的输入不是 raw truth，而是：

- scheduler 调度后的观测流；
- 经 estimator（Kalman）融合后得到的估计状态序列；
- 再拼接物理派生特征与可选 missing-aware 特征。

当前基础状态列包括：

- `wind_speed_ms`
- `wind_direction_deg`
- `air_temperature_c`
- `relative_humidity`
- `air_pressure_pa`
- `solar_radiation_wm2`
- `snow_surface_temperature_c`
- `snow_particle_mean_diameter_mm`
- `snow_particle_mean_velocity_ms`
- `snow_mass_flux_kg_m2_s`

当前默认物理派生特征包括：

- `wind_dir_sin`
- `wind_dir_cos`
- `wind_u`
- `wind_v`
- `surface_air_temp_gap`
- `particle_kinetic_proxy`
- `size_velocity_interaction`
- `transport_exceedance`

因此，标准 predictor 默认看到的是“估计状态 + 物理增强特征”，而不是简单的原始单变量时间序列。

### 2.2 当前 forecast targets

当前 windblown 主实验已经不再把 `solar_radiation_wm2` 作为预测目标。原因很简单：在当前 truth 生成方式下，辐射呈现尖峰稀疏分布，现有预测器无法对它形成可信预测，因此继续把它放在主目标集合里会扭曲评价。

当前 forecast target 列为：

- `air_temperature_c`
- `snow_surface_temperature_c`
- `wind_speed_ms`
- `wind_dir_sin`
- `wind_dir_cos`
- `snow_mass_flux_kg_m2_s`
- `snow_particle_mean_velocity_ms`

其中，当前主任务口径更偏向：

- 微气候主状态预测：
  - `air_temperature_c`
  - `snow_surface_temperature_c`
  - `wind_speed_ms`
- 吹雪辅助状态预测：
  - `wind_dir_sin`
  - `wind_dir_cos`
  - `snow_mass_flux_kg_m2_s`
  - `snow_particle_mean_velocity_ms`

### 2.3 评估指标

当前不能只看 `dRMSE`。现已统一补充：

- `RMSE`
- `MAE`
- `sMAPE`
- `Pearson`
- `DTW`

原因：

- `RMSE/MAE` 看点误差；
- `sMAPE` 看相对尺度；
- `Pearson` 看同步性和形状相关；
- `DTW` 看局部相位偏移后的整体轨迹相似性。

对于你现在这种多传感器调度问题，只用 `dRMSE` 证据不够。

## 3. 各预测模型说明

### 3.1 `naive`

- 配置：`configs/predictor/naive.yaml`
- 实现：`src/forecasting/baselines.py`

逻辑：

- 复制最后一个观测时间步；
- 对所有 horizon 直接外推。

作用：

- persistence baseline；
- 用来判断目标变量的短时自相关是否已经足够强。

### 3.2 `mlp`

- 配置：`configs/predictor/mlp.yaml`
- 实现：`src/forecasting/mlp.py`

逻辑：

- 将 `lookback × feature_dim` 展平；
- 用全连接网络直接回归未来多步目标。

作用：

- 检验输入表达本身是否已经足够可分；
- 若它表现异常波动，通常说明输入尺度或噪声处理仍有问题。

### 3.3 `lstm`

- 配置：`configs/predictor/lstm.yaml`
- 实现：`src/forecasting/lstm.py`

逻辑：

- 用 LSTM 编码历史窗口；
- 用最终隐藏状态回归未来多步目标。

作用：

- 经典深度时序基线；
- 对短时和中短期依赖通常比较稳。

### 3.4 `transformer`

- 配置：`configs/predictor/transformer.yaml`
- 实现：`src/forecasting/transformer.py`

逻辑：

- 输入投影到 `d_model`；
- 加位置编码；
- 用 encoder 提取时序表示并回归未来目标。

作用：

- 检验更强的全局依赖建模是否能更好利用调度后输入。

### 3.5 `informer`

- 配置：`configs/predictor/informer.yaml`
- 实现：`src/forecasting/informer.py`

逻辑：

- 当前是简化的 informer-like 工程基线；
- 结构风格近似 transformer family，但不是完整论文级 Informer 复现。

作用：

- 提供一个长依赖 transformer 家族的对照模型。

### 3.6 `tcn`

- 配置：`configs/predictor/tcn.yaml`
- 实现：`src/forecasting/tcn.py`

逻辑：

- 使用时序卷积提取局部模式；
- 用卷积堆叠后的表示回归未来多步目标。

作用：

- 对局部平滑结构和短时动态通常较强；
- 在高自相关任务上常常是强基线。

### 3.7 `pinn`

- 配置：`configs/predictor/pinn.yaml`
- 实现：
  - `src/forecasting/pinn.py`
  - `src/forecasting/physics_constraints.py`

逻辑：

- 主体是 LSTM；
- 训练时加入轻量 physics-informed 约束。

当前约束包括：

- `nonnegative_target`
- `threshold_transport`
- 事件段加权

作用：

- 检验物理约束是否能提高对调度后数据的利用效率；
- 更适合看“补强 forecasting 端”的价值，而不是替代调度侧修复。

### 3.8 `sert_like`

- 配置：`configs/predictor/sert_like.yaml`
- 实现：`src/forecasting/sert_like.py`

逻辑：

- transformer 风格；
- 显式依赖 `observed_mask` 与 `time_delta`；
- 面向缺测感知输入。

作用：

- 检验 missing-aware 输入建模是否比普通 transformer 更稳。

### 3.9 `s4m_like`

- 配置：`configs/predictor/s4m_like.yaml`
- 实现：`src/forecasting/s4m_like.py`

逻辑：

- 状态空间风格递推更新；
- 同样依赖 `observed_mask` 与 `time_delta`。

作用：

- 作为 missing-aware state-space-like baseline。

## 4. 论文中建议的模型分层

### 第一组：极简基线

- `naive`

### 第二组：标准深度时序基线

- `mlp`
- `lstm`
- `tcn`
- `transformer`
- `informer`

### 第三组：物理约束扩展

- `pinn`

### 第四组：缺失感知扩展

- `sert_like`
- `s4m_like`

主结论更应依赖第二组与第三组，而不是只看 `naive` 或单一 missing-aware 模型。

## 5. 当前解释建议

当前若出现：

- 某些简单模型上 `round_robin` 或 `periodic` 略优于 `full_open`

更合理的解释是：

- 全开观测的额外信息尚未被当前输入表示充分利用；
- 某些结构化调度产生了轻度去噪/正则化效应；
- 不应直接写成“轮询优于全开”。

因此，预测模型部分的正确使用方式是：

- 分模型分别报告；
- 在 `RMSE + DTW + Pearson` 三类指标上同时看；
- 优先看主任务目标集合，而不是只盯单一雪通量目标；
- 把局部负增幅视为输入表示和目标定义仍需优化的信号，而不是最终结论本身。
