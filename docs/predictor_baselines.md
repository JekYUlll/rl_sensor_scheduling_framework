# 时序预测模型基线说明

## 1. 文档目的

本文件用于说明 `rl_sensor_scheduling_framework` 中各个下游时序预测模型的定位、实现形式与论文比较意义。

当前实验中，预测模型的作用是：

- 接收某个 scheduler 生成的估计状态序列；
- 在统一的 train/val/test 窗口切分下进行训练和测试；
- 比较不同调度策略对预测性能的影响。

因此，这些模型不用于替代调度算法，而是用于回答：

> 某种调度策略生成的数据，对不同类型的时序预测模型是否都仍然有效。

## 2. 统一比较原则

所有预测模型都共享以下基本条件：

- 输入来自同一个 scheduler-specific dataset；
- 使用相同的 `lookback` 与 `horizon` 设置；
- 目标变量默认为 `snow_mass_flux_kg_m2_s`；
- 指标统一使用 `RMSE / MAE / MAPE`，并相对 `full_open` 做比较。

需要强调：

- 这些模型应当分别报告结果，而不应简单做模型平均来定义论文主结论；
- 如果某个调度策略只对单一模型有效，而对其它模型明显失效，那么其结论不能视为稳健。

## 3. 各预测模型说明

### 3.1 `naive`

配置：

- `configs/predictor/naive.yaml`

实现：

- `src/forecasting/baselines.py`

逻辑：

- 直接复制最后一个观测时间步；
- 对所有 horizon 重复该值。

作用：

- 最基础的 persistence baseline；
- 用来衡量数据本身的短时可预测性。

解释意义：

- 如果 `naive` 已经很强，说明目标变量具有很强的短时自相关；
- 这时更复杂模型的收益应重点体现在事件段、峰值段或更长 horizon 上。

### 3.2 `mlp`

配置：

- `configs/predictor/mlp.yaml`

实现：

- `src/forecasting/mlp.py`

逻辑：

- 将 `lookback × feature_dim` 展平；
- 通过全连接网络直接回归未来多步目标。

作用：

- 作为不显式建模时序结构的简单神经网络基线；
- 用于检验输入特征本身的线性/非线性可分性。

特点：

- 实现简单；
- 对输入噪声较敏感；
- 若其性能异常波动，往往说明输入表示或归一化存在问题。

### 3.3 `lstm`

配置：

- `configs/predictor/lstm.yaml`

实现：

- `src/forecasting/lstm.py`

逻辑：

- 通过单向 LSTM 编码历史窗口；
- 使用最后一个时间步的隐藏状态回归未来多步输出。

作用：

- 作为经典时序神经网络基线；
- 用于检验历史依赖与短期动态模式是否重要。

特点：

- 对短时依赖建模稳定；
- 对小样本通常比大 transformer 更稳；
- 常可作为论文中的“标准深度时序基线”。

### 3.4 `transformer`

配置：

- `configs/predictor/transformer.yaml`

实现：

- `src/forecasting/transformer.py`

逻辑：

- 先将输入投影到 `d_model`；
- 加入位置编码；
- 再通过 Transformer encoder 提取时序表示；
- 使用最后一个时间步编码回归未来多步目标。

作用：

- 作为自注意力架构基线；
- 用于检验长依赖建模是否优于 RNN/TCN。

特点：

- 表达力强；
- 对数据量和输入表示更敏感；
- 若输入中冗余信息较多，性能不一定稳定优于 LSTM。

### 3.5 `informer`

配置：

- `configs/predictor/informer.yaml`

实现：

- `src/forecasting/informer.py`

逻辑：

- 当前版本是简化的 informer-like baseline；
- 实现上继承 `TransformerPredictor`，本质仍是 encoder-style transformer。

作用：

- 保留一个“长序列 transformer 家族”的对照模型；
- 方便与标准 transformer 比较。

说明：

- 当前实现不是论文级完整 Informer 复现；
- 更适合作为结构风格接近的工程基线。

### 3.6 `tcn`

配置：

- `configs/predictor/tcn.yaml`

实现：

- `src/forecasting/tcn.py`

逻辑：

- 使用 1D 卷积堆叠提取时序局部模式；
- 以最后时刻的高层表示回归未来多步目标。

作用：

- 作为卷积式时序建模基线；
- 特别适合检验局部平滑趋势与短期动态是否主导预测。

特点：

- 对短时局部模式通常较强；
- 训练稳定；
- 在高自相关任务上常可与 LSTM 接近甚至更优。

### 3.7 `pinn`

配置：

- `configs/predictor/pinn.yaml`

实现：

- `src/forecasting/pinn.py`
- `src/forecasting/physics_constraints.py`

逻辑：

- 主体是 LSTM 回归器；
- 训练时额外加入 physics-informed 约束项；
- 当前支持的约束包括：
  - `nonnegative_target`
  - `threshold_transport`
  - 事件段加权

作用：

- 作为 physics-informed forecasting baseline；
- 用于验证：加入轻量物理约束后，是否能缓解调度诱导的观测缺失对预测的影响。

解释注意：

- 当前版本属于“轻量物理正则”而非完整物理模型；
- 如果它没有明显改变 scheduler 排名，更可能说明瓶颈在调度/输入表示，而不是仅仅缺少物理约束。

### 3.8 `sert_like`

配置：

- `configs/predictor/sert_like.yaml`

实现：

- `src/forecasting/sert_like.py`

逻辑：

- 基于 transformer encoder；
- 对输入先做 value projection 与 gating；
- 设计上强调 missing-aware 输入；
- 期望调用方预先拼接：
  - observed mask
  - time delta

作用：

- 作为缺失感知 transformer 风格基线；
- 用于检验“显式利用缺测结构”是否比普通 transformer 更稳。

说明：

- 当前是 scoped baseline，不是论文级严格复现；
- 更适合作为“missing-aware architecture family”的代表。

### 3.9 `s4m_like`

配置：

- `configs/predictor/s4m_like.yaml`

实现：

- `src/forecasting/s4m_like.py`

逻辑：

- 使用状态空间风格的递推更新；
- 通过 gated state update 累积时序信息；
- 同样期望调用方预先拼接：
  - observed mask
  - time delta

作用：

- 作为 missing-aware state-space-like baseline；
- 用于检验状态空间风格方法在不规则观测输入下的稳健性。

说明：

- 当前也是 scoped baseline；
- 更强调“建模风格”而非论文级严格复现。

## 4. 建议的模型分层

为了让论文比较更清晰，建议将预测模型分成四组：

### 第一组：极简基线

- `naive`

作用：

- 给出最小可接受基准；
- 判断任务是否被强自相关主导。

### 第二组：标准深度基线

- `mlp`
- `lstm`
- `tcn`
- `transformer`
- `informer`

作用：

- 覆盖常见的多种时序建模范式；
- 是论文主结果最应依赖的一组。

### 第三组：物理约束扩展

- `pinn`

作用：

- 检验物理先验是否能提高调度生成数据的利用效率；
- 也可为后续 reward oracle 设计提供参考。

### 第四组：缺失感知扩展

- `sert_like`
- `s4m_like`

作用：

- 检验显式 missing-aware 输入增强是否有效；
- 当前更适合作为扩展实验，而不是论文主结论的唯一依据。

## 5. 当前实验解释建议

当前实验若出现如下现象：

- 某些简单模型上 `round_robin` 略优于 `full_open`

不应直接解释为：

- `round_robin` 的信息量高于 `full_open`

更合理的解释是：

- 现有输入表达尚未充分利用 `full_open` 的额外信息；
- 结构化低功耗调度可能带来了轻度正则化或去噪效应；
- 简单模型对这种输入更敏感。

因此，预测模型部分的正确使用方式应当是：

- 分模型分别报告；
- 强调跨模型一致性；
- 将 `full_open` 视作理论上限；
- 把局部负增幅解释为“输入表示问题需要进一步优化”，而不是直接作为调度优越性的最终结论。

## 6. 当前使用建议

如果篇幅允许，建议完整保留：

- `naive`
- `mlp`
- `lstm`
- `tcn`
- `transformer`
- `informer`
- `pinn`

其中：

- 主结论建议优先依据：
  - `lstm`
  - `tcn`
  - `transformer`
  - `informer`
- `naive` 用于解释目标的短时可预测性；
- `pinn` 用于解释物理约束是否有额外收益；
- `sert_like / s4m_like` 可作为扩展实验保留，不必强行写成论文主结论核心。
