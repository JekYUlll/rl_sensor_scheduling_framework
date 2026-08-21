# 仿真数据生成器缺口分析：合理性评估与优先级排序

## 总体判断

用户提出的五个缺口**全部成立**，但优先级和紧迫性差异显著。其中缺口 5（事件条件动态调度 claim 未被实验支持）是**当前最危险的论文风险**，必须在提交修订版前解决；缺口 1（能量过程未生成）和缺口 2（事件信号免费可见）是**中期架构问题**，影响论文的物理合理性声明；缺口 3（功耗抽象混合）已在 turn_69 的 physical_event_value_v2 方案中部分解决；缺口 4（多年份宣称过度）是**措辞问题**，修改成本最低。

以下逐项评估，并给出可操作的修复路径。

---

## 缺口 1：供电/能量过程未生成（中期架构问题）

### 合理性评估：完全成立，但紧迫性取决于论文声明的强度

用户的诊断准确。当前 V3.1 主线使用固定 `per_step_budget`（B=1.65/1.70/1.75），不随天气变化，也没有 battery SOC。这与论文中反复出现的"tight power budget"、"energy-harvesting AWS"等表述存在张力。

然而，这个缺口的紧迫性取决于论文**实际声明了什么**：

若论文仅声明"本文采用固定功耗预算作为简化模型，代表极夜期间风能发电的保守估计"，则缺口 1 是一个**局限性**，可以在 §7 中承认，不需要立即修复。

若论文声明"本框架建模了 Antarctic AWS 的能量收集动态"或"time-varying power budget"，则缺口 1 是一个**虚假声明**，必须修复或删除。

**修复路径（按工程量排序）**：

最小改动（推荐用于当前修订版）：在 §3.2 明确说明"本文采用固定功耗预算 B 作为简化模型，对应太阳能/风能系统的保守平均发电功率估计，未建模电池 SOC 动态和能量收集过程"，并在 §7 局限性中列出"未来工作可引入能量账户模型（SOC 作为观测输入），以更真实地建模南极 AWS 的供电动态"。

中等改动（推荐用于论文最终版，若 pilot 成功）：实现能量账户模型（turn_69 §4.4 的方案二），将 `energy_account_t` 作为观测空间输入，允许 `laser=1.35` 在平均预算 `B_avg=1.00` 下可行。这与 Fernandez-Bes et al. (2025) 的能量收集传感器 MDP 框架一致，论文贡献也更有说服力。

**当前建议**：采用最小改动，修改 §3.2 和 §7 措辞，不在当前修订版中实现能量账户模型。

---

## 缺口 2：事件信号可能过于"先验可见"（中期架构问题）

### 合理性评估：成立，但存在可辩护的物理解释

用户的担忧有道理：若 `event_flag` 直接来自被调度的事件传感器（如 Parsivel²），则在传感器未开启时策略不应免费知道事件是否发生。这会造成**信息泄露**，使策略在实际部署中无法复现训练时的性能。

然而，这个问题有一个关键的物理辩护路径：**事件前兆可以从低成本基础气象传感器推断**。具体而言：

南极吹雪事件通常伴随风速超过阈值（约 7–10 m/s）、温度低于某值、辐射骤降等前兆信号，这些信号来自 `met_station_core`（GMX500 MaxiMet，成本 0.18，始终开启）。因此，若论文明确说明"事件前兆特征由始终开启的基础气象传感器（met_station_core）提供，而非来自被调度的高成本传感器"，则信息泄露问题可以被辩护。

**需要检查的关键问题**：当前实现中，`event_flag` 或 `storm_regime_belief` 是否来自 met_station_core 的输出，还是来自 Parsivel²/FC4 的输出？若来自后者，则存在真实的信息泄露问题。

**修复路径**：

若 `event_flag` 来自 met_station_core（始终开启）：在论文中明确说明这一点，问题可以辩护。

若 `event_flag` 来自 Parsivel²/FC4（被调度传感器）：需要将其替换为"基于 met_station_core 输出的事件概率估计"（例如，风速超过阈值的概率），并在论文中说明这是一个保守的信息假设。

**当前建议**：在代码中确认 `event_flag` 的来源，若存在信息泄露，在 §3.1 中明确说明观测空间的信息假设，并在局限性中承认"真实部署中事件检测依赖于传感器可用性"。

---

## 缺口 3：传感器物理功耗与调度成本混合（已部分解决）

### 合理性评估：成立，但 physical_event_value_v2 已提供可辩护的解决方案

用户的诊断准确：当前成本设置与真实传感器功耗排序不一致，且 FC4/Parsivel 的解释会引发审稿人追问。

然而，turn_69 的 physical_event_value_v2 方案已经提供了一个可辩护的解决路径：

`laser=0.90`（对应 Parsivel² 稳态功耗含基础加热维持，不含峰值加热激活），`B=1.20`（对应极夜期间风能发电的保守估计）。这个解释在论文中是可以辩护的，前提是在 §3.2 明确说明"本文的功耗成本代表传感器稳态运行功耗的归一化值，峰值加热功耗通过 `startup_peak_power` 参数单独建模，但在当前简化模型中不直接参与预算约束"。

**当前建议**：采用 physical_event_value_v2 配置，在 §3.2 中明确功耗抽象层次，在局限性中承认"Parsivel² 加热器的间歇性峰值功耗未被完整建模"。

---

## 缺口 4："多年份/多季节"宣称过度（措辞问题，修复成本最低）

### 合理性评估：完全成立，措辞需要精确降级

用户的诊断准确。当前生成器基于 AntAWS 多年统计数据（DFT phase randomization），但没有真实建模季节演化、极昼极夜切换、年度太阳高度变化等。

**精确的可辩护措辞**：

可以说："calibrated to multi-year Antarctic AWS statistics from the AntAWS dataset"

不能说："multi-season deployment validation"、"cross-season generalization"、"validated across multiple Antarctic seasons"

**具体修改建议**：

在 §4 中将"multi-year Antarctic AWS data"改为"multi-year Antarctic AWS statistics"，并在脚注中说明"合成序列通过 DFT phase randomization 保留了原始信号的统计特性（均值、方差、自相关结构），但不建模季节性趋势或极昼极夜切换"。

**当前建议**：这是修复成本最低的缺口，可以在论文修改阶段直接处理，不需要任何代码改动。

---

## 缺口 5：事件条件动态调度 claim 未被实验支持（最危险的论文风险）

### 合理性评估：完全成立，这是当前最紧迫的问题

用户的诊断完全正确，且这是五个缺口中**最危险的**。论文中存在若干表述（"selectively warms event-sensitive instruments"、"adaptive mechanism for event-conditioned warm-up decisions"）目前没有实验证据支持，甚至与 Figure 8 的 rollout 数据相矛盾（PD-PPO 6/8 传感器近常量，laser/FC4 事件 lift 很弱甚至负）。

这类"claim 了但未实现"的表述是审稿人最容易抓住的弱点，也是 Major Revision 意见的核心来源之一。

### 五类未实现 claim 的精确诊断

**Claim A：time-varying power budget**

论文架构图/文字暗示功耗预算随时间变化，但主实现是固定 B。这是一个**过度声明**，需要在 §3.2 中明确说明"本文采用固定预算作为简化模型"。

**Claim B：event-conditioned warm-up decisions**

论文说 PD-PPO 学会了"在事件期间激活高成本传感器并等待预热完成"。但 V3.1 S2 rollout 显示 PD-PPO 大多是准静态，laser 的事件 lift 很弱。这是一个**实验未支持的 claim**，需要通过 physical_event_value_v2 pilot 验证，或在论文中降级为"理论上 PD-PPO 具备此能力，但在当前参数配置下未充分显现"。

**Claim C：adaptive scheduling necessary**

论文介绍里说"normalized costs and warm-up delays make adaptive scheduling necessary"，但实验显示 strong static already nearly optimal（feasible_static_projected FW-MAE=0.4352，PD-PPO FW-MAE=0.4398，差距仅 0.0046）。这是一个**与实验结果矛盾的 claim**，需要修改为"adaptive scheduling provides directional but not uniformly significant gains over strong static baselines"。

**Claim D：sustained exploration / not deterministic policy**

训练诊断文字说没有 collapsed/deterministic policy，但 Figure 8 和 score 诊断显示至少部分 run 是硬常量掩码（6/8 传感器近常量）。这是一个**与 Figure 8 矛盾的 claim**，需要在 §5 中承认"在当前参数配置下，PD-PPO 学到接近强静态参考的策略，这在能量约束非激活时是理论最优的"。

**Claim E：physical sensor/power constraints**

论文可以说"informed by datasheets"，但不能说"真实功耗模型"；heater、battery、电源电子都未实现。这是一个**措辞问题**，将"physical power constraints"改为"power budget constraints informed by sensor datasheets"即可。

### 修复路径：两阶段策略

**第一阶段（当前修订版，必须完成）**：

将所有未实现的 claim 降级为诚实的表述。具体替换如下：

"selectively warms event-sensitive instruments" → "learns scheduling policies that can selectively activate event-sensitive instruments when energy budget permits"

"adaptive mechanism for event-conditioned warm-up decisions" → "prediction-driven mechanism designed to support event-conditioned warm-up decisions"

"adaptive scheduling necessary" → "adaptive scheduling provides a principled framework for balancing measurement quality and power constraints"

"PD-PPO significantly outperforms all baselines" → "PD-PPO achieves directional but not uniformly significant gains over strong static baselines"

在 §5 中新增一段：

"The near-static behavior observed in Figure 8 is consistent with theoretical predictions: when the energy budget B=1.70 exceeds the power consumption of the optimal feasible sensor subset (1.46), the energy constraint is inactive, and the optimal policy degenerates to a fixed subset selection (Fernandez-Bes et al., 2025, Theorem 1). This finding motivates the tighter budget scenario (B=1.20) examined in Section 6.X, where the energy constraint is genuinely active and dynamic scheduling provides measurable advantages."

**第二阶段（若 physical_event_value_v2 pilot 成功）**：

若 pilot 显示 event-conditional laser activation ratio > 3:1 且 event snow FW-MAE 改善 > 2%，则可以将降级的 claim 恢复为实验支持的表述，并将 physical_event_value_v2 场景作为主实验场景。

---

## 综合优先级排序与行动计划

### 优先级 1（必须在提交修订版前完成）：修改论文措辞

以下修改不需要任何代码改动，可以立即执行：

将所有"event-conditioned warm-up decisions"等未实现 claim 降级（见缺口 5 第一阶段）。在 §3.2 明确说明固定预算简化假设（缺口 1）。将"multi-year"改为"calibrated to multi-year statistics"（缺口 4）。在 §3.2 说明功耗抽象层次（缺口 3）。在 §7 局限性中承认能量缓冲未建模、事件信号来源假设（缺口 1、2）。

### 优先级 2（oracle lift 诊断，约 1 小时，无需重训）

在 physical_event_value_v2 成本向量 + B=1.00/1.10/1.20 下，计算不同子集的最优固定子集及其 oracle loss，确认约束激活边界，量化 laser_event_lift。

若 laser_event_lift ≥ 0.005：进入 pilot 阶段。若 laser_event_lift < 0.005：问题在合成数据生成器的观测噪声模型，需先修正 laser_disdrometer 的信息价值建模，再运行 pilot。

### 优先级 3（physical_event_value_v2 pilot，8–12 小时）

单 seed pilot（seed=41），B=1.20，physical_event_value_v2 配置。验证 event-conditional laser activation ratio > 3:1，laser event lift > 0，warmup abort rate < round_robin。

### 优先级 4（中期架构改进，论文最终版）

实现能量账户/SOC 模型（缺口 1），确认 event_flag 来源并修复信息泄露（缺口 2）。这两项改动可以合并为一个"物理合理性增强"版本，在 pilot 成功后实施。

---

## 关于"不要只改权重"的核心判断

用户的建议完全正确。当前问题的根源不是权重设置，而是**实验场景的结构性缺陷**：能量约束非激活（B=1.70 >> 最优固定子集功耗 1.46），导致 MDP 退化为无约束固定子集选择问题。在这个退化的 MDP 上，无论如何调整奖励权重，最优策略都是固定子集，PPO 会正确地学到这个策略。

两阶段目标的核心逻辑是：

第一阶段（physical_event_value_v2）：通过降低预算（B=1.20）和调整成本向量（laser=0.90），使能量约束真正激活，产生真正的调度张力。这是验证 PD-PPO 动态调度能力的**必要前提**。

第二阶段（能量账户模型）：通过实现 SOC 动态，使调度问题更接近真实 Antarctic AWS 的物理场景，同时允许 laser=1.35 在平均预算下可行，产生更丰富的调度行为。这是提升论文物理合理性和贡献深度的**长期目标**。

在第一阶段完成之前，继续在 B=1.70 上调整权重是无效的——这等价于在一个退化的 MDP 上寻找不存在的动态最优策略。
