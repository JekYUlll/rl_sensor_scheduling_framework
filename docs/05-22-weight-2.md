# 修正评估：laser 可行性约束、YAML 格式、soft coverage 实现，以及瞬时 vs 平均功耗预算的根本选择

## 总体判断

用户的三项修正**全部成立**，且第一项（laser 成本不能高于预算）是整个方案能否产生调度价值的**前提条件**，不是细节问题。第四个问题（瞬时 vs 平均功耗预算）是一个更深层的架构选择，对论文的物理合理性和实验可行性都有根本影响，需要单独讨论。

---

## 一、最重要的修正：laser 必须在预算内可行

### 1.1 问题的精确表述

用户的诊断完全正确。在当前**硬 per-step budget** 实现下，若 `laser_disdrometer.power_cost > B`，则 laser 在任何时间步都无法被选择，MaskedActor 会将其 logit 置为 `-inf`，智能体永远不会探索激活 laser 的状态。这不是"稀缺资源"，而是"不存在的选项"。

turn_68 的配置（`laser=1.35, B=0.90/1.00/1.15`）犯了这个错误：在所有三个预算点，laser 都完全不可行。这意味着整个 physical_event_value_v1 方案在当前实现下**无法产生任何关于 laser 调度的学习信号**。

### 1.2 可行性约束的精确计算

设 `met_station_core = 0.18`（required），`fc4_flux = 0.05`（期望作为哨兵），则 laser 的可行性条件为：

**最小可行组合**（仅 met + laser）：
```
laser_max = B - 0.18
```

**期望组合**（met + fc4 + laser，哨兵同时开）：
```
laser_max = B - 0.18 - 0.05 = B - 0.23
```

在不同预算点下，laser 的最大可行成本：

| B    | met+laser 最大 laser | met+fc4+laser 最大 laser |
|------|---------------------|--------------------------|
| 0.90 | 0.72                | 0.67                     |
| 1.00 | 0.82                | 0.77                     |
| 1.10 | 0.92                | 0.87                     |
| 1.20 | 1.02                | 0.97                     |
| 1.35 | 1.17                | 1.12                     |

### 1.3 修正后的推荐配置

用户建议的 `laser=0.90, B=1.00/1.10/1.20` 是正确的。在这个配置下：

- `met(0.18) + fc4(0.05) + laser(0.90)` = 1.13，在 B=1.20 可行，在 B=1.10 边界（超出 0.03），在 B=1.00 不可行
- `met(0.18) + fc4(0.05) + snow_particle(0.50)` = 0.73，在所有预算点可行
- `met(0.18) + fc4(0.05) + laser(0.90) + radio(0.06)` = 1.19，在 B=1.20 边界可行

这产生了真正的调度张力：在 B=1.10 下，laser 只能与 met 单独配对（不能同时开 fc4），智能体必须在"laser 精度"和"fc4 哨兵"之间权衡；在 B=1.20 下，laser+fc4 可以共存，但不能再加其他传感器。

**最终推荐的成本/预算组合**：

```yaml
# 修正版：laser 在预算内可行但昂贵
laser_disdrometer:
  power_cost: 0.90
  startup_peak_power: 1.25
  warmup_steps: 4

snow_particle_counter:
  power_cost: 0.50
  startup_peak_power: 0.65
  warmup_steps: 2

budget:
  pilot:         1.00   # laser 仅能与 met 配对，snow_particle 可与 fc4 共存
  main_tight:    1.10   # laser+met+fc4 超出 0.03，snow_particle 可与多传感器共存
  main_moderate: 1.20   # laser+met+fc4 可行，但不能再加其他
  reference:     1.70   # 宽松对照
```

### 1.4 为什么 turn_68 的 laser=1.35 在物理上仍然合理，但在当前实现下不可用

turn_68 的 `laser=1.35` 对应 Parsivel² 加热器激活的等效平均功耗（≈4.5 W），物理上是正确的。但这个数字只有在**平均功耗预算**下才有意义——加热器是间歇性激活的，单步瞬时功耗可能高达 50–100 W，但 30 分钟平均下来才是 4.5 W。

在当前硬 per-step budget 实现下，`laser=1.35` 意味着每一步都消耗 1.35 单位，这实际上对应的是"加热器持续全功率运行"的场景，而非间歇激活。这是一个**功耗抽象层次的错误**，不是数值错误。

这个问题引出了第四个讨论点（瞬时 vs 平均预算），见第四节。

---

## 二、YAML 格式修正：必须按 loader 格式写

用户的观察完全正确。turn_68 的 YAML 使用了概念格式（`steady_cost`、`peak_cost`），而当前 loader 使用的字段名是 `power_cost`、`startup_peak_power`、`warmup_steps`，且传感器列表是数组格式而非字典格式。

**正确的 YAML 格式**（参照 `windblown_sensors_*.yaml` 的实际结构）：

```yaml
sensors:
  - sensor_id: met_station_core
    power_cost: 0.18
    startup_peak_power: 0.22
    warmup_steps: 0

  - sensor_id: radiometer_basic
    power_cost: 0.06
    startup_peak_power: 0.08
    warmup_steps: 0

  - sensor_id: surface_temp_ir
    power_cost: 0.08
    startup_peak_power: 0.10
    warmup_steps: 1

  - sensor_id: ultrasonic_anemometer_hd
    power_cost: 0.09
    startup_peak_power: 0.12
    warmup_steps: 0

  - sensor_id: shielded_thermo_hygro
    power_cost: 0.09
    startup_peak_power: 0.12
    warmup_steps: 0

  - sensor_id: snow_particle_counter
    power_cost: 0.50
    startup_peak_power: 0.65
    warmup_steps: 2

  - sensor_id: laser_disdrometer
    power_cost: 0.90        # 修正：从 1.35 降至 0.90，确保在 B=1.20 下可行
    startup_peak_power: 1.25
    warmup_steps: 4

  - sensor_id: fc4_flux
    power_cost: 0.05
    startup_peak_power: 0.07
    warmup_steps: 0

budget: 1.20   # 或 1.10/1.00，取决于实验阶段

required_sensors:
  - met_station_core

coverage_groups: []   # 禁用所有硬覆盖组约束
```

**关键字段名对照**：

| turn_68 概念格式 | 实际 loader 字段名 |
|-----------------|-------------------|
| `steady_cost`   | `power_cost`      |
| `peak_cost`     | `startup_peak_power` |
| `warmup_steps`  | `warmup_steps`（相同）|

---

## 三、coverage_groups 的 soft_preferred 当前未实现

用户的观察完全正确。`soft_preferred` 不是当前 `PowerProjector` 支持的结构，turn_68 的 YAML 中写入 `soft_preferred: [radiometer_basic]` 会被 loader 忽略或报错。

**推荐的替代方案**（按实现复杂度排序）：

**方案 A（最简单，推荐）**：完全禁用 coverage_groups，仅保留 `required_sensors = [met_station_core]`。

```yaml
required_sensors: [met_station_core]
coverage_groups: []
```

这样 `radiometer_basic` 的激活完全由奖励函数中的 `solar_radiation_wm2` 权重（0.55）驱动。若权重设置合理，智能体会学到"辐射计成本低（0.06），信息价值足够，应该常开"，不需要硬约束。

**方案 B（中等复杂度）**：在 `PowerProjector` 中新增 soft coverage penalty：

```python
# 在 reward 计算中加入 soft coverage 惩罚
if 'radiometer_basic' not in active_sensors:
    reward -= soft_coverage_penalty  # 例如 0.01
```

这等价于给 `radiometer_basic` 一个"不激活的小惩罚"，物理上对应"辐射计是基础气象监测的标准配置，不激活需要额外理由"。但这会引入一个新的超参数，审稿人可能质疑其设置依据。

**方案 C（不推荐）**：实现完整的 soft coverage group 机制。工程量大，且对当前实验目标（验证 laser 动态调度）没有直接帮助。

**结论**：采用方案 A，在 pilot 阶段先验证 laser 的动态调度行为，再考虑是否需要 soft coverage 机制。

---

## 四、核心架构问题：瞬时功耗硬约束 vs 长期平均功耗预算

这是整个讨论中最深层的问题，直接影响论文的物理合理性和实验设计的可行性。

### 4.1 两种预算模型的本质区别

**瞬时功耗硬约束（当前实现）**：
```
∀t: Σ_i a_i(t) · c_i ≤ B
```
每一个时间步的功耗之和不超过 B。这是一个**逐步可行性约束**，等价于"电池在每个时间步都不能过载"。

**长期平均功耗预算（能量收集场景）**：
```
(1/T) Σ_t Σ_i a_i(t) · c_i ≤ B
```
或等价地，在滚动窗口内的平均功耗不超过 B。这是一个**时间平均约束**，等价于"太阳能板的平均发电量能支撑平均功耗"。

### 4.2 南极 AWS 的真实物理场景

南极 AWS 的供电系统通常是**太阳能 + 风能 + 电池缓冲**的混合系统。其物理约束更接近长期平均预算，而非瞬时硬约束，原因如下：

第一，电池缓冲允许短时间超过平均功耗。Parsivel² 加热器激活时瞬时功耗可达 50–100 W，但持续时间短（几分钟到几十分钟），电池可以吸收这个峰值。

第二，极夜期间太阳能不可用，系统依赖风能和电池。此时的约束是"电池电量不能耗尽"，这是一个**状态约束**（电池 SOC ≥ 阈值），而非逐步功耗约束。

第三，真实 AWS 运营中，传感器的开关决策通常基于**预测的能量可用性**（未来几小时的风速预测），而非当前瞬时功耗。

### 4.3 瞬时硬约束的问题

当前实现的瞬时硬约束存在两个根本性问题：

**问题一：物理不合理**。Parsivel² 加热器的 50–100 W 瞬时功耗在真实系统中是通过电池缓冲处理的，不会直接触发"传感器不可选"的约束。将其归一化为 `power_cost=1.35` 并施加逐步硬约束，相当于假设系统没有任何能量缓冲，这与南极 AWS 的实际设计不符。

**问题二：产生不可学习的约束**。如第一节所述，若 `laser.power_cost > B`，laser 永远不可选，智能体无法学习任何关于 laser 的调度策略。这不是"稀缺资源"，而是"被禁止的选项"。

### 4.4 长期平均预算的优势

若改为长期平均预算（或滚动窗口平均），则：

- `laser=1.35` 在 `B=1.00` 下仍然可行，只要其激活频率足够低（例如每 10 步激活 1 步，平均功耗 = 1.35/10 = 0.135）
- 智能体可以学习"在事件期间激活 laser，在非事件期间关闭"，这正是期望的动态调度行为
- 物理解释更合理：平均功耗预算对应太阳能/风能的平均发电量

**长期平均预算的实现方式**：

```python
# 滚动窗口平均功耗约束
window_size = 48  # 例如 48 步 = 24 小时（若每步 30 分钟）
rolling_power = deque(maxlen=window_size)
rolling_power.append(current_step_power)
avg_power = sum(rolling_power) / len(rolling_power)

# 约束：滚动平均功耗 ≤ B
if avg_power > B:
    # 软约束：加入惩罚项
    reward -= lambda_power * (avg_power - B)
    # 或硬约束：限制可选传感器集合
```

或者更简单的**能量账户模型**：

```python
# 能量账户：每步补充 B 单位能量，消耗 current_power 单位
energy_account += B - current_step_power
energy_account = min(energy_account, max_battery_capacity)
# 约束：energy_account ≥ 0（不能透支）
```

### 4.5 两种方案的实验设计建议

**方案一（短期修复，推荐用于当前 pilot）**：保留瞬时硬约束，但将 `laser=0.90`，`B=1.10/1.20`，确保 laser 在预算内可行但昂贵。这是用户建议的方案，工程改动最小，可以立即运行。

物理辩护：将 `laser=0.90` 解释为"Parsivel² 在低温环境下的稳态功耗（含基础加热维持），不含峰值加热激活"，`B=1.20` 对应"极夜期间风能发电的保守估计"。这个解释在论文中是可以辩护的。

**方案二（长期改进，推荐用于论文最终版）**：实现能量账户模型，允许 `laser=1.35`，`B_avg=1.00`（平均功耗预算）。这在物理上更合理，且能产生更丰富的调度行为（智能体学习"储能-消耗"的时间策略）。

物理辩护：`B_avg=1.00` 对应太阳能/风能的平均发电功率，`laser=1.35` 对应 Parsivel² 加热器激活的等效平均功耗（间歇激活，30 分钟平均）。能量账户模型对应电池 SOC 的动态变化。

**方案二的额外优势**：能量账户状态可以作为观测空间的一部分输入给智能体，使其能够学习"当电量充足时激活高功耗传感器，当电量不足时切换到低功耗模式"。这是一个更接近真实 AWS 运营逻辑的决策问题，论文贡献也更有说服力。

### 4.6 对论文叙事的影响

**若采用方案一（瞬时硬约束 + laser=0.90）**：

论文需要在 §3.2 明确说明"本文采用逐步功耗约束作为简化模型，其中传感器成本代表稳态功耗的归一化值"，并在局限性中承认"真实系统的能量缓冲机制未被建模"。

**若采用方案二（能量账户模型）**：

论文可以声称"本文的能量账户模型捕捉了太阳能/风能供电系统的动态特性，允许短时间超过平均功耗（通过电池缓冲），同时保证长期能量平衡"。这是一个更强的物理合理性声明，且与 Fernandez-Bes et al. (2025) 的能量收集传感器 MDP 框架更一致。

---

## 五、关于 event multiplier 的建议：完全同意分两步

用户建议先用固定训练权重，再在评估时额外报告 event-conditional 指标，这是正确的策略。

**训练权重（固定）**：
```
0.8, 0.8, 1.2, 0.4, 0.4, 0.55, 4.0, 2.5, 2.5
```

**评估时额外报告的指标**：
```
event_snow_fw_mae          # 事件期间 snow 目标的 FW-MAE
non_event_snow_fw_mae      # 非事件期间 snow 目标的 FW-MAE
laser_event_activation_ratio  # 事件期 laser 激活比例 / 非事件期 laser 激活比例
laser_event_lift           # 激活 laser 的事件期 FW-MAE 改善
fc4_event_lift             # FC4 在事件前驱期的 FW-MAE 改善
warmup_success_rate        # warmup 完成后传感器被使用的比例
warmup_abort_rate          # warmup 中途被中断的比例
```

将 event multiplier 写进训练 reward 的风险：审稿人会质疑"为什么事件期间的奖励要乘以 1.8？这个系数是如何确定的？是否是为了让 PPO 赢而调整的？"这个质疑很难在论文中完全消除。

---

## 六、修正后的完整配置（可直接写入 YAML）

综合以上所有修正，给出最终的 pilot 配置：

```yaml
# physical_event_value_v2.yaml
# 修正：laser=0.90（在 B=1.20 下可行），格式符合 loader 要求，禁用 coverage_groups

sensors:
  - sensor_id: met_station_core
    power_cost: 0.18
    startup_peak_power: 0.22
    warmup_steps: 0

  - sensor_id: radiometer_basic
    power_cost: 0.06
    startup_peak_power: 0.08
    warmup_steps: 0

  - sensor_id: surface_temp_ir
    power_cost: 0.08
    startup_peak_power: 0.10
    warmup_steps: 1

  - sensor_id: ultrasonic_anemometer_hd
    power_cost: 0.09
    startup_peak_power: 0.12
    warmup_steps: 0

  - sensor_id: shielded_thermo_hygro
    power_cost: 0.09
    startup_peak_power: 0.12
    warmup_steps: 0

  - sensor_id: snow_particle_counter
    power_cost: 0.50
    startup_peak_power: 0.65
    warmup_steps: 2

  - sensor_id: laser_disdrometer
    power_cost: 0.90
    startup_peak_power: 1.25
    warmup_steps: 4

  - sensor_id: fc4_flux
    power_cost: 0.05
    startup_peak_power: 0.07
    warmup_steps: 0

budget: 1.20

required_sensors:
  - met_station_core

coverage_groups: []

forecast_weights:
  air_temperature_c: 0.8
  snow_surface_temperature_c: 0.8
  wind_speed_ms: 1.2
  wind_dir_sin: 0.4
  wind_dir_cos: 0.4
  solar_radiation_wm2: 0.55
  snow_mass_flux_kg_m2_s: 4.0
  snow_particle_mean_diameter_mm: 2.5
  snow_particle_mean_velocity_ms: 2.5
```

**约束激活验证**（B=1.20）：
- `met(0.18) + fc4(0.05) + snow_particle(0.50)` = 0.73，可行，是低功耗固定子集
- `met(0.18) + fc4(0.05) + laser(0.90)` = 1.13，可行，是事件期高精度子集
- `met(0.18) + fc4(0.05) + laser(0.90) + radio(0.06)` = 1.19，可行（边界）
- `met(0.18) + fc4(0.05) + laser(0.90) + snow_particle(0.50)` = 1.63，**不可行** ✓
- `met(0.18) + fc4(0.05) + laser(0.90) + surface_temp(0.08)` = 1.21，**不可行** ✓

这产生了真正的调度张力：laser 和 snow_particle_counter 不能同时激活，智能体必须在事件期间选择其中一个。

---

## 七、结论：用户修正的核心洞察

用户提出的关键区别——"不可行的传感器不会产生调度价值；只有'可行但昂贵'的传感器，才会逼 PPO 学会什么时候值得打开"——是整个实验设计的核心原则，完全正确。

这个原则可以形式化为：**有效稀缺资源的定义**是存在一个可行的固定子集 $S^*$ 使得 $\sum_{i \in S^*} c_i \leq B$，但 $S^* \cup \{\text{laser}\}$ 不可行（$\sum_{i \in S^* \cup \{\text{laser}\}} c_i > B$）。只有在这个条件下，激活 laser 才需要牺牲其他传感器，动态调度才有价值。

turn_68 的 `laser=1.35, B=1.15` 违反了这个定义的前提（laser 本身就不可行），而用户建议的 `laser=0.90, B=1.20` 满足这个定义：`met+fc4+snow_particle`（0.73）是可行固定子集，`met+fc4+snow_particle+laser`（1.63）不可行，因此 laser 是真正的稀缺资源。

**关于瞬时 vs 平均预算的最终建议**：当前 pilot 阶段采用瞬时硬约束（方案一），以最小工程改动验证动态调度行为；若 pilot 成功，在论文最终版中考虑实现能量账户模型（方案二），以提升物理合理性和论文贡献的深度。
