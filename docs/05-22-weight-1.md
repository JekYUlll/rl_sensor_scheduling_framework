# 物理功耗抽象与权重方案评估：`physical_event_value_v1`

## 总体判断

用户提出的方案在**方向上完全正确**，物理推理链条自洽，且与 turn_67 的理论分析高度一致。但方案中存在若干需要修正的细节，以及一个被低估的结构性风险。以下逐项评估，最后给出可直接写入 YAML 的最终配置。

---

## 一、功耗抽象评估：哪些合理，哪些需要修正

### 1.1 GMX500 逻辑流拆分：合理，但边际成本需要重新校准

将 GMX500 拆成 `met_station_core`（基础气象）、`ultrasonic_anemometer_hd`（高精度风速流）、`shielded_thermo_hygro`（温湿度流）三个逻辑通道，并给后两者赋予"低边际成本"（0.04–0.06），物理逻辑是正确的：GMX500 的基础功耗（≈0.3 W）在设备开机后已经支付，额外激活某个数据流的边际成本确实接近零。

**需要修正的地方**：用户建议的 `ultrasonic_anemometer_hd = 0.04` 和 `shielded_thermo_hygro = 0.04` 过低，会导致这两个通道几乎"免费"，智能体会无差别地常开它们，不产生任何调度决策。建议将这两个通道的边际成本设为 **0.08–0.10**，足够低以反映边际特性，但足够高以让预算约束在极紧情况下仍然约束它们。

更重要的是：如果 `met_station_core` 已经包含了 GMX500 的基础气象输出（温度、湿度、风速基础精度），那么 `ultrasonic_anemometer_hd` 和 `shielded_thermo_hygro` 的**信息增量**（相对于 `met_station_core` 的边际预测改善）需要在奖励函数中有对应体现，否则这两个通道即使被激活也不会影响 FW-MAE，智能体会学到"激活它们没有意义"，最终还是常关。这是一个**奖励-成本对齐问题**，不是成本设置问题。

### 1.2 Parsivel² 拆分为两个通道：合理，但成本比例需要调整

将 Parsivel² 拆成 `snow_particle_counter`（低速摘要模式，0.45）和 `laser_disdrometer`（全粒子模式，0.90）是一个有创意的设计，物理上对应 Parsivel² 的两种工作模式（低功耗统计输出 vs 全分辨率粒子谱）。

**需要修正的地方**：用户建议的 `snow_particle_counter = 0.45` 和 `laser_disdrometer = 0.90` 的比值（1:2）与真实 Parsivel² 的两种模式功耗比不符。Parsivel² 的主要功耗差异来自**加热器**（50–100 W），而非测量模式本身（测量模式功耗差异约 0.5–1.5 W）。更准确的抽象是：

- `snow_particle_counter`：对应 Parsivel² 无加热器的基础测量模式，归一化成本 ≈ **0.50**（1.5 W / 0.3 W × 0.1 归一化因子）
- `laser_disdrometer`：对应 Parsivel² 加热器激活的全功率模式，归一化成本 ≈ **1.20–1.50**（考虑加热器的间歇性激活，取 5–7 W 等效平均功耗）

这个调整的关键意义是：`laser_disdrometer` 的成本必须**高于 B 的一半**，才能成为真正的稀缺资源。在 B=1.10 的设定下，`laser_disdrometer = 1.20` 意味着激活它就消耗了预算的 109%，必须关闭其他所有传感器——这才是真正的稀缺资源动态。

### 1.3 FC4 成本：这是整个方案最关键的修正

用户建议 `fc4_flux = 0.04–0.06`，这与真实 FC4 FlowCapt 的平均功耗（≈0.06 W）完全一致，**方向完全正确**。这是对 `windblown_sensors_complex.yaml` 中 FC4 被设为高成本（1.46）这一严重错误的直接修正。

**但需要注意一个陷阱**：FC4 成本极低（0.04–0.06）意味着它几乎可以"免费"常开。如果 FC4 的信息价值（对 FW-MAE 的贡献）在非事件期间接近零，智能体会学到"FC4 虽然便宜但没用，不如不开"。这会导致 FC4 被常关，而不是用户期望的"低功耗吹雪哨兵"角色。

**解决方案**：FC4 的"哨兵"价值需要在奖励函数中显式体现。具体来说，FC4 的 `snow_mass_flux` 测量在**事件前驱期**（风速开始上升但尚未达到吹雪阈值）应该有较高的预测价值，因为它能提前检测到低强度吹雪通量。如果当前奖励函数只在事件发生后才给 FC4 高奖励，那么 FC4 的"哨兵"功能就无法被学到。这需要检查事件模型中 FC4 的信息价值时序结构。

### 1.4 LPS10 辐射计：建议降低成本

用户建议 `radiometer_basic = 0.06`，这是合理的。LPS10 的实际功耗约 0.1–0.3 W，归一化后确实应该是低成本通道。当前 `windblown_sensors_complex.yaml` 中辐射计的成本设置需要核查，但方向正确。

### 1.5 SI-111 表面温度：warmup=1 的设置合理

`surface_temp_ir` 保留轻微 warmup（1步）是合理的，SI-111 确实需要热稳定时间。成本 0.08 也符合其实际功耗（约 0.25 W）。

---

## 二、预算区间评估：B=1.10 是否真正有效

用户建议的三个预算点（B=0.95, 1.10, 1.25）需要逐一验证其约束激活性。

**关键计算**：在用户建议的成本向量下，最低可行子集的成本是多少？

假设覆盖组约束放宽（不强制每步选 snow_transport），最低可行子集为 `met_station_core + radiometer_basic`：
- 成本 = 0.18 + 0.06 = **0.24**

这意味着即使 B=0.95，也能维持基础气象监测。真正的约束激活边界取决于"什么是最优固定子集"：

- 若最优固定子集为 `met_station_core + radiometer_basic + fc4_flux + snow_particle_counter`：成本 = 0.18 + 0.06 + 0.05 + 0.50 = **0.79**
- 若最优固定子集包含 `laser_disdrometer`：成本 = 0.18 + 0.06 + 0.05 + 1.35 = **1.64**（使用修正后的 laser 成本 1.35）

在 B=1.10 下：
- `met + radio + fc4 + snow_particle_counter`（成本 0.79）：可行，且可能是最优固定子集
- `met + radio + fc4 + laser_disdrometer`（成本 1.64）：**不可行**，laser 成为真正稀缺资源

**结论**：B=1.10 在修正后的成本向量下，确实能使 `laser_disdrometer` 成为稀缺资源，但需要确认 `snow_particle_counter`（成本 0.50）是否也会被约束。若 `met + radio + fc4 + snow_particle_counter`（成本 0.79）是可行固定子集，则 B=1.10 仍然允许这个子集常开，静态陷阱可能以新的形式出现（只是换了一个更便宜的固定子集）。

**建议**：将 B 设为 **0.85–0.95**，使得即使 `snow_particle_counter` 也无法与 `met + radio + fc4` 同时常开，智能体必须在事件期间动态选择激活哪个 Parsivel² 模式。

---

## 三、覆盖组约束评估：放宽方向正确，但需要明确边界

用户建议放宽覆盖组约束，不强制每步选 snow_transport，这与 turn_67 的分析完全一致。**这是整个方案中最重要的结构性改变**，比任何成本调整都更关键。

**具体建议**：
- 保留 `met_station_core` 作为 required（或 soft-required，违反有小惩罚），因为基础气象是 AWS 的核心功能
- 将 `fc4_flux` 从 snow_transport 覆盖组中移出，允许其独立决策（它的低成本使其可以近常开，但不应被强制）
- 完全移除"每步必须选一个 snow_transport"的硬约束，改为通过奖励函数的 snow 权重来激励 snow 传感器的使用

这样做的物理动机是：在极夜极紧预算条件下，AWS 运营者的首要目标是维持基础气象监测，snow 传感器的激活是**条件性的**（取决于事件预测和预算余量），而非强制性的。

---

## 四、评价权重评估：方向正确，但 solar 降幅过大

用户建议的权重方案：

| 变量 | 当前权重 | 建议权重 | 评估 |
|------|---------|---------|------|
| `air_temperature_c` | 1.0 | 0.8 | 合理，轻微降低 |
| `snow_surface_temperature_c` | 1.0 | 0.8 | 合理 |
| `wind_speed_ms` | 1.2 | 1.2 | 保持，正确 |
| `wind_dir_sin` | 0.6 | 0.4 | 合理，风向对吹雪预测贡献有限 |
| `wind_dir_cos` | 0.6 | 0.4 | 同上 |
| `solar_radiation_wm2` | 1.0 | 0.3 | **过低，见下文** |
| `snow_mass_flux_kg_m2_s` | 3.0 | 4.0 | 合理，提升吹雪核心目标 |
| `snow_particle_mean_diameter_mm` | 2.0 | 2.5 | 合理 |
| `snow_particle_mean_velocity_ms` | 2.0 | 2.5 | 合理 |

**关于 solar 权重**：将 `solar_radiation_wm2` 从 1.0 降至 0.3 的幅度过大，存在两个风险：

第一，`radiometer_basic`（LPS10）的主要输出是辐射数据。若 solar 权重降至 0.3，则 `radiometer_basic` 对 FW-MAE 的贡献极小，智能体可能学到"辐射计可以关掉"，导致 `radiometer_basic` 被常关。这会破坏基础气象监测的完整性，且在论文中难以辩护（审稿人会问：为什么辐射计几乎不被使用？）。

第二，在南极极夜场景中，solar 辐射确实接近零，但 `solar_radiation_wm2` 在仿真中可能还包含了长波辐射或其他辐射分量，其预测价值不应被过度压低。

**建议**：将 `solar_radiation_wm2` 权重设为 **0.5–0.6**，而非 0.3。这样 `radiometer_basic` 仍然有足够的激励被保持激活，同时 snow 目标的相对权重仍然显著提升。

**关于 event-balanced reward 的建议**：用户提出的"事件窗口和非事件窗口各占一半，或事件期 snow targets 乘以 1.5–2.0"是一个更优雅的解决方案，比单纯调整全局权重更有针对性。这与 EventAwareCritic 的设计意图完全一致，建议作为主要方案实施。

---

## 五、成功判据评估：完全正确，需要补充一条

用户提出的成功判据：

> PPO 不一定必须 beat static overall，但必须在 event snow-target FW-MAE 上优于 static，同时 laser/FC4 event lift > 0，warmup abort rate 低于 round-robin/random。

这个判据是**诚实且可防御的**，完全正确。需要补充一条：

> **event-conditional scheduling ratio**：在事件发生的时间步中，PD-PPO 激活 `laser_disdrometer` 的比例应显著高于非事件时间步（建议 > 3:1 的比值），以证明调度行为确实是事件条件的，而非随机的。

这条判据能直接回应审稿人关于"动态调度是否真的由事件驱动"的质疑。

---

## 六、最终推荐配置：`physical_event_value_v1`

综合以上评估，给出可直接写入 YAML 的最终配置：

### 6.1 传感器成本配置

```yaml
# physical_event_value_v1.yaml
# 物理依据：GMX500(0.3W), LPS10(0.1W), SI-111(0.25W), Parsivel²(1.5W测量/5-7W加热等效), FC4(0.06W)
# 归一化基准：GMX500基础功耗 0.3W → 归一化成本 0.18

sensors:
  met_station_core:
    steady_cost: 0.18
    peak_cost: 0.22
    warmup_steps: 0
    description: "GMX500 base meteorological output"

  radiometer_basic:
    steady_cost: 0.06
    peak_cost: 0.08
    warmup_steps: 0
    description: "LPS10 shortwave/longwave radiation"

  surface_temp_ir:
    steady_cost: 0.08
    peak_cost: 0.10
    warmup_steps: 1
    description: "SI-111 IR surface temperature"

  ultrasonic_anemometer_hd:
    steady_cost: 0.09
    peak_cost: 0.12
    warmup_steps: 0
    description: "GMX500 high-resolution wind stream (marginal cost)"

  shielded_thermo_hygro:
    steady_cost: 0.09
    peak_cost: 0.12
    warmup_steps: 0
    description: "GMX500 shielded T/RH stream (marginal cost)"

  snow_particle_counter:
    steady_cost: 0.50
    peak_cost: 0.65
    warmup_steps: 2
    description: "Parsivel² low-rate summary mode (no heater)"

  laser_disdrometer:
    steady_cost: 1.35
    peak_cost: 1.60
    warmup_steps: 4
    description: "Parsivel² full particle mode with heater activation"

  fc4_flux:
    steady_cost: 0.05
    peak_cost: 0.07
    warmup_steps: 0
    description: "FC4 FlowCapt blowing snow flux (continuous low-power sentinel)"
```

**归一化说明**：以 GMX500 基础功耗 0.3 W 为参考，`laser_disdrometer` 的 1.35 对应约 4.5 W 等效平均功耗（加热器间歇激活的等效值），`snow_particle_counter` 的 0.50 对应约 1.5 W（纯测量模式）。

### 6.2 预算设置

```yaml
budget:
  pilot: 0.90          # 极紧：强制动态选择，验证约束激活
  main_tight: 1.00     # 紧：laser 稀缺，snow_particle_counter 可用
  main_moderate: 1.15  # 中：laser 偶尔可用，测试事件条件激活
  reference: 1.70      # 宽松参考：复现当前静态行为，作为对照
```

**约束激活验证**（需在运行前确认）：
- B=0.90：`met(0.18) + radio(0.06) + fc4(0.05) + snow_particle(0.50)` = 0.79，可行；`+ laser(1.35)` = 2.14，不可行 ✓
- B=1.00：同上，laser 仍不可行；但 `met + radio + fc4 + snow_particle + surface_temp(0.08)` = 0.87，可行 ✓
- B=1.15：`met + radio + fc4 + snow_particle + surface_temp + thermo(0.09)` = 0.96，可行；laser 仍不可行 ✓

在所有三个预算点，`laser_disdrometer` 都无法与其他传感器同时常开，必须通过动态调度才能在事件期间激活。

### 6.3 覆盖组配置

```yaml
coverage_groups:
  # 仅保留基础气象的软约束
  basic_meteorology:
    required: [met_station_core]   # 硬约束：基础气象必须激活
    soft_preferred: [radiometer_basic]  # 软约束：辐射计优先，但可牺牲

  # 移除 snow_transport 硬约束
  # snow 传感器的激活完全由奖励函数驱动
  # （不再强制每步选一个 snow_transport 传感器）
```

### 6.4 评价权重配置

```yaml
forecast_weights:
  air_temperature_c: 0.8
  snow_surface_temperature_c: 0.8
  wind_speed_ms: 1.2
  wind_dir_sin: 0.4
  wind_dir_cos: 0.4
  solar_radiation_wm2: 0.55      # 保留适度权重，维持 radiometer 激励
  snow_mass_flux_kg_m2_s: 4.0
  snow_particle_mean_diameter_mm: 2.5
  snow_particle_mean_velocity_ms: 2.5

# 可选：event-balanced reward multiplier
event_reward_multiplier:
  during_event: 1.8              # 事件期间 snow targets 权重乘以 1.8
  outside_event: 1.0
```

**权重归一化后的有效比例**：
- 基础气象（temp×2 + wind×1.2 + dir×0.8 + solar×0.55）≈ 5.35
- 吹雪目标（flux×4.0 + diameter×2.5 + velocity×2.5）≈ 9.0
- 吹雪目标占总权重约 **63%**，相比当前（约 50%）有显著提升，但不至于完全牺牲基础气象监测质量

---

## 七、方案的局限性与风险

**仍然不保证成功的原因**：即使采用上述配置，如果事件模型中 `laser_disdrometer` 在事件期间的 oracle loss 改善（相对于仅使用 `snow_particle_counter`）小于其 warmup 成本（4步 × 预算占用），智能体仍然会选择不激活 `laser_disdrometer`。这是一个**信息价值问题**，不是成本问题。

**建议在运行 pilot 前先做的诊断**（无需重训，约 1 小时）：

```python
# 诊断：在 B=0.90 下，事件期间激活 laser_disdrometer 的 oracle loss 改善
event_oracle_with_laser = oracle_loss(
    sensors=['met_station_core', 'fc4_flux', 'laser_disdrometer'],
    condition='during_event'
)
event_oracle_without_laser = oracle_loss(
    sensors=['met_station_core', 'fc4_flux', 'snow_particle_counter'],
    condition='during_event'
)
laser_event_lift = event_oracle_without_laser - event_oracle_with_laser
print(f"Laser event lift: {laser_event_lift:.4f}")
# 若 laser_event_lift < 0.005，则即使在事件期间，laser 的信息价值也不足以
# 抵消其 warmup 成本，动态激活永远不值得
```

**若诊断显示 laser_event_lift < 0.005**：问题不在成本/预算设置，而在于仿真数据中 Parsivel² 全粒子模式相对于摘要模式的**信息增量过小**。此时需要检查合成数据生成器中 `laser_disdrometer` 和 `snow_particle_counter` 的观测噪声模型是否有足够的差异化。

---

## 八、与论文叙事的对接

采用 `physical_event_value_v1` 配置后，论文的叙事框架可以更新为：

> "We adopt a physically-grounded sensor cost model based on measured hardware specifications: the Parsivel² disdrometer in full particle mode (with heater activation) consumes approximately 4.5× the power of the GMX500 base station, while the FC4 FlowCapt operates at approximately 1/6 of the GMX500 power draw. Under tight power budgets representative of Antarctic polar night conditions (B ≤ 1.15), the Parsivel² full mode becomes a genuinely scarce resource that cannot be continuously activated. In this regime, PD-PPO's prediction-driven scheduling demonstrates measurable advantages over static baselines: it achieves [X]% lower event-conditional FW-MAE by activating the Parsivel² full mode during predicted blowing snow events while maintaining the FC4 as a continuous low-power sentinel."

这个叙事有三个优点：物理依据明确（真实硬件规格）、约束激活有理论支撑（Fernandez-Bes et al., 2025）、动态调度的价值有具体的机制解释（预测驱动的稀缺资源分配）。

---

## 九、执行优先级

**第一步（无需重训，1–2小时）**：运行 oracle 诊断，确认在 B=0.90/1.00/1.15 下 `laser_disdrometer` 的事件期 oracle lift > 0.005。若不满足，需要先修正合成数据生成器的观测噪声模型。

**第二步（需重训，8–12小时）**：在 `physical_event_value_v1` 配置下运行单 seed pilot（seed=41），验证：(a) PD-PPO 的 event-conditional laser activation ratio > 3:1；(b) laser event lift > 0；(c) warmup abort rate < round_robin。

**第三步（需重训，40–60小时）**：若 pilot 成功，扩展到 n=10 seeds，更新论文主结果表格。

**不推荐**：在没有完成第一步诊断的情况下直接运行全量实验。oracle lift 诊断是整个方案的前提条件，成本极低但信息价值极高。
