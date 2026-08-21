# V3.1 S2 调度行为诊断备忘录

Last updated: 2026-05-22

本备忘录对应根目录 `task_plan.md` 的 Phase 2，用已有 V3.1 S2 结果检验
`docs/05-22-plan-1.md` 中关于“静态陷阱”的因果判断。诊断脚本为：

- `scripts/47_v31_behavior_diagnostics.py`

主要输出目录：

- `reports/v31_s2_main/behavior_diagnostics/`
- 摘要文件：`reports/v31_s2_main/behavior_diagnostics/diagnostic_summary.md`

## 1. 配置事实

V3.1 S2 使用的传感器配置是：

- `configs/sensors/windblown_sensors_balanced.yaml`

该配置明确是 **weakly heterogeneous / balanced** 成本设定，而不是物理真实功耗比例。
传感器功耗和暖机如下：

| sensor | power | peak | warmup |
|---|---:|---:|---:|
| met_station_core | 0.42 | 0.55 | 0 |
| radiometer_basic | 0.36 | 0.48 | 0 |
| surface_temp_ir | 0.38 | 0.50 | 1 |
| ultrasonic_anemometer_hd | 0.58 | 0.78 | 1 |
| shielded_thermo_hygro | 0.52 | 0.70 | 1 |
| snow_particle_counter | 0.68 | 0.92 | 2 |
| laser_disdrometer | 0.82 | 1.15 | 4 |
| fc4_flux | 0.86 | 1.20 | 5 |

V3.1 S2 默认预算：

- `B in {1.65, 1.70, 1.75}`
- `startup_peak_budget = 3.2`
- `max_active = 4`

目标权重不是 snow-light，而是 snow-heavy-ish：

- `snow_mass_flux_kg_m2_s = 3.0`
- `snow_particle_mean_diameter_mm = 2.0`
- `snow_particle_mean_velocity_ms = 2.0`

因此，`05-22-plan-1.md` 中“β 权重导致吹雪传感器贡献被稀释”不是当前最强解释。
当前更强解释是：coverage groups + balanced costs + oracle/candidate prior 共同形成了强固定核心子集。

## 2. Static projection 的真实机制

`src/v2/policies.py` 中：

```python
class FeasibleStaticPriorityPolicy(V2Policy):
    def act_scores(self, env):
        return np.linspace(1.0, 0.0, n_sensors)
```

所以 `feasible_static_projected` 是真正的固定优先级策略，不使用事件状态，也不会随时间主动切换。

`PowerProjector` 会先满足 coverage groups：

- weather: `met_station_core`, `ultrasonic_anemometer_hd`, `shielded_thermo_hygro`
- surface_forcing: `radiometer_basic`, `surface_temp_ir`
- snow_transport: `snow_particle_counter`, `laser_disdrometer`, `fc4_flux`

在固定线性分数下，各组被选中的最高分传感器分别是：

- `met_station_core`
- `radiometer_basic`
- `snow_particle_counter`

三者总 steady power：

- `0.42 + 0.36 + 0.68 = 1.46`

这解释了 Figure 8 中 static projection 的行为：三个传感器常开，五个传感器常关。

## 3. 对“收紧预算 A”的重要修正

`05-22-plan-1.md` 建议新增 `B in {1.50, 1.55, 1.60}`。现有诊断显示：

- 当前 static 核心子集功耗为 `1.46`
- 因此 `B=1.50/1.55/1.60` 仍然允许同一个 static 子集
- 单纯把预算收紧到 `1.50` 不会打破 static projection 的固定三传感器解

如果只采用当前 balanced costs，预算必须低于约 `1.46` 才会迫使 static reference 放弃该核心子集。
但在当前 coverage groups 下，`1.46` 同时也是可行调度的最低 steady-power 门槛：

- cheapest weather = `met_station_core = 0.42`
- cheapest surface_forcing = `radiometer_basic = 0.36`
- cheapest snow_transport = `snow_particle_counter = 0.68`
- total = `1.46`

因此，若保留 coverage groups，`B < 1.46` 会使每步覆盖约束不可满足；若移除 coverage groups，
则问题定义又发生变化，不能直接作为 V3.1 主结果的简单预算延伸。

结论：**策略 A 单独执行不足。** 若要让预算变化产生机制差异，应至少同时考虑：

1. 重新定义物理成本向量，使 `met_station_core + radiometer_basic + snow_particle_counter`
   不再稳定地低于紧预算；或
2. 设计 `B < 1.46` 的诊断性 pilot，并先验证 coverage feasibility；或
3. 保持预算不变，但改变 scenario 到更物理真实的 cost-value tradeoff。

## 4. V3.1 事件条件是否不足？

V3.1 truth 事件并不稀少：

- truth event rate mean: `0.289`
- 512-step max event fraction mean: `0.795`
- `P(event_fraction_512 > 0.75)`: `0.091`
- event duration median: `18` steps

因此，V3.1 已经有 event-heavy windows。当前 PD-PPO 没有表现出强事件条件激活，不能简单归因于
“事件太少”。更可能的原因是：当前 reward/oracle/candidate prior 下，事件条件切换 high-latency sensors
的边际收益不足以超过固定核心观测与暖机持有成本。

## 5. 行为统计

V3.1 S2 全 30 runs 的行为摘要：

| budget | policy | near-constant sensors | switches/step | warmup abort rate | mean power |
|---:|---|---:|---:|---:|---:|
| 1.65 | static | 8.0 | 0.0005 | 0.000 | 1.460 |
| 1.65 | PD-PPO | 5.1 | 0.305 | 0.020 | 1.561 |
| 1.65 | round-robin | 0.0 | 2.000 | 0.249 | 1.555 |
| 1.65 | AoI | 1.0 | 2.998 | 0.166 | 1.600 |
| 1.70 | static | 8.0 | 0.0005 | 0.000 | 1.460 |
| 1.70 | PD-PPO | 5.4 | 0.337 | 0.017 | 1.559 |
| 1.70 | round-robin | 0.0 | 2.000 | 0.249 | 1.555 |
| 1.70 | AoI | 1.0 | 3.647 | 0.419 | 1.623 |
| 1.75 | static | 8.0 | 0.0005 | 0.000 | 1.460 |
| 1.75 | PD-PPO | 5.4 | 0.341 | 0.022 | 1.546 |
| 1.75 | round-robin | 0.0 | 2.000 | 0.249 | 1.555 |
| 1.75 | AoI | 1.0 | 3.000 | 0.010 | 1.668 |

解读：

- static projection 是完全静态参考。
- PD-PPO 不是完全静态，但仍有约 5.1--5.4 个近常量传感器。
- round-robin/AoI 的高动态性主要表现为高切换和高暖机浪费，不等于更好的 adaptive scheduling。
- PD-PPO 的低 warmup abort rate 是正面结果，但它来自保守持有/少切换，而不是明显事件触发。

## 6. 高延迟传感器的事件条件使用

PD-PPO 对 high-latency sensors 的选择没有表现出强正向事件条件性：

| budget | sensor | selected event | selected non-event | event lift |
|---:|---|---:|---:|---:|
| 1.65 | laser_disdrometer | 0.653 | 0.743 | -0.090 |
| 1.65 | snow_particle_counter | 0.342 | 0.255 | +0.087 |
| 1.70 | laser_disdrometer | 0.683 | 0.694 | -0.011 |
| 1.70 | snow_particle_counter | 0.315 | 0.305 | +0.010 |
| 1.75 | laser_disdrometer | 0.564 | 0.615 | -0.051 |
| 1.75 | snow_particle_counter | 0.435 | 0.384 | +0.051 |
| all | fc4_flux | approximately 0 | approximately 0 | approximately 0 |

这说明当前 PD-PPO 的动态部分更多是在 `laser_disdrometer` 与 `snow_particle_counter` 之间做有限替换，
但不是清晰的“事件来临 -> 激活高价值吹雪传感器”。`fc4_flux` 基本未被使用。

## 7. 对下一步实验的决策建议

不建议立刻按 `05-22-plan-1.md` 的“策略 A + B”完整重训，因为 A 中的 `B=1.50--1.60`
不会打破当前 static 核心子集。

推荐下一步做两个轻量 pilot，而不是 full grid：

1. **Physical-cost pilot**
   - 新建一个物理成本配置，不覆盖 `windblown_sensors_balanced.yaml`。
   - 让高延迟/高价值传感器形成真实 opportunity cost。
   - 在 `B in {1.50, 1.60, 1.70}` 或按新成本归一化后的等价预算上跑少量 seeds。
   - 仍然重新优化 static projection，不把 static 当弱 baseline。

2. **Optional coverage-ablation diagnostic**
   - 仅作为机制诊断，可试 `--disable-coverage-groups` + tight budgets。
   - 不建议直接写入主结果，因为它改变了问题定义。
   - 用于回答：coverage groups 是否是 static core 的主要结构来源。

事件频率提升可以作为第二阶段变量，而不是第一阶段就叠加。V3.1 当前事件率已经不低；
如果先叠加高事件频率，会混淆“成本/预算”与“事件分布”的因果。

奖励改动（warmup reward）应暂缓。先用配置级诊断确认动态价值是否能在物理合理约束下自然出现。

## 8. Phase 3 场景闸门

基于本诊断，下一轮不应直接运行 `05-22-plan-1.md` 的 full grid。建议的决策闸门如下。

### 8.1 当前不通过的方案

**Balanced-cost tight-budget full sweep (`B=1.50--1.60`)：不通过。**

原因：

- 当前 static core `met_station_core + radiometer_basic + snow_particle_counter` 的功耗是 `1.46`。
- `B=1.50--1.60` 不会改变该 static core。
- 若降到 `B<1.46`，当前 coverage groups 不可满足。
- 因此该 sweep 很可能消耗计算但无法回答 adaptive value 问题。

**直接叠加 high-event-frequency：暂缓。**

原因：

- V3.1 当前 event rate 已约 `0.289`，且存在 event-heavy windows。
- 当前问题不是没有事件，而是事件条件激活没有被策略充分利用。
- 先叠加事件频率会混淆成本/预算机制与事件机制。

**直接修改 reward 加 warmup bonus：暂缓。**

原因：

- 这是最容易被审稿人理解为 post hoc tuning 的改动。
- 应先验证配置级物理约束是否足以暴露动态价值。

### 8.2 推荐下一步

**下一步应是 physical-cost pilot + static-reference clarification。**

具体要求：

1. 新建独立 sensor config，例如 `windblown_sensors_physical_pilot.yaml`，不要覆盖 V3.1 主配置。
2. 在训练前先 dry-run/枚举该配置下的可行子集、coverage feasibility、fixed-priority static subset。
3. 明确 static baseline 的定义：
   - 若继续使用当前 `FeasibleStaticPriorityPolicy`，应称为 fixed-priority projected baseline，而非 oracle-optimized static projection。
   - 若论文需要 strong static reference，应实现/使用真正按 oracle/validation loss 选择的 static subset，并对每个 scenario 重新优化。
4. 只跑 pilot seeds，不启动 full grid。
5. 先看行为图和行为统计，再看显著性。

### 8.3 Pilot 接受标准

pilot 不是为了保证 PD-PPO 击败 static reference。通过标准应是：

- **可行性**：coverage groups 可满足，steady/peak violation 率为 0。
- **物理解释性**：成本向量能对应传感器硬件等级，而不是任意调参。
- **行为差异**：PD-PPO 不能只是更频繁切换；应显示更低 warmup abort rate 或更高 ready/active yield。
- **事件条件性**：至少一个高价值 snow-transport sensor 的 event-selection lift 为正，或 event-heavy windows 中 active rate 有清晰提升。
- **预测质量**：PD-PPO 至少保持优于 random/AoI，并报告相对 round-robin 的均值差异；不要求显著优于 static reference。
- **叙事安全**：若 static reference 仍强，论文应写成 regime-dependent 或 static-boundary diagnosis，而不是“RL 显著优于所有 baseline”。
