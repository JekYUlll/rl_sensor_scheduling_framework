# 修改意见（第二轮）：更广泛的审稿

本轮审稿基于新提交版本（1782153518023-79b89ef7_main.pdf）进行全面评估。与上一轮相比，新版本已完成若干重要改进：增加了符号表（Table 1）、超参数表（Table 7）、逐事件类型误差分解（Table 10）、完整的固定掩码候选集透明化（Appendix B，Table B.13），并将可行动作掩码、可观测性审计、比较参考等信息系统化为独立表格。这些改进显著提升了论文的可重复性和透明度。

以下意见在延续上轮未解决问题的基础上，进一步覆盖新版本中出现的新问题，按模块分节陈述。

---

## 一、实验设计与统计严谨性

### 1.1 消融实验仍仅使用 6 粒种子（高优先级）

**现状**：Table 11（机制检验）仍使用 6 粒种子（pilot），而主实验使用 24 粒种子。这一不对称性在新版本中依然存在，且是最可能引发审稿人质疑的单一问题。6 粒种子的消融结论（例如"No event-context support 导致 5/6 通过"）在统计上极不稳定：一粒种子的差异即可改变结论方向。

**建议**：将消融实验扩展至 24 粒种子，并在 Table 11 中同时报告连续指标（macro margin 均值、中位数及 95% bootstrap 置信区间），而非仅报告通过计数。若计算资源确实有限，至少应在正文中明确标注"6-seed pilot"并量化其统计局限性（例如，报告在 6 粒种子下检测到 0.01 macro margin 差异所需的最小效应量）。

### 1.2 事件混合鲁棒性检验仍仅覆盖一种变体（中优先级）

**现状**：Table 12 报告了一种"flux 事件更频繁（0.35/0.30/0.35）"的变体，6 粒种子。这是对主基准（0.40/0.20/0.40）的温和扰动，两者差异较小，鲁棒性结论的说服力有限。

**建议**：增加至少一种更极端的变体，例如：
- **稀有事件场景**：将某一事件类型（如 flux）的频率降低至 0.10，测试 PD-PPO 在低频事件下是否仍能学到有效的专家选择；
- **均匀混合场景**：三种事件类型频率相等（0.33/0.33/0.33），测试在无主导事件时的表现。

每种变体报告与 Table 12 相同的指标，并在正文中讨论性能变化的方向和幅度。

### 1.3 缺少训练曲线与收敛分析（中优先级）

**现状**：新版本仍未展示 PD-PPO 的训练过程。Table 7 报告了"200,000 policy timesteps"和"10,000 steps behaviour-cloning pretraining"，但读者无法判断策略是否稳定收敛，AWBC 辅助项是否加速了收敛，以及不同变体的收敛速度是否有差异。

**建议**：增加一张训练曲线图（可作为附录图），展示：
- 横轴：训练步数（或 PPO 更新轮次）；
- 纵轴：验证集上的 macro score（或 step loss）；
- 曲线：Full PD-PPO 与 No imitation guide 变体的对比，以多粒种子的均值 ± 标准差呈现。

这一图表直接支持"AWBC 提升样本效率"的 claim，目前该 claim 在正文中有所暗示但缺乏实证支撑。

### 1.4 缺少与更强基线的比较（低优先级，但建议在 Discussion 中讨论）

**现状**：当前基线包括验证选择的固定掩码、规则驱动动态策略（循环、随机、陈旧优先）和事件感知回放（特权信息上界）。缺少以下两类基线：

**（a）启发式自适应基线**：基于简单阈值规则的自适应策略（例如"当风速超过阈值时切换到 flux 专家"）。这类基线不需要学习，但利用了领域知识，是实际部署中的常见竞争对手。

**（b）上下文赌博机（contextual bandit）基线**：将每个调度步骤视为独立的上下文赌博机问题，忽略跨步骤的时序依赖。与 PD-PPO 的对比可以说明时序建模相对于无记忆策略的增益。

若计算资源有限，至少应在 Discussion 中讨论为何未包含这些基线，以及预期的性能差距。

### 1.5 固定预测器性能基准仍缺失（中优先级）

**现状**：Table 7 报告了 TCN 预测器的架构（residual TCN, 3 levels, 64 channels, kernel size 3, dropout 0.05）和训练设置（18 epochs, batch 512, 53,274 steps），但没有报告其预测性能（如 MAE 或归一化 MAE）。读者无法判断预测器本身的质量是否足以作为可靠的奖励信号。

**建议**：在 Section 5.3 中增加一个小表，报告 TCN 预测器在验证分区上对各目标变量（wind、temperature、snow flux 等）的预测误差，并与简单基线（如持久性预测）对比。这一表格不需要很大，但对于建立读者对奖励信号可靠性的信心至关重要。

---

## 二、图表形式的拓展与完善

### 2.1 Figure 6（行为诊断）缺少最直观的可视化（高优先级）

**现状**：Figure 6 Panel A 展示了种子级掩码熵、事件-掩码互信息和事件条件传感器使用分离度，Panel B 展示了种子级成对边际分布。这些指标是间接的行为证据。

**建议**：增加一个**专家选择频率热图**作为新面板，横轴为事件类型（particle、flux、thermal、non-event），纵轴为专家通道（radiometer、thermo-hygro、surface IR、laser disdrometer、FC4 flux），颜色表示在该事件类型下选择该专家的频率（跨 24 粒种子的均值）。这一热图直接可视化"策略是否学会了将专家与事件类型匹配"，是行为诊断最直观的证据，也是审稿人最容易理解的图表。

同时，建议将 Panel A 的三个诊断指标改为**箱线图**，展示 24 粒种子的分布（中位数、四分位距、异常值），并叠加固定掩码基线（熵为 0）和简单循环基线（MI 为 0）的参考线。

### 2.2 Figure 7（机制检验）缺少连续指标的可视化（中优先级）

**现状**：Figure 8（新版本编号）Panel A 比较四种变体在 6 粒种子上的表现，以通过计数为主。图表缺少连续指标的对比，无法直观比较各变体的性能量级。

**建议**：将通过计数改为**分组条形图**，每组对应一种变体，条形高度为 macro margin 均值，误差棒为标准差（6 粒种子）。若消融实验扩展至 24 粒种子（见 §1.1），则改为箱线图，展示各变体的 macro margin 分布。

### 2.3 Figure 1（框架图）仍未区分三个阶段（中优先级）

**现状**：Figure 1 描述为"预测驱动调度框架，运行时控制器将估计器摘要映射到可行通道子集；训练时，固定评估预测器对部分观测轨迹评分，PPO 更新调度策略"。该图仅展示了数据流，未明确区分预测器拟合、策略训练和最终评估三个阶段。

**建议**：重新设计 Figure 1，明确区分三个阶段并在图中标注：
- **阶段 1（预测器拟合）**：输入原始时序数据（前 35% 分区），输出固定 TCN 预测器；
- **阶段 2（策略训练）**：输入固定预测器 + 模拟器（35%–85% 分区），输出 PD-PPO 策略；包含 AWBC 标签生成的子流程；
- **阶段 3（最终评估）**：输入 PD-PPO 策略 + 固定预测器 + 保留测试窗口（最后 7.5% 分区），输出 macro score 和 step loss。

三个阶段用不同颜色的边框区分，数据流用箭头连接，并标注哪些数据分区用于哪个阶段（对应 Figure 2 的时序分割）。

### 2.4 Figure 2（时序分割）仍缺少实际步数标注（低优先级）

**现状**：Figure 2 展示了时序分割，但仅标注了百分比，没有标注实际步数或时间长度。Table 7 中已有分区信息（[0,24500)、[24500,59500)、[59500,64750)、[64750,70000)），但图中未体现。

**建议**：在图中增加每个分区的实际步数（如"24,500 steps"）以及对应的实际时间跨度（如"约 1,021 天"，按每步 1 小时计算）。这对于读者评估训练数据量是否充足至关重要。

### 2.5 Figure 5（种子级证据）信息密度偏低（低优先级）

**现状**：Figure 5 Panel A 是按固定掩码 step margin 排序的条形图，Panel B 是成对边际分布的汇总。Panel A 仅展示了 step margin，未同时展示 macro margin。

**建议**：将 Panel A 改为**散点图 + 误差棒**，横轴为种子编号（按 step margin 排序），纵轴同时展示 step margin（主轴）和 macro margin（副轴或不同颜色的点），并叠加预设优越性边界线（$\epsilon_s$）。可选择增加一个 CDF 面板，展示 24 粒种子的 macro margin 分布，并与事件感知诊断参考的 macro margin 分布并排，直观显示 PD-PPO 与特权信息上界之间的差距。

---

## 三、写作与表述的完善

### 3.1 Eq. (11) 中 $\mathcal{L}_\text{event}$ 仍无显式公式（高优先级）

**现状**：Eq. (11) 列出了 PD-PPO 的完整损失函数，包含六个项，其中 $\mathcal{L}_\text{event}$（事件类型辅助项）在正文中仅描述为"encourages the policy representation to distinguish particle, flux, and thermal event regimes"，没有给出显式公式。$\mathcal{L}_\text{guide}$（引导先验项）在正文中注明"not part of the reported 24-seed configuration"，但仍出现在主方程中。

**建议**：
- 为 $\mathcal{L}_\text{event}$ 增加一个方程，明确其形式（例如，是交叉熵损失还是 KL 散度？事件类型标签如何获得？标签是否来自 Table 3 中的"Event-type labels"？）。
- 将 $\mathcal{L}_\text{guide}$ 移至附录，并在正文中注明"available in the codebase but not used in the reported configuration"，避免在主方程中引入未使用的项造成混淆。

### 3.2 Proposition 1 和 Proposition 2 缺少 Remark 段落（中优先级）

**现状**：两个命题的证明草图在正文中给出，完整证明在附录中。但命题的条件陈述较为抽象，读者难以快速判断这些条件在实验设置中是否满足。

**建议**：在每个命题之后增加一段"Remark"，明确说明：（a）实验基准满足该命题条件的具体方式（例如，三种事件类型对应三个不兼容的最优专家子集，满足 Proposition 1 的条件）；（b）该命题对实验结果的解释意义（即理论结果如何支持实验观察）。

### 3.3 Section 6 缺少与 Section 4.5 评估框架的对应说明（中优先级）

**现状**：Section 4.5 按顺序描述了评估框架的四个部分（固定掩码基线、动态参考、固定掩码回放检验、行为诊断），但 Section 6 的小节顺序（6.1 主要结果 → 6.2 固定掩码回放 → 6.3 事件感知诊断 → 6.4 行为诊断 → 6.5 机制检验）与之不完全对应，且没有引导性文字说明对应关系。

**建议**：在 Section 6 开头增加一段引导性文字，明确说明各小节与 Section 4.5 评估框架的对应关系，例如："Section 6.1 reports the main paired comparisons (criteria 1 and 2 in Section 4.5); Section 6.2 reports the fixed-mask replay check (criterion 3); Sections 6.3 and 6.4 report the diagnostic references and behavioural checks (criterion 4); Section 6.5 reports the pilot mechanism and robustness checks."

### 3.4 Discussion 需要更直接地回应潜在质疑（中优先级）

**现状**：Section 7 的四个小节写得较为保守，主要是对已有结果的重述。7.4 节已提到固定预测器和模拟器的局限性，但缺少对以下潜在审稿人质疑的直接回应：

**（a）模拟器的真实性**：审稿人可能质疑"基于模拟数据的结论能否推广到真实 AWS 数据"。建议在 7.4 节中增加一段，说明 Table 6 的统计检验如何保证模拟数据与真实 Antarctic AWS 数据的统计一致性，以及哪些方面仍需实地验证。

**（b）固定预测器的偏差方向**：7.4 节已提到"固定评估预测器不是端到端双层优化"，但没有讨论这一设计选择对结果的潜在影响方向（即固定预测器是否可能系统性地低估或高估某些调度策略的价值，以及这种偏差是否对 PD-PPO 有利）。建议增加一段分析。

**（c）one-specialist 约束的普遍性**：审稿人可能质疑"one-specialist 设置是否过于特殊"。建议在 7.1 节中增加一段，说明 one-specialist 约束如何代表一类更广泛的调度问题（即任何 $r < K$ 的情况），以及该设置在实际 AWS 部署中的合理性。

### 3.5 Table 10 的事件类型分解需要更清晰的解读（低优先级）

**现状**：Table 10 报告了三种事件类型（particle、flux、thermal）的逐类型损失分解，并注明"The event-type rows are diagnostic: the main claim is the held-out macro and step-loss comparison, not per-event all-seed dominance"。但 flux 类型的 Seeds 列（12/24）和 thermal 类型（16/24）远低于 macro average（24/24），这一差异在正文中没有得到充分解释。

**建议**：在 Section 6.2 中增加一段，解释为何 macro average 在 24/24 种子上均有改进，而单独的 flux 和 thermal 事件类型仅在部分种子上改进。这一解释应涉及：（a）macro score 是三种事件类型的均值，单类型的局部退化可以被其他类型的改进所补偿；（b）per-event 比较的统计功效较低，因为每粒种子中各事件类型的样本量不均等。

### 3.6 Abstract 与 Introduction 的 claim 范围需要核查（低优先级）

**现状**：Abstract 中的 claim 表述为"PD-PPO improved both ordinary step-weighted forecast loss and the static-reference-normalised event-regime macro score over validation-selected fixed-mask and rule-based dynamic references in every seed"。Introduction 的贡献列表（第 1–4 条）与 Abstract 基本一致。

**潜在问题**：Introduction 第 4 条贡献声称"fixed-mask replay and behavioural diagnostics showing, over 24 seeds, that the learned scheduler is not explained by a constant specialist choice or a simple periodic rotation"。这一表述将行为诊断作为独立贡献，但行为诊断本质上是对主要性能 claim 的支撑证据，而非独立贡献。建议重新表述，将其定位为"evaluation methodology contribution"而非独立的"result contribution"，以避免审稿人认为贡献列表存在重复计数。

---

## 四、符号与术语一致性

### 4.1 Eq. (11) 中的系数命名与 Table 7 不完全一致（低优先级）

**现状**：Eq. (11) 使用 $\beta_\text{BC}$、$\beta_\text{event}$、$\beta_\text{guide}$ 作为系数名称，而 Table 7 报告"Imitation and auxiliary weights: advantage-weighted imitation 2.5; event-type auxiliary 5.0"，未使用方程中的符号名称。

**建议**：在 Table 7 中使用与 Eq. (11) 一致的符号名称（$\beta_\text{BC} = 2.5$，$\beta_\text{event} = 5.0$），并补充 $\beta_\text{guide}$ 的取值（即使该项未使用，也应注明"not used in reported configuration"）。

### 4.2 "event-context support" 与 "event-context auxiliary signal" 的术语不一致（低优先级）

**现状**：Table 11 中的消融变体名称为"No event-context auxiliary signal"，而 Section 6.5 正文中描述为"Removing the regime-aware support"，Introduction 中描述为"event context"。这三种表述指向同一组件，但措辞不一致。

**建议**：统一使用一个术语（建议使用"event-context auxiliary signal"，与 Eq. (11) 中的 $\mathcal{L}_\text{event}$ 对应），并在首次出现时明确定义。

---

## 五、优先级总结

**高优先级（强烈建议在修订中完成）**：
- 消融实验扩展至 24 粒种子，并增加连续指标（§1.1）；
- 增加专家选择频率热图（§2.1）；
- 为 $\mathcal{L}_\text{event}$ 增加显式公式，将 $\mathcal{L}_\text{guide}$ 移至附录（§3.1）。

**中优先级（建议完成，可显著提升论文质量）**：
- 增加训练曲线图（§1.3）；
- 增加固定预测器性能基准表（§1.5）；
- 重新设计 Figure 1 为三阶段架构图（§2.3）；
- Figure 7/8 改为分组条形图（§2.2）；
- Proposition 1 和 2 增加 Remark 段落（§3.2）；
- Section 6 开头增加评估框架对应说明（§3.3）；
- Discussion 增加对潜在质疑的直接回应（§3.4）。

**低优先级（有余力时完成，可进一步完善）**：
- 事件混合鲁棒性的更极端变体（§1.2）；
- 增加启发式自适应基线或在 Discussion 中讨论（§1.4）；
- Figure 2 增加实际步数标注（§2.4）；
- Figure 5 改为散点图并增加 CDF 面板（§2.5）；
- Table 10 事件类型分解的解读补充（§3.5）；
- Abstract/Introduction 贡献表述核查（§3.6）；
- Eq. (11) 系数与 Table 7 符号对齐（§4.1）；
- 术语统一（§4.2）。

---

## 附：已解决的上轮问题

以下上轮建议在新版本中已得到有效解决，无需再次修改：
- 符号表（Table 1）——已添加；
- 超参数表（Table 7）——已添加，覆盖 PPO 和 TCN 参数；
- 逐事件类型误差分解（Table 10）——已添加；
- 固定掩码候选集透明化（Appendix B，Table B.13）——已完整报告 24 粒种子的逐粒选择结果；
- 可行动作掩码表（Table 2）——已添加，包含 6 个候选掩码及其功率和解释；
- 可观测性审计表（Table 3）——已添加，明确区分在线/离线信息；
- 比较参考角色说明（Table 4）——已添加，清晰区分各基线的角色。
