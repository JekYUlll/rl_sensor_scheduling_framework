# Phase 1 执行结果评价

## 总体判断

**执行质量：良好，但存在若干需要在投稿前处理的遗留问题。**

Phase 1 的三项实验（A1/H1/G1）在技术层面均已完成，结果内部一致，与 Phase 0 的 V2 基线衔接合理。论文集成工作也已推进。但以下几个问题需要在提交前明确处理，否则会在审稿阶段引发可预见的质疑。

---

## A1 完整组件消融：评价

### 正面

结果结构清晰，Full PD-PPO 在所有 8 个配置中最优，符合预期。oracle prior 和 AWBC 的贡献最大，与 A2 staged diagnostic 的方向一致，两组实验互相印证，增强了可信度。action mask 和 ActionEmbedding 的独立显著性也是新增的有价值发现。

### 需要注意的问题

**EventAwareCritic 的处理**：A1 结果显示 No EventAwareCritic 的 FW-MAE 为 0.4004，p=0.01953，Bonferroni 校正后不显著（α_adj ≈ 0.0071）。这与 A2 D2 的 −0.1% 结论方向一致，但 A1 的 delta 是 +2.38%，比 A2 的 −0.1% 大得多。这个差异需要在正文中解释，否则审稿人会追问两组实验为何结论不同。

合理解释是：A2 是 staged diagnostic（在 D1 基础上逐步添加），A1 是 remove-one-component（从 Full PD-PPO 移除），两种消融路径不等价，特别是当组件之间存在交互效应时。这个解释需要明确写入论文，不能让读者自己猜。

**"No AWBC/prior"的 delta 为 +13.97%**：这个数字远大于单独移除任一组件的效果，说明 AWBC 和 oracle prior 之间存在正向交互。这是一个值得在正文中点出的发现，而不仅仅是表格里的一行数字。

**n=10 vs n=5 的混用**：A1 用 n=10，A2 用 n=5。如果两个表格都出现在论文中，需要在表格标题或正文中明确说明，避免读者误以为两组实验的统计功效相同。

---

## H1 超参数敏感性：评价

### 正面

所有 9 个格点（含默认）都在 +5% 范围内，这是一个干净的结果，支持"局部鲁棒"的声称。heatmap 图的加入也是好的，视觉上直观。

### 需要注意的问题

**措辞边界**：(0.10, 0.5) 格点的 delta 为 +4.70%，非常接近 +5% 上限。在正文中应该用"all configurations remain within 5% of the default"而不是"robust"或"insensitive"。"robust"这个词在审稿人眼中意味着更强的保证，而 4.70% 接近上限的格点会被用来反驳这个声称。

**n=5 的统计功效**：H1 非默认格点只有 n=5 seeds。对于 FW-MAE 标准差约 0.035–0.047 的情况，n=5 的置信区间相当宽。论文中不应该对 H1 做显著性检验，只应该报告描述性统计（mean ± std）并声称"all within 5%"。如果当前论文中有对 H1 格点做 Wilcoxon 检验的内容，应该删除。

**默认格点的 mean 值**：H1 报告默认格点 (0.10, 1.0) 的 mean 为 0.3901，而 S1 主结果中 PD-PPO B=1.70 的 mean 为 0.3911。差异为 0.001，在 n=5 vs n=10 的范围内可以接受，但需要在论文中说明 H1 默认格点复用 S1 的 n=10 结果，而不是重新跑了 5 seeds。如果实际上是重新跑了 5 seeds，那么 0.3901 和 0.3911 的差异需要解释。

---

## G1 V3.1 生成器验证：评价

### 正面

所有 8 项验收指标均通过，且多数有较大余量（ACF 偏差 0.048 vs 上限 0.05，PSD log-MSE 0.042 vs 上限 0.10）。semi-Markov + CRED hysteresis + minimum duration/gap 的设计修正了 V2 的两个核心缺陷，这是实质性的技术进步。

### 需要注意的问题

**G1 通过 ≠ V3.1 可以替换 V2 作为主结果**：这一点在报告中已经正确指出，但需要确认论文正文中没有任何地方暗示 V3.1 生成器的验证结果支持了 V2 实验的结论。G1 是生成器的统计验证，与 PD-PPO 在 V3.1 上的性能完全无关。

**ACF 偏差 0.048 非常接近上限 0.05**：这个余量只有 4%。如果审稿人要求重新验证或使用不同的随机种子，结果可能会超过上限。建议在论文中报告具体数值（0.048），而不是只说"passed"，让读者自己判断余量是否充足。

**G1-V2 的 P(ef>0.75) = 0.065**：这个值刚好超过 0.05 的下限，余量同样很小。如果这个指标是用单次生成的样本估计的，置信区间可能很宽。建议报告估计的置信区间，或者说明是基于多少个独立窗口计算的。

**V2 与 V3.1 的对比**：报告中提到 V3.1 解决了 V2 的两个缺陷（event-heavy 窗口不可达、风速 ACF 被事件覆写），但没有给出 V2 在这些指标上的对应数值。如果论文中要声称 V3.1 优于 V2，需要提供 V2 的对比数据，否则这个声称是无法验证的。

---

## 论文集成：需要立即检查的问题

### A1 与 A2 的共存

论文中现在同时有 A2 staged diagnostic 表（D1–D4，n=5）和 A1 full ablation 表（8 variants，n=10）。这两个表格的共存需要在正文中有清晰的叙述逻辑：A2 是早期的 staged diagnostic，用于理解组件添加顺序的效果；A1 是后续的完整 remove-one-component 消融，用于量化每个组件的独立贡献。两者结论互补而非矛盾。如果正文中没有这个解释，审稿人会问为什么要做两种消融。

### EventAwareCritic 的措辞一致性

A2 D2 的结论是"contributes negligibly in isolation (−0.1%)"，A1 的结论是"not significant after Bonferroni correction (+2.38%)"。这两个表述需要在正文中统一，不能在不同地方给出看似矛盾的描述。推荐的统一表述：

> EventAwareCritic does not reach statistical significance in either the staged diagnostic (A2: −0.1% in isolation) or the full ablation (A1: +2.38% when removed, p=0.020, Bonferroni-corrected α=0.0071). This is consistent with the V2 environment's limited event-heavy episodes; the component is retained for architectural completeness and is expected to contribute more substantially in V3.1 environments with higher event fractions.

### H1 heatmap 的图例

需要确认 heatmap 的颜色轴方向是否直观：较低的 FW-MAE 应该对应较深/较冷的颜色（因为 FW-MAE 越低越好）。如果颜色轴方向相反，读者会误读。

### G1 表格的位置

G1 验证表格放在 §4（仿真环境）是合理的，但如果表格较大，可以考虑移入附录，在正文中只保留关键指标的文字描述和通过/未通过的结论。Cold Regions S&T 的篇幅限制需要考虑。

---

## 关于 S2 的决策

报告中的判断是正确的：**不能直接把 V2 策略拿到 V3.1 上评估后称为 V3 主结果**。这是一个严格的方法论边界，必须坚守。

当前的投稿策略建议维持不变：以 V2 为主结果投稿，G1 验证作为生成器改进的证据，S2 明确标注为 future work。这个策略在 Cold Regions S&T 的审稿语境下是合理的，审稿人通常接受"仿真环境有已知局限性，已识别并计划改进"的表述，前提是局限性被诚实地写出来而不是被掩盖。

---

## 优先级排序的建议

在提交前，按以下顺序处理剩余问题：

第一优先级（影响论文可信度，必须修正）：在正文中明确解释 A1 与 A2 消融路径的差异，以及 EventAwareCritic 在两组实验中结论看似不同的原因。

第二优先级（影响审稿人印象，强烈建议修正）：将 H1 的措辞从"robust"改为"all configurations within 5% of default"；在 G1 表格中报告具体数值而非只报告"passed"；确认 H1 默认格点的数据来源（复用 S1 还是重新运行）。

第三优先级（影响论文完整性，建议处理）：在 §4 中补充 V2 在 G1 指标上的对应数值，以支持"V3.1 改进了 V2"的声称；确认 A1/H1/G1 表格在编译后的 PDF 中没有浮动位置异常或横向溢出。

第四优先级（投稿后处理）：S2 全量重跑；T1 跨站点泛化。
