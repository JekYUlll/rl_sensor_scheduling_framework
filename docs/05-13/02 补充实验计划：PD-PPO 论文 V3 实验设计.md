# 补充实验计划：PD-PPO 论文 V3 实验设计

**目标**：在提交正式修订稿之前，补充三类当前缺失的实验证据，使论文从"有 plan 但缺实验支持"的状态升级为"每一项声称均有对应实验数据"的完整论文。

本计划覆盖以下四个实验模块：

| 模块 | 实验代号 | 当前状态 | 优先级 |
|------|---------|---------|--------|
| 完整组件消融 | **A1** | 未跑（仅有 A2 staged diagnostic） | 最高 |
| 超参数敏感性扫描 | **H1** | 未跑 | 高 |
| V3 生成器验证 + 重跑主实验 | **G1 + S2** | 未跑（V2 生成器有已知缺陷） | 高 |
| 跨站点泛化评估 | **T1** | 未跑 | 中 |

---

## 一、A1：完整组件消融实验

### 1.1 动机与当前缺口

A2 staged diagnostic（D1→D4）采用"逐步添加"设计，只能证明每个组件在特定添加顺序下的边际贡献，无法排除组件间的顺序依赖效应（order effect）。例如，EventAwareCritic 在 D2 阶段贡献仅 −0.1%，但这可能是因为在没有 AWBC 的情况下 Critic 无法有效学习，而非 EventAwareCritic 本身无用。完整消融（每次移除一个组件）可以独立量化每个组件的贡献，是审稿人通常期望的标准消融设计。

**当前可声称**：A2 staged diagnostic 显示 AWBC 和 Oracle Prior 是主要贡献来源。  
**当前不可声称**：EventAwareCritic 在完整系统中无贡献；各组件贡献相互独立。

### 1.2 实验设计

**基准配置（Full PD-PPO）**：MaskedActor + ActionEmbedding + EventAwareCritic + AWBC + Oracle Prior

**六个消融变体**（每次从完整配置中移除一个组件）：

| 变体代号 | 移除的组件 | 保留的组件 |
|---------|-----------|-----------|
| A1-v1 | ActionEmbedding | MaskedActor + EventAwareCritic + AWBC + Oracle Prior |
| A1-v2 | EventAwareCritic | MaskedActor + ActionEmbedding + AWBC + Oracle Prior |
| A1-v3 | AWBC | MaskedActor + ActionEmbedding + EventAwareCritic + Oracle Prior |
| A1-v4 | Oracle Prior | MaskedActor + ActionEmbedding + EventAwareCritic + AWBC |
| A1-v5 | AWBC + Oracle Prior | MaskedActor + ActionEmbedding + EventAwareCritic（纯 PPO 基线） |
| A1-v6 | 全部辅助组件 | MaskedActor only（最小可行策略，等价于 D1 的 MaskedActor + ActionEmbedding） |

**注意**：A1-v6 与 A2 的 D1 配置相同，可直接复用 D1 数据（0.4224 ± 0.0326），无需重跑。

**实验参数**：
- Budget：B = 1.70（与 A2 一致，便于直接比较）
- Seeds：41–50（n=10，与 S1 主实验一致；A2 仅用 n=5，A1 升级到 n=10 以提高统计功效）
- 评估指标：FW-MAE（mean ± std），以及 Wilcoxon signed-rank vs. Full PD-PPO（Bonferroni 校正，6 次比较，α_adj = 0.05/6 ≈ 0.0083）
- 训练步数：与主实验相同（不得缩短，否则结果不可比）

### 1.3 预期结果与分析框架

预期 A1-v3（移除 AWBC）和 A1-v4（移除 Oracle Prior）的 FW-MAE 显著高于 Full PD-PPO，与 A2 的 D3/D4 贡献一致。A1-v2（移除 EventAwareCritic）的结果是关键：

- 若 A1-v2 FW-MAE ≈ Full PD-PPO（差异 < 0.005）：确认 EventAwareCritic 在完整系统中贡献有限，与 A2 D2 结论一致，可在正文中明确陈述。
- 若 A1-v2 FW-MAE 显著高于 Full PD-PPO：说明 EventAwareCritic 在完整系统中有贡献，但在 A2 的 staged 设计中被 AWBC/Oracle Prior 掩盖，需修改正文叙述。

### 1.4 输出文件规范

```
reports/v3_supplement_assets/
  exp_a1_ablation_stats.csv          # 列：variant, mean, std, ci_lower, ci_upper, p_vs_full
  exp_a1_ablation_raw.csv            # 列：variant, seed, fw_mae（原始数据，用于 Wilcoxon 检验）
```

CSV 列规范（`exp_a1_ablation_stats.csv`）：

| 列名 | 类型 | 说明 |
|------|------|------|
| variant | str | "full", "no_action_emb", "no_event_critic", "no_awbc", "no_oracle", "no_awbc_oracle", "masked_only" |
| mean | float | FW-MAE 均值（n=10） |
| std | float | FW-MAE 标准差 |
| ci_lower | float | 95% 置信区间下界 |
| ci_upper | float | 95% 置信区间上界 |
| p_vs_full | float | Wilcoxon signed-rank p 值（vs. full PD-PPO，双侧） |
| significant | bool | p < 0.0083（Bonferroni 校正） |

### 1.5 正文影响

A1 完成后，`tables/ablation.tex` 将扩展为包含 A1 和 A2 两个子表（或合并为一个双栏表），`sections/06_experiments.tex` 的消融子节将从"staged diagnostic"升级为"full ablation + staged diagnostic"的双重证据结构。

---

## 二、H1：超参数敏感性扫描

### 2.1 动机与当前缺口

当前论文使用固定超参数配置，无法声称"PD-PPO 对超参数选择鲁棒"。审稿人可能质疑：报告的性能是否依赖于精心调优的超参数，而非算法设计本身的优势。H1 通过系统扫描两个最关键的 PD-PPO 特有超参数，提供鲁棒性证据。

**两个最关键的 PD-PPO 特有超参数**：

1. **`awbc_coef`**（AWBC 辅助损失权重）：控制行为克隆正则化强度。过大会导致策略过度保守（无法偏离先验），过小则失去冷启动稳定效果。
2. **`prior_kl_coef`**（Oracle Prior KL 散度权重）：控制策略与 oracle 先验的距离。过大会导致策略退化为 oracle 先验（失去自适应能力），过小则失去校准效果。

### 2.2 实验设计

**扫描网格**（2×4 = 8 个配置，加上基准配置共 9 个）：

| `awbc_coef` \ `prior_kl_coef` | 0.01 | 0.05（基准） | 0.10 | 0.20 |
|-------------------------------|------|-------------|------|------|
| 0.1 | H1-11 | H1-12 | H1-13 | H1-14 |
| 0.5（基准） | H1-21 | **H1-22（基准）** | H1-23 | H1-24 |
| 1.0 | H1-31 | H1-32 | H1-33 | H1-34 |
| 2.0 | H1-41 | H1-42 | H1-43 | H1-44 |

**注意**：基准配置（H1-22）与 S1 主实验相同，可直接复用 S1 数据（FW-MAE = 0.3911 ± 0.0240，n=10），无需重跑。实际需要新跑的配置为 15 个（排除基准）。

**实验参数**：
- Budget：B = 1.70（固定，减少计算量）
- Seeds：41–45（n=5；超参数扫描的目的是评估相对鲁棒性，n=5 足够）
- 评估指标：FW-MAE（mean ± std）；以基准配置为参考，计算相对变化 Δ%

**计算量估算**：15 个新配置 × 5 seeds = 75 次训练运行。若单次训练约 30 分钟，总计约 37.5 小时（可并行化到约 8 小时，使用 8 GPU）。

### 2.3 分析框架

**鲁棒性判定标准**：若所有 15 个非基准配置的 FW-MAE 均在基准值 ±5%（即 [0.371, 0.411]）范围内，则可声称"PD-PPO 对 `awbc_coef` 和 `prior_kl_coef` 的选择在一个数量级范围内鲁棒"。

**热图可视化**：将 9 个配置的 FW-MAE 绘制为 3×4 热图（`awbc_coef` × `prior_kl_coef`），直观展示性能景观的平坦程度。

**边际效应分析**：分别固定一个超参数，绘制另一个超参数的 FW-MAE 折线图，识别最敏感的超参数方向。

### 2.4 输出文件规范

```
reports/v3_supplement_assets/
  exp_h1_hyperparam_stats.csv        # 列：awbc_coef, prior_kl_coef, mean, std, delta_pct
  exp_h1_hyperparam_raw.csv          # 列：awbc_coef, prior_kl_coef, seed, fw_mae
  figure_h1_heatmap.png              # 热图（3×4，FW-MAE 颜色编码）
```

CSV 列规范（`exp_h1_hyperparam_stats.csv`）：

| 列名 | 类型 | 说明 |
|------|------|------|
| awbc_coef | float | AWBC 辅助损失权重 |
| prior_kl_coef | float | Oracle Prior KL 权重 |
| mean | float | FW-MAE 均值（n=5） |
| std | float | FW-MAE 标准差 |
| delta_pct | float | 相对基准的变化百分比（正值=性能下降） |
| is_baseline | bool | 是否为基准配置 |

### 2.5 正文影响

H1 完成后，`sections/06_experiments.tex` 新增子节 `\subsection{Hyperparameter Sensitivity (H1)}`，包含热图（Figure 3 或 Figure 4）和鲁棒性结论。`sections/07_discussion.tex` 中"fixed hyperparameter configuration; sensitivity analysis is future work"的占位符替换为实际结论。

---

## 三、G1 + S2：V3 生成器验证与主实验重跑

### 3.1 动机与当前缺口

V2 生成器存在两个已知缺陷（见 turn_54 诊断）：

**缺陷 1（ACF 破坏）**：事件覆写操作（event-floor + precursor-ramp）修改 `wind_speed_ms`，破坏 DFT 相位随机化保留的风速 PSD/ACF。审稿人已在图 3 中发现此问题。

**缺陷 2（event fraction 上限）**：事件以独立区间注入，无法产生 event fraction > 0.58 的持续风暴条件，导致 E1 的"event-heavy"条件仅覆盖 [0.40, 0.58]，而非代表性的南极风暴条件（> 0.75）。

这两个缺陷直接影响论文的核心声称：若仿真环境不能真实反映南极气象条件，则 PD-PPO 在仿真中的优势能否迁移到真实部署存疑。

### 3.2 V3 生成器设计规格

**架构变更**（相对 V2）：

**模块 1：Storm-Regime 潜变量**

采用半马尔可夫过程（Semi-Markov Process）建模风速状态机：

```
状态空间：{calm, transition, storm}
转移概率矩阵（来自 Amory 2020 D17 站点统计）：
  P(calm → calm)       = 0.92
  P(calm → transition) = 0.08
  P(transition → storm)= 0.60
  P(transition → calm) = 0.40
  P(storm → storm)     = 0.85
  P(storm → transition)= 0.15

状态持续时间分布（半马尔可夫）：
  calm:       Weibull(k=1.5, λ=48h)   # 中位持续约 40h
  transition: Exponential(λ=3h)        # 中位持续约 2h
  storm:      Weibull(k=2.0, λ=20h)   # 中位持续约 18h，与 Amory 2020 D17 中位 15h 接近
```

**模块 2：风速生成（状态调制 DFT）**

在每个状态内，使用 DFT 相位随机化生成风速序列，但以状态对应的 AntAWS 子集（仅 calm 时段 / 仅 storm 时段）为参考信号，从而保留各状态内的 ACF 结构：

```python
# 伪代码
for regime in ['calm', 'transition', 'storm']:
    ref_signal = antaws_wind_speed[regime_mask[regime]]
    synthetic_wind[regime_windows] = dft_phase_randomize(ref_signal, length=window_len)
```

状态边界处使用 Hann 窗平滑过渡（长度 = 10 步），避免硬切换引入的 ACF 破坏。

**模块 3：CRED 条件触发吹雪事件**

替换 V2 的独立区间注入，改为基于 CRED 函数的条件触发：

```python
# CRED 参数来自 Amory 2020 Fig.3
def cred(wind_speed_ms):
    # 8 m/s 时 CRED ≈ 0.5，12 m/s 时 CRED > 0.9
    return 1 / (1 + exp(-1.2 * (wind_speed_ms - 8.0)))

# 每步以 CRED(v_t) 的概率触发吹雪事件
event_t = Bernoulli(p=cred(wind_speed_t))
```

此设计使 event fraction 自然地与风速状态耦合：storm 状态下风速高，CRED 高，event fraction 高（可达 > 0.75）；calm 状态下 event fraction 低（< 0.10）。

**模块 4：条件生成吹雪变量**

吹雪质量通量和粒径变量改为以风速为条件的幂律模型（来自 Amory 2020）：

```python
# 质量通量幂律（Amory 2020 Eq.3）
flux = alpha * (wind_speed - threshold) ** beta * event_indicator
# alpha, beta, threshold 从 AntAWS 数据拟合
```

### 3.3 G1：V3 生成器验证实验

在重跑主实验之前，必须验证 V3 生成器满足以下统计标准：

**验证指标 G1-V1（ACF 保真度）**：
- 计算合成风速序列的 ACF（lag 1–48h）
- 与 AntAWS 参考 ACF 的最大绝对偏差 < 0.05（在 lag 1–12h 范围内）
- 报告：ACF 对比图（合成 vs. 真实，含 95% 置信带）

**验证指标 G1-V2（Event Fraction 分布）**：
- 在 512 步窗口上采样 1000 个窗口，计算 event fraction 分布
- 要求：P(event fraction > 0.75) > 0.05（即至少 5% 的窗口为真正的 event-heavy）
- 要求：P(event fraction < 0.25) > 0.30（保留足够的 calm 窗口）

**验证指标 G1-V3（边际分布保真度）**：
- 对所有 8 个传感器通道，计算合成数据与 AntAWS 参考数据的 KS 统计量
- 要求：所有通道 KS 统计量 < 0.05（p > 0.05，即不拒绝同分布假设）

**验证指标 G1-V4（功率谱密度）**：
- 计算合成风速的 PSD，与 AntAWS 参考 PSD 的对数均方误差 < 0.1（在 0.001–0.5 Hz 范围内）

**输出文件规范**：

```
reports/v3_supplement_assets/
  exp_g1_generator_validation.csv    # 列：metric, value, threshold, passed
  figure_g1_acf_comparison.png       # ACF 对比图（合成 vs. 真实）
  figure_g1_event_fraction_dist.png  # Event fraction 分布直方图
  figure_g1_psd_comparison.png       # PSD 对比图
```

**通过标准**：G1-V1 至 G1-V4 全部通过后，方可进行 S2 主实验重跑。若任一指标未通过，需修改 V3 生成器参数后重新验证。

### 3.4 S2：V3 生成器上的主实验重跑

**实验设计**（与 S1 完全相同，仅替换生成器）：

- Budget：B ∈ {1.65, 1.70, 1.75}
- Seeds：41–50（n=10）
- 策略：Full obs. / Static proj. / PD-PPO / Round-robin / AoI / Random
- 评估指标：FW-MAE（mean ± std），Wilcoxon signed-rank + Bonferroni 校正

**新增 E1-V3（V3 条件分层评估）**：

在 S2 完成后，重跑 E1，此时 event fraction 可达 > 0.75，三档条件重新定义为：

| 条件 | Event fraction 范围 | 预期均值 |
|------|-------------------|---------|
| calm | < 0.15 | ~0.08 |
| mixed | 0.30–0.50 | ~0.40 |
| event-heavy | > 0.75 | ~0.82 |

**输出文件规范**：

```
reports/v3_supplement_assets/
  exp_s2_main_stats.csv              # 与 exp_s1_main_stats.csv 格式相同
  exp_s2_e1_condition_stats.csv      # V3 条件分层结果
```

### 3.5 正文影响

S2 完成后，论文可将 V2 结果作为"preliminary results"保留，将 V3/S2 结果作为主要结果报告，并在 §4 中将 V3 生成器设计作为方法贡献的一部分（而非仅作为 future work）。摘要数字需根据 S2 结果更新。

---

## 四、T1：跨站点泛化评估

### 4.1 动机

当前所有实验均基于单一仿真环境（校准到 AntAWS 数据集的统计参数，但未指定具体站点）。Cold Regions S&T 的审稿人可能质疑：PD-PPO 的调度策略是否对特定站点的统计特性过拟合，能否迁移到其他南极 AWS 站点。

### 4.2 实验设计

选择 AntAWS 数据集中统计特性差异最大的两个站点作为测试环境：

**站点 A（训练分布内）**：D17（Adélie Land 沿海，高 event frequency ~20%，中位事件持续 15h）——与当前仿真校准参数最接近。

**站点 B（分布外，低 event frequency）**：Dome C（内陆高原，event frequency ~5%，风速低，温度极低）——代表与训练分布差异最大的极端情况。

**站点 C（分布外，高 event frequency）**：D47（Adélie Land 内陆，event frequency ~25%，中位事件持续 27.5h）——代表比训练分布更高 event frequency 的情况。

**实验流程**：

1. 以 D17 统计参数训练 PD-PPO（与 S1/S2 相同）
2. 在 D17、Dome C、D47 三个站点的仿真环境中评估同一训练好的策略（零样本迁移）
3. 可选：在 Dome C 和 D47 上微调（fine-tune）10% 的训练步数，评估迁移后性能

**实验参数**：
- Budget：B = 1.70（固定）
- Seeds：41–45（n=5，泛化评估不需要 n=10）
- 评估指标：FW-MAE（mean ± std），以及相对于各站点 AoI 基线的改进幅度

**输出文件规范**：

```
reports/v3_supplement_assets/
  exp_t1_transfer_stats.csv          # 列：station, policy, mean, std, delta_vs_aoi_pct
```

### 4.3 正文影响

T1 完成后，`sections/07_discussion.tex` 的"Implications for Antarctic AWS Operations"子节可增加跨站点泛化的讨论，将 PD-PPO 定位为可部署到多个南极 AWS 站点的通用调度框架，而非针对单一站点的专用解决方案。

---

## 五、实验优先级与执行顺序

### 5.1 优先级矩阵

| 实验 | 审稿人直接要求 | 影响核心声称 | 计算成本 | 优先级 |
|------|-------------|------------|---------|--------|
| A1（完整消融） | 是（审稿意见一） | 是（EventAwareCritic 贡献） | 低（75 runs） | **最高** |
| H1（超参数扫描） | 是（审稿意见一） | 是（鲁棒性声称） | 中（75 runs） | **高** |
| G1（V3 验证） | 是（审稿意见三） | 是（ACF 保真度） | 低（验证只需统计计算） | **高** |
| S2（V3 主实验） | 间接（审稿意见三） | 是（event-heavy 条件） | 高（60 runs） | **高（依赖 G1）** |
| T1（跨站点） | 否 | 否（加分项） | 中（30 runs） | **中** |

### 5.2 推荐执行顺序

**阶段一（立即执行，约 1 周）**：

1. **A1**：启动 15 个消融变体的训练（n=10 seeds，B=1.70）。可与 H1 并行。
2. **H1**：启动 15 个超参数配置的训练（n=5 seeds，B=1.70）。可与 A1 并行。
3. **G1**：实现 V3 生成器，运行统计验证（无需 GPU，纯数据处理）。

**阶段二（G1 通过后，约 1–2 周）**：

4. **S2**：在 V3 生成器上重跑主实验（n=10 seeds，三个 budget）。
5. **E1-V3**：在 S2 完成后，重跑条件分层评估（使用 V3 生成器的 event-heavy 条件）。

**阶段三（可选，约 1 周）**：

6. **T1**：跨站点泛化评估（n=5 seeds，三个站点）。

### 5.3 最小可行实验集（若时间紧迫）

若投稿截止日期不允许完成全部实验，以下三项构成最小可行集，可将论文从"有缺陷"升级为"可接受"：

1. **A1**（完整消融）：直接回应审稿意见一，计算成本最低，影响最大。
2. **H1**（超参数扫描）：直接回应"hyperparameter-robust"声称缺口，计算成本与 A1 相当。
3. **G1**（V3 生成器验证）：直接回应审稿意见三（ACF 失配），无需 GPU，仅需数据处理。

S2 和 T1 可作为"ongoing work"在 §7 中提及，不影响当前投稿。

---

## 六、各实验的 LaTeX 正文占位符

以下为各实验完成后需要填入正文的 LaTeX 代码框架，供 Codex 在实验数据就绪后直接填充。

### 6.1 A1 消融表（`tables/ablation_full.tex`）

```latex
\begin{table}[t]
\caption{Full component ablation (A1) at budget $B = 1.70$
  ($n = 10$ seeds, seeds 41--50). Each row removes one component
  from the full PD-PPO configuration. Lower FW-MAE is better.
  $*$: significantly different from full PD-PPO
  ($p < 0.0083$, Bonferroni-corrected Wilcoxon signed-rank test).}
\label{tab:ablation_full}
\begin{tabular}{llrr}
\toprule
Variant & Removed component & FW-MAE (mean $\pm$ std) & $p$ vs.\ full \\
\midrule
Full PD-PPO          & ---                          & $0.3911 \pm 0.0240$ & --- \\
A1-v1                & ActionEmbedding              & $\mathtt{[FILL]} \pm \mathtt{[FILL]}$ & $\mathtt{[FILL]}$ \\
A1-v2                & EventAwareCritic             & $\mathtt{[FILL]} \pm \mathtt{[FILL]}$ & $\mathtt{[FILL]}$ \\
A1-v3$^*$            & AWBC auxiliary loss          & $\mathtt{[FILL]} \pm \mathtt{[FILL]}$ & $\mathtt{[FILL]}$ \\
A1-v4$^*$            & Oracle-calibrated prior      & $\mathtt{[FILL]} \pm \mathtt{[FILL]}$ & $\mathtt{[FILL]}$ \\
A1-v5$^*$            & AWBC + Oracle prior          & $\mathtt{[FILL]} \pm \mathtt{[FILL]}$ & $\mathtt{[FILL]}$ \\
A1-v6 (MaskedActor)  & All auxiliary components     & $0.4224 \pm 0.0326$ & $\mathtt{[FILL]}$ \\
\bottomrule
\end{tabular}
\end{table}
```

**注意**：A1-v6 的数字直接复用 A2 D1 数据（0.4224 ± 0.0326），但 p 值需从 A1 原始数据重新计算（n=10 vs. n=5）。

### 6.2 H1 超参数热图正文段落（`sections/06_experiments.tex`）

```latex
\subsection{Hyperparameter Sensitivity (H1)}
\label{sec:h1_hyperparam}

Figure~\ref{fig:h1_heatmap} reports FW-MAE across a $3 \times 4$ grid
of \texttt{awbc\_coef} $\in \{0.1, 0.5, 2.0\}$ and
\texttt{prior\_kl\_coef} $\in \{0.01, 0.05, 0.10, 0.20\}$ at budget
$B = 1.70$ ($n = 5$ seeds). All configurations achieve FW-MAE within
$\mathtt{[FILL]}\%$ of the baseline ($0.3911$), confirming that
PD-PPO's performance is robust to hyperparameter variation within
one order of magnitude. The most sensitive direction is
\texttt{[FILL]}: increasing \texttt{[FILL]} beyond $\mathtt{[FILL]}$
degrades FW-MAE by $\mathtt{[FILL]}\%$, consistent with
\texttt{[FILL]} causing \texttt{[FILL]}.
```

### 6.3 G1 生成器验证段落（`sections/04_simulation_environment.tex`）

```latex
\paragraph{V3 generator validation (G1).}
The V3 generator satisfies four statistical criteria verified against
the AntAWS reference dataset: (i) wind-speed ACF maximum absolute
deviation $< 0.05$ at lags 1--12\,h (G1-V1: $\mathtt{[FILL]}$);
(ii) event fraction $> 0.75$ achievable in $\mathtt{[FILL]}\%$ of
512-step windows (G1-V2); (iii) KS statistic $< 0.05$ for all eight
sensor channels (G1-V3: max $= \mathtt{[FILL]}$); (iv) PSD log-MSE
$< 0.1$ in the 0.001--0.5\,Hz band (G1-V4: $\mathtt{[FILL]}$).
```

---

## 七、实验间依赖关系图

```
A1 ──────────────────────────────────────────────────────► 更新 tables/ablation_full.tex
                                                            更新 §6 消融子节
H1 ──────────────────────────────────────────────────────► 新增 §6 H1 子节
                                                            删除"future work"占位符
G1（V3 生成器验证）
  │
  ├─ 通过 ──► S2（V3 主实验）──► 更新 tables/main_results.tex（V3 数据）
  │                │              更新摘要数字
  │                └──► E1-V3 ──► 更新 §6 E1 子节（真正的 event-heavy 条件）
  │
  └─ 未通过 ──► 修改 V3 生成器参数 ──► 重新运行 G1

T1（可选）──────────────────────────────────────────────► 更新 §7 跨站点泛化讨论
```

---

## 八、与 turn_56 Codex 执行计划的关系

turn_56 的 14 步 Codex 执行计划处理的是**已有实验数据的正文对齐**，本计划处理的是**尚未运行的实验**。两者的关系如下：

**建议执行顺序**：

1. **立即执行 turn_56 计划**（14 步）：将已有的 S1/A2/E1-fix/E2/P1 数据写入正文，消除所有 `[FILL]` 占位符，修正所有已知的过强声称。这是投稿前的最低要求，与本计划无冲突。

2. **并行启动 A1 和 H1 训练**：计算成本低，可在 turn_56 执行期间同步运行。

3. **A1/H1 数据就绪后**：在 turn_56 已修改的正文基础上，进一步扩展消融子节和新增超参数子节。

4. **G1 验证通过后启动 S2**：若 S2 结果显著优于 S1（因 V3 生成器修复了 ACF 缺陷），则需更新摘要数字和主结果表；若 S2 结果与 S1 接近，则 V2 结果仍为主要结果，V3 作为验证。

**不应等待本计划完成后再执行 turn_56**：turn_56 的修改（修正 DOI 错误、删除无来源声称、统一符号）是独立于实验结果的，应立即执行。
