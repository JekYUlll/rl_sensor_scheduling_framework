# 论文图表清单与理论证明要求

## 一、必须输出的图表

### 图（Figures）

**Figure 1 — 系统架构总览图**（Introduction或Section 3）
展示"传感器层 → 状态机 → 调度策略（RL Agent）→ Frozen Oracle → 奖励反馈"的完整闭环流程。这是全文的核心示意图，审稿人第一眼需要看到整体逻辑。

**Figure 2 — 传感器状态机转移图**（Section 3.2）
三状态有向图：OFF → WARMING_UP → ACTIVE，标注各转移条件（启动触发、warmup倒计时、连续供电保持、断电回到OFF、暖机中途放弃）。v2主线不再要求实现OTT低温强制下线；低温可用性仅作为未来扩展或数据标记保留。

**Figure 3 — 合成数据统计对比图**（Section 4.1）
合成数据 vs. AntAWS真实数据的分布对比，至少包含：温度、风速、snow mass flux三个变量的概率密度曲线（PDF）或Q-Q图，以及自相关函数（ACF）对比。这是回应"合成数据可信度"质疑的核心证据图。

**Figure 4 — 学习曲线图**（Section 5或Section 6）
PPO、DQN、CMDP-DQN三种算法的训练收敛曲线（episode reward vs. training steps），展示收敛速度和稳定性差异。

**Figure 5 — 调度策略可视化图**（Section 6）
选取一段典型时间窗口（如含吹雪事件的48小时），展示PPO/DQN/规则基线的传感器状态时序（OFF/WARMING/ACTIVE Gantt图或热力图），直观呈现高功耗传感器（FlowCapt FC4、粒子传感器）的调度模式、暖机保持/中断行为，以及功率预算曲线。

**Figure 6 — 真实气象锚定泛化验证图**（可选，Section 6.5）
在AntAWS/ERA5真实气象背景驱动的数据上，对比"PPO调度输入"与"全观测参考输入"在可观测变量子集上的预测MAE。该图用于展示真实气象背景下的泛化趋势，不写成完整真实部署验证。

### 表（Tables）

**Table 1 — 传感器物理参数表**（Section 3.1）
列出v2实验所用传感器的功耗（W或归一化功耗）、采样间隔、启动延迟、启动峰值功耗、对应预测变量，作为全文参数基准。第一版可使用8传感器complex配置；若论文压缩实验规模，也可报告5传感器核心配置。

**Table 2 — 主结果表**（Section 6.2）
行：6种调度策略（Random / Fixed-Period / AoI-Bandit / DQN / PPO / CMDP-DQN）；列：各变量MAE、聚合MAE、平均功耗、约束违反率。这是全文最重要的表格，需在固定能耗预算下比较预测质量。

**Table 3 — 逐变量分析表**（Section 6.3）
重点呈现高功耗传感器变量（snow_particle_mean_diameter_mm、snow_particle_mean_velocity_ms、snow_mass_flux_kg_m2_s）在不同调度策略下的MAE，并额外报告吹雪事件时段与非事件时段的条件MAE。

**Table 4 — 消融实验结果表**（Section 6.4）
5项消融的结果汇总：(1) 去除状态机warmup逻辑；(2) reward horizon从10步缩短至5步；(3) 硬约束改为软惩罚；(4) oracle从多模型/强模型退化为轻量线性或TCN单模型；(5) 去除事件/吹雪变量权重。

---

## 二、需要证明的理论

### 理论1 — 预测驱动奖励与KF后验协方差奖励的不等价性（Section 3.4，必须）

**需要证明的命题**：最优KF调度策略（最小化后验协方差迹$\text{tr}(P_t)$）与最优预测驱动调度策略（最小化H步预测MAE）在一般情况下不等价，仅在线性高斯系统且H=1时趋于等价。

**证明路径**：(1) 写出KF奖励的数学形式$r_t^{\text{KF}} = -\text{tr}(P_t^+)$，指出其依赖线性高斯状态空间模型假设；(2) 写出本研究奖励的数学形式$r_t = -\frac{1}{|\mathcal{V}_t|}\sum_{v}w_v\cdot\text{MAE}(\hat{y}_{t+1:t+H}^v, y_{t+1:t+H}^v)$，指出oracle为非参数模型，无动力学假设；(3) 举反例：在吹雪变量的重尾非高斯分布下，KF的$P_t$低估真实不确定性，导致KF调度策略系统性地低估FlowCapt FC4的调度优先级；(4) 引用Alali et al. (2024, id_002)和Al Ahdab et al. (2025, id_025)作为KF范式代表，引用Amory (2020, id_015)作为吹雪非高斯分布的实证依据。

### 理论2 — CMDP-DQN三层约束架构的可行性（Section 5.4，必须）

**需要证明的命题**：硬约束投影器保证每步动作满足瞬时功率约束；若加入CMDP-DQN扩展，则Primal-Dual Lagrangian和约束Q网络可用于处理长期平均功率约束。

**证明路径**：(1) 投影器逐步构造可行子集，因此每次加入传感器前都会验证 `max_active`、`per_step_budget` 和 `startup_peak_budget`；(2) 空动作集天然可行，因此可行集非空；(3) 对于CMDP-DQN扩展，对偶变量$\lambda_{\text{cost}}$可按episode平均功耗约束进行次梯度更新，动作选择规则为$\text{argmax}[Q_{\text{reward}} - \lambda_{\text{cost}} Q_{\text{cost}}]$。主线论文可先证明投影器的瞬时可行性，CMDP部分作为扩展说明。

### 理论3 — 合成数据的统计保真度（Section 4.1，必须）

**需要证明的命题**：基于DFT的合成时序数据（Aloni et al., 2024, id_024）在功率谱密度、自相关结构和边缘分布上与AntAWS真实数据统计等价，足以作为RL训练环境。

**证明路径**：定量报告合成数据与真实数据在以下指标上的差异：Kolmogorov-Smirnov检验p值（边缘分布）、功率谱密度曲线的均方误差（频域结构）、ACF在lag 1–20的相关系数差异。这是实验性证明，非数学推导，但必须有具体数值支撑。

### 理论4 — 预测驱动调度优于AoI代理指标（Section 7.1，支撑性）

**需要证明的命题**：AoI（Age of Information）作为调度奖励的代理指标，在多变量异构传感器场景下与预测MAE的相关性弱，导致AoI-Bandit基线的预测性能系统性劣于PPO。

**证明路径**：这是实验性论证，不需要数学证明。在Table 2中展示AoI-Bandit的MAE显著高于PPO，并在Discussion中解释原因：AoI假设"越新鲜的数据越有价值"，但对于采样间隔为15s的FlowCapt FC4，AoI会过度调度该传感器；而预测驱动奖励能识别在低吹雪活动期间FlowCapt的边际预测贡献下降，从而动态降低其调度频率。

---

## 三、优先级排序

**必须有、无可替代**：Figure 1（架构图）、Figure 2（状态机）、Figure 3（合成数据统计对比）、Table 1（传感器参数）、Table 2（主结果）、理论1（KF不等价性）、理论3（合成数据保真度）。

**强烈建议有**：Figure 3（分布对比）、Figure 5（调度可视化）、Table 4（消融）、理论2（CMDP-DQN可行性）。

**可选**：Figure 4（学习曲线）、Figure 6（Sim-to-Real）、Table 3（逐变量）、理论4（AoI对比）——这些在实验结果充分时加入，篇幅受限时可压缩为正文叙述。
