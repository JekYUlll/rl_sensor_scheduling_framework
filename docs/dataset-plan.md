# 数据集生成工作清单与公开数据增强方案

## 一、当前数据集生成的已有基础与缺口诊断

根据已锁定的实验架构，仿真环境需要提供以下9个变量的多变量时序数据，供Frozen Oracle训练和RL策略评估使用：

- **气象基础变量（4个）**：`air_temperature_c`、`wind_speed_ms`、`wind_dir_sin`、`wind_dir_cos`
- **雪面/辐射变量（2个）**：`snow_surface_temperature_c`、`solar_radiation_wm2`
- **吹雪专项变量（3个）**：`snow_particle_mean_diameter_mm`、`snow_particle_mean_velocity_ms`、`snow_mass_flux_kg_m2_s`

当前方案（id_024，Aloni et al., 2024）采用基于离散傅里叶变换（DFT）的合成时序生成方法，其核心机制是：保留原始信号的DFT幅度谱不变，对相位分量进行随机化，从而在保留均值、标准差和自相关函数（ACF）的前提下生成统计等价的新序列（Lemma 1–3，已有解析证明）。该方法对气象基础变量（温度、风速、辐射）的生成具有良好的理论保障，但存在以下三个关键缺口：

**缺口1：吹雪变量的非高斯重尾特性无法被DFT方法完整捕获。** DFT方法保留的是功率谱密度（PSD）和ACF，但不保留高阶矩（偏度、峰度）。Amory（2020，id_015）在Adélie Land的多年观测表明，吹雪质量通量（snow mass flux）呈现显著的重尾分布，偏度超过2，且在低风速区间（<8 m/s）几乎为零、在高风速区间（>12 m/s）急剧增大，呈现强非线性的幂律关系。纯DFT合成无法重现这种条件分布结构。

**缺口2：吹雪变量与气象变量之间的物理耦合关系缺乏约束。** 真实数据中，`snow_mass_flux`与`wind_speed`之间存在幂律耦合（$\eta_{DR} \propto u^{\alpha}$，$\alpha \approx 3$–5），`snow_particle_mean_diameter`与`wind_speed`之间存在负相关（高风速下粒径分布向小粒径偏移）。若各变量独立合成，这些物理约束将被破坏，导致仿真环境物理不可信。

**缺口3：OTT Parsivel²低温不可用期间（$T_{\text{env}} < -30°C$）的数据分布需要专项处理。** 在低温时段，`snow_particle_*`变量应被标记为缺失（传感器强制OFF），合成数据需正确反映这一条件缺失模式，而非生成虚假的粒子测量值。

---

## 二、可行的数据增强方案（三层架构）

整体方案分为三层：**公开数据锚定层**（提供真实统计先验）→ **物理约束合成层**（生成符合物理规律的多变量序列）→ **统计保真度验证层**（量化验证合成质量）。

---

### 第一层：公开数据锚定——建立真实统计先验

#### 1.1 AntAWS数据集（首选，气象基础变量）

AntAWS数据集（Wang et al., 2023，id_017）整合了267个南极AWS站点1980–2021年的气温、气压、相对湿度、风速和风向观测，以3小时、日和月分辨率提供，可通过 `https://doi.org/10.48567/key7-ch19` 公开获取。该数据集直接提供了`air_temperature_c`、`wind_speed_ms`、`wind_dir_*`的真实分布参数（均值、标准差、季节性周期、极端值频率），可作为DFT合成的**输入基信号**，确保合成序列的一阶和二阶统计量与南极真实气候一致。

具体操作：从AntAWS中选取地理位置和海拔与目标站点最接近的1–3个站点（优先选取东南极内陆站，与id_008 SAE-LSTM的训练站点Panda 300、Kunlun保持一致），提取多年逐小时或逐3小时观测，作为DFT方法的基信号 $S$，生成合成序列 $\hat{S}$。

#### 1.2 Amory（2020）Adélie Land吹雪统计（吹雪变量先验）

Amory（2020，id_015）提供了D17和D47两个站点2010–2018年的2G-FlowCapt™多年连续观测，包含吹雪质量通量的完整统计特征：风速阈值（CRED > 0.5对应风速约8 m/s，CRED > 0.9对应约12 m/s）、幂律指数、年际变化范围（年总质量输运约$1.6$–$2.7 \times 10^6$ kg m$^{-2}$）。这些统计量可直接用于参数化吹雪变量的条件分布。

#### 1.3 PANGAEA吹雪粒径数据集（粒径分布先验）

PANGAEA数据库（`https://doi.pangaea.de/10.1594/PANGAEA.992701`）提供了Ny-Ålesund站点2025年冬季的开路光学雪粒计数器（SPC）观测，包含36–500 µm范围内65个粒径区间的粒径分布和质量通量（id_050，Monrad-Krohn et al., 2026）。尽管该数据来自北极而非南极，其粒径分布的形态特征（对数正态或Gamma分布）可作为`snow_particle_mean_diameter_mm`合成的分布形状先验，结合Amory（2020）的南极风速条件进行参数调整。

#### 1.4 ERA5再分析数据（辐射和温度的长期背景场）

ERA5（ECMWF，通过Copernicus CDS API获取，`https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels`）提供1940年至今的逐小时全球再分析数据，南极区域的`2m_temperature`、`surface_solar_radiation_downwards`、`10m_u_component_of_wind`、`10m_v_component_of_wind`均可免费下载。ERA5在南极降水事件识别率超过85%（MDPI Geosciences, 2019），可用于：（a）验证AntAWS基信号的季节性模式是否合理；（b）为`solar_radiation_wm2`提供极夜/极昼的年周期模板（南极极夜期间辐射为零，极昼期间辐射峰值可达300–400 W/m²）；（c）提供温度场的长期趋势背景，避免合成序列出现物理上不可能的温度跳变。

---

### 第二层：物理约束合成——生成多变量耦合序列

#### 2.1 气象基础变量（DFT方法，直接适用）

对`air_temperature_c`、`wind_speed_ms`、`wind_dir_sin`、`wind_dir_cos`、`solar_radiation_wm2`、`snow_surface_temperature_c`六个变量，直接采用Aloni et al.（2024，id_024）的DFT方法，以AntAWS真实观测为基信号生成合成序列。该方法在解析上保证均值、标准差和ACF三项统计量的精确保留（Lemma 1–3），满足理论3的验证标准（KS检验 $p_v > 0.05$，ACF差异 < 0.1）。

相似度控制参数 $m$（保留的原始相位数量）建议设置为 $m = N/10$ 至 $N/5$，在保持统计相似性的同时引入足够的序列多样性，避免训练集过拟合。

#### 2.2 吹雪变量（条件生成，物理约束驱动）

吹雪三变量（`snow_mass_flux`、`snow_particle_mean_diameter`、`snow_particle_mean_velocity`）不能独立使用DFT方法生成，需采用**条件参数化生成**策略：

**步骤1：风速阈值门控。** 根据Amory（2020，id_015）的CRED分布，当合成风速 $u < 8$ m/s时，令`snow_mass_flux = 0`，`snow_particle_*`变量置为NaN（模拟传感器无信号状态）；当 $u \geq 8$ m/s时进入步骤2。

**步骤2：幂律参数化质量通量。** 在吹雪发生条件下，采用幂律模型生成质量通量：
$$\hat{\eta}_{DR} = a \cdot u^{\alpha} \cdot \epsilon$$
其中 $a$ 和 $\alpha$ 从Amory（2020）的D17/D47拟合结果中取值（$\alpha \approx 3$–5），$\epsilon$ 为对数正态噪声项（捕获重尾特性，$\log \epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2)$，$\sigma_\epsilon$ 从观测数据的残差方差估计）。

**步骤3：粒径与速度的条件生成。** `snow_particle_mean_diameter`与风速负相关，可采用线性回归加噪声模型：$\hat{d} = \beta_0 - \beta_1 \cdot u + \delta$，参数从PANGAEA粒径数据（id_050）或文献中的经验关系估计。`snow_particle_mean_velocity`与风速正相关，采用类似参数化。

**步骤4：OTT低温强制缺失。** 当合成温度 $T_{\text{env}} < -30°C$ 时，将`snow_particle_*`所有变量强制置为NaN，与代码修改M1/M3保持一致。

#### 2.3 多变量协方差结构保留

为保证9个变量之间的跨变量相关性（如温度与辐射的日周期同步、风速与吹雪的耦合），在DFT合成阶段采用**多变量联合相位随机化**：对所有变量的DFT系数使用同一组随机相位偏移矩阵，而非各变量独立随机化。这一操作在保留各变量边际分布的同时，近似保留变量间的互相关结构（cross-correlation）。

---

### 第三层：统计保真度验证——量化合成质量（对应理论3）

按照已锁定的理论3验证标准，对每个变量 $v$ 执行以下三项检验：

**检验1：KS检验（边际分布）。** 对合成序列 $\hat{S}_v$ 与真实AntAWS观测 $S_v$ 执行双样本Kolmogorov-Smirnov检验，要求 $p_v > 0.05$（不拒绝同分布假设）。对吹雪变量，在 $u \geq 8$ m/s的条件子集上分别检验。

**检验2：PSD MSE（频域结构）。** 计算合成序列与真实序列的功率谱密度（PSD）之差的均方误差，要求 $\text{MSE}_{\text{PSD}} < \epsilon_{\text{PSD}}$（阈值 $\epsilon_{\text{PSD}}$ 由真实数据的PSD方差的10%确定）。DFT方法在理论上保证PSD完全一致（Lemma 3的推论），因此该项对气象基础变量应自动满足。

**检验3：ACF差异（时序依赖结构）。** 计算lag 1–20的自相关函数差异，要求 $\max_{l=1}^{20} |ACF_{\hat{S}}(l) - ACF_S(l)| < 0.1$。同样由DFT方法的Lemma 3保证。

**补充检验4（吹雪变量专项）：条件分布检验。** 在 $u \in [8, 12)$ m/s和 $u \geq 12$ m/s两个风速区间内，分别对`snow_mass_flux`执行KS检验，验证幂律参数化的条件分布保真度。

验证结果（KS检验p值、PSD MSE数值、ACF最大差异）将作为理论3的定量支撑填入Section 4.1，并在Figure 3中以分布对比图（真实 vs. 合成）可视化展示。

---

## 三、公开数据集获取清单（可操作）

以下为各数据源的具体获取方式，按优先级排序：

**P0（必须获取）：**

AntAWS数据集（Wang et al., 2023，id_017）可通过 `https://doi.org/10.48567/key7-ch19` 直接下载。本地 `data/AntAWS/3_hourly/` 已包含CSV格式的逐3小时站点观测文件，当前无需手动重新下载。建议第一版优先使用完整性较好的 `Panda100`、`Panda200`、`Taishan` 作为DFT气象基信号；`Kunlun`、`Panda1100` 可作为更冷、更高海拔场景的稳健性补充；`D-17`、`D-47` 更适合用于吹雪文献背景和风速阈值参照，而不是直接作为完整气象基信号。

ERA5再分析数据通过Copernicus CDS Python API（`cdsapi`库）批量下载，目标变量包括`2m_temperature`、`surface_solar_radiation_downwards`、`10m_u_component_of_wind`、`10m_v_component_of_wind`，空间范围限定为目标站点±2°经纬度框，时间分辨率选1小时，时间范围2010–2021年。本地 `~/.cdsapirc` 已配置CDS API地址和密钥；后续只需确认账号已接受目标ERA5数据集条款，并用小范围请求验证下载链路，避免一开始发起多年大文件下载。

**P1（强烈建议获取）：**

Amory（2020，id_015）的D17站点原始数据已通过论文附录和Supplement提供部分统计参数，可直接从论文中提取幂律拟合参数（$a$、$\alpha$）和CRED分布数值，无需重新下载原始观测。

PANGAEA粒径数据集（id_050，`https://doi.pangaea.de/10.1594/PANGAEA.992701`）提供CC-BY-4.0开放许可，可直接下载CSV格式的粒径分布时序数据，用于估计`snow_particle_mean_diameter`的分布形状参数。

**P2（可选，用于Sim-to-Real验证增强）：**

AMRDC（Antarctic Meteorological Research and Data Center，University of Wisconsin，`https://amrdc.ssec.wisc.edu/`）提供美国南极计划（USAP）AWS网络的实时和历史数据，部分站点提供逐分钟分辨率观测，可用于Section 6.5的Sim-to-Real泛化验证，作为AntAWS 3小时分辨率数据的高频补充。

MERRA-2吹雪诊断数据（Bhatta & Yang, 2024，id_054）提供0.5°×0.625°空间分辨率、逐小时的南极吹雪发生概率、高度和光学厚度，可用于验证合成吹雪事件的时空分布是否与再分析产品一致，作为吹雪变量合成质量的独立交叉验证。

---

## 四、实施顺序建议（对应实验重跑Step 1）

与已锁定的实验重跑7步顺序中的Step 1（合成数据生成与统计验证）对应，建议按以下子步骤执行：

**子步骤1.1**：复用本地AntAWS目标站点CSV，下载或抽取ERA5背景场，执行数据清洗（去除缺测、异常值标记）。AntAWS第一版不再阻塞于人工下载；ERA5应先进行小范围链路验证，再批量下载。

**子步骤1.2**：对气象基础变量执行DFT合成，相似度参数 $m$ 扫描 $\{N/20, N/10, N/5\}$ 三档，选取KS检验通过且序列多样性最高的配置。当前已新增第一版实现入口 `scripts/20_build_public_weather_truth.py`，可从本地AntAWS站点CSV生成与现有truth replay兼容的CSV，并输出 `synthetic_validation.csv`。实现上默认在相位随机化后执行经验分布重映射，以兼顾时序结构和边际分布保真度。

**子步骤1.3**：基于Amory（2020）参数执行吹雪变量条件参数化生成，加入OTT低温缺失逻辑。当前第一版为保持现有Kalman/oracle链路稳定，不在truth state列中直接写入NaN，而是输出数值化的物理变量，并额外写入 `blowing_snow_active`、`parsivel_available` 标记；后续传感器层应根据这些标记决定观测是否可用。

**子步骤1.4**：执行三项统计保真度验证（KS、PSD MSE、ACF），记录所有变量的定量结果，填入理论3数值占位符。

**子步骤1.5**：生成Figure 3（合成数据 vs. AntAWS真实数据分布对比图），包含边际分布直方图、PSD曲线对比和ACF对比三个子图。

---

## 五、关键引用定位

上述方案中涉及的文献引用位置建议如下：Aloni et al.（2024，id_024）的DFT方法引用于Section 4.1合成数据生成方法段落；Amory（2020，id_015）的吹雪统计引用于Section 4.1吹雪变量参数化段落和Section 3.4理论1反例的物理依据；Wang et al.（2023，id_017）的AntAWS数据集引用于Section 4.1数据来源说明；ERA5引用于Section 4.1背景场验证段落。PANGAEA粒径数据（id_050）若作为参数估计来源，引用于Section 4.1粒径分布参数化段落。
