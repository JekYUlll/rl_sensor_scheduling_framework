# 2026-07-02 参考文献联网校对报告（ESWA 投稿版）

对象稿件：`paper/main.tex`，目标期刊：*Expert Systems with Applications*（ESWA）。

本报告只审计当前 active manuscript 会进入编译链的参考文献，不把历史草稿、翻译镜像、归档文件、未 include 的旧 section 计入 active citation。第一次抽取时漏掉了跨行 `\citep{...}` 组；已修正为跨行解析后重新统计。

## 0. 结论先行

active 引用数：24 条。

`paper/references.bib` 总条目数：38 条。

当前未被 active manuscript 引用的 BibTeX 条目：14 条，列在本文末尾“附录 A”。这些条目不会出现在当前 PDF 的 References 中，除非后续显式引用或 `\nocite`。

总体判断：现有 bibliography 大体可用，文献方向也能支撑论文主线，但在 ESWA 投稿前需要修几个硬问题：

1. 严重元数据问题：
   - `FernandezBes2015` 的 DOI `10.1109/JSAC.2015.2430512` 在 doi.org 返回 DOI Not Found。IEEE Xplore record 存在，document id 是 `7009961`。建议删除 DOI 或用 IEEE 官方导出的正确 DOI 替换；至少加入 URL `https://ieeexplore.ieee.org/document/7009961`。
   - `Golovin2011` 的 DOI `10.1613/jair.3278` 在 doi.org 也返回 DOI Not Found；JAIR 页面显示该 DOI，但 resolver 不通。建议加入 JAIR URL 和 arXiv URL；若提交前仍不 resolve，去掉 DOI 字段。

2. 会影响 ESWA reference linking 的字段缺失：
   - `Murad2020` 缺页码，官方 ACM 页面为 `1--8`。
   - `Wei2020` 缺页码，IEEE INFOCOM 页面为 `864--873`。
   - `Pendyala2024` 缺页码，Springer LNCS 页面为 `150--165`。
   这些缺失也解释了之前 BibTeX 编译里的 empty pages warning。

3. preprint / arXiv 元数据需要规范：
   - `Schulman2017` 应加 `eprint={1707.06347}`、`archivePrefix={arXiv}`、`doi={10.48550/arXiv.1707.06347}`。
   - `Liu2024` 应加 OpenReview URL 和 arXiv `2310.06625`。
   - `Tran2026` 是 arXiv-only，应明确标为 arXiv preprint，并加 DOI `10.48550/arXiv.2601.21482`。
   - `Jonah2026` 已有 IEEE Access 正式版本，arXiv 只可作为补充，不应替代正式版本。

4. Related Work 第一段 citation pile 太重：
   - 当前一句话里堆了 8 个参考文献：`Shi_2011, Kaul_2012, FernandezBes2015, Qu2022, Alali2024, AlAhdab2025, Jonah2026, Tran2026`。
   - ESWA 读者更偏应用型，建议按 objective family 拆成 2–3 句：estimation/covariance、AoI/AoII/freshness、learning-based scheduling，而不是一括号塞 8 篇。

5. `Pendyala2024` 不是 sensor scheduling 文献，只能支撑“真实 PPO 优化/调度问题”的宽泛背景；如果正文仍写“adaptive sensing / informative path planning / combinatorial scheduling”，它是弱支撑。建议删除或换成更贴近 sensor/monitoring 的文献。

6. `Monrad2026` 是 Ny-Ålesund / Arctic PANGAEA 数据集，不是 Antarctic。它可以支撑 blowing-snow particle size / snow mass flux measurement，但正文不要让读者误以为它是 Antarctic AWS 证据。

7. 若正文继续写 “masked PPO policy”，建议新增一条 action masking 文献：Huang & Ontañón (2022), “A Closer Look at Invalid Action Masking in Policy Gradient Algorithms”, DOI `10.32473/flairs.v35i.130584`。这比只引用 PPO 原论文更准确。

## 1. ESWA 官方格式要求与本稿风险点

依据 ESWA / ScienceDirect Guide for Authors：

- 文中引用与参考文献表必须一一对应。官方原文：
  > “Any references cited within your article should also be present in your reference list and vice versa.”

- ESWA 鼓励 DOI，因为 DOI 支持 permanent linking。官方原文：
  > “We encourage the use of Digital Object Identifiers (DOIs) as reference links, as they provide a permanent link to the electronic article referenced.”

- 提交前要检查作者姓氏、期刊/书名、年份、页码等。官方原文：
  > “Before submission, check that all data provided in your reference list are correct... Any incorrect surnames, journal or book titles, publication years or pagination within your references may prevent link creation.”

- ESWA reference style 使用 APA 7。官方原文：
  > “Citations in the text should follow the referencing style used by the American Psychological Association... Seventh Edition (2020).”

- 参考文献表按字母顺序、同作者按时间顺序。官方原文：
  > “The reference list should be arranged alphabetically and then chronologically.”

- dataset 要显式作为 data reference 处理。官方原文：
  > “When citing data references, you should include: author name(s), dataset title, data repository, version (where available), year, global persistent identifier.”
  > “Add [dataset] immediately before your reference.”

- preprint 要明确标注，并提供 preprint DOI；若已有正式发表版，应引用正式发表版。官方原文：
  > “We ask you to mark preprints clearly. You should include the word ‘preprint’ or the name of the preprint server as part of your reference and provide the preprint DOI.”
  > “Where a preprint has subsequently become available as a peer-reviewed publication, use the formal publication as your reference.”

- ESWA 对 research data 是 Option C：要求存储、引用并链接数据，或说明不可共享原因。官方原文：
  > “For this journal, Option C instructions ... apply. This means that you are required to: Deposit your research data in a relevant data repository. Cite and link to this dataset in your article. If this is not possible, make a statement explaining why research data cannot be shared.”

本稿对应风险：

- 两个 DOI resolver 不通：`FernandezBes2015`, `Golovin2011`。这直接违反“correct reference data / DOI linking”的目标。
- 若 `Monrad2026` 作为 dataset，应按 dataset citation 规范处理，最好在 BibTeX note 或 title 中标识 `[dataset]`。
- 当前 active manuscript 没有引用 14 个 BibTeX 条目。它们不会出现在 References；如果这些条目是 sensor manual / data source，必须决定是删掉还是重新引用。

## 2. active references 逐条校对

### 2.1 `Schulman2017`

真实文档：

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017.
- arXiv DOI：`10.48550/arXiv.1707.06347`。

本稿引用处：

- `sections/01_introduction.tex`：
  > “PD-PPO is the implementation used to test these hypotheses. It is a masked PPO policy over candidate channel masks \citep{Schulman2017}.”

原文证据：

> “The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically).”

是否支持本稿论据：

- 支持 PPO 方法来源。
- 不支持 “masked” 这个实现细节；action masking 不是 Schulman et al. 的核心论点。
- 当前句子可以保留，因为 citation 可理解为只支撑 PPO；但若审稿人严格追问 invalid action masking，最好新增 Huang & Ontañón (2022)。

格式/作者校对：

- 作者、题名正确。
- BibTeX 建议补：`eprint`, `archivePrefix`, `primaryClass`, `doi`, `url`。

ESWA 适配性：

- 核心方法文献，必须保留。
- 作为 preprint，应按 ESWA preprint 规范清楚标注 arXiv 和 DOI。

建议：保留，并补全 arXiv 字段；新增 action masking 文献作为并列支撑。

### 2.2 `Shi_2011`

真实文档：

- Ling Shi, Peng Cheng, Jiming Chen. “Sensor data scheduling for optimal state estimation with communication energy constraint.” *Automatica* 47(8):1693–1698, 2011. DOI `10.1016/j.automatica.2011.02.037`。

本稿引用处：

- `sections/02_related_work.tex`：用于支撑 resource-constrained sensing 中 estimation/covariance/energy-constrained scheduling 背景。

原文证据：

> “In this paper, we consider sensor data scheduling with communication energy constraint. A sensor has to decide whether to send its data to a remote estimator or not due to the limited available communication energy.”
>
> “We construct effective sensor data scheduling schemes that minimize the estimation error and satisfy the energy constraint.”

是否支持本稿论据：

- 强支持。它正是 estimation error + communication energy constraint 下的 sensor data scheduling。
- 不支持 DRL 或 forecast-loss objective，但本稿只把它放在 classical objective 背景中，合理。

格式/作者校对：

- 作者、题名、期刊、卷期、页码、DOI 正确。

ESWA 适配性：

- foundational control/sensor scheduling 文献，适合保留。

建议：保留。

### 2.3 `Kaul_2012`

真实文档：

- Sanjit Kaul, Roy Yates, Marco Gruteser. “Real-time status: How often should one update?” IEEE INFOCOM 2012, pp. 2731–2735. DOI `10.1109/INFCOM.2012.6195689`。

本稿引用处：

- `sections/02_related_work.tex`：用于支撑 AoI/freshness objective。

原文证据：

> “we employ a time-average age metric for the performance evaluation of status update systems.”
>
> “We show the existence of an optimal rate at which a source must generate its information to keep its status as timely as possible at all its monitors.”

是否支持本稿论据：

- 强支持 AoI / timeliness / freshness 这一 objective family。
- 它不是 sensor selection/forecasting 论文，但在 AoI 背景中引用准确。

格式/作者校对：

- IEEE Xplore 元数据、页码、DOI 正确。

ESWA 适配性：

- 虽是通信会议，但 AoI 是经典概念来源；建议保留。

建议：保留。

### 2.4 `FernandezBes2015`

真实文档：

- Jesús Fernández-Bes, Jesús Cid-Sueiro, Antonio G. Marques. “An MDP Model for Censoring in Harvesting Sensors: Optimal and Approximated Solutions.” *IEEE Journal on Selected Areas in Communications* 33(8):1717–1729, 2015。
- IEEE Xplore record：`https://ieeexplore.ieee.org/document/7009961`。
- 当前 BibTeX DOI：`10.1109/JSAC.2015.2430512`，但 doi.org 返回 DOI Not Found。

本稿引用处：

- `sections/02_related_work.tex`：用于支撑 censoring utility / energy-harvesting sensor scheduling 背景。

原文证据：

> “we propose a novel censoring policy for energy-efficient transmissions in energy-harvesting sensors. The problem is formulated as an infinite-horizon Markov Decision Process (MDP).”

是否支持本稿论据：

- 支持 energy-harvesting sensors、censoring policy、MDP scheduling。
- 适合作为 resource-constrained sensing objective 的一条背景文献。

格式/作者校对：

- 作者、题名、期刊、卷期、页码基本正确。
- DOI 字段严重可疑：resolver 不通。ESWA 强调 DOI/元数据正确性，这个不能原样提交。

ESWA 适配性：

- 内容相关，但如果不修 DOI，会降低 reference quality。

建议：保留但必须修 DOI。若找不到可解析 DOI，则删除 DOI 字段，加入 IEEE URL。

### 2.5 `Qu2022`

真实文档：

- Zhiyi Qu, Xue Zhao, Huihui Xu, Hongying Tang, Jiang Wang, Baoqing Li. “An Improved Q-Learning-Based Sensor-Scheduling Algorithm for Multi-Target Tracking.” *Sensors* 22, Article 6972, 2022. DOI `10.3390/s22186972`。

本稿引用处：

- `sections/02_related_work.tex`：resource-constrained / learning-based sensor scheduling 背景。

原文证据：

> “we propose in this paper an improved Q-learning-based sensor-scheduling algorithm for multi-target tracking (MTT-SS).”
>
> “Simulation results demonstrate that our proposed algorithm can obtain a significant enhancement in terms of tracking accuracy and energy efficiency compared with the existing sensor-scheduling algorithms.”

是否支持本稿论据：

- 支持 Q-learning sensor scheduling 和 tracking/energy efficiency。
- 但它偏 multi-target tracking，和本稿 forecast-loss scheduling 不是直接同类。

格式/作者校对：

- 元数据正确。

ESWA 适配性：

- 可保留，但如果要压缩 Related Work，它优先级低于 `Alali2024`, `Murad2020`, `Ogbodo2025`。

建议：可保留；若 citation pile 过重，可删。

### 2.6 `Alali2024`

真实文档：

- Mohammad Alali, Armita Kazeminajafabadi, Mahdi Imani. “Deep Reinforcement Learning Sensor Scheduling for Effective Monitoring of Dynamical Systems.” *Systems Science & Control Engineering* 12(1):2329260, 2024. DOI `10.1080/21642583.2024.2329260`。

本稿引用处：

- `sections/02_related_work.tex`：learning-based / DRL sensor scheduling 背景。

原文证据：

> “This paper focuses on sensor scheduling for systems modeled by hidden Markov models.”
>
> “This paper formulates optimal sensor scheduling as a reinforcement learning problem defined over the posterior distribution of system states.”
>
> “The proposed method applies to any monitoring objective that can be expressed in terms of the posterior distribution of the states (e.g., state estimation, information gain, etc.).”

是否支持本稿论据：

- 强支持 DRL sensor scheduling。
- 它的 objective 是 posterior/state monitoring，不是 downstream forecast loss；但作为近邻工作非常合适。

格式/作者校对：

- 元数据正确。

ESWA 适配性：

- 近期、相关、应用型，适合 ESWA Related Work。

建议：保留。

### 2.7 `AlAhdab2025`

真实文档：

- Mohamad Al Ahdab, John Leth, Zheng-Hua Tan. “Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics.” ICML 2025, PMLR 267:703–729。
- 官方 PMLR 页面作者写作 `Mohamad Al Ahdab, John Leth, Zheng-Hua Tan`。

本稿引用处：

- `sections/02_related_work.tex`：resource-constrained sensing / sensor selection / filtering background。

原文证据：

> “continuous-time dynamics are observed via multiple sensors with discrete, irregularly timed measurements.”
>
> “Each sensor thus carries distinct costs and constraints associated with its measurement rate and additional constraints and costs on the auxiliary state.”
>
> “Empirical results in state-space filtering and dynamic temporal Gaussian process regression demonstrate that our approach achieves improved trade-offs between resource usage and estimation accuracy.”

是否支持本稿论据：

- 支持 sensor scheduling/selection、costs/constraints、estimation accuracy/resource trade-off。
- 不是 DRL 文献，但正文用作 objective/scheduling 背景时合理。

格式/作者校对：

- 页码、PMLR volume 正确。
- 当前 BibTeX 如果写 `John-Josef Leth`，与 PMLR 官方页面不一致；建议改为 `John Leth`，除非作者官网另有确认。

ESWA 适配性：

- ICML/PMLR 高质量近期文献，适合保留。

建议：保留；修正作者名。

### 2.8 `Jonah2026`

真实文档：

- Sokipriala Jonah, Seong Ki Yoo, Saurav Sthapit. “Adaptive Scheduling: A Reinforcement Learning Whittle Index Approach for Wireless Sensor Networks.” *IEEE Access* 14:40042–40059, 2026. DOI `10.1109/ACCESS.2026.3673220`。
- arXiv preprint：`2601.01179`。

本稿引用处：

- `sections/02_related_work.tex`：RL/Whittle-index/AoII wireless sensor scheduling 背景。

原文证据：

> “We propose a Reinforcement Learning (RL)-based scheduling framework for Restless Multi-Armed Bandit (RMAB) problems, centred on a Whittle Index Q-Learning policy with Upper Confidence Bound ... exploration.”
>
> “We evaluate WIQL-UCB on standard RMAB benchmarks and on a practical sensor scheduling application based on the Age of Incorrect Information (AoII).”

是否支持本稿论据：

- 支持 RL-based sensor scheduling 和 AoII/freshness 背景。
- 和本稿 forecast-loss scheduling 目标不同，但属于相近问题族。

格式/作者校对：

- IEEE published metadata 正确。
- 因已有正式 IEEE Access 版本，ESWA 应引用正式版本，不应只引用 arXiv。

ESWA 适配性：

- 相关且新；但 2026 文献较新，引用数量应控制。

建议：可保留；如果压缩 citation pile，可考虑删 `Tran2026` 而保留此正式发表版本。

### 2.9 `Tran2026`

真实文档：

- Nho-Duc Tran, Aamir Mahmood, Mikael Gidlund. “Learning-Based Sensor Scheduling for Delay-Aware and Stable Remote State Estimation.” arXiv:2601.21482, 2026. DOI `10.48550/arXiv.2601.21482`。

本稿引用处：

- `sections/02_related_work.tex`：delay-aware / learning-based sensor scheduling 背景。

原文证据：

> “Unpredictable sensor-to-estimator delays fundamentally distort what matters for wireless remote state estimation: not just freshness, but how delay interacts with sensor informativeness and energy efficiency.”
>
> “we cast scheduling as a Markov decision process and develop a proximal policy optimization (PPO) scheduler.”

是否支持本稿论据：

- 支持 learning-based sensor scheduling、PPO scheduler、freshness/energy/informativeness trade-off。
- 它是 arXiv-only，且任务是 remote state estimation with delays，不是 forecast-loss scheduling。

格式/作者校对：

- 作者/题名正确。
- 必须按 ESWA preprint 要求加入 arXiv 和 DOI 字段。

ESWA 适配性：

- 可作为非常新的近邻工作，但 peer-reviewed status 弱。

建议：如果 Related Work 需要最新 PPO sensor-scheduling 背景，则保留；若要减少 preprint 和 citation density，建议删除。

### 2.10 `Bajcsy2018`

真实文档：

- Ruzena Bajcsy, Yiannis Aloimonos, John K. Tsotsos. “Revisiting active perception.” *Autonomous Robots* 42:177–196, 2018. DOI `10.1007/s10514-017-9615-3`。

本稿引用处：

- `sections/02_related_work.tex`：active perception / observations as actions。

原文证据：

> “Despite the recent successes in robotics, artificial intelligence and computer vision, a complete artificial agent necessarily must include active perception.”
>
> “This is the essence of active perception—to set up a goal based on some current belief about the world and to put in motion the actions that may achieve it.”

是否支持本稿论据：

- 支持 active perception 背景。
- 不直接支持 sensor scheduling algorithm，但作为概念桥接合理。

格式/作者校对：

- 作者、题名、期刊、页码、DOI 正确。
- online-first 年份可能是 2017，issue 年份 2018；BibTeX 用 2018 可接受。

ESWA 适配性：

- 基础概念文献，保留即可；不要让 active perception 段落过长。

建议：保留。

### 2.11 `Lauri_2023`

真实文档：

- Mikko Lauri, David Hsu, Joni Pajarinen. “Partially Observable Markov Decision Processes in Robotics: A Survey.” *IEEE Transactions on Robotics* 39(1):21–40, 2023. DOI `10.1109/TRO.2022.3200138`。

本稿引用处：

- `sections/02_related_work.tex`：POMDP / partially observable decision making。

原文证据：

> “Noisy sensing, imperfect control, and environment changes are defining characteristics of many real-world robot tasks.”
>
> “The partially observable Markov decision process (POMDP) provides a principled mathematical framework for modeling and solving robot decision and control tasks under uncertainty.”

是否支持本稿论据：

- 支持 partial observability / sensing under uncertainty 背景。
- 是 robotics survey，不是本稿直接竞品。

格式/作者校对：

- 元数据正确。

ESWA 适配性：

- 合理，但可与 `Bajcsy2018`、`Golovin2011` 一起压缩成一小段，不宜喧宾夺主。

建议：保留。

### 2.12 `Golovin2011`

真实文档：

- Daniel Golovin, Andreas Krause. “Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization.” *Journal of Artificial Intelligence Research* 42:427–486, 2011。
- JAIR page：`https://www.jair.org/index.php/jair/article/view/10731`。
- arXiv：`1003.3967`。
- 当前 DOI `10.1613/jair.3278` 在 doi.org 返回 DOI Not Found。

本稿引用处：

- `sections/02_related_work.tex`：adaptive sensing / partial observability / active learning background。

原文证据：

> “Many problems in artificial intelligence require adaptively making a sequence of decisions with uncertain outcomes under partial observability.”
>
> “we introduce the concept of adaptive submodularity, generalizing submodular set functions to adaptive policies.”
>
> “examples of adaptive submodular objectives arising in diverse AI applications including management of sensing resources, viral marketing and active learning.”

是否支持本稿论据：

- 支持 adaptive decision under partial observability 和 management of sensing resources。
- 不支持 DRL/forecast objective；作为 theoretical background 合理。

格式/作者校对：

- 作者、题名、期刊、卷、页码正确。
- DOI resolver 不通，是严重格式风险。ESWA 强调 DOI link 和 correct reference data。

ESWA 适配性：

- JAIR 是高质量 AI 文献，可以保留。

建议：保留；加入 JAIR URL / arXiv URL；若 DOI 提交前仍不 resolve，去掉 DOI。

### 2.13 `Murad2020`

真实文档：

- Abdulmajid Murad, Frank Alexander Kraemer, Kerstin Bach, Gavin Taylor. “Information-driven adaptive sensing based on deep reinforcement learning.” Proceedings of the 10th International Conference on the Internet of Things, ACM, 2020, pp. 1–8. DOI `10.1145/3410992.3411001`。

本稿引用处：

- `sections/02_related_work.tex`：DRL for adaptive sensing。

原文证据：

> “In order to make better use of deep reinforcement learning in the creation of sensing policies for resource-constrained IoT devices, we present and study a novel reward function based on the Fisher information value.”
>
> “This reward function enables IoT sensor devices to learn to spend available energy on measurements at otherwise unpredictable moments, while conserving energy at times when measurements would provide little new information.”

是否支持本稿论据：

- 强支持 DRL adaptive sensing / resource-constrained IoT sensing。
- 和本稿非常贴近，建议作为 Related Work 核心文献之一。

格式/作者校对：

- 作者、题名、DOI 正确。
- 当前 BibTeX 若缺页码，应补 `pages={1--8}`。

ESWA 适配性：

- 应用型、DRL、sensing，非常适合。

建议：保留；补页码。

### 2.14 `Wei2020`

真实文档：

- Yongyong Wei, Rong Zheng. “Informative Path Planning for Mobile Sensing with Reinforcement Learning.” IEEE INFOCOM 2020, pp. 864–873. DOI `10.1109/INFOCOM41043.2020.9155528`。

本稿引用处：

- `sections/02_related_work.tex`：informative path planning / mobile sensing with RL。

原文证据：

> “we propose a novel IPP algorithm using reinforcement learning (RL).”
>
> “Extensive experiments using real-world measurement data demonstrate that the proposed algorithm outperforms state-of-the-art algorithms in most test cases.”

是否支持本稿论据：

- 支持 RL for informative path planning / mobile sensing。
- 不直接是 fixed sensor schedule；放在 broad adaptive sensing/path planning 句子中合理。

格式/作者校对：

- 元数据正确。
- 补 `pages={864--873}`。

ESWA 适配性：

- 相关但不是最直接。若要压缩引用，可保留 `Murad2020`，删或保留 `Wei2020` 取决于是否保留 path planning 这条支线。

建议：保留；补页码。

### 2.15 `Ogbodo2025`

真实文档：

- Collins O. Ogbodo, Timothy J. Rogers, Mattia Dal Borgo, David J. Wagg. “Adaptive sensor steering strategy using deep reinforcement learning for dynamic data acquisition in digital twins.” *Proceedings of the Royal Society A* 482(2329):20250326, 2026. DOI `10.1098/rspa.2025.0326`。

本稿引用处：

- `sections/02_related_work.tex`：DRL for adaptive sensing / dynamic data acquisition。

原文证据：

> “This paper introduces a sensor steering methodology based on deep reinforcement learning (DRL) to enhance the predictive accuracy and decision support capabilities of digital twins by optimizing the data acquisition process.”

是否支持本稿论据：

- 强支持 DRL adaptive sensor steering / dynamic data acquisition。
- 和本稿 forecast-loss/sensor acquisition 的概念接近。

格式/作者校对：

- DOI suffix 含 2025，但正式 issue 年份为 2026；BibTeX 年份 2026 可接受。
- 期刊名建议统一为全称或标准缩写。

ESWA 适配性：

- 高质量近邻应用文献，建议保留。

建议：保留。

### 2.16 `Pendyala2024`

真实文档：

- Abhijeet Pendyala, Asma Atamna, Tobias Glasmachers. “Solving a Real-World Optimization Problem Using Proximal Policy Optimization with Curriculum Learning and Reward Engineering.” ECML PKDD 2024, LNCS 14950, pp. 150–165. DOI `10.1007/978-3-031-70381-2_10`。

本稿引用处：

- `sections/02_related_work.tex`：DRL has been used for adaptive sensing, informative path planning, and other combinatorial scheduling problems。

原文证据：

> “We present a proximal policy optimization agent trained through curriculum learning (CL) principles and meticulous reward engineering to optimize a real-world high-throughput waste sorting facility.”
>
> “This problem is particularly difficult due to the environment’s extremely delayed rewards with long time horizons and class (or action) imbalance.”

是否支持本稿论据：

- 只支持“PPO 用于真实复杂优化问题 / delayed reward / reward engineering”。
- 不支持 sensor scheduling、adaptive sensing、informative path planning。
- 若正文说 “other real-world PPO optimization problems”，则合理；若当前句子重点是 sensing/scheduling，则支撑较弱。

格式/作者校对：

- 作者、题名、DOI 正确。
- 应补 `pages={150--165}`，可补 `volume={14950}`。

ESWA 适配性：

- 非 sensor，且是会议章；对 ESWA 读者不是最直接。

建议：优先删除或替换。若保留，正文必须改写为“other real-world PPO optimization problems”，不能让它承担 sensor scheduling 证据。

### 2.17 `Lim2021`

真实文档：

- Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister. “Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.” *International Journal of Forecasting* 37(4):1748–1764, 2021. DOI `10.1016/j.ijforecast.2021.03.012`。

本稿引用处：

- `sections/02_related_work.tex`：modern time-series models can score prediction consequences of a schedule。

原文证据：

> “we introduce the Temporal Fusion Transformer (TFT) – a novel attention-based architecture that combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics.”
>
> “TFT utilizes specialized components to select relevant features and a series of gating layers to suppress unnecessary components.”

是否支持本稿论据：

- 支持 modern multi-horizon forecasting model。
- 不直接支持 “score schedules”；这是本稿把 forecaster 用作 evaluator 的贡献。当前上下文若下一句说明 forecaster 被本稿冻结用于 reward/evaluation，则引用合理。

格式/作者校对：

- 元数据正确。

ESWA 适配性：

- 强 forecasting reference，保留。

建议：保留。

### 2.18 `Bai2018`

真实文档：

- Shaojie Bai, J. Zico Kolter, Vladlen Koltun. “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.” arXiv:1803.01271, 2018. DOI `10.48550/arXiv.1803.01271`。

本稿引用处：

- `sections/02_related_work.tex`：modern time-series models。
- `sections/04_framework_protocol.tex`：evaluation forecaster is a residual temporal convolutional network。

原文证据：

> “We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling.”
>
> “Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory.”
>
> “convolutional networks should be regarded as a natural starting point for sequence modeling tasks.”

是否支持本稿论据：

- 强支持 TCN / residual temporal convolutional forecaster 作为序列建模架构。
- 是本稿具体 forecaster architecture 的关键引用。

格式/作者校对：

- 元数据正确。

ESWA 适配性：

- 必须保留。虽是 arXiv，但已广泛使用；按 preprint 格式保留 DOI/eprint。

建议：保留。

### 2.19 `Liu2024`

真实文档：

- Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long. “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.” ICLR 2024 / OpenReview；arXiv:2310.06625。

本稿引用处：

- `sections/02_related_work.tex`：modern time-series models。

原文证据：

> “We propose iTransformer that simply applies the attention and feed-forward network on the inverted dimensions.”
>
> “the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations.”
>
> “The iTransformer model achieves state-of-the-art on challenging real-world datasets.”

是否支持本稿论据：

- 支持 modern time-series forecasting。
- 不是本稿实际 forecaster，所以不是核心引用；但用于 Related Work 合理。

格式/作者校对：

- 作者/题名正确。
- BibTeX 应补 OpenReview URL 和 arXiv eprint/DOI。ICLR 通常无页码，empty pages warning 可以接受，但 URL/eprint 应清楚。

ESWA 适配性：

- 近期高影响 forecasting 文献，保留。

建议：保留并补 metadata。

### 2.20 `AntAWS2023`

真实文档：

- Yetang Wang et al. “The AntAWS dataset: a compilation of Antarctic automatic weather station observations.” *Earth System Science Data* 15:411–429, 2023. DOI `10.5194/essd-15-411-2023`。

本稿引用处：

- `sections/02_related_work.tex`：Antarctic AWS networks provide long-term observations, but coverage uneven。
- `sections/05_simulation_setup.tex`：generator anchored to Antarctic AWS and blowing-snow statistics。

原文证据：

> “A new meteorological dataset derived from records of Antarctic automatic weather stations ... at 3 h, daily and monthly resolutions including quality control information is presented here.”
>
> “This dataset integrates the measurements of air temperature, air pressure, relative humidity, and wind speed and direction from 267 Antarctic AWSs obtained from 1980 to 2021.”
>
> “The AWS spatial distribution remains heterogeneous, with the majority of instruments located in near-coastal areas and only a few inland on the East Antarctic Plateau.”

是否支持本稿论据：

- 强支持 Antarctic AWS observation、long-term dataset、spatial heterogeneity。
- 不支持 particle counter / specialist blowing-snow sensors；这些需要 Amory/Monrad 等支撑。

格式/作者校对：

- 元数据正确。
- 当前 BibTeX 若用 `and others`，建议替换为完整作者列表，或至少确认 APA 渲染不会异常。

ESWA 适配性：

- 领域数据核心引用，必须保留。
- ESWA 要求引用和链接数据；如果 benchmark 数据来源依赖 AntAWS，应在 data availability 里对应说明。

建议：保留；补全作者列表更稳。

### 2.21 `Amory2020`

真实文档：

- Charles Amory. “Drifting-snow statistics from multiple-year autonomous measurements in Adélie Land, East Antarctica.” *The Cryosphere* 14:1713–1725, 2020. DOI `10.5194/tc-14-1713-2020`。

本稿引用处：

- `sections/02_related_work.tex`：blowing-snow variables and regimes。
- `sections/05_simulation_setup.tex`：generator anchored to blowing-snow statistics。

原文证据：

> “two remote locations in coastal Adélie Land (East Antarctica) ... were instrumented ... with meteorological and second-generation IAV Engineering acoustic FlowCapt™ sensors.”
>
> “This paper presents an assessment of drifting-snow occurrences and snow mass transport from up to 9 years (2010–2018) of half-hourly observational records.”

是否支持本稿论据：

- 强支持 Antarctic drifting/blowing-snow statistics、snow mass transport、FlowCapt 相关背景。

格式/作者校对：

- 元数据正确。

ESWA 适配性：

- 领域 grounding 必要。保留。

建议：保留。

### 2.22 `Sharma2023`

真实文档：

- Varun Sharma, Franziska Gerber, Michael Lehning. “Introducing CRYOWRF v1.0: multiscale atmospheric flow simulations with advanced snow cover modelling.” *Geoscientific Model Development* 16:719–749, 2023. DOI `10.5194/gmd-16-719-2023`。

本稿引用处：

- `sections/02_related_work.tex`：snow-atmosphere / blowing-snow regime background。
- `sections/05_simulation_setup.tex`：generator physical motivation。

原文证据：

> “CRYOWRF couples the state-of-the-art and widely used atmospheric model WRF ... with the detailed snow cover model SNOWPACK.”
>
> “Additionally, a new blowing snow scheme is introduced in CRYOWRF and is discussed in detail.”
>
> “Three case studies showcasing envisaged use cases for CRYOWRF for polar ice sheets and alpine snowpacks are provided.”

是否支持本稿论据：

- 支持 snow-atmosphere coupling / blowing snow modeling。
- 不直接支持 sensor scheduling；作为 domain/modeling background 合理。

格式/作者校对：

- 元数据正确。

ESWA 适配性：

- 领域支撑合理，保留。

建议：保留。

### 2.23 `Monrad2026`

真实文档：

- Lukas J. Monrad-Krohn, Christian Buhren, Laura Eickelmann, Andreas Foth, Markus M. Frey, Maximilian Maahn, Wenceslas Marie-Sainte, Mario Mech, Thomas Poinsot, Andreas Walbröl. “Measurements of blowing snow particle size distribution using an open-path optical snow particle counter in Ny-Ålesund during winter 2025.” PANGAEA dataset, 2026. DOI `10.1594/PANGAEA.992701`。

本稿引用处：

- `sections/02_related_work.tex`：particle microstructure / snow particle statistics。
- `sections/05_simulation_setup.tex`：generator anchored to blowing-snow statistics。

原文证据：

> “an open-path optical Snow Particle Counter ... measured particle size distributions of snow particles close to the snow surface in Ny-Ålesund from 31st January until 3rd March 2025.”
>
> “The detected particles, with diameters ranging from 36 to 500 µm, are sorted into 65 size bins.”
>
> “The snow mass flux ... is calculated by assuming spherical particles with an ice density of 917 kg m⁻³.”

是否支持本稿论据：

- 支持 particle-size distribution / snow mass flux measurement。
- 不支持 Antarctic；Ny-Ålesund 是 Arctic/Svalbard。正文必须避免把它并入 Antarctic evidence。

格式/作者校对：

- 当前 `author={Monrad-Krohn, L. J. and others}` 太粗略。ESWA 对 dataset citation 要求 author names、dataset title、repository、year、persistent identifier。
- 建议完整作者列表，并在 BibTeX note 或 type 中标识 `[dataset]`。

ESWA 适配性：

- 如果本稿需要支撑 particle counter / snow particle distribution，保留合理。
- 数据引用应符合 ESWA dataset reference 规则。

建议：保留但改为完整 dataset citation；正文避免 Antarctic 暗示。

### 2.24 `Aloni2024`

真实文档：

- Ofek Aloni, Gal Perelman, Barak Fishbain. “Synthetic random environmental time series generation with similarity control, preserving original signal’s statistical characteristics.” *Environmental Modelling & Software* 185:106283, 2025. DOI `10.1016/j.envsoft.2024.106283`。
- arXiv：`2502.02392`。

本稿引用处：

- `sections/05_simulation_setup.tex`：generator checks，包括 autocorrelation、event frequency、distribution agreement、durations、flux–wind scaling、particle correlation、spectra 等。

原文证据：

> “Synthetic datasets are widely used in many applications, such as missing data imputation, examining non-stationary scenarios, in simulations, training data-driven models, and analyzing system robustness.”
>
> “This paper presents a method, based on discrete Fourier transform, for generating synthetic time series with similar statistical moments for any given signal.”
>
> “Proof shows analytically that this method preserves the first two statistical moments of the input signal, and its autocorrelation function.”

是否支持本稿论据：

- 支持 synthetic environmental time series 和 statistical similarity / autocorrelation 检查。
- 不支持本稿具体的八项 acceptance checks；这些是本稿自己的 validation protocol。
- 若正文让读者以为 Aloni et al. 是本稿 exact protocol 来源，则过强；当前更合适的写法是“in the spirit of statistical-similarity validation for synthetic environmental time series”。

格式/作者校对：

- 作者/题名/DOI 正确。
- Bib key `Aloni2024` 和 year 2025 不一致不影响渲染，但可改 key 为 `Aloni2025` 以减少维护混乱。

ESWA 适配性：

- 环境模拟和 synthetic time series 文献，适合保留。

建议：保留；如果改正文，避免让它承担 exact validation protocol 的责任。

## 3. 建议新增文献

### 3.1 强烈建议新增：invalid action masking

候选文献：

- Shengyi Huang, Santiago Ontañón. “A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.” *International FLAIRS Conference Proceedings* 35, 2022. DOI `10.32473/flairs.v35i.130584`；arXiv `2006.14171`。

原文证据：

> “The usual approach to deal with this problem in policy gradient algorithms is to ‘mask out’ invalid actions and just sample from the set of valid actions.”
>
> “we 1) show theoretical justification for such a practice, 2) empirically demonstrate its importance as the space of invalid actions grows...”

为什么要加：

- 当前 `Schulman2017` 只支撑 PPO，不支撑 masked action distribution。
- 本稿有 feasibility mask / candidate channel masks / masked PPO policy。Huang & Ontañón 是直接证据。

建议引用位置：

- Introduction 中 “masked PPO policy over candidate channel masks” 后面：`\citep{Schulman2017,Huang2022InvalidMasking}`。
- 或 Method 中第一次定义 feasibility mask / masked categorical policy 的地方。

ESWA 适配性：

- FLAIRS 会议，不是 ESWA，但 methodological justification 很直接；比泛泛再加一个 DRL scheduling paper 更有用。

### 3.2 可选新增：ESWA 目标期刊桥接文献

已核验到的 ESWA 近邻但非 sensor-specific 文献：

1. Kun Lei et al. “A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem.” *Expert Systems with Applications* 205:117796, 2022. DOI `10.1016/j.eswa.2022.117796`。
2. Samuel Yanes Luis et al. “Variational model-based Deep Reinforcement Learning for Non-Homogeneous Patrolling aquatic environments with multiple unmanned surface vehicles.” *Expert Systems with Applications* 270:126483, 2025. DOI `10.1016/j.eswa.2025.126483`。
3. Peng Song et al. “Quantization-aware distributed deep reinforcement learning for dynamic multi-robot scheduling.” *Expert Systems with Applications* 296:129027, 2026. DOI `10.1016/j.eswa.2025.129027`。
4. Yong Lei et al. “Deep reinforcement learning for dynamic distributed job shop scheduling problem with transfers.” *Expert Systems with Applications* 251:123970, 2024. DOI `10.1016/j.eswa.2024.123970`。

判断：

- 这些文献可以证明 ESWA 接受 DRL + scheduling/resource allocation 类问题。
- 但它们不是 sensor scheduling 或 forecast-driven acquisition。不能用来替代 `Murad2020`, `Alali2024`, `Ogbodo2025` 这类近邻文献。
- 如果要增强 ESWA fit，建议最多加 1 条，且正文应明确写成 “DRL scheduling has also been studied in ESWA-style applied scheduling domains”，不要把它包装成 sensor scheduling precedent。

推荐优先级：

- 若只加一条 ESWA bridge：优先 `Lei2022`，因为它是明确 DRL scheduling，年份稳定、正式发表。
- 若强调 sensing/patrolling/resource collection：可考虑 `YanesLuis2025`，但需要先看全文确认其 sensing/patrolling目标是否与本稿叙事匹配。
- 不建议为了“贴 ESWA”堆 2026 future/online-first 文献。

## 4. 建议删除或替换

建议优先删除/替换：

1. `Pendyala2024`
   - 删除理由：非 sensor / sensing / path planning；只支撑 PPO real-world optimization。
   - 如果保留，需要改正文措辞，不要让它支撑 adaptive sensing。

2. `Tran2026`
   - 删除理由：arXiv-only、2026、与本稿目标不同；如果 citation pile 要瘦身，这是比 `Jonah2026` 更容易删的一条。
   - 保留理由：它直接用了 PPO scheduler 和 sensor scheduling，若要强调最新工作，可保留。

3. `Qu2022`
   - 删除理由：MDPI Sensors、multi-target tracking-specific，在当前 8 文献堆里不是必要。
   - 保留理由：Q-learning sensor scheduling 明确，若需要 classical RL sensor scheduling 可保留。

4. `FernandezBes2015`
   - 不是内容问题，而是 DOI 问题。若不修 DOI/URL，建议删；修好后可保留。

不建议删除：

- `Schulman2017`, `Bai2018`, `AntAWS2023`, `Amory2020`, `Murad2020`, `Alali2024`, `Ogbodo2025`, `Shi_2011`, `Kaul_2012`。

## 5. 建议重写的 citation clusters

### 5.1 Related Work 第一段

当前问题：一句话混合 covariance、information gain、censoring utility、AoI、learning-based scheduling，括号里 8 个 citation。

建议拆法：

- Estimation / covariance / filtering:
  `\citep{Shi_2011,AlAhdab2025}`
- AoI/AoII/freshness:
  `\citep{Kaul_2012,Jonah2026}`
- MDP/censoring / energy harvesting:
  `\citep{FernandezBes2015}`，前提是 DOI/URL 修好。
- Learning-based sensor scheduling:
  `\citep{Alali2024,Tran2026}` 或只保留 `Alali2024`。

可执行改写方向：

> Earlier sensor-scheduling work often optimizes estimation error or filtering uncertainty under communication and energy constraints \citep{Shi_2011,AlAhdab2025}. Other formulations emphasize freshness or censoring decisions, including age-based status-update metrics and MDP models for energy-harvesting sensors \citep{Kaul_2012,FernandezBes2015,Jonah2026}. Recent learning-based schedulers cast monitoring and remote-estimation problems as reinforcement-learning tasks \citep{Alali2024,Tran2026}.

如果删 `Tran2026`，最后一组改为 `\citep{Alali2024}` 或 `\citep{Alali2024,Qu2022}`。

### 5.2 DRL adaptive sensing cluster

当前：

> “Deep reinforcement learning has been used for adaptive sensing, informative path planning, and other combinatorial scheduling problems \citep{Murad2020,Wei2020,Ogbodo2025,Pendyala2024}.”

建议：

- 若删 `Pendyala2024`：
  > Deep reinforcement learning has been used for adaptive sensing, informative path planning, and dynamic data acquisition \citep{Murad2020,Wei2020,Ogbodo2025}.

- 若保留 `Pendyala2024`：
  > Deep reinforcement learning has been used for adaptive sensing, informative path planning, dynamic data acquisition, and other real-world PPO optimization problems \citep{Murad2020,Wei2020,Ogbodo2025,Pendyala2024}.

### 5.3 Forecasting model cluster

当前：

> “Modern time-series models provide a direct way to score the prediction consequences of a schedule \citep{Lim2021,Bai2018,Liu2024}.”

建议：

- 内容可保留，但“provide a direct way to score schedules”是本稿使用方式，不是这三篇共同论点。
- 更稳的表述：
  > Modern time-series models provide accurate multi-horizon sequence forecasts \citep{Lim2021,Bai2018,Liu2024}. This paper uses a frozen forecaster as a common evaluator of sensing schedules.

### 5.4 Domain cluster

当前：

> “Blowing-snow studies use variables whose predictive value changes with storm conditions, near-surface transport, and particle microstructure \citep{Amory2020,Sharma2023,Monrad2026}.”

建议：

- 避免暗示 Monrad 是 Antarctic：
  > Antarctic drifting-snow records and polar snow models document event-dependent transport regimes \citep{Amory2020,Sharma2023}, while particle-counter datasets provide size-distribution and flux measurements for blowing-snow studies \citep{Monrad2026}.

## 6. BibTeX 修复清单（不直接修改，只给建议）

高优先级：

- `FernandezBes2015`
  - 删除或修正 `doi={10.1109/JSAC.2015.2430512}`。
  - 添加 `url={https://ieeexplore.ieee.org/document/7009961}`。

- `Golovin2011`
  - 若 DOI 提交前仍不 resolve，删除 DOI。
  - 添加 `url={https://www.jair.org/index.php/jair/article/view/10731}`。
  - 可添加 `eprint={1003.3967}`, `archivePrefix={arXiv}`。

- `Murad2020`
  - 添加 `pages={1--8}`。

- `Wei2020`
  - 添加 `pages={864--873}`。

- `Pendyala2024`
  - 添加 `pages={150--165}`。
  - 添加/确认 `volume={14950}`。

- `Schulman2017`
  - 添加 `eprint={1707.06347}`。
  - 添加 `archivePrefix={arXiv}`。
  - 添加 `primaryClass={cs.LG}`。
  - 添加 `doi={10.48550/arXiv.1707.06347}`。
  - 添加 `url={https://arxiv.org/abs/1707.06347}`。

- `Liu2024`
  - 添加 `url={https://openreview.net/forum?id=JePfAI8fah}`。
  - 添加 `eprint={2310.06625}`。
  - 添加 `archivePrefix={arXiv}`。
  - 添加 `doi={10.48550/arXiv.2310.06625}`。

- `Tran2026`
  - 添加 `eprint={2601.21482}`。
  - 添加 `archivePrefix={arXiv}`。
  - 添加 `primaryClass={cs.IT}`。
  - 添加 `doi={10.48550/arXiv.2601.21482}`。
  - 添加 `url={https://arxiv.org/abs/2601.21482}`。
  - 若删除该引用，则同时删 BibTeX 或留作 unused 条目。

- `Monrad2026`
  - 完整作者列表。
  - 按 ESWA dataset reference 加 `[dataset]` 标识，例如 `note={[dataset] PANGAEA dataset}` 或在 entry type / title 中处理，具体取决于 BibTeX style 渲染。

- `AlAhdab2025`
  - 作者名建议改成 PMLR 官方写法 `John Leth`。

中优先级：

- `AntAWS2023`：把 `and others` 展开为完整作者列表。
- `Ogbodo2025`：统一期刊名全称/缩写。
- 全库统一 journal title 风格：当前混合了全称、缩写、首字母大小写；ESWA/Elsevier 可接受，但最终 PDF 观感会不一致。

## 7. 附录 A：当前 unused BibTeX 条目

这些条目在 `paper/references.bib` 中存在，但不被 `paper/main.tex` 及递归 include 文件引用。

| Key | 本地文档 | 建议 |
|---|---|---|
| `VanHasselt2016` | Double DQN, AAAI 2016 | 若正文不讨论 DQN baseline，删除。 |
| `Wang2016` | Dueling DQN, ICML 2016 | 若正文不讨论 dueling DQN，删除。 |
| `DeLaFuente2024` | DQN/PPO/A2C comparative arXiv | 泛泛 RL 比较，当前不用，删除。 |
| `Ibrahim2024` | Reward engineering/shaping overview | 如果正文讨论 reward shaping 可加；否则删除。 |
| `Ying2022` | CMDP entropy-regularized dual approach | 当前方法没有以 CMDP 理论展开，删除。 |
| `Chen2026` | Primal-dual CMDP, Management Science | 同上；若重引需先核对年份/DOI。 |
| `Wang2021` | East Antarctic sensor environment ML prediction | 可用于 polar ML 背景，但当前不用。 |
| `Ding2025` | Antarctic SAT reconstruction, Scientific Data | 可用于 Antarctic ML/data 背景，但当前不用。 |
| `Wang2025` | multivariate time-series imputation survey | 当前稿件不是 imputation paper，删除。 |
| `OTT2022` | Parsivel manual | 若 sensor spec table 需仪器依据，可重引；否则删除。 |
| `IAV2024` | FlowCapt FC4 product description | 若正文/表格声称 FC4 specs，可重引；否则删除。 |
| `GillGMX500` | MaxiMet GMX500 station manual | 同上。 |
| `SensecaLPS10` | pyranometer datasheet | 同上。 |
| `ApogeeSI111` | infrared radiometer manual | 同上。 |

如果论文定位为“真实传感器配置驱动的 benchmark”，建议重新引用相关 manual 来支撑 sensor table；如果定位为“抽象 schedulable benchmark”，删除 unused manuals 更干净。

## 8. 最终建议路线

最小改动版：

1. 修 `FernandezBes2015` 和 `Golovin2011` 的 DOI/URL。
2. 补 `Murad2020`, `Wei2020`, `Pendyala2024` 页码。
3. 补 `Schulman2017`, `Liu2024`, `Tran2026` 的 arXiv/URL/DOI。
4. `Monrad2026` 改成 dataset-style citation，并修正文案避免 Antarctic 暗示。
5. Related Work 第一段拆 citation pile。
6. 新增 Huang & Ontañón (2022) 支撑 masked PPO。

更强 ESWA 投稿版：

1. 删除或弱化 `Pendyala2024`。
2. 删除 `Tran2026` 或把它明确标为 arXiv-only recent work。
3. 在 ESWA bridge 上最多新增 1 条正式 ESWA DRL scheduling 文献，例如 Lei et al. (2022), DOI `10.1016/j.eswa.2022.117796`，但只用于说明 DRL scheduling 在 ESWA 应用问题中的接受度，不当作 sensor scheduling 直接竞品。
4. 清理 unused 14 条 BibTeX，或者有选择地把 sensor manuals 重新引用到 sensor specification table。

## 9. 证据文件

机器可读抽取与联网元数据位于：

- `.planning/2026-07-02-reference-audit/citation_extraction.json`
- `.planning/2026-07-02-reference-audit/online_metadata_raw_24.json`
- `.planning/2026-07-02-reference-audit/online_metadata_summary_24.json`

本报告没有直接修改 `paper/references.bib` 或正文引用；它是审计与修改建议。