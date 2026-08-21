# 参考文献全网校对核验报告

## 核验说明

本次核验对象为你上传的论文 PDF《Reinforcement learning with forecast-loss rewards for sensor scheduling under power constraints》。从正文可见，文献引用主要集中在 Section 2 的背景与相关工作、Section 4 对 forecaster 架构的说明、Section 5 对生成器校验的说明；参考文献表位于 PDF 末尾，共 23 条。fileciteturn0file1

本报告按三条主线执行：其一，逐条核验参考文献的真实性、作者、题名、年份、卷期页、DOI 或 URL、文献类型与发表状态；其二，逐个核验正文中每处引用论点是否获得原文直接支持、间接支持，或存在错引/过度外推；其三，给出统一化、可直接回填到稿件中的标准化参考文献建议稿。对能访问到出版社/官方数据库全文或摘要页的条目，我优先使用官方页面；对只能访问 arXiv、PMLR、OpenReview、机构知识库、PANGAEA、Copernicus、IEEE/Elsevier/Springer 元数据页的条目，我只依据这些可核验内容下结论，不做超范围推断。citeturn14search1turn0search3turn1search1turn0search1turn3search1turn14search0turn2search7turn15search0turn9search7turn3search0turn5search3turn12search0turn4search0turn15search4turn6search2turn16search3turn8search0turn10search2turn8search2turn15search2turn10search0turn15search1

## 总体结论

结论先行：23/23 条参考文献均能在权威学术来源中找到真实对应记录；当前参考文献表里**没有发现伪造文献**，但存在**1 处明确 DOI 错误**、**2 处发表状态/文献类型标注不足**、**2 处书目信息不完整**、以及**数处正文论点表述强于原文证据**的问题。最严重的问题是 Fernández-Bes et al. 2015 的 DOI 写错了，稿中写为 `10.1109/JSAC.2015.2430512`，官方记录显示正确 DOI 为 `10.1109/JSAC.2015.2391792`。此外，Tran et al. 2026 目前是 arXiv 预印本，稿中未明确标注预印本状态且漏掉了 arXiv DOI；Monrad-Krohn et al. 2026 实际是 PANGAEA 数据集记录，不是期刊论文；Wei and Zheng 2020 缺失页码；Pendyala et al. 2024 的书目信息缺少完整书名与页码。citeturn14search0turn14search2turn15search2turn4search0turn15search1turn16search3

正文论点层面，背景文献中关于“资源受限调度目标”“主动感知/POMDP”“DRL 已用于自适应感知与路径规划”的三组引用，整体上是成立的；但“现代时间序列模型可直接用于评分调度后果”和“吹雪研究中的变量预测价值会随工况变化”这两组表述，原始文献更多提供的是**模型家族或应用背景**，稿件当前写法属于**作者综合推断**，建议收窄措辞，避免被审稿人认为“引用支持强度不足”。fileciteturn0file1 citeturn5search3turn2search3turn12search0turn10search0turn0search1turn10search2turn4search0

## 参考文献逐条核验

### 调度与状态估计相关文献

**Ahdab, Leth, Tan 2025**：真实存在，正式发表于 ICML 2025 的 PMLR 卷 267，页码 703–729。官方记录给出的作者标准写法是 *Mohamad Al Ahdab, John-Josef Leth, Zheng-Hua Tan*；稿件写成 “M.A., J.J., Z.H.” 不算事实错误，但如果要做严格标准化，建议把 Leth 记为 **J.-J.** 或按官方全名回填。PMLR 页面未给出 DOI，当前保留 URL 是合理的。citeturn14search1turn14search3

**Alali, Kazeminajafabadi, Imani 2024**：真实存在，Taylor & Francis 正式发表，期刊为 *Systems Science & Control Engineering*，卷 12，文章号 2329260，DOI 有效。作者、题名、年份、DOI 均正确。citeturn0search2turn0search3

**Fernández-Bes, Cid-Sueiro, Marques 2015**：真实存在，正式发表于 *IEEE Journal on Selected Areas in Communications*，33(8): 1717–1729。稿件**DOI 明确错误**；官方记录显示正确 DOI 为 **10.1109/JSAC.2015.2391792**。这条必须改。作者顺序与题名正确。citeturn14search0turn14search2turn14search4

**Shi, Cheng, Chen 2011**：真实存在，正式发表于 *Automatica*，47(8): 1693–1698，DOI 正确。稿件条目信息基本无误。citeturn8search2turn8search4

**Kaul, Yates, Gruteser 2012**：真实存在，为 INFOCOM 2012 会议论文，页码 2731–2735，DOI 正确。稿件当前条目基本正确。citeturn9search7turn9search3turn9search6

**Qu, Zhao, Xu, Tang, Wang, Li 2022**：真实存在，正式发表于 *Sensors*，22(18): 6972，DOI 正确。作者、题名、年份、DOI 均正确。citeturn8search0

**Jonah, Yoo, Sthapit 2026**：真实存在，正式发表于 *IEEE Access*，卷 14，页码 40042–40059，DOI 正确。稿件作者缩写为 “S.” 略显宽泛，官方标准缩写更接近 **S. S. Jonah**；建议按官方篇目统一。citeturn15search0

**Tran, Mahmood, Gidlund 2026**：真实存在，但当前状态是 **arXiv 预印本**，不是已正式发表期刊论文。稿件没有明确“preprint / arXiv”状态，同时漏掉了 arXiv DOI **10.48550/arXiv.2601.21482**。这一条需要把文献状态写清楚。citeturn15academia48turn15search2

### 主动感知、POMDP 与 DRL 相关文献

**Bajcsy, Aloimonos, Tsotsos 2018**：真实存在，正式发表于 *Autonomous Robots*，42: 177–196，DOI 正确。注意该文在线发表时间是 2017，但卷页归属 2018；稿件采用 2018 是可以接受的。citeturn3search1

**Lauri, Hsu, Pajarinen 2023**：真实存在，正式发表于 *IEEE Transactions on Robotics*，39(1): 21–40，DOI 正确。稿件条目可信。citeturn3search0

**Golovin, Krause 2011**：真实存在，正式发表于 *Journal of Artificial Intelligence Research*，42: 427–486，DOI 正确。稿件信息正确。citeturn2search7

**Murad, Kraemer, Bach, Taylor 2020**：真实存在，为 ACM IoT’20 会议论文，DOI 正确。稿件条目基本正确；若按更完整格式，可补会议届次与出版社信息。citeturn15academia49turn15search4

**Wei, Zheng 2020**：真实存在，为 IEEE INFOCOM 2020 会议论文，DOI 正确；官方元数据显示页码为 **864–873**，稿件当前**缺页码**。应补全。citeturn15search1

**Ogbodo, Rogers, Dal Borgo, Wagg 2026**：真实存在，正式发表于 *Proceedings of the Royal Society A*，482(2329): 20250326，DOI 正确。稿件目前只写 “Proc. R. Soc. A 482, 20250326” 尚可识别，但建议补上期号 **2329** 以及期刊全名，以提升规范性。citeturn6search2turn6search5

**Pendyala, Atamna, Glasmachers 2024**：真实存在，已收录于 Springer 出版的 *Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track*，页码 **150–165**，DOI **10.1007/978-3-031-70381-2_10**。稿件当前写成“Proc. ECML/PKDD 2024, Springer”过于粗略，建议改成书章/会议论文标准格式并补页码。citeturn16search1turn16search3turn16search0

### 预测模型、仿真与南极观测相关文献

**Bai, Kolter, Koltun 2018**：真实存在，但其状态是 **arXiv technical report / preprint**，不是期刊论文。稿件写成 “arXiv preprint arXiv:1803.01271” 是事实正确的；不过末尾同时写 DOI 和 arXiv 号两次，有重复。可精简。citeturn2search3turn2search0

**Lim, Arık, Loeff, Pfister 2021**：真实存在，正式发表于 *International Journal of Forecasting*，37(4): 1748–1764，DOI 正确。稿件条目正确。citeturn5search3turn5search0

**Liu et al. 2024**：真实存在，发表于 ICLR 2024；ICLR proceedings 与作者代码仓库都显示其为 **ICLR 2024 Spotlight**。稿件当前作为会议论文引用是成立的。官方 proceedings 页面未显示 DOI；用 ICLR 2024 proceedings URL 是合理的。citeturn12search0turn12search2turn12search5

**Aloni, Perelman, Fishbain 2025**：真实存在，正式发表于 *Environmental Modelling & Software*，185: 106283，DOI 正确。稿件信息正确。citeturn1search1

**Wang et al. 2023**：真实存在，正式发表于 *Earth System Science Data*，15: 411–429，DOI 正确。该文作者名单较长，稿件使用 “et al.” 可以识别，但若按严格样式要求，需要遵守目标期刊对长作者列表的截断规则。citeturn10search0turn10search1

**Amory 2020**：真实存在，正式发表于 *The Cryosphere*，14: 1713–1725，DOI 正确。稿件条目正确。citeturn0search0turn0search1

**Sharma, Gerber, Lehning 2023**：真实存在，正式发表于 *Geoscientific Model Development*，16: 719–749，DOI 正确。稿件条目正确。citeturn10search2

**Monrad-Krohn et al. 2026**：真实存在，但文献类型是 **PANGAEA 数据集记录**，不是传统期刊论文。稿件目前把它列在参考文献里是可接受的，但应显式标注 **[dataset]**，并最好保留平台名 PANGAEA、发布日期与 DOI。稿件当前 “et al.” 省略了大量作者，风格上是否允许取决于期刊，但从严格核验角度看，这属于“作者信息未完整展示”。citeturn4search0turn4search1

## 正文引文论点匹配核验

### 资源受限调度目标这一组引用

稿件在背景部分写道：资源约束下的传感器调度历史悠久，代表性目标包括协方差降低、信息增益、censoring utility 与 AoI，并在一句中合并引用 Shi、Kaul、Fernández-Bes、Qu、Alali、Ahdab、Jonah、Tran。该引用位置位于 Section 2.1。fileciteturn0file1

这一组总体上**成立**。可核验的原文支持如下：Shi 的摘要明确写到“**minimize the estimation error**”；Kaul 的摘要写到“**time-average age metric**”；Fernández-Bes 的摘要写到“**importance (utility) of all transmitted messages**”；Ahdab 的摘要写到“**posterior covariance matrix**”；Jonah 的摘要写到传感器调度应用“**based on the Age of Incorrect Information**”；Tran 的摘要则写明其目标为“**information-per-joule scheduling objective beyond age of information proxies**”。因此，这一句作为“该领域存在多类优化目标”的背景性综述是有支撑的。需要收紧的是**精确归属**：Qu 更偏向 multi-target tracking 中的 Q-learning 调度，Alali 更偏 HMM belief-space 下的 RL sensor scheduling；若保留这一长串引文，最好把各类目标与代表文献一一对应，而不要让审稿人误以为每一篇都同时支持“协方差、信息增益、censoring、AoI”四类目标。citeturn8search2turn9search7turn14search0turn8search0turn0search3turn14search1turn15search0turn15academia48

### 主动感知与 POMDP 这一组引用

稿件称该问题“connects to active perception and partially observable decision making, where observations are actions selected for a future task”，并引 Bajcsy、Lauri、Golovin and Krause。该处位于 Section 2.2。fileciteturn0file1

这组引用**直接支持度较高**。Bajcsy 的摘要直接强调“**a complete artificial agent necessarily must include active perception**”；Lauri 的综述指出 “**POMDP provides a principled mathematical framework**” 来刻画不确定机器人决策；Golovin and Krause 的摘要则明确讨论“**sequence of decisions with uncertain outcomes under partial observability**”。因此，把当前工作放到主动感知与部分可观测决策框架下，是有文献依据的。稿件里“observations are actions selected for a future task”这句，更严格地说是对该文献线索的归纳性转述，不是三篇文献逐字共同给出的统一定义，但不构成错引。citeturn3search1turn3search0turn2search7

### DRL 自适应感知与路径规划这一组引用

稿件称“Deep reinforcement learning has been used for adaptive sensing, informative path planning, and other combinatorial scheduling problems”，并引 Murad、Wei、Ogbodo、Pendyala。该处位于 Section 2.2。fileciteturn0file1

这组引用**基本成立，但内部支持强度不等**。Murad 的摘要明确是“**sensing policies for resource-constrained IoT devices**”；Wei 的摘要写明“**informative path planning ... using reinforcement learning**”；Ogbodo 的摘要为“**sensor steering methodology based on deep reinforcement learning**”。这三篇都能直接支撑“adaptive sensing / path planning”。Pendyala 的论文则是 PPO 在**真实工业优化问题**上的应用，摘要里写的是“**optimize a real-world high-throughput waste sorting facility**”，它可以支撑稿中的“other combinatorial scheduling problems”，但**不宜被读成 sensing precedent**。如果你要降低被质疑风险，建议把这句改成“DRL has been used in adaptive sensing, informative path planning, and related real-world scheduling/optimization problems”。citeturn15academia49turn15search1turn6search2turn13academia29turn16search1

### 预测模型作为调度评分器这一组引用

稿件称“Modern time-series models provide a direct way to score the prediction consequences of a schedule”，并引 Lim、Bai、Liu。该处位于 Section 2.3。fileciteturn0file1

这里存在**明显的“表述强于证据”**。Lim 的 TFT 文章确实是多步时间序列预测模型；Bai 的文章是泛序列建模中的 TCN 架构比较；Liu 的 iTransformer 是时间序列预测骨干模型。但这三篇文献本身都**没有研究“调度诱导的部分观测序列如何被统一 forecaster 评分”**。它们支持的是“可以选用这些 forecasting backbones 来做 downstream scoring”，支持不到“直接提供评分调度后果的方法论”。因此这一句不算错引，但属于**作者推论型引用**。推荐改写为：“Modern forecasting architectures such as TFT, TCN-based models, and iTransformer provide candidate backbones for evaluating schedule-induced observation streams.” 这样就与原文证据强度一致了。citeturn5search3turn2search3turn12search0turn11academia38

### 南极 AWS 与吹雪应用背景这一组引用

稿件称 Antarctic AWS 网络覆盖“uneven in space, season, and measurement type”，并进一步说吹雪研究中某些变量的“predictive value changes with storm conditions, near-surface transport, and particle microstructure”，引 Wang、Amory、Sharma、Monrad-Krohn。该处位于 Section 2.4。fileciteturn0file1

其中前半句对 Wang 2023 是**直接支持**：官方摘要明确写到 AWS spatial distribution “**remains heterogeneous**”，且多数站点集中在近海区域。后半句则是**领域综合判断**，不是四篇文献任何一篇的直述结论。Amory 提供的是长期 drifting-snow occurrence 与 mass transport 统计；Sharma 提供的是带先进 snow cover modelling 的多尺度大气流模拟平台；Monrad-Krohn 提供的是近地表 blowing-snow 粒径分布与 snow mass flux 数据集。它们合起来能够支撑“粒子、输运、热力相关变量在吹雪监测中很重要”，但“predictive value changes ...”是作者概括，建议改成更保守的说法，例如：“Blowing-snow monitoring relies on variables linked to transport conditions, near-surface fluxes, and particle microphysics.” 这样更贴近原文证据。citeturn10search0turn0search1turn10search2turn4search0

### Bai 2018 在方法部分的使用

稿件在方法部分写明 frozen forecaster 是 “a residual temporal convolutional network (Bai et al., 2018) with three levels, 64 channels per level, kernel size 3, and dropout rate 0.05”。该处位于 Section 4.3。fileciteturn0file1

这条引用**部分直接、部分不直接**。Bai 2018 的原文确实提出并系统评估了 TCN 家族，摘要与正文都能支持“residual temporal convolutional network / TCN”这一架构来源；原文还强调 causal convolution、dilated convolution、residual layers 等关键设计。但稿件中的“三层、64 通道、kernel size 3、dropout 0.05”是你们自己的具体超参数，不是 Bai 2018 的固定推荐值。因此该引用可保留，但建议把句子改成：“The frozen forecaster uses a TCN-style residual temporal convolutional architecture inspired by Bai et al. (2018)...” 这样不会给人“这些超参数出自 Bai 2018”的误解。citeturn2search3turn2search0

### Aloni 2025 在生成器校验部分的使用

稿件在 Section 5.2 说生成器校验覆盖风速自相关、分布一致性等，并引 Aloni et al. 2025。fileciteturn0file1

这条引用属于**方法学层面的间接支持**。Aloni 的摘要明确写到其合成时间序列方法“**preserves the first two statistical moments and the autocorrelation function**”，这可以支撑你们用统计特征保持与相似性控制来设计生成器校验思想；但它并**不直接给出**你们稿件中的那 8 个 acceptance checks，也不对应南极吹雪专门场景。所以这里建议把引用语气改成“in line with statistics-preserving synthetic environmental time-series generation work”之类，而不是让读者以为 8 个 checks 出自 Aloni 本文。citeturn1search1

## 高风险问题与修正建议

最需要优先修的只有五处，但每一处都值得在正式投稿前处理干净。

第一，**Fernández-Bes et al. 2015 的 DOI 必须更正**，这是确定性错误。保留错误 DOI 会直接影响检索与审稿信任。正确 DOI 是 `10.1109/JSAC.2015.2391792`。citeturn14search0turn14search2

第二，**Tran et al. 2026 必须显式标注为预印本**。当前它仍是 arXiv 记录，稿件若把它排成正式发表论文，会被视为文献状态不清。建议写成 “arXiv preprint arXiv:2601.21482. doi:10.48550/arXiv.2601.21482”。citeturn15academia48turn15search2

第三，**Monrad-Krohn et al. 2026 的文献类型应改为 dataset**。当前标题与 DOI 都是真实的，但本质上是 PANGAEA 数据集记录，建议在条目末尾显式加 `[dataset]` 或按期刊数据引用规范重写。citeturn4search0turn4search1

第四，**Pendyala 2024 与 Wei 2020 的条目信息不完整**。Pendyala 缺完整书名和页码 150–165；Wei 缺页码 864–873。会议论文在 ESWA 这类期刊投稿里通常会被审稿人顺手检查，建议一次性补全。citeturn16search3turn16search1turn15search1

第五，正文里有两处建议收窄措辞：一个是 forecast models 作为“直接评分方法”的表述，一个是 Antarctic/blowing-snow 变量“predictive value changes” 的表述。这两处并非造假引用，但都存在**论证强度大于原始证据强度**的问题。fileciteturn0file1 citeturn5search3turn2search3turn12search0turn10search0turn0search1turn10search2turn4search0

## 标准化参考文献建议稿

下面给出一版可直接回填的、较为统一的参考文献建议稿。为避免与目标期刊最终样式冲突，我采用“作者—年份—题名—来源—卷页/文章号—DOI/URL—状态说明”的保守写法；若你后续使用目标期刊官方 BibTeX/EndNote 样式，可再做机械转换。

1. Ahdab, M. A., Leth, J.-J., & Tan, Z.-H. (2025). *Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics*. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR, Vol. 267, pp. 703–729). URL retained; no DOI shown on official PMLR record. citeturn14search1turn14search3

2. Alali, M., Kazeminajafabadi, A., & Imani, M. (2024). Deep reinforcement learning sensor scheduling for effective monitoring of dynamical systems. *Systems Science & Control Engineering*, 12, 2329260. https://doi.org/10.1080/21642583.2024.2329260 citeturn0search2turn0search3

3. Aloni, O., Perelman, G., & Fishbain, B. (2025). Synthetic random environmental time series generation with similarity control, preserving original signal’s statistical characteristics. *Environmental Modelling & Software*, 185, 106283. https://doi.org/10.1016/j.envsoft.2024.106283 citeturn1search1

4. Amory, C. (2020). Drifting-snow statistics from multiple-year autonomous measurements in Adélie Land, East Antarctica. *The Cryosphere*, 14, 1713–1725. https://doi.org/10.5194/tc-14-1713-2020 citeturn0search0turn0search1

5. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint* arXiv:1803.01271. https://doi.org/10.48550/arXiv.1803.01271 citeturn2search3turn2search0

6. Bajcsy, R., Aloimonos, Y., & Tsotsos, J. K. (2018). Revisiting active perception. *Autonomous Robots*, 42, 177–196. https://doi.org/10.1007/s10514-017-9615-3 citeturn3search1

7. Fernández-Bes, J., Cid-Sueiro, J., & García-Marques, A. (2015). An MDP model for censoring in harvesting sensors: Optimal and approximated solutions. *IEEE Journal on Selected Areas in Communications*, 33(8), 1717–1729. https://doi.org/10.1109/JSAC.2015.2391792 citeturn14search0turn14search2turn14search4

8. Golovin, D., & Krause, A. (2011). Adaptive submodularity: Theory and applications in active learning and stochastic optimization. *Journal of Artificial Intelligence Research*, 42, 427–486. https://doi.org/10.1613/jair.3278 citeturn2search7

9. Jonah, S. S., Yoo, S., & Sthapit, S. (2026). Adaptive Scheduling: A Reinforcement Learning Whittle Index Approach for Wireless Sensor Networks. *IEEE Access*, 14, 40042–40059. https://doi.org/10.1109/ACCESS.2026.3673220 citeturn15search0

10. Kaul, S. K., Yates, R. D., & Gruteser, M. (2012). Real-time status: How often should one update? In *2012 Proceedings IEEE INFOCOM* (pp. 2731–2735). IEEE. https://doi.org/10.1109/INFCOM.2012.6195689 citeturn9search7turn9search3

11. Lauri, M., Hsu, D., & Pajarinen, J. (2023). Partially observable Markov decision processes in robotics: A survey. *IEEE Transactions on Robotics*, 39(1), 21–40. https://doi.org/10.1109/TRO.2022.3200138 citeturn3search0

12. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012 citeturn5search3turn5search0

13. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. In *International Conference on Learning Representations 2024*. URL retained from official proceedings. citeturn12search0turn12search5

14. Monrad-Krohn, L. J., Buhren, C., Eickelmann, L., Foth, A., Frey, M. M., Maahn, M., Marie-Sainte, W., Mech, M., Poinsot, T., & Walbröl, A. (2026). *Measurements of blowing snow particle size distribution using an open-path optical snow particle counter in Ny-Ålesund during winter 2025* [dataset]. PANGAEA. https://doi.org/10.1594/PANGAEA.992701 citeturn4search0turn4search1

15. Murad, A., Kraemer, F. A., Bach, K., & Taylor, G. (2020). Information-driven adaptive sensing based on deep reinforcement learning. In *Proceedings of the 10th International Conference on the Internet of Things* (IoT’20). ACM. https://doi.org/10.1145/3410992.3411001 citeturn15academia49turn15search4

16. Ogbodo, C. O., Rogers, T. J., Dal Borgo, M., & Wagg, D. J. (2026). Adaptive sensor steering strategy using deep reinforcement learning for dynamic data acquisition in digital twins. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 482(2329), 20250326. https://doi.org/10.1098/rspa.2025.0326 citeturn6search2turn6search5

17. Pendyala, A., Atamna, A., & Glasmachers, T. (2024). Solving a Real-World Optimization Problem Using Proximal Policy Optimization with Curriculum Learning and Reward Engineering. In *Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track* (ECML PKDD 2024, pp. 150–165). Springer Nature Switzerland. https://doi.org/10.1007/978-3-031-70381-2_10 citeturn16search1turn16search3turn16search0

18. Qu, Z., Zhao, X., Xu, H., Tang, H., Wang, J., & Li, B. (2022). An improved Q-learning-based sensor-scheduling algorithm for multi-target tracking. *Sensors*, 22(18), 6972. https://doi.org/10.3390/s22186972 citeturn8search0

19. Sharma, V., Gerber, F., & Lehning, M. (2023). Introducing CRYOWRF v1.0: multiscale atmospheric flow simulations with advanced snow cover modelling. *Geoscientific Model Development*, 16, 719–749. https://doi.org/10.5194/gmd-16-719-2023 citeturn10search2

20. Shi, L., Cheng, P., & Chen, J. (2011). Sensor data scheduling for optimal state estimation with communication energy constraint. *Automatica*, 47(8), 1693–1698. https://doi.org/10.1016/j.automatica.2011.02.037 citeturn8search2turn8search4

21. Tran, N.-D., Mahmood, A., & Gidlund, M. (2026). Learning-Based Sensor Scheduling for Delay-Aware and Stable Remote State Estimation. *arXiv preprint* arXiv:2601.21482. https://doi.org/10.48550/arXiv.2601.21482 citeturn15academia48turn15search2

22. Wang, Y., Zhang, X., Ning, W., Lazzara, M. A., Ding, M., Reijmer, C. H., Smeets, P. C. J. P., Grigioni, P., Heil, P., Thomas, E. R., Mikolajczyk, D., Welhouse, L. J., Keller, L. M., Zhai, Z., Sun, Y., & Hou, S. (2023). The AntAWS dataset: a compilation of Antarctic automatic weather station observations. *Earth System Science Data*, 15, 411–429. https://doi.org/10.5194/essd-15-411-2023 citeturn10search0turn10search1

23. Wei, Y., & Zheng, R. (2020). Informative Path Planning for Mobile Sensing with Reinforcement Learning. In *IEEE INFOCOM 2020 - IEEE Conference on Computer Communications* (pp. 864–873). IEEE. https://doi.org/10.1109/INFOCOM41043.2020.9155528 citeturn15search1

综合判断：这篇稿件的参考文献体系**可以修到很干净**，主要不是“真假文献”问题，而是**一个明确 DOI 错误、若干状态/类型标注不足，以及两处正文论点表述偏强**的问题。把这几处处理掉之后，参考文献层面的审稿风险会明显下降。citeturn14search0turn15search2turn4search0turn16search3turn15search1turn5search3turn10search0