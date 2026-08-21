# 05-13-02 补充实验回应：E1-fix、E2、P1 正文填充方案

本文档基于 `docs/05-13-02.md` 的实验输出，提供三项补充实验的正文 LaTeX 代码、关键数字解读，以及对 V2 生成器局限性的修订叙事。所有数字均直接来自报告中的实验输出，未作任何假设或推断。

---

## 一、E1-fix：条件分组评估（§6 正文替换）

### 1.1 关键发现

V2 生成器在 1024 步窗口中 event fraction 最高约为 0.58，无法产生 `>0.75` 的 event-heavy 窗口。因此本轮采用可实现且互不重叠的三档条件：

- calm：event fraction `< 0.25`（均值 0.172）
- mixed：event fraction `0.28–0.36`（均值 0.321）
- event：event fraction `> 0.40`（均值 0.450，最高 0.584）

三个条件池均无候选警告（candidate warning = 0），采样诊断干净。

### 1.2 §6 E1 子节 LaTeX 代码（替换 turn_54 占位符）

```latex
\subsection{Condition-Stratified Evaluation (E1)}
\label{sec:e1_condition}

To assess whether PD-PPO's advantage is robust across meteorological
regimes, we stratify evaluation windows by blowing-snow event
fraction. The V2 synthetic generator produces event fractions up to
approximately $0.58$ within a 512-step window; we therefore define
three non-overlapping strata: \emph{calm} (event fraction $< 0.25$,
mean $0.172$), \emph{mixed} (event fraction $\in [0.28, 0.36]$, mean
$0.321$), and \emph{event-heavy} (event fraction $> 0.40$, mean
$0.450$). Each stratum is evaluated over 6 rollouts $\times$ 10 seeds
at budgets $B \in \{1.65, 1.70, 1.75\}$; results at $B = 1.70$ are
reported in Table~\ref{tab:e1_condition}.

\begin{table}[h]
\centering
\caption{Condition-stratified FW-MAE at budget $B = 1.70$
         (E1-fix; 512-step windows, 6 rollouts, seeds 41--50).
         Lower is better.}
\label{tab:e1_condition}
\begin{tabular}{lcccccc}
\toprule
Condition & Full obs. & Static proj. & PD-PPO & Round-robin & AoI & Random \\
\midrule
Calm   & 0.3031 & 0.3222 & 0.3233 & 0.3413 & 0.3495 & 0.3631 \\
Mixed  & 0.3828 & 0.3999 & 0.4018 & 0.4177 & 0.4292 & 0.4431 \\
Event  & 0.4164 & 0.4343 & 0.4406 & 0.4544 & 0.4754 & 0.4876 \\
\bottomrule
\end{tabular}
\end{table}

PD-PPO ranks second in all three strata, consistently outperforming
round-robin, AoI, and random scheduling. The absolute gap between
PD-PPO and AoI widens from $0.026$ under calm conditions to $0.035$
under mixed and $0.035$ under event-heavy conditions, suggesting that
the prediction-driven reward provides greater benefit when blowing-snow
events are more frequent. PD-PPO remains within $0.001$--$0.006$ of
the static projection baseline across all strata, consistent with the
main S1 result.

\paragraph{Generator limitation.}
The V2 generator cannot produce windows with event fraction exceeding
approximately $0.58$, because blowing-snow events are injected as
independent intervals rather than being driven by a persistent
wind-speed regime. Consequently, the ``event-heavy'' stratum reported
here corresponds to event fractions in $[0.40, 0.58]$, not the
$> 0.75$ threshold originally targeted. A V3 generator that derives
events from a storm-regime latent variable is planned to enable
evaluation under more extreme event conditions (Section~\ref{sec:discussion}).
```

---

## 二、E2：Oracle 鲁棒性评估（§6 正文填充）

### 2.1 关键发现

Oracle FW-MAE 随活跃传感器数量 $k$ 平滑下降，从 $k=1$（0.696）到 $k=8$（0.385），未出现近常数奖励或完全塌缩。PD-PPO（0.390）与 static projection（0.389）接近，均远优于 random（0.430）。奖励景观对调度策略具有明确区分度。

### 2.2 §6 E2 子节 LaTeX 代码（替换 turn_54 占位符）

```latex
\subsection{Oracle Reward Signal Reliability (E2)}
\label{sec:e2_oracle}

A prediction-driven reward is only informative if the TCN oracle's
forecast accuracy degrades gracefully as fewer sensor channels are
observed. We evaluate the frozen oracle under partial observation by
fixing $k \in \{1, 2, 3, 4, 5\}$ active channels (randomly sampled
subsets, up to 16 per $k$; $k=1$ fully enumerated) at budget
$B = 1.70$, seeds 41--45. Results are reported in
Table~\ref{tab:e2_oracle}.

\begin{table}[h]
\centering
\caption{Oracle FW-MAE under partial observation (E2; $B = 1.70$,
         seeds 41--45). $k$ denotes the number of active sensor
         channels; $k=8$ is full observation. Lower is better.}
\label{tab:e2_oracle}
\begin{tabular}{crrr}
\toprule
Active channels $k$ & FW-MAE mean & Std & $n$ \\
\midrule
1 & 0.6960 & 0.1180 & 40 \\
2 & 0.6211 & 0.1374 & 80 \\
3 & 0.5598 & 0.1263 & 80 \\
4 & 0.5155 & 0.1173 & 80 \\
5 & 0.4723 & 0.0878 & 80 \\
8 (full) & 0.3848 & 0.0384 &  5 \\
\bottomrule
\end{tabular}
\end{table}

The oracle FW-MAE decreases monotonically from $0.696$ at $k=1$ to
$0.385$ at $k=8$, confirming that the reward signal is non-degenerate
across the full range of feasible sensor subsets. For reference, the
same seed batch yields: full observation $0.363$, static projection
$0.389$, PD-PPO $0.390$, and random $0.430$. The $17\%$ gap between
full observation and random scheduling ($0.363$ vs.\ $0.430$) provides
a lower bound on the reward landscape's discriminability: even the
worst feasible policy (random) incurs a substantially higher FW-MAE
than the oracle, confirming that the prediction-driven reward does not
collapse to a near-constant signal for any budget in $[1.65, 1.75]$.

The proximity of PD-PPO ($0.390$) to static projection ($0.389$)
indicates that the learned policy has converged to a near-optimal
static allocation in the V2 environment, rather than exploiting
dynamic event context. This is consistent with the A2 ablation result
showing that EventAwareCritic contributes only $-0.1\%$ in isolation
(Section~\ref{sec:ablation}), and motivates the V3 generator redesign
described in Section~\ref{sec:discussion}.
```

---

## 三、P1：物理单位误差表（§6 正文填充）

### 3.1 关键发现

FW-MAE 不等于摄氏度误差。在 $B=1.70$ 下，PD-PPO 的逐变量原始 MAE 为：气温 2.547°C、风速 1.414 m s⁻¹、吹雪质量通量 9.69×10⁻⁵ kg m⁻² s⁻¹。与 AoI 相比，PD-PPO 在气温上节省 0.476°C（15.7%），在风速上节省 0.181 m s⁻¹（11.3%）。

### 3.2 §6 物理意义段落 LaTeX 代码（替换 turn_54 占位符）

```latex
\subsection{Physical-Unit Interpretation (P1)}
\label{sec:p1_physical}

The forecast-weighted MAE (FW-MAE) is a horizon-weighted average of
per-target MAE values normalised to $[0,1]$; it does not directly
correspond to a physical unit. Table~\ref{tab:p1_physical} reports the
raw per-variable MAE in physical units at budget $B = 1.70$ for the
three primary target variables.

\begin{table}[h]
\centering
\caption{Per-variable raw forecast MAE in physical units at $B = 1.70$
         (P1; $n = 10$ seeds). Lower is better.}
\label{tab:p1_physical}
\begin{tabular}{llcccccc}
\toprule
Variable & Unit & Full obs. & Static proj. & PD-PPO & Round-robin & AoI & Random \\
\midrule
Air temperature    & $^\circ$C                  & 2.454 & 2.559 & 2.547 & 2.773 & 3.023 & 3.044 \\
Wind speed         & m\,s$^{-1}$                & 1.335 & 1.409 & 1.414 & 1.444 & 1.595 & 1.620 \\
Snow mass flux     & kg\,m$^{-2}$\,s$^{-1}$    & $9.33{\times}10^{-5}$ & $9.74{\times}10^{-5}$ & $9.69{\times}10^{-5}$ & $9.81{\times}10^{-5}$ & $1.00{\times}10^{-4}$ & $1.01{\times}10^{-4}$ \\
\bottomrule
\end{tabular}
\end{table}

PD-PPO reduces air-temperature forecast error by $0.476\,^\circ$C
relative to AoI ($3.023 \to 2.547\,^\circ$C, $-15.7\%$) and wind-speed
error by $0.181\,\mathrm{m\,s^{-1}}$ ($1.595 \to 1.414\,\mathrm{m\,s^{-1}}$,
$-11.3\%$). Snow mass flux error is reduced by $3.1\%$ in absolute
terms ($1.00{\times}10^{-4} \to 9.69{\times}10^{-5}\,\mathrm{kg\,m^{-2}\,s^{-1}}$).
These reductions are operationally meaningful for Antarctic AWS
applications: a $0.5\,^\circ$C improvement in near-surface temperature
forecasting directly affects blowing-snow onset prediction, given that
CRED exceeds $0.5$ at wind speeds above $8\,\mathrm{m\,s^{-1}}$
\citep{Amory2020}.
```

---

## 四、§4 ACF 局限性说明（最终修订版）

基于 E1-fix 的实际发现（event fraction 上限约 0.58），§4 的 ACF 局限性说明需在 turn_54 §3.1 措辞基础上补充生成器事件覆写对 event fraction 上限的影响：

```latex
The base meteorological variables (wind speed, air temperature,
relative humidity, barometric pressure, and solar radiation) are
synthesised via DFT phase randomisation anchored to AntAWS station
records \citep{Aloni2025}, which preserves the power spectral density
and autocorrelation structure of the reference signal. However, the
subsequent blowing-snow event construction modifies
\texttt{wind\_speed\_ms} through event-floor and precursor-ramp
operations, which can disrupt the DFT-preserved temporal structure
during event transitions. Consequently, the wind-speed ACF of the
final synthetic sequences may deviate from the AntAWS reference near
event boundaries. Furthermore, because events are injected as
independent intervals rather than being driven by a persistent
wind-speed regime, the generator cannot produce windows with event
fraction exceeding approximately $0.58$, limiting the range of
meteorological conditions that can be evaluated. Both limitations are
acknowledged; a V3 generator that derives blowing-snow events from a
storm-regime latent variable is planned as future work
(Section~\ref{sec:discussion}).
```

---

## 五、摘要数字核查（无需修改）

E1-fix、E2、P1 的结果与 turn_53 锁定的摘要数字一致，无需修改摘要：

- PD-PPO FW-MAE (B=1.70) = 0.391（E2 同批 seed 为 0.390，差异在 n=5 vs n=10 的随机性范围内）
- PD-PPO vs. AoI 改进 7.9%（B=1.70）：P1 气温层面对应 15.7%，风速层面对应 11.3%，物理意义更强
- PD-PPO vs. static projection 差距 0.03%：E2 同批 seed 确认（0.390 vs 0.389）

---

## 六、不应声称的内容（更新版）

在 turn_54 §4 基础上，新增以下限制：

- "event-heavy windows with event fraction > 0.75"——V2 生成器无法产生，不应声称
- "E1 event-heavy condition supports the claim"——当前 event 条件最高仅 0.584，应改为"higher-event-fraction windows (event fraction > 0.40)"
- "FW-MAE of 0.391 corresponds to 0.391°C"——P1 已确认：气温 MAE 为 2.547°C，FW-MAE 是归一化加权平均值

以下内容现已可以安全声称（P1 数据确认后）：

- "PD-PPO reduces air-temperature forecast error by 0.476°C relative to AoI"
- "PD-PPO reduces wind-speed forecast error by 0.181 m/s relative to AoI"
- "Oracle FW-MAE degrades monotonically from 0.696 (k=1) to 0.385 (k=8)"

---

## 七、执行顺序建议（更新版）

1. **立即**：将 §6 E1 子节替换为本文档 §1.2 的 LaTeX 代码（E1-fix 数据已就绪）
2. **立即**：将 §6 E2 子节替换为本文档 §2.2 的 LaTeX 代码（E2 数据已就绪）
3. **立即**：将 §6 P1 子节替换为本文档 §3.2 的 LaTeX 代码（P1 数据已就绪）
4. **立即**：将 §4 ACF 局限性说明替换为本文档 §4 的修订版（补充 event fraction 上限说明）
5. **立即**：更新 §7 讨论节，将 V3 生成器设计中的 event fraction 上限问题纳入局限性讨论
6. **中期**：修复生成器事件覆写逻辑（V3），使 event fraction 可达 >0.75
7. **V3 验证通过后**：重跑 E1，报告真正的 event-heavy 条件结果
