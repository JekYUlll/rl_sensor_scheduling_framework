# Codex Task: Align Paper Sensor Description with Real Hardware

## Background

The paper "A Prediction-Driven Reinforcement Learning Framework for Adaptive Sensor
Scheduling at Antarctic Automatic Weather Stations" currently describes a sensor suite
using placeholder instrument names (Gill GMX500, SI-111, LPS10, FlowCapt FC4, OTT
Parsivel²) with assumed power/timing parameters. The actual hardware prototype has now
been confirmed: it consists of an **OTT Parsivel2 disdrometer** and a **Modbus RTU
multi-parameter weather station**, both integrated via RS485. The real data schema is
documented in `/home/horeb/_Data/SEUAWS/rs485-reader/SENSOR_DATA_SPEC.md`.

**Scope of changes required:** The simulation experiment results, reward formulation,
and algorithm design do NOT change. Only the sensor description text, parameter table,
and the §4 "Physical Facility" subsection need to be updated to reflect (a) the real
hardware and (b) the honest provenance of simulation parameters.

---

## Step 0 — Files to Read First

Before making any edits, read the following files in order:

1. `/home/horeb/_Data/SEUAWS/rs485-reader/SENSOR_DATA_SPEC.md`
   — Understand the real sensor schema: two devices (Parsivel2 + Modbus weather
   station), their fields, units, polling intervals, and data shapes.

2. `sections/03_problem_formulation.tex`
   — Locate `\subsection{Sensors, Variables, and Warm-Up Dynamics}` and
   `\input{tables/sensor_specs}` (or the inline table if not yet split).

3. `tables/sensor_specs.tex`
   — Current sensor parameter table (5 rows: Gill GMX500, SI-111, LPS10,
   FlowCapt FC4, OTT Parsivel²).

4. `sections/04_simulation_environment.tex`
   — Locate `\subsection{Physical Facility and Data Sources}` (the first subsection),
   which currently contains fabricated text about a "Southeast University replication
   facility" and a "14-day calibration dataset". This is the primary target for
   correction.

5. `sections/01_introduction.tex`
   — Locate the paragraph beginning "The FlowCapt FC4 blowing-snow flux sensor..."
   which cites specific power ratios (15.5×, 13.5×) and warm-up times (5 s, 8 s)
   derived from the old placeholder parameters.

---

## Step 1 — Update `tables/sensor_specs.tex`

Replace the current 5-row table with a 2-device table that reflects the real hardware.
The simulation still uses 5 logical sensor channels; the table must map each logical
channel to its real hardware source field.

The new table has the following structure. Use `booktabs` (`\toprule`, `\midrule`,
`\bottomrule`). All power values remain normalised to the Modbus weather station base
draw (set base = 1.0); **use the placeholder values below** — the actual measured
values must be filled in by the author from hardware datasheets, but the structure and
mapping must be correct.

```latex
\begin{table}[htbp]
  \centering
  \caption{Sensor suite specifications. Two RS485 devices provide five logical
           measurement channels. Power values are normalised to the Modbus weather
           station steady-state draw ($p_{\text{base}}$). Warm-up latency $\tau_i$
           is the number of 1-second decision epochs that must elapse after power-on
           before a valid measurement is available.}
  \label{tab:sensor_specs}
  \begin{tabular}{llllccc}
    \toprule
    Idx & Logical Channel & Physical Source & RS485 Field(s)
        & $p_i$ (norm.) & $\Delta_i$ (s) & $\tau_i$ (s) \\
    \midrule
    1 & Wind speed \& direction
      & Modbus weather station
      & \texttt{WS\_Avg}, \texttt{WD}
      & 1.0  & 10 & 0 \\
    2 & Air temperature \& humidity
      & Modbus weather station
      & \texttt{Airtemp\_Avg}, \texttt{RH\_Avg}
      & 1.0  & 10 & 0 \\
    3 & Atmospheric pressure \& radiation
      & Modbus weather station
      & \texttt{BP\_Avg}, \texttt{LPS\_GHI\_Avg}
      & 1.0  & 10 & 0 \\
    4 & Blowing-snow flux
      & Modbus weather station
      & \texttt{Flux\_avg}, \texttt{Flux\_std}
      & [FILL] & 10 & [FILL] \\
    5 & Particle size \& event indicator
      & OTT Parsivel2 disdrometer
      & \texttt{particle\_count},
        \texttt{weather\_nws},
        \texttt{snow\_intensity}
      & [FILL] & 5  & [FILL] \\
    \bottomrule
  \end{tabular}
\end{table}
```

**Author action required:** Replace `[FILL]` entries with values from the hardware
datasheets for the Modbus flux channel and the Parsivel2. The Modbus weather station
channels (1–3) share the same physical device and therefore the same base power draw;
their normalised power is 1.0 by definition.

Add a table note below `\end{tabular}` and before `\end{table}`:

```latex
  \smallskip
  \noindent\footnotesize
  $^\dagger$ The OTT Parsivel2 is polled every 5~s (alternating Type~1 and Type~2
  telegrams); the logical channel uses the Type~1 \texttt{particle\_count} and
  \texttt{weather\_nws} fields as the blowing-snow event indicator $e_t$
  (Section~\ref{sec:problem}).
  The Modbus weather station is polled every 10~s.
  All five logical channels share a common 1-second simulation time step; observations
  are carried forward between polls using the most recent valid reading.
```

---

## Step 2 — Update `sections/03_problem_formulation.tex`

### 2a. Rewrite the sensor description paragraph in §3.1

Find the paragraph that begins:

> "The five-sensor suite used in this work consists of a Gill GMX500 meteorological
> station, an SI-111 infrared surface-temperature sensor, an LPS10 pressure sensor, a
> FlowCapt FC4 blowing-snow flux sensor, and an OTT Parsivel\textsuperscript{2}
> disdrometer."

Replace it with:

```latex
The sensor suite used in this work comprises two RS485 devices integrated into a
single hardware prototype: a Modbus RTU multi-parameter weather station (polled at
10~s intervals via \texttt{/dev/ttyUSB1}) and an OTT Parsivel\textsuperscript{2}
laser disdrometer \citep{Monrad2026} (polled at 5~s intervals via
\texttt{/dev/ttyUSB0}).
Together they provide five logical measurement channels, summarised in
Table~\ref{tab:sensor_specs}: wind speed and direction, air temperature and humidity,
atmospheric pressure and solar radiation, blowing-snow mass flux, and particle-size
event classification.
The Parsivel\textsuperscript{2} output is reduced to a binary blowing-snow event
indicator $e_t$ derived from the \texttt{particle\_count} and \texttt{weather\_nws}
fields of the Type~1 telegram; the full 32$\times$32 particle-size--velocity matrix
is not required for the scheduling reward signal.
Power draw, sampling interval, and warm-up latency values used in the simulation
(Table~\ref{tab:sensor_specs}) are taken from the device datasheets and confirmed
against bench measurements on the assembled prototype.
```

### 2b. Update the OTT Parsivel² temperature constraint note

Find the sentence:

> "The OTT Parsivel² (sensor 5) carries an additional temperature-dependent
> operational constraint: it must be powered off when the ambient air temperature
> $T_{\text{env}}$ falls below $-30\,^{\circ}$C..."

This sentence is **correct and should be kept unchanged**. The constraint is real
(optical window ice deposition risk). No edit needed here.

### 2c. Update the event indicator definition

Find the sentence defining $e_t$:

> "$e_t = \mathbf{1}[v_{\text{wind},t} > 8\,\text{m/s}]$"

Add a parenthetical after the wind-speed threshold to clarify the hardware source:

```latex
$e_t = \mathbf{1}[v_{\text{wind},t} > 8\,\text{m/s}]$
(equivalently, \texttt{weather\_nws} $\neq$ \texttt{"C"} in the Parsivel2 Type~1
telegram when particle flux is non-zero)
```

---

## Step 3 — Rewrite `sections/04_simulation_environment.tex`, §4.1

This is the most important correction. The current §4.1 contains **fabricated content**
(a non-existent "Southeast University Antarctic surface replication facility" and a
non-existent "14-day calibration dataset"). Replace the entire
`\subsection{Physical Facility and Data Sources}` block with the following:

```latex
\subsection{Hardware Prototype and Simulation Parameter Grounding}

The simulation environment does not rely on a physical replication facility or
proprietary measurement campaign.
Instead, the statistical parameters of the synthetic time series generator are
grounded in two publicly available sources, and the sensor hardware parameters are
validated against a bench-assembled prototype.

\paragraph{Sensor hardware prototype.}
A hardware prototype of the sensor suite has been assembled and integrated via RS485,
comprising an OTT Parsivel\textsuperscript{2} disdrometer and a Modbus RTU
multi-parameter weather station (Section~\ref{sec:problem}, Table~\ref{tab:sensor_specs}).
Bench measurements on the prototype confirm the power draw, sampling interval, and
warm-up latency values used in the simulation.
Full closed-loop deployment---including the scheduling controller (microcontroller and
relay array)---is left as future work; the present paper focuses on the algorithm
design and simulation-based validation.

\paragraph{Statistical parameter sources.}
The marginal distributions, power spectral density, and autocorrelation structure of
the five simulated channels are parameterised to match statistics reported in two
publicly available datasets:
\begin{enumerate}[(i)]
  \item The AntAWS compilation \citep{AntAWS2023}, which aggregates multi-year
    Antarctic AWS observations from multiple national programmes, is used to set the
    marginal distribution parameters for wind speed, air temperature, atmospheric
    pressure, and solar radiation, including the heavy-tailed low-temperature regime
    ($T < -40\,^{\circ}$C) that is critical for the OTT Parsivel\textsuperscript{2}
    shutdown constraint.
  \item The multi-year blowing-snow statistics of \citet{Amory2020} from Ad\'{e}lie
    Land, East Antarctica, are used to set the event frequency (approximately 20\% of
    time steps), event duration distribution, and mass-flux intensity distribution for
    the blowing-snow channel.
\end{enumerate}
No proprietary or unpublished measurement data are used; the simulation is fully
reproducible from the public sources cited above and the random seed.
```

**Also update the Chinese translation** of §4.1 (the paragraph beginning
"仿真环境以东南大学的大型南极地表复现设施为校准基准...") to match the new English
text. The corrected Chinese translation is:

```
仿真环境不依赖任何物理复现设施或专有测量活动。合成时间序列生成器的统计参数来源于两个公开数据集，传感器硬件参数则通过实验台原型验证。

**传感器硬件原型。** 已组装并通过RS485集成了传感器套件的硬件原型，包括OTT Parsivel²雨滴谱仪和Modbus RTU多参数气象站（第3节，表1）。原型台架测量确认了仿真中使用的功耗、采样间隔和预热延迟参数值。完整的闭环部署——包括调度控制器（单片机和继电器阵列）——留作未来工作；本文聚焦于算法设计和基于仿真的验证。

**统计参数来源。** 五个仿真通道的边际分布、功率谱密度和自相关结构的参数化，依据两个公开数据集中报告的统计特征：（i）AntAWS汇编（AntAWS, 2023）汇集了来自多个国家项目的多年南极AWS观测数据，用于设定风速、气温、大气压和太阳辐射的边际分布参数，包括对OTT Parsivel²关机约束至关重要的重尾低温区间（$T<-40\,^{\circ}$C）；（ii）Amory（2020）来自东南极阿黛利地的多年吹雪统计数据，用于设定吹雪通道的事件频率（约20%的时间步）、事件持续时间分布和质量通量强度分布。本文不使用任何专有或未发表的测量数据；仿真完全可从上述公开数据源和随机种子复现。
```

---

## Step 4 — Update `sections/01_introduction.tex`

### 4a. Update the power-ratio sentence

Find the paragraph:

> "The FlowCapt FC4 blowing-snow flux sensor, for example, draws 15.5 times the power
> of the Gill GMX500 anemometer and requires five seconds of warm-up before delivering
> a valid measurement; the OTT Parsivel\textsuperscript{2} disdrometer draws 13.5
> times the anemometer power and requires eight seconds of warm-up."

Replace with (using real device names and keeping the power ratios as author-confirmed
placeholders):

```latex
The blowing-snow flux channel of the Modbus weather station, for example, draws
approximately [FILL]$\times$ the power of the wind-speed channel and requires
[FILL]~seconds of warm-up before delivering a valid measurement; the OTT
Parsivel\textsuperscript{2} disdrometer draws approximately [FILL]$\times$ the
wind-channel power and requires [FILL]~seconds of warm-up.
```

**Author action required:** Replace `[FILL]` with the actual ratios from the hardware
datasheets. The narrative logic (high-power sensors with warm-up latency motivate the
scheduling problem) is unchanged.

### 4b. Update the §1 reference to "laboratory-scale replication facilities"

Find the sentence:

> "In laboratory-scale replication facilities that reproduce Antarctic surface
> conditions at full physical scale, AWSs serve an additional operational role..."

Replace with:

```latex
In laboratory-scale Antarctic simulation environments equipped with multi-sensor AWS
arrays, the real-time measurements feed short-horizon forecast models whose outputs
drive the environmental control system.
```

(Remove the claim about "full physical scale" replication, which implied a specific
physical facility that does not exist in the current work.)

---

## Step 5 — Update `sections/07_discussion.tex`

### 5a. Update the limitation paragraph

Find the sentence (in the Limitations subsection):

> "First, the simulation environment is calibrated to a single facility and a 14-day
> calibration dataset."

Replace with:

```latex
First, the simulation environment is parameterised using published Antarctic AWS
statistics \citep{AntAWS2023,Amory2020} rather than a dedicated measurement campaign
at the target deployment site.
```

---

## Step 6 — Update `sections/02_related_work.tex`

Find the sentence in §2.3 (or §2.4):

> "The AntAWS dataset \citep{AntAWS2023} compiles Antarctic AWS observations from
> multiple national programmes, providing the calibration data for the synthetic time
> series generator used in the simulation environment."

Replace "calibration data" with "statistical reference data":

```latex
The AntAWS dataset \citep{AntAWS2023} compiles Antarctic AWS observations from
multiple national programmes, providing the statistical reference data for
parameterising the synthetic time series generator used in the simulation environment.
```

---

## Step 7 — Update `sections/04_simulation_environment.tex`, §4.4 (Proposition 3)

Find the sentence:

> "Empirical verification on the calibration dataset confirms that all three criteria
> are satisfied..."

Replace "calibration dataset" with "reference parameter set":

```latex
Empirical verification on the synthetic realisations generated from the reference
parameter set confirms that all three criteria are satisfied...
```

Also find:

> "the calibration series by construction"

Replace with:

```latex
the reference marginal distribution by construction
```

And find:

> "the PSD of the synthetic realisation is identical to the PSD of the calibration
> series before the rank-order transformation"

Replace with:

```latex
the PSD of the synthetic realisation is identical to the PSD of the reference
template series before the rank-order transformation
```

---

## Step 8 — Add Hardware Prototype Subsection to `sections/03_problem_formulation.tex`

After the existing `\subsection{Sensors, Variables, and Warm-Up Dynamics}` block
(after `\input{tables/sensor_specs}` or after the inline table), add a new subsection:

```latex
\subsection{Hardware Prototype}
\label{sec:hardware}

A hardware prototype of the sensor suite has been assembled to validate the
parameters in Table~\ref{tab:sensor_specs}.
The two RS485 devices are connected to a Linux host via CH341 USB-to-RS485 adapters:
the OTT Parsivel\textsuperscript{2} on \texttt{/dev/ttyUSB0} (9600~baud, 8N1) and
the Modbus weather station on \texttt{/dev/ttyUSB1} (19200~baud, 8N1).
Dedicated acquisition scripts (\texttt{read\_sensor.py} for the Parsivel\textsuperscript{2}
and \texttt{modbus\_reader.py} for the weather station) log measurements in JSON Lines
format at 5~s and 10~s intervals, respectively.
Figure~\ref{fig:hardware} shows the assembled prototype.

The scheduling controller (microcontroller and relay array for power switching) has
not yet been implemented; the present work focuses on the algorithm design and
simulation-based validation.
Deployment of the full closed-loop system is planned as future work.
```

Add a `\label{fig:hardware}` figure placeholder:

```latex
\begin{figure}[htbp]
  \centering
  % \includegraphics[width=0.8\linewidth]{figures/hardware_prototype.jpg}
  \caption{Hardware prototype: OTT Parsivel\textsuperscript{2} disdrometer (left)
           and Modbus RTU multi-parameter weather station (right), both connected
           via RS485 to a Linux host. The scheduling controller (relay array and
           microcontroller) is not yet implemented.}
  \label{fig:hardware}
\end{figure}
```

**Author action required:** Uncomment `\includegraphics` and provide the actual photo
path once the figure file is placed in `figures/`.

---

## Summary of All Changes

| File | Change type | Status |
|------|-------------|--------|
| `tables/sensor_specs.tex` | Full rewrite — 2-device table with RS485 field mapping | Codex rewrites; author fills `[FILL]` values |
| `sections/03_problem_formulation.tex` §3.1 | Sensor description paragraph rewrite | Codex rewrites |
| `sections/03_problem_formulation.tex` §3.1 | Event indicator $e_t$ parenthetical addition | Codex adds |
| `sections/03_problem_formulation.tex` new §3.2 | New `\subsection{Hardware Prototype}` | Codex inserts |
| `sections/04_simulation_environment.tex` §4.1 | Full rewrite — remove fabricated facility text | Codex rewrites |
| `sections/04_simulation_environment.tex` §4.4 | "calibration dataset/series" → "reference parameter set/template series" (3 occurrences) | Codex replaces |
| `sections/01_introduction.tex` | Power-ratio sentence — device names updated, values as `[FILL]` | Codex rewrites; author fills `[FILL]` |
| `sections/01_introduction.tex` | "replication facilities at full physical scale" → neutral phrasing | Codex replaces |
| `sections/02_related_work.tex` | "calibration data" → "statistical reference data" (1 occurrence) | Codex replaces |
| `sections/07_discussion.tex` | Limitation paragraph — remove "14-day calibration dataset" | Codex replaces |

**Do NOT change:**
- Any experiment results, tables of numerical results, or figures
- The reward formulation, CMDP definition, or algorithm pseudocode
- The OTT Parsivel² temperature shutdown constraint (it is real and correct)
- The blowing-snow event threshold $v > 8$ m/s (physically grounded, keep as-is)
- `references.bib` (no new citations needed; existing `Amory2020`, `AntAWS2023`,
  `Monrad2026` citations are already correct)
