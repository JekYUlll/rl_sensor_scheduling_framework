# 两张图的命名、插入位置与修改意见（Agent 执行文档）

本文档供 agent 直接执行，涵盖两张图的文件命名、LaTeX 插入代码、正文引用句、以及所有待修改事项。

---

## 图 1：南极 AWS 部署平台渲染图

### 文件命名

```
figures/aws_deployment.png
```

（若为 TIFF 格式则命名为 `figures/aws_deployment.tif`；`\label` 统一使用 `fig:aws_deployment`。）

### 插入位置

**`sections/01_introduction.tex`**，在描述多传感器功耗超出太阳能预算的段落末尾，作为 **Figure 1**。

### 完整 LaTeX 代码块

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/aws_deployment}
  \caption{Three-dimensional rendering of the Antarctic automatic weather
    station (AWS) deployment platform used in this study. The station is
    housed in a solar-powered enclosure equipped with five sensor
    subsystems: an OTT~Parsivel2 laser disdrometer for precipitation
    particle-size detection, a Modbus multi-parameter weather station
    providing wind speed and direction, air temperature, relative
    humidity, barometric pressure, and global solar radiation, an FC4
    blowing-snow flux sensor, an infrared snow surface temperature
    sensor, and a pyranometer. The simultaneous power draw of all five
    subsystems exceeds the available solar budget under typical Antarctic
    winter conditions, motivating the adaptive sensor scheduling problem
    studied in this paper.}
  \label{fig:aws_deployment}
\end{figure}
```

### 正文引用句（插入图环境之前的段落末尾）

```latex
Figure~\ref{fig:aws_deployment} shows the physical deployment platform,
illustrating the spatial arrangement of the five sensor subsystems whose
combined power consumption motivates the scheduling problem addressed in
this work.
```

### 修改意见（按优先级）

**[必须] 标注语言：5 处中文 → 英文**

图中当前存在 5 处中文标注，投稿至英文期刊必须全部替换。若 Blender 重新渲染成本较高，可在 LaTeX 中使用 `\usepackage{overpic}` 或 TikZ overlay 叠加英文标注，无需重渲染。对应翻译如下：

| 当前中文标注 | 替换为英文标注 |
|------------|--------------|
| 激光雨滴谱仪 | OTT Parsivel2 Laser Disdrometer |
| 红外雪温传感器 | IR Snow Surface Temperature Sensor |
| 全辐射仪 | Pyranometer (Global Solar Radiation) |
| 小型气象站 | Modbus Multi-parameter Weather Station |
| FC4风雪通量传感器 | FC4 Blowing-Snow Flux Sensor |

**[建议] 标注样式：深色背景标签 → 学术图注风格**

当前深蓝色矩形背景+白色文字的标签样式偏向产品宣传图风格。建议改为白色或浅灰背景小方框+黑色文字，或使用引线（leader line）指向部件、文字置于图外侧无背景框。若在 Blender 中修改成本较高，同样可通过 `overpic`/TikZ overlay 在 LaTeX 层覆盖。

**[必须] 分辨率与格式**

- 格式：PNG 或 TIFF（禁止 JPEG，避免压缩伪影）
- 分辨率：≥ 300 DPI（彩色图 Elsevier 最低要求）
- 建议渲染至 3000 × 2000 像素以上（对应 10 × 6.67 英寸 @ 300 DPI）

---

## 图 2：PD-PPO 算法架构图

### 文件命名

```
figures/pdppo_architecture.pdf
```

（TikZ 编译输出为 PDF 矢量格式；若使用 Inkscape/Illustrator 导出则同名 PDF 或 EPS；`\label` 统一使用 `fig:pdppo_arch`。）

### 插入位置

**`sections/05_methodology.tex`**，在 PD-PPO 五组件首次完整介绍段落之前，作为 **Figure 2**。

具体位置：在"We propose PD-PPO comprising five components..."段落的前一行插入图环境，使读者在阅读组件逐一描述之前先获得整体架构的视觉印象。

### 完整 LaTeX 代码块

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/pdppo_architecture}
  \caption{Architecture of PD-PPO (Prediction-Driven Proximal Policy
    Optimisation). Information flows left to right through five
    components: (\textit{i})~the Antarctic AWS environment providing
    observations $\mathbf{o}_t$ and power budget $B_t$;
    (\textit{ii})~the State Encoder combining an ActionEmbedding module
    with blowing-snow event context $z_t$ to form the state
    $s_t = [\mathbf{o}_t, z_t]$;
    (\textit{iii})~the PD-PPO Core (purple border) containing the
    MaskedActor $\pi_\theta$ enforcing the hard budget constraint
    $\sum a_i \leq B_t$, the EventAwareCritic $V_\phi(s_t, z_t)$, the
    AWBC auxiliary loss with warmup decay $\lambda_{\mathrm{BC}}(t)$,
    and the oracle-calibrated prior $\pi_0$ providing a
    $\mathrm{KL}(\pi_\theta \| \pi_0)$ penalty at cold-start;
    (\textit{iv})~the PPO Update engine computing the clipped surrogate
    objective and gradient updates $\nabla\theta$, $\nabla\phi$; and
    (\textit{v})~the Prediction-Driven Reward module computing
    $r_t = -\mathrm{MAE}(\hat{y}_{t+h}, y_{t+h})$ via the forecasting
    model $f_\theta$. Dashed arrows denote gradient and initialisation
    flows; solid arrows denote forward data flow.}
  \label{fig:pdppo_arch}
\end{figure*}
```

注：使用 `figure*` 环境（双栏跨全宽），与 TikZ 规格中总图宽约 20 cm 一致。

### 正文引用句（插入图环境之后的段落首句）

```latex
Figure~\ref{fig:pdppo_arch} provides an overview of the complete
PD-PPO architecture; the following subsections describe each component
in detail.
```

### 修改意见（按优先级，基于 turn_50 核查结果）

**[最高优先级] 问题 1：`e_t` 定义缺失**

图中 State Encoder 右侧括注写作 `s_t = [o_t, z_t, e_t]`，但摘要和规格均未定义 `e_t`。

执行方案（二选一，需作者确认）：

- **方案 A（推荐）**：将图中括注改为 `s_t = [o_t, z_t]`，与摘要保持一致，删除 `e_t`。
- **方案 B**：在 `sections/03_problem_formulation.tex` 或 `sections/05_methodology.tex` 中明确定义 `e_t`（例如"let $e_t$ denote the event embedding derived from $z_t$"），并在图注中补充说明。

**[最高优先级] 问题 2：Oracle Prior 箭头目标**

图中紫色虚线同时指向 MaskedActor 和 EventAwareCritic，但规格仅描述指向 MaskedActor。

执行方案（二选一，需作者确认）：

- **方案 A（推荐）**：若 Oracle Prior 仅初始化 Actor，删除指向 EventAwareCritic 的虚线箭头，保留 `π_0 init → MaskedActor` 一条。
- **方案 B**：若同时初始化 Actor 和 Critic，在 `sections/05_methodology.tex` 中补充说明，并将图注中对应句改为"initialising both $\pi_\theta$ and $V_\phi$"。

**[中优先级] 问题 3：两条反馈箭头拥挤**

`∇θ update`（紫色虚线）和 `∇φ update`（绿色虚线）从 PPO Update 框左侧出发点重叠。

在 TikZ 源码中将两条箭头的出发 y 坐标错开 0.4 cm：
- `∇θ update`：从 `(col4.west |- actor.east)` 出发，即 y ≈ 4.2
- `∇φ update`：从 `(col4.west |- critic.east)` 出发，即 y ≈ 3.6

**[低优先级] 问题 4：AWBC 子框颜色**

图中 AWBC 框为蓝色边框+白色填充，规格要求 steel blue（`#4472C4`）实色填充。在 TikZ 中将对应 `\node` 的 `fill` 属性改为 `fill=steelblue!60` 即可。

**[低优先级] 问题 5：Total Loss 公式颜色编码**

PPO Update 框内 Total Loss 公式各项建议分色，在 TikZ 中使用 `\textcolor`：

```latex
$\mathcal{L} = \textcolor{orange}{\mathcal{L}^{\mathrm{CLIP}}}
             - c_1\,\textcolor{darkgreen}{\mathcal{L}^V}
             + c_2\,\textcolor{purple}{H}
             + \lambda_{\mathrm{BC}}\,\textcolor{steelblue}{\mathcal{L}^{\mathrm{BC}}}$
```

---

## 图号汇总与 paper.tex 全局图号确认

| 图号 | 文件名 | label | 所在节 |
|------|--------|-------|--------|
| Figure 1 | `figures/aws_deployment` | `fig:aws_deployment` | `sections/01_introduction.tex` |
| Figure 2 | `figures/pdppo_architecture` | `fig:pdppo_arch` | `sections/05_methodology.tex` |

若 `sections/` 中其他节已有图（如 §6 实验结果图、§4 仿真环境示意图），需在插入上述两图后对全文图号重新排序，确保 Figure 1 和 Figure 2 按出现顺序正确编号。

---

## overpic 叠加方案（仅当 Blender 无法重渲染时使用）

若渲染图中文标注无法在 Blender 中修改，可在 LaTeX 中使用以下 overpic 代码叠加英文标注（坐标为百分比，需根据实际图片位置微调）：

```latex
\usepackage{overpic}

\begin{figure}[t]
  \centering
  \begin{overpic}[width=\columnwidth]{figures/aws_deployment}
    \put(18, 72){\small OTT Parsivel2 Laser Disdrometer}
    \put(62, 68){\small IR Snow Surface Temperature Sensor}
    \put(75, 45){\small Pyranometer}
    \put(10, 38){\small Modbus Multi-parameter Weather Station}
    \put(45, 20){\small FC4 Blowing-Snow Flux Sensor}
  \end{overpic}
  \caption{...}  % 同上完整图题
  \label{fig:aws_deployment}
\end{figure}
```

坐标 `\put(x, y)` 中 x、y 为图片宽度和高度的百分比（0–100），需在编译后目视调整至与对应部件对齐。
