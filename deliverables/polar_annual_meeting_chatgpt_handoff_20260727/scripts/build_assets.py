#!/usr/bin/env python3
"""Build presentation-ready assets from the frozen PD-PPO evidence package."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = FRAMEWORK_ROOT / "paper"
REPORT_ROOT = FRAMEWORK_ROOT / "reports" / "aggregate"
IMAGE_DIR = PACKAGE_ROOT / "images"
DATA_DIR = PACKAGE_ROOT / "data"

FONT_PATH = Path("/usr/share/fonts/truetype/winfonts/msyh.ttf")
font_manager.fontManager.addfont(FONT_PATH)
FONT_FAMILY = font_manager.FontProperties(fname=FONT_PATH).get_name()
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
GRAY = "#7A8793"
DARK = "#28323C"
LIGHT_BLUE = "#EAF3FA"
LIGHT_GREEN = "#E7F5F0"
LIGHT_ORANGE = "#FFF3D9"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(IMAGE_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(IMAGE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rasterize_pdf(source: Path, target_stem: Path, dpi: int = 180) -> Path:
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-singlefile",
            str(source),
            str(target_stem),
        ],
        check=True,
    )
    return target_stem.with_suffix(".png")


def annotate_aws_render() -> None:
    source = PAPER_ROOT / "figures" / "aws_deployment.png"
    target = IMAGE_DIR / "01_极地多传感器平台渲染图_中文标注.png"
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 25)
    labels = [
        ((114, 185, 307, 301), "红外雪面\n温度传感器"),
        ((114, 330, 307, 426), "总辐射\n传感器"),
        ((114, 451, 307, 579), "多参数\n气象站"),
        ((701, 167, 881, 273), "激光雨滴谱仪"),
        ((1104, 272, 1324, 362), "FC4风吹雪\n通量传感器"),
    ]
    for box, label in labels:
        draw.rectangle(box, fill="#243650")
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=5, align="left")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = box[0] + 14
        y = box[1] + (box[3] - box[1] - text_h) / 2 - bbox[1]
        if text_w > box[2] - box[0] - 20:
            x = box[0] + 8
        draw.multiline_text((x, y), label, font=font, fill="white", spacing=5)
    image.save(target, quality=95)


def export_existing_vector_assets() -> None:
    framework_pdf = PAPER_ROOT / "figures" / "figure_pdppo_framework_drawio.pdf"
    full_png = rasterize_pdf(
        framework_pdf, IMAGE_DIR / "02_PD-PPO完整框架图_高清", dpi=200
    )
    shutil.copy2(framework_pdf, IMAGE_DIR / "02_PD-PPO完整框架图_矢量.pdf")

    image = Image.open(full_png)
    width, height = image.size
    crops = {
        "02a_时间分区_框架图裁剪.png": (0, 0, width, int(height * 0.19)),
        "02b_在线调度闭环_框架图裁剪.png": (
            0,
            int(height * 0.18),
            width,
            int(height * 0.79),
        ),
        "02c_离线训练与评分_框架图裁剪.png": (
            0,
            int(height * 0.77),
            width,
            height,
        ),
    }
    for name, box in crops.items():
        image.crop(box).save(IMAGE_DIR / name)

    proposition_pdf = PAPER_ROOT / "figures" / "proposition_dynamic_value_standalone.pdf"
    rasterize_pdf(
        proposition_pdf, IMAGE_DIR / "03_动态调度价值理论示意图_高清", dpi=220
    )
    shutil.copy2(proposition_pdf, IMAGE_DIR / "03_动态调度价值理论示意图_矢量.pdf")

    shutil.copy2(
        PAPER_ROOT / "figures" / "figure3_synthetic_statistics.png",
        IMAGE_DIR / "11_仿真数据与南极AWS统计对比_英文原图.png",
    )
    shutil.copy2(
        PAPER_ROOT / "figures" / "figure3_synthetic_statistics.pdf",
        IMAGE_DIR / "11_仿真数据与南极AWS统计对比_矢量.pdf",
    )


def load_evidence() -> dict[str, object]:
    claim_dir = REPORT_ROOT / "pdppo_clean_validation_frozen_24seed_20260718"
    claim = json.loads((claim_dir / "validation_frozen_claim_summary.json").read_text())
    seeds = pd.read_csv(claim_dir / "validation_frozen_seed_metrics.csv")
    mechanism_dir = REPORT_ROOT / "pdppo_clean_mechanism_24seed_20260718"
    mechanism = json.loads(
        (mechanism_dir / "clean_policy_mechanism_summary.json").read_text()
    )
    duty = pd.read_csv(
        mechanism_dir / "clean_policy_subtype_sensor_duty_summary.csv"
    )
    framework = pd.read_csv(
        REPORT_ROOT
        / "pdppo_framework_baselines_clean_24seed_20260718"
        / "framework_baseline_summary.csv"
    )
    dqn = pd.read_csv(
        REPORT_ROOT
        / "pdppo_matched_dqn_clean_24seed_20260718"
        / "matched_dqn_summary.csv"
    ).iloc[0]
    rewards = pd.read_csv(
        REPORT_ROOT
        / "pdppo_clean_matched_reward_24seed_20260718"
        / "matched_reward_summary.csv"
    )
    ridge = pd.read_csv(
        REPORT_ROOT
        / "pdppo_secondary_forecaster_24seed_20260718"
        / "secondary_forecaster_summary.csv"
    ).iloc[0]
    full = json.loads(
        (
            REPORT_ROOT
            / "pdppo_full_final_partition_24seed_20260718"
            / "validation_frozen_claim_summary.json"
        ).read_text()
    )
    return {
        "claim": claim,
        "seeds": seeds,
        "mechanism": mechanism,
        "duty": duty,
        "framework": framework,
        "dqn": dqn,
        "rewards": rewards,
        "ridge": ridge,
        "full": full,
    }


def write_compact_data(evidence: dict[str, object]) -> None:
    seeds = evidence["seeds"]
    assert isinstance(seeds, pd.DataFrame)
    claim = evidence["claim"]
    assert isinstance(claim, dict)

    static_step = seeds["validation_selected_static_step_loss"].mean()
    pdppo_step = seeds["custom_ppo_step_loss"].mean()
    static_macro = seeds[
        "validation_selected_static_oracle_loss_macro_subtype_event_validationnorm"
    ].mean()
    pdppo_macro = seeds[
        "custom_ppo_oracle_loss_macro_subtype_event_validationnorm"
    ].mean()
    headline = pd.DataFrame(
        [
            {
                "metric_cn": "平均多步预测损失",
                "pdppo": pdppo_step,
                "fixed_schedule": static_step,
                "relative_reduction_percent": 100 * (static_step - pdppo_step) / static_step,
                "wins": 24,
                "total_seeds": 24,
                "mean_paired_difference": claim[
                    "step_pdppo_vs_validation_selected_static"
                ]["mean_margin"],
                "ci95_low": claim[
                    "step_pdppo_vs_validation_selected_static"
                ]["bootstrap_95_ci"][0],
                "ci95_high": claim[
                    "step_pdppo_vs_validation_selected_static"
                ]["bootstrap_95_ci"][1],
            },
            {
                "metric_cn": "事件类型等权归一化损失",
                "pdppo": pdppo_macro,
                "fixed_schedule": static_macro,
                "relative_reduction_percent": 100
                * (static_macro - pdppo_macro)
                / static_macro,
                "wins": 24,
                "total_seeds": 24,
                "mean_paired_difference": claim[
                    "pdppo_vs_validation_selected_static"
                ]["mean_margin"],
                "ci95_low": claim[
                    "pdppo_vs_validation_selected_static"
                ]["bootstrap_95_ci"][0],
                "ci95_high": claim[
                    "pdppo_vs_validation_selected_static"
                ]["bootstrap_95_ci"][1],
            },
        ]
    )
    headline.to_csv(DATA_DIR / "01_核心结果.csv", index=False)

    event_rows = []
    event_meta = {
        "particle": ("粒子事件", 20, 0.0152, 0.0459),
        "flux": ("通量事件", 13, 0.0557, 0.1786),
        "thermal": ("热事件", 17, 0.0492, 0.1422),
    }
    for subtype, (name_cn, wins, ci_low, ci_high) in event_meta.items():
        normalizer = seeds[f"validation_normalizer_{subtype}"]
        pdppo = (
            seeds[f"custom_ppo_oracle_loss_subtype_{subtype}"] / normalizer
        ).mean()
        static = (
            seeds[f"validation_selected_static_oracle_loss_subtype_{subtype}"]
            / normalizer
        ).mean()
        event_rows.append(
            {
                "event_type_cn": name_cn,
                "pdppo_normalized_loss": pdppo,
                "fixed_schedule_normalized_loss": static,
                "wins": wins,
                "total_seeds": 24,
                "mean_normalized_paired_difference": static - pdppo,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    pd.DataFrame(event_rows).to_csv(DATA_DIR / "02_分事件类型结果.csv", index=False)

    baseline_rows = [
        ("固定调度", claim["pdppo_vs_validation_selected_static"]["mean_margin"], 24),
        ("AoI优先", claim["pdppo_vs_aoi"]["mean_margin"], 24),
        ("轮转调度", claim["pdppo_vs_round_robin"]["mean_margin"], 24),
        ("随机可行调度", claim["pdppo_vs_random"]["mean_margin"], 24),
        ("单步预测贪心", 0.17898896386137564, 24),
        ("Double-DQN", float(evidence["dqn"]["macro_margin_dqn_minus_ppo_mean"]), 24),
        ("预警规则", 0.0012217345505791, 11),
        ("真实事件标签参考", 0.0017683245171340, 12),
    ]
    baseline = pd.DataFrame(
        baseline_rows,
        columns=["baseline_cn", "macro_margin_baseline_minus_pdppo", "pdppo_wins"],
    )
    baseline["total_seeds"] = 24
    baseline.to_csv(DATA_DIR / "03_基线对比.csv", index=False)

    duty = evidence["duty"]
    assert isinstance(duty, pd.DataFrame)
    duty_out = duty[
        ~duty["sensor"].isin(["met_station_core", "radiometer_basic"])
    ].copy()
    duty_out.to_csv(DATA_DIR / "04_事件类型与传感器选择频率.csv", index=False)

    channel = pd.DataFrame(
        [
            ("核心气象通道", "风速、气温、湿度、气压", 0.25, 0.30, 0, "始终开启"),
            ("总辐射通道", "太阳辐射", 0.08, 0.10, 0, "可选"),
            ("温湿度通道", "气温、湿度", 0.18, 0.22, 0, "可选"),
            ("红外雪温通道", "雪面温度", 0.49, 0.62, 1, "可选；热事件专用"),
            ("激光粒子通道", "粒径、粒子速度、微结构", 0.49, 0.62, 2, "可选；粒子事件专用"),
            ("FC4通量通道", "风吹雪质量通量", 0.49, 0.62, 1, "可选；通量事件专用"),
        ],
        columns=[
            "logical_channel_cn",
            "main_variables_cn",
            "steady_cost",
            "startup_peak_cost",
            "warmup_steps",
            "role_cn",
        ],
    )
    channel.to_csv(DATA_DIR / "05_六通道配置.csv", index=False)

    mechanism = evidence["mechanism"]
    assert isinstance(mechanism, dict)
    ridge = evidence["ridge"]
    full = evidence["full"]
    pd.DataFrame(
        [
            ("行为判据通过", mechanism["behavior_gate_passes"], "个种子", "24/24"),
            ("近似固定序列", mechanism["fixed_like_count"], "个种子", "0/24"),
            ("近似周期序列", mechanism["simple_cycle_like_count"], "个种子", "0/24"),
            ("平均切换率", mechanism["switches_per_step"]["mean"], "每步", "0.00368"),
            ("暖机中断", 0, "次", "全部种子为0"),
            (
                "完整测试区间胜固定调度",
                full["pdppo_vs_validation_selected_static"]["wins"],
                "个种子",
                "24/24",
            ),
            (
                "替代预测器胜重新选择固定调度",
                int(ridge["macro_margin_vs_secondary_static_wins"]),
                "个种子",
                "23/24",
            ),
        ],
        columns=["metric_cn", "value", "unit_cn", "display_cn"],
    ).to_csv(DATA_DIR / "06_行为与稳健性.csv", index=False)

    pd.DataFrame(
        [
            ("预测器训练", 0, 24500, "拟合并冻结预测器"),
            ("策略训练", 24500, 59500, "训练PD-PPO"),
            ("验证", 59500, 64750, "选择固定基线并计算归一化因子"),
            ("最终测试", 64750, 70000, "不使用测试反馈调参"),
        ],
        columns=["partition_cn", "start", "end", "purpose_cn"],
    ).to_csv(DATA_DIR / "07_时间分区.csv", index=False)

    # Preserve the exact 24-seed paired margins for independent plotting.
    seeds[
        [
            "seed",
            "macro_margin_pdppo_vs_validation_selected_static",
            "step_margin_pdppo_vs_validation_selected_static",
            "macro_margin_pdppo_vs_aoi",
            "macro_margin_pdppo_vs_round_robin",
            "macro_margin_pdppo_vs_random",
            "switches_per_step",
            "warmup_abort_count",
        ]
    ].to_csv(DATA_DIR / "08_逐种子核心指标.csv", index=False)


def plot_headline_results(evidence: dict[str, object]) -> None:
    headline = pd.read_csv(DATA_DIR / "01_核心结果.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
    for ax, row in zip(axes, headline.to_dict("records")):
        values = [row["fixed_schedule"], row["pdppo"]]
        bars = ax.bar(
            ["固定调度", "PD-PPO"],
            values,
            color=[GRAY, BLUE],
            width=0.58,
            edgecolor="white",
        )
        ax.set_title(row["metric_cn"], fontsize=24, fontweight="bold", pad=18)
        ax.set_ylim(0, max(values) * 1.32)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=14)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.4f}",
                ha="center",
                fontsize=17,
                fontweight="bold",
            )
        ax.text(
            0.5,
            0.91,
            f"降低 {row['relative_reduction_percent']:.1f}%",
            transform=ax.transAxes,
            ha="center",
            fontsize=25,
            color=GREEN,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.82,
            "24/24 个随机种子均更低",
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
            color=DARK,
        )
    fig.suptitle("预测驱动调度稳定优于验证集选择的固定调度", fontsize=28, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=4)
    save_figure(fig, "04_核心结果_固定调度对比")


def plot_event_heatmap(evidence: dict[str, object]) -> None:
    duty = evidence["duty"]
    assert isinstance(duty, pd.DataFrame)
    subtype_order = ["calm", "particle", "flux", "thermal"]
    sensor_order = [
        "shielded_thermo_hygro",
        "surface_temp_ir",
        "laser_disdrometer",
        "fc4_flux",
    ]
    row_names = ["平静期", "粒子事件", "通量事件", "热事件"]
    col_names = ["温湿度", "红外雪温", "激光粒子", "FC4通量"]
    matrix = np.zeros((4, 4))
    for i, subtype in enumerate(subtype_order):
        for j, sensor in enumerate(sensor_order):
            value = duty.loc[
                (duty["subtype"] == subtype) & (duty["sensor"] == sensor),
                "selection_fraction_mean",
            ]
            matrix[i, j] = float(value.iloc[0]) if not value.empty else 0.0

    fig, ax = plt.subplots(figsize=(12, 6.75))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), col_names, fontsize=19)
    ax.set_yticks(range(4), row_names, fontsize=19)
    for i in range(4):
        for j in range(4):
            color = "white" if matrix[i, j] > 0.55 else DARK
            ax.text(
                j,
                i,
                f"{100 * matrix[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold" if matrix[i, j] > 0.9 else "normal",
                color=color,
            )
    ax.set_title("PD-PPO按事件类型分配唯一可选传感器通道", fontsize=28, fontweight="bold", pad=20)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("通道开启比例", fontsize=17)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel("可选传感器通道", fontsize=19, labelpad=12)
    ax.set_ylabel("运行状态", fontsize=19, labelpad=12)
    fig.tight_layout()
    save_figure(fig, "05_事件类型与传感器选择热图")


def plot_baseline_comparison() -> None:
    baseline = pd.read_csv(DATA_DIR / "03_基线对比.csv")
    main = baseline.iloc[:6].copy()
    main = main.sort_values("macro_margin_baseline_minus_pdppo")
    fig, ax = plt.subplots(figsize=(12, 6.75))
    colors = [GREEN if name == "固定调度" else BLUE for name in main["baseline_cn"]]
    bars = ax.barh(
        main["baseline_cn"],
        main["macro_margin_baseline_minus_pdppo"],
        color=colors,
        alpha=0.9,
    )
    ax.axvline(0, color=DARK, linewidth=1.5)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("基线损失 − PD-PPO损失（越大表示PD-PPO优势越大）", fontsize=17)
    ax.set_title("PD-PPO对主要固定、规则、贪心与RL基线的宏平均优势", fontsize=27, fontweight="bold", pad=18)
    xmax = main["macro_margin_baseline_minus_pdppo"].max()
    ax.set_xlim(0, xmax * 1.35)
    for bar, (_, row) in zip(bars, main.iterrows()):
        value = row["macro_margin_baseline_minus_pdppo"]
        ax.text(
            value + xmax * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"+{value:.3f}；{int(row['pdppo_wins'])}/24",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
    ax.text(
        0.99,
        0.03,
        "单步贪心使用测试时刻的未来目标，仅作诊断参考",
        transform=ax.transAxes,
        ha="right",
        fontsize=13,
        color=GRAY,
    )
    fig.tight_layout()
    save_figure(fig, "06_主要基线宏平均对比")


def plot_behavior_cards(evidence: dict[str, object]) -> None:
    mechanism = evidence["mechanism"]
    assert isinstance(mechanism, dict)
    cards = [
        ("24/24", "行为判据通过", "非固定、非简单周期"),
        ("0 次", "暖机中断", "所有测试种子"),
        (f"{mechanism['switches_per_step']['mean'] * 100:.3f}%", "平均切换率", "每步通道状态变化比例"),
        ("99.3 / 97.8 / 99.1%", "事件匹配通道开启率", "粒子 / 通量 / 热事件"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    positions = [(0.05, 0.54), (0.53, 0.54), (0.05, 0.10), (0.53, 0.10)]
    colors = [LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, "#F1F3F5"]
    borders = [BLUE, GREEN, ORANGE, GRAY]
    for (x, y), color, border, (value, title, subtitle) in zip(
        positions, colors, borders, cards
    ):
        rect = plt.Rectangle(
            (x, y),
            0.42,
            0.34,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=border,
            linewidth=2.2,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.21,
            y + 0.235,
            value,
            transform=ax.transAxes,
            ha="center",
            fontsize=27,
            fontweight="bold",
            color=border,
        )
        ax.text(
            x + 0.21,
            y + 0.135,
            title,
            transform=ax.transAxes,
            ha="center",
            fontsize=20,
            fontweight="bold",
        )
        ax.text(
            x + 0.21,
            y + 0.065,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
            color=GRAY,
        )
    ax.set_title("调度序列满足工程行为要求，同时保持状态依赖性", fontsize=27, fontweight="bold", pad=18)
    fig.tight_layout()
    save_figure(fig, "07_调度行为与工程约束摘要")


def plot_protocol_timeline() -> None:
    partitions = pd.read_csv(DATA_DIR / "07_时间分区.csv")
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 70000)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = [LIGHT_BLUE, LIGHT_GREEN, "#F1F3F5", LIGHT_ORANGE]
    borders = [BLUE, GREEN, GRAY, ORANGE]
    for index, ((_, row), color, border) in enumerate(
        zip(partitions.iterrows(), colors, borders), start=1
    ):
        start, end = row["start"], row["end"]
        width = end - start
        rect = plt.Rectangle(
            (start, 0.50),
            width,
            0.17,
            facecolor=color,
            edgecolor=border,
            linewidth=2.2,
        )
        ax.add_patch(rect)
        ax.text(
            start + width / 2,
            0.585,
            str(index),
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=border,
        )
    card_centres = [8750, 26250, 43750, 61250]
    for index, ((_, row), color, border, centre) in enumerate(
        zip(partitions.iterrows(), colors, borders, card_centres), start=1
    ):
        ax.text(
            centre,
            0.39,
            f"{index}  {row['partition_cn']}",
            ha="center",
            fontsize=17,
            fontweight="bold",
            color=border,
        )
        ax.text(
            centre,
            0.315,
            f"{int(row['start']):,}–{int(row['end']):,}",
            ha="center",
            fontsize=13,
            color=GRAY,
        )
        ax.text(
            centre,
            0.245,
            row["purpose_cn"],
            ha="center",
            fontsize=12,
        )
    ax.annotate(
        "",
        xy=(70000, 0.13),
        xytext=(0, 0.13),
        arrowprops={"arrowstyle": "->", "linewidth": 2, "color": DARK},
    )
    ax.text(35000, 0.055, "严格按时间顺序分区", ha="center", fontsize=17, fontweight="bold")
    ax.text(
        69000,
        0.77,
        "最终测试不参与模型、策略或基线选择",
        ha="right",
        fontsize=16,
        color=ORANGE,
        fontweight="bold",
    )
    ax.set_title("训练、选择与测试严格分离", fontsize=28, fontweight="bold", pad=20)
    fig.tight_layout()
    save_figure(fig, "08_实验时间分区")


def plot_channel_configuration() -> None:
    channels = pd.read_csv(DATA_DIR / "05_六通道配置.csv")
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    table_data = [
        [
            row["logical_channel_cn"],
            row["main_variables_cn"],
            f"{row['steady_cost']:.2f}",
            f"{row['startup_peak_cost']:.2f}",
            str(int(row["warmup_steps"])),
            row["role_cn"],
        ]
        for _, row in channels.iterrows()
    ]
    column_labels = ["逻辑通道", "主要变量", "稳态代价", "启动峰值", "暖机步数", "角色"]
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc="center",
        cellLoc="left",
        colWidths=[0.16, 0.26, 0.10, 0.10, 0.09, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if r == 0:
            cell.set_facecolor("#243650")
            cell.set_text_props(color="white", weight="bold")
        elif r == 1:
            cell.set_facecolor(LIGHT_BLUE)
            cell.set_text_props(weight="bold")
        elif r in (4, 5, 6):
            cell.set_facecolor(LIGHT_GREEN)
    ax.set_title(
        "六个逻辑通道：核心气象通道始终开启，五个可选通道竞争一个名额",
        fontsize=25,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.5,
        0.06,
        "预算0.75；核心通道代价0.25；任一事件专用通道代价0.49，因此两个事件专用通道不能同时开启",
        transform=ax.transAxes,
        ha="center",
        fontsize=15,
        color=GRAY,
    )
    fig.tight_layout()
    save_figure(fig, "09_六通道配置与调度约束")


def plot_robustness_cards(evidence: dict[str, object]) -> None:
    ridge = evidence["ridge"]
    full = evidence["full"]
    cards = [
        ("22/22", "模型选择后新种子", "宏平均损失均优于固定调度"),
        (
            f"{int(full['pdppo_vs_validation_selected_static']['wins'])}/24",
            "完整测试区间",
            "每个种子覆盖5,242个可评估时刻",
        ),
        (
            f"{int(ridge['macro_margin_vs_secondary_static_wins'])}/24",
            "替代预测器",
            "相对重新选择的固定调度",
        ),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.axis("off")
    xs = [0.04, 0.355, 0.67]
    colors = [LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE]
    borders = [BLUE, GREEN, ORANGE]
    for x, color, border, (value, title, subtitle) in zip(xs, colors, borders, cards):
        rect = plt.Rectangle(
            (x, 0.25),
            0.29,
            0.48,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=border,
            linewidth=2.4,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.145,
            0.57,
            value,
            transform=ax.transAxes,
            ha="center",
            fontsize=36,
            color=border,
            fontweight="bold",
        )
        ax.text(
            x + 0.145,
            0.44,
            title,
            transform=ax.transAxes,
            ha="center",
            fontsize=21,
            fontweight="bold",
        )
        ax.text(
            x + 0.145,
            0.33,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
            color=GRAY,
            wrap=True,
        )
    ax.set_title("主要结论不依赖两个试验种子、选定窗口或单一预测器", fontsize=27, fontweight="bold", pad=18)
    fig.tight_layout()
    save_figure(fig, "10_稳健性检查摘要")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    annotate_aws_render()
    export_existing_vector_assets()
    evidence = load_evidence()
    write_compact_data(evidence)
    plot_headline_results(evidence)
    plot_event_heatmap(evidence)
    plot_baseline_comparison()
    plot_behavior_cards(evidence)
    plot_protocol_timeline()
    plot_channel_configuration()
    plot_robustness_cards(evidence)
    print(f"Built assets under {PACKAGE_ROOT}")


if __name__ == "__main__":
    main()
