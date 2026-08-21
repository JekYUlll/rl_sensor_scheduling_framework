#!/usr/bin/env python3
"""Build supporting figures for the PD-PPO CRST rewrite.

The figures are intentionally manuscript-facing rather than experiment-log
artefacts:

* operational behaviour: channel-use validity and switching rates under the
  deployment-constrained result;
* fixed-budget trade-off: forecast loss versus realised power in the
  fixed-budget reference experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

OPERATIONAL_RAW = (
    ROOT
    / "reports"
    / "v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced"
    / "raw"
)
FIXED_STATS = ROOT / "reports" / "v31_split_protocol_main" / "v31_s2_main_stats.csv"

PALETTE = {
    "blue": "#4A90D9",
    "blue_dark": "#2F5F9F",
    "green": "#5BA58B",
    "green_fill": "#DEEFDE",
    "amber": "#D4A252",
    "amber_fill": "#FFF4CC",
    "red": "#C76F64",
    "red_fill": "#FFE2DE",
    "purple": "#B77AA8",
    "slate": "#7B8794",
    "gray": "#D8DCE0",
    "ink": "#2F3437",
    "panel": "#FAFAF8",
}

METHOD_LABELS = {
    "custom_ppo": "PD-PPO",
    "validation_selected_static": "Compact static",
    "duty_constrained_validation_selected_static": "Deployable static",
    "duty_constrained_round_robin": "Duty round-robin",
    "duty_constrained_aoi": "Duty AoI",
    "round_robin": "Round-robin",
    "aoi": "AoI",
    "random": "Random",
    "full_open_unconstrained": "Full observation",
}

METHOD_COLORS = {
    "custom_ppo": PALETTE["blue_dark"],
    "validation_selected_static": PALETTE["slate"],
    "duty_constrained_validation_selected_static": PALETTE["green"],
    "duty_constrained_round_robin": PALETTE["amber"],
    "duty_constrained_aoi": PALETTE["purple"],
    "round_robin": PALETTE["amber"],
    "aoi": PALETTE["green"],
    "random": PALETTE["red"],
    "full_open_unconstrained": PALETTE["blue"],
}

METHOD_MARKERS = {
    "validation_selected_static": "D",
    "custom_ppo": "o",
    "round_robin": "s",
    "aoi": "^",
    "random": "X",
    "full_open_unconstrained": "o",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.9,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.9,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5C6368",
            "axes.linewidth": 0.65,
            "axes.facecolor": PALETTE["panel"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "errorbar.capsize": 3.5,
        }
    )


def load_operational_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in range(41, 51):
        path = OPERATIONAL_RAW / f"budget1p70_seed{seed}" / "v2_custom_ppo_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_csv(path)
        part["seed"] = seed
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def plot_operational_behaviour() -> None:
    set_style()
    df = load_operational_metrics()
    policies = [
        "custom_ppo",
        "validation_selected_static",
        "duty_constrained_validation_selected_static",
        "duty_constrained_round_robin",
        "duty_constrained_aoi",
    ]
    plot_df = df[df["policy"].isin(policies)].copy()

    mean_counts = (
        plot_df.groupby("policy", as_index=False)[
            [
                "always_off_sensor_count",
                "mid_duty_sensor_count",
                "always_on_sensor_count",
            ]
        ]
        .mean()
        .set_index("policy")
        .loc[policies]
    )

    fig, (ax_count, ax_switch) = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.15),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.08)

    y = np.arange(len(policies))
    left = np.zeros(len(policies))
    stack_items = [
        ("always_off_sensor_count", "Always off", PALETTE["red_fill"]),
        ("mid_duty_sensor_count", "Intermediate duty", PALETTE["green_fill"]),
        ("always_on_sensor_count", "Always on", "#D9E7FC"),
    ]
    for col, label, color in stack_items:
        values = mean_counts[col].to_numpy(dtype=float)
        ax_count.barh(
            y,
            values,
            left=left,
            height=0.64,
            color=color,
            edgecolor=PALETTE["ink"],
            linewidth=0.45,
            label=label,
        )
        for yi, x0, val in zip(y, left, values, strict=False):
            if val >= 0.55:
                ax_count.text(
                    x0 + val / 2,
                    yi,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    color=PALETTE["ink"],
                    fontsize=7.4,
                )
        left += values
    ax_count.set_yticks(y, [METHOD_LABELS[p] for p in policies])
    ax_count.invert_yaxis()
    ax_count.set_xlim(0, 8)
    ax_count.set_xlabel("logical channels")
    ax_count.set_title("(a) Channel-use validity")
    ax_count.grid(axis="x", color=PALETTE["gray"], linewidth=0.55, alpha=0.65)
    ax_count.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, -0.30),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.8,
    )

    x_positions = np.arange(len(policies))
    for idx, policy in enumerate(policies):
        sub = plot_df[plot_df["policy"] == policy].sort_values("seed")
        x = np.full(len(sub), idx, dtype=float)
        # Deterministic small offsets make seed-level dots visible without random jitter.
        offsets = np.linspace(-0.12, 0.12, len(sub))
        yvals = sub["switches_per_step"].to_numpy(dtype=float) * 100.0
        ax_switch.scatter(
            x + offsets,
            yvals,
            s=26,
            color=METHOD_COLORS[policy],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        mean = float(yvals.mean())
        ax_switch.plot(
            [idx - 0.18, idx + 0.18],
            [mean, mean],
            color=PALETTE["ink"],
            linewidth=1.1,
            zorder=4,
        )
        ax_switch.text(
            idx,
            mean + 0.22,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=PALETTE["ink"],
        )
    ax_switch.set_xticks(x_positions, [METHOD_LABELS[p] for p in policies], rotation=28, ha="right")
    ax_switch.set_ylabel("switches per step (%)")
    ax_switch.set_title("(b) Switching rate across seeds")
    ax_switch.set_ylim(-0.45, 6.55)
    ax_switch.grid(axis="y", color=PALETTE["gray"], linewidth=0.55, alpha=0.65)
    ax_switch.tick_params(width=0.6, length=2.4, color="#5C6368")
    ax_count.tick_params(width=0.6, length=2.4, color="#5C6368")

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "figure_operational_behavior.png"
    svg = OUT / "figure_operational_behavior.svg"
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(png)
    print(svg)


def plot_fixed_budget_tradeoff() -> None:
    set_style()
    if not FIXED_STATS.exists():
        raise FileNotFoundError(FIXED_STATS)
    stats = pd.read_csv(FIXED_STATS)
    rename = {
        "forecast_weighted_mae_overall_mean": "fwmae_mean",
        "forecast_weighted_mae_overall_std": "fwmae_std",
        "power_mean_mean": "power_mean",
        "power_mean_std": "power_std",
    }
    stats = stats.rename(columns=rename)
    methods = [
        "validation_selected_static",
        "custom_ppo",
        "round_robin",
        "aoi",
        "random",
    ]
    fig, (ax, ax_ref) = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.25),
        sharey=True,
        gridspec_kw={"width_ratios": [4.0, 0.82], "wspace": 0.05},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.04, wspace=0.05)

    for method in methods:
        sub = stats[stats["policy"] == method].sort_values("budget")
        if sub.empty:
            continue
        color = METHOD_COLORS[method]
        ax.errorbar(
            sub["power_mean"],
            sub["fwmae_mean"],
            yerr=sub["fwmae_std"],
            xerr=sub["power_std"],
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=5.7,
            capsize=3.5,
            capthick=1.35,
            elinewidth=1.85,
            linewidth=2.05,
            color=color,
            label=METHOD_LABELS[method],
            alpha=0.96,
            markeredgewidth=1.0,
            markeredgecolor="white" if method != "validation_selected_static" else color,
            markerfacecolor=color if method != "validation_selected_static" else "white",
            zorder=5 if method == "custom_ppo" else 4,
        )

    ref = stats[stats["policy"] == "full_open_unconstrained"].sort_values("budget")
    ax_ref.errorbar(
        ref["power_mean"],
        ref["fwmae_mean"],
        yerr=ref["fwmae_std"],
        marker=METHOD_MARKERS["full_open_unconstrained"],
        markersize=5.7,
        capsize=3.5,
        capthick=1.35,
        elinewidth=1.85,
        linewidth=2.05,
        color=METHOD_COLORS["full_open_unconstrained"],
        markeredgewidth=1.0,
        markeredgecolor="white",
        label=METHOD_LABELS["full_open_unconstrained"],
        zorder=5,
    )

    ax.set_xlabel("mean power (normalised units)")
    ax.set_ylabel("forecast-weighted MAE")
    ax.set_title("(a) Constrained policies")
    ax_ref.set_xlabel("full obs.")
    ax_ref.set_title("(b) Reference")
    ax.set_xlim(1.435, 1.69)
    ax_ref.set_xlim(4.57, 4.67)
    for item in (ax, ax_ref):
        item.grid(color=PALETTE["gray"], linewidth=0.45, alpha=0.45)
        item.tick_params(width=0.6, length=2.4, color="#5C6368")
    ax.spines["right"].set_visible(False)
    ax_ref.spines["left"].set_visible(False)
    ax_ref.tick_params(labelleft=False, left=False)

    d = 0.014
    kwargs = dict(transform=ax.transAxes, color=PALETTE["ink"], clip_on=False, linewidth=0.9)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_ref.transAxes)
    ax_ref.plot((-d, +d), (-d, +d), **kwargs)
    ax_ref.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    handles, labels = ax.get_legend_handles_labels()
    ref_handles, ref_labels = ax_ref.get_legend_handles_labels()
    fig.legend(
        handles + ref_handles,
        labels + ref_labels,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.18),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.3,
        handletextpad=0.35,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "figure_fixed_budget_power_error.png"
    svg = OUT / "figure_fixed_budget_power_error.svg"
    fig.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(png)
    print(svg)


def main() -> None:
    plot_operational_behaviour()
    plot_fixed_budget_tradeoff()


if __name__ == "__main__":
    main()
