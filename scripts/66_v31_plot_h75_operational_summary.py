#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PAPER = {
    "blue_fill": "#D9E7FC",
    "green_fill": "#DEEFDE",
    "yellow_fill": "#FFF4CC",
    "red_fill": "#FFE2DE",
    "cyan_fill": "#DDF2F0",
    "gray_fill": "#EEF1F4",
    "blue": "#4A90D9",
    "green": "#5BA58B",
    "amber": "#D4A252",
    "red": "#C76F64",
    "slate": "#7B8794",
    "purple": "#B77AA8",
    "ink": "#2F3437",
    "grid": "#D8DCE0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the deployment-constrained PD-PPO result.")
    parser.add_argument(
        "--summary",
        default=(
            "reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/"
            "env_dwell12_h75_operational_summary_10seed.csv"
        ),
    )
    parser.add_argument(
        "--comparisons",
        default=(
            "reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/"
            "env_dwell12_h75_operational_summary_10seed_comparisons.csv"
        ),
    )
    parser.add_argument(
        "--out",
        default="paper/figures/figure_operational_summary.png",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    return out


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(resolve(args.summary))
    comparisons = pd.read_csv(resolve(args.comparisons))
    summary = summary[summary["complete"].astype(bool)].copy()

    method_cols = [
        ("PD-PPO", "pdppo_oracle_loss"),
        ("Compact static", "selected_static_oracle_loss"),
        ("Deployable static", "deployable_selected_static_oracle_loss"),
        ("Best dynamic", "best_original_dynamic_oracle_loss"),
        ("Best duty baseline", "best_duty_non_pdppo_oracle_loss"),
    ]
    colors = {
        "PD-PPO": PAPER["blue"],
        "Compact static": PAPER["slate"],
        "Deployable static": PAPER["green"],
        "Best dynamic": PAPER["amber"],
        "Best duty baseline": PAPER["purple"],
    }
    markers = {
        "PD-PPO": "o",
        "Compact static": "D",
        "Deployable static": "s",
        "Best dynamic": "^",
        "Best duty baseline": "P",
    }
    offsets = {
        "PD-PPO": -0.20,
        "Compact static": -0.10,
        "Deployable static": 0.00,
        "Best dynamic": 0.10,
        "Best duty baseline": 0.20,
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5C6368",
            "axes.linewidth": 0.65,
            "axes.facecolor": "#FAFAF8",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 180,
        }
    )

    fig = plt.figure(figsize=(7.35, 3.20), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.05, hspace=0.04, wspace=0.05)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])

    seeds = summary["seed"].astype(int).to_numpy()
    for label, col in method_cols:
        values = summary[col].astype(float).to_numpy()
        size = 42 if label == "PD-PPO" else 32
        alpha = 1.0 if label == "PD-PPO" else 0.86
        x = seeds + offsets[label]
        if label == "Compact static":
            ax_loss.scatter(
                x,
                values,
                marker=markers[label],
                s=size,
                facecolor="white",
                edgecolor=colors[label],
                linewidth=1.05,
                alpha=0.92,
                label=label,
                zorder=4,
            )
            continue
        ax_loss.scatter(
            x,
            values,
            marker=markers[label],
            s=size,
            color=colors[label],
            edgecolor="white",
            linewidth=0.35,
            alpha=alpha,
            label=label,
            zorder=5 if label == "PD-PPO" else 4,
        )
    ax_loss.set_title("Held-out oracle loss by seed")
    ax_loss.set_xlabel("seed")
    ax_loss.set_ylabel("oracle loss (lower is better)")
    ax_loss.set_xticks(seeds)
    ax_loss.grid(axis="y", color=PAPER["grid"], linewidth=0.55, alpha=0.68)
    ax_loss.tick_params(width=0.6, length=2.4, color="#5C6368")
    ax_loss.title.set_fontweight("semibold")

    wanted = [
        ("Deployable static", "deployable_selected_static"),
        ("Best dynamic", "best_original_dynamic"),
        ("Best duty baseline", "best_duty_non_pdppo"),
        ("Compact static", "selected_static"),
    ]
    comp_rows = []
    for label, key in wanted:
        row = comparisons.loc[comparisons["comparison"].astype(str) == key]
        if row.empty:
            continue
        item = row.iloc[0]
        comp_rows.append(
            {
                "label": label,
                "wins": int(item["pdppo_win_count"]),
                "n": int(item["n"]),
                "delta": float(item["mean_delta_baseline_minus_pdppo"]),
            }
        )
    comp = pd.DataFrame(comp_rows)
    y = np.arange(len(comp))
    bar_colors = [colors[item] for item in comp["label"]]
    ax_delta.axvline(0.0, color=PAPER["ink"], linewidth=0.75)
    ax_delta.barh(y, comp["delta"], color=bar_colors, alpha=0.82, edgecolor="none")
    for yi, row in comp.iterrows():
        x = float(row["delta"])
        if x <= 0.0004:
            label_x = 0.00022
            ha = "left"
        else:
            label_x = x + 0.00028
            ha = "left"
        ax_delta.text(
            label_x,
            yi,
            f"{row['wins']}/{row['n']}",
            va="center",
            ha=ha,
            fontsize=7.2,
            color=PAPER["ink"],
        )
    ax_delta.set_yticks(y, comp["label"])
    ax_delta.invert_yaxis()
    ax_delta.set_title("Mean advantage and wins")
    ax_delta.set_xlabel("comparator loss - PD-PPO loss")
    ax_delta.grid(axis="x", color=PAPER["grid"], linewidth=0.55, alpha=0.68)
    ax_delta.tick_params(width=0.6, length=2.4, color="#5C6368")
    ax_delta.title.set_fontweight("semibold")

    handles, labels = ax_loss.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.055),
        ncol=5,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.35,
        handletextpad=0.35,
    )
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=360, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
