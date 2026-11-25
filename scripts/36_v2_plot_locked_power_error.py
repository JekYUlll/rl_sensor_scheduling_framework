#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

POLICY_LABELS = {
    "full_open_unconstrained": "Full observation",
    "validation_selected_static": "Val.-selected static",
    "custom_ppo": "PD-PPO",
    "round_robin": "Round-robin",
    "aoi": "AoI",
    "random": "Random",
}

POLICY_ORDER = [
    "full_open_unconstrained",
    "validation_selected_static",
    "custom_ppo",
    "round_robin",
    "aoi",
    "random",
]

POLICY_COLORS = {
    "full_open_unconstrained": "#4C78A8",
    "validation_selected_static": "#72B7B2",
    "custom_ppo": "#1F4E8C",
    "round_robin": "#F58518",
    "aoi": "#54A24B",
    "random": "#8C8C8C",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot chronological final-test V3.1 power-error tradeoff for the paper."
    )
    parser.add_argument(
        "--stats-csv",
        default=str(ROOT / "reports/v31_split_protocol_main/v31_s2_main_stats.csv"),
        help="Corrected split-protocol statistics CSV.",
    )
    parser.add_argument(
        "--power-csv",
        default=str(ROOT / "reports/v31_split_protocol_main/v31_s2_overall_long.csv"),
        help="Corrected final-test rollout CSV used for power coordinates.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "paper/figures/figure6_power_error_tradeoff_v31.png"),
    )
    parser.add_argument(
        "--out-svg",
        default=str(ROOT / "paper/figures/figure6_power_error_tradeoff_v31.svg"),
    )
    args = parser.parse_args()

    stats = pd.read_csv(args.stats_csv)
    power = pd.read_csv(args.power_csv)
    if "policy" in stats.columns:
        stats = stats.rename(
            columns={
                "policy": "method",
                "forecast_weighted_mae_overall_mean": "mean_fwmae",
                "forecast_weighted_mae_overall_std": "std_fwmae",
            }
        )

    required = {"method", "budget", "mean_fwmae", "std_fwmae"}
    missing = required.difference(stats.columns)
    if missing:
        raise ValueError(f"{args.stats_csv} missing columns: {sorted(missing)}")
    if "power_mean" not in power.columns:
        raise ValueError(f"{args.power_csv} missing column: power_mean")

    power_mean = (
        power.groupby(["policy", "budget"], as_index=False)["power_mean"].mean()
    )
    plot_df = stats.merge(
        power_mean,
        left_on=["method", "budget"],
        right_on=["policy", "budget"],
        how="left",
    )
    plot_df = plot_df[plot_df["method"].isin(POLICY_ORDER)].copy()
    if plot_df["power_mean"].isna().any():
        missing_rows = plot_df[plot_df["power_mean"].isna()][["method", "budget"]]
        raise ValueError(f"Missing power coordinates:\n{missing_rows}")

    fig, (ax, ax_full) = plt.subplots(
        1,
        2,
        figsize=(8.0, 4.8),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4.0, 1.0], "wspace": 0.06},
    )
    for policy in POLICY_ORDER:
        subset = plot_df[plot_df["method"] == policy].sort_values("budget")
        if subset.empty:
            continue
        target_ax = ax_full if policy == "full_open_unconstrained" else ax
        target_ax.errorbar(
            subset["power_mean"],
            subset["mean_fwmae"],
            yerr=subset["std_fwmae"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            markersize=5,
            color=POLICY_COLORS.get(policy),
            label=POLICY_LABELS.get(policy, policy),
        )

    ax.set_xlabel("Mean power (normalised units)")
    ax_full.set_xlabel("Full obs.")
    ax.set_ylabel("Forecast-weighted MAE")
    fig.suptitle("Chronological final-test power-error tradeoff")
    ax.set_xlim(1.42, 1.72)
    ax_full.set_xlim(4.55, 4.69)
    for item in (ax, ax_full):
        item.grid(alpha=0.25)

    # Broken-axis visual cue: full observation is an unconstrained diagnostic point
    # far outside the feasible power range.
    ax.spines["right"].set_visible(False)
    ax_full.spines["left"].set_visible(False)
    ax_full.tick_params(labelleft=False, left=False)
    d = 0.015
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_full.transAxes)
    ax_full.plot((-d, +d), (-d, +d), **kwargs)
    ax_full.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    handles, labels = [], []
    for item in (ax, ax_full):
        h, l = item.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=True,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    out_png = Path(args.out_png)
    out_svg = Path(args.out_svg)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_svg)
    plt.close(fig)
    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
