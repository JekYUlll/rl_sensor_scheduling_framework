#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_diagnostics(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "ppo_training_diagnostics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "timesteps" not in df.columns:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    series = [
        ("oracle_loss_mean", "Oracle loss"),
        ("instant_mae", "Instant MAE"),
        ("dtw_mean", "DTW"),
        ("warmup_abort_count", "Warmup aborts"),
    ]
    for ax, (column, title) in zip(axes.reshape(-1), series, strict=True):
        if column in df.columns:
            ax.plot(df["timesteps"], df[column], marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("PPO training timesteps")
    axes[-1, 1].set_xlabel("PPO training timesteps")
    fig.suptitle("Figure 4 draft: PPO training diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "figure4_ppo_training_diagnostics.png", dpi=200)
    plt.close(fig)


def plot_main_table_summary(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "evaluation" / "v2_eval_overall.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    score_col = "weighted_normalized_mae" if "weighted_normalized_mae" in df.columns else "mae"
    ordered = df.sort_values(score_col)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ordered.plot.bar(x="policy", y=score_col, ax=axes[0], legend=False, color="#4C78A8")
    axes[0].set_title("Main metric under fixed budget")
    axes[0].set_ylabel(score_col)
    axes[0].tick_params(axis="x", rotation=35)
    if "power_mean" in df.columns:
        df.plot.scatter(x="power_mean", y=score_col, ax=axes[1], color="#F58518")
        for _, row in df.iterrows():
            axes[1].annotate(str(row["policy"]), (float(row["power_mean"]), float(row[score_col])), fontsize=8)
        axes[1].set_title("Power vs forecast error")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "table2_main_result_summary.png", dpi=200)
    plt.close(fig)


def plot_sensor_usage_heatmap(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "evaluation" / "v2_eval_sensor_usage.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or not {"policy", "sensor", "active_rate"}.issubset(df.columns):
        return
    pivot = df.pivot(index="policy", columns="sensor", values="active_rate").fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * pivot.shape[1]), max(4, 0.45 * pivot.shape[0] + 2)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Figure 5 companion: active-rate heatmap")
    fig.colorbar(im, ax=ax, label="active rate")
    fig.tight_layout()
    fig.savefig(out_dir / "figure5_sensor_active_rate_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-draft figures from a v2 run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures_paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_diagnostics(run_dir, out_dir)
    plot_main_table_summary(run_dir, out_dir)
    plot_sensor_usage_heatmap(run_dir, out_dir)
    print(out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"failed to build figures: {exc}", file=sys.stderr)
        raise
