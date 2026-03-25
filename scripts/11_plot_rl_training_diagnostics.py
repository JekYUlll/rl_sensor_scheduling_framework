from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCHEDULERS = ("dqn", "cmdp_dqn", "ppo")


def _rolling_stats(values: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = pd.to_numeric(values, errors="coerce")
    mean = clean.rolling(window=window, min_periods=1).mean()
    std = clean.rolling(window=window, min_periods=1).std().fillna(0.0)
    return clean.to_numpy(dtype=float), mean.to_numpy(dtype=float), std.to_numpy(dtype=float)


def _plot_band(ax: plt.Axes, x: np.ndarray, series: pd.Series, *, color: str, label: str, window: int) -> None:
    raw, mean, std = _rolling_stats(series, window)
    mask = np.isfinite(mean)
    if not np.any(mask):
        return
    ax.plot(x[mask], raw[mask], color=color, alpha=0.18, linewidth=1.0)
    ax.plot(x[mask], mean[mask], color=color, linewidth=2.0, label=label)
    lower = mean - std
    upper = mean + std
    ax.fill_between(x[mask], lower[mask], upper[mask], color=color, alpha=0.18)


def _scheduler_log(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "training_log.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_single_scheduler(run_tag: str, scheduler: str, df: pd.DataFrame, out_dir: Path, window: int) -> None:
    x = np.arange(len(df), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.reshape(-1)

    metric_specs = [
        ("reward", "Episode reward", "tab:blue"),
        ("val_objective", "Validation objective", "tab:orange"),
        ("power", "Average power", "tab:green"),
        ("coverage", "Coverage", "tab:red"),
    ]
    fallback_specs = {
        "val_objective": "trace_P",
    }

    for ax, (column, title, color) in zip(axes, metric_specs, strict=True):
        use_col = column if column in df.columns and df[column].notna().any() else fallback_specs.get(column)
        if use_col is None or use_col not in df.columns or not df[use_col].notna().any():
            ax.set_visible(False)
            continue
        label = use_col if use_col != column else title
        _plot_band(ax, x, df[use_col], color=color, label=label, window=window)
        ax.set_title(title if use_col == column else f"{title} (fallback: {use_col})")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=False)
    axes[2].set_xlabel("episode")
    axes[3].set_xlabel("episode")
    fig.suptitle(f"{scheduler} training diagnostics | {run_tag}", fontsize=14)
    _save_fig(fig, out_dir / f"{scheduler}_training_shaded.png")

    constraint_cols = [
        ("lambda_avg", "Lambda avg", "tab:purple"),
        ("avg_power_violation", "Avg-power violation", "tab:brown"),
        ("peak_violation_rate", "Peak violation rate", "tab:olive"),
        ("total_energy", "Episode energy", "tab:cyan"),
    ]
    active = [spec for spec in constraint_cols if spec[0] in df.columns and df[spec[0]].notna().any()]
    if active:
        fig2, axes2 = plt.subplots(len(active), 1, figsize=(10, 2.2 * len(active)), sharex=True)
        if not isinstance(axes2, np.ndarray):
            axes2 = np.array([axes2])
        for ax, (column, title, color) in zip(axes2, active, strict=True):
            _plot_band(ax, x, df[column], color=color, label=column, window=window)
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", frameon=False)
        axes2[-1].set_xlabel("episode")
        fig2.suptitle(f"{scheduler} constraint diagnostics | {run_tag}", fontsize=14)
        _save_fig(fig2, out_dir / f"{scheduler}_constraints_shaded.png")


def _plot_compare(run_tag: str, logs: dict[str, pd.DataFrame], out_dir: Path, window: int) -> None:
    colors = {
        "dqn": "tab:blue",
        "cmdp_dqn": "tab:orange",
        "ppo": "tab:green",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.reshape(-1)
    metrics = [
        ("reward", "Episode reward"),
        ("power", "Average power"),
        ("coverage", "Coverage"),
        ("val_objective", "Validation objective"),
    ]
    fallback = {"val_objective": "trace_P"}
    for ax, (column, title) in zip(axes, metrics, strict=True):
        any_plotted = False
        for scheduler, df in logs.items():
            use_col = column if column in df.columns and df[column].notna().any() else fallback.get(column)
            if use_col is None or use_col not in df.columns or not df[use_col].notna().any():
                continue
            x = np.arange(len(df), dtype=float)
            _plot_band(ax, x, df[use_col], color=colors.get(scheduler, None) or f"C{len(colors)}", label=scheduler, window=window)
            any_plotted = True
        if not any_plotted:
            ax.set_visible(False)
            continue
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=False)
    axes[2].set_xlabel("episode")
    axes[3].set_xlabel("episode")
    fig.suptitle(f"RL training comparison | {run_tag}", fontsize=14)
    _save_fig(fig, out_dir / "rl_training_compare_shaded.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot shaded RL training diagnostics from training_log.csv files")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--schedulers", nargs="+", default=list(DEFAULT_SCHEDULERS))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--reports-dir", default="reports/runs")
    parser.add_argument("--aggregate-dir", default="reports/aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / args.reports_dir
    out_dir = repo_root / args.aggregate_dir / f"posthoc_{args.run_tag}" / "rl_training"
    logs: dict[str, pd.DataFrame] = {}
    for scheduler in args.schedulers:
        run_dir = runs_dir / f"{args.run_tag}_{scheduler}"
        df = _scheduler_log(run_dir)
        if df is None:
            print(f"[skip] missing training log for {scheduler}: {run_dir / 'training_log.csv'}")
            continue
        logs[scheduler] = df
        _plot_single_scheduler(args.run_tag, scheduler, df, out_dir, args.window)
        print(f"[ok] wrote shaded RL plots for {scheduler} -> {out_dir}")
    if logs:
        _plot_compare(args.run_tag, logs, out_dir, args.window)
        print(f"[ok] wrote RL comparison plot -> {out_dir}")
    else:
        raise SystemExit("No training logs found for requested schedulers")


if __name__ == "__main__":
    main()
