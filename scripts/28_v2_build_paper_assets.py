#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


POLICY_LABELS = {
    "full_open_unconstrained": "Full open",
    "feasible_static_projected": "Static oracle",
    "custom_ppo": "PD-PPO",
    "dqn": "DQN",
    "cmdp_dqn": "CMDP-DQN",
    "round_robin": "Round robin",
    "aoi": "AoI",
    "random": "Random",
}

POLICY_ORDER = (
    "full_open_unconstrained",
    "feasible_static_projected",
    "custom_ppo",
    "dqn",
    "cmdp_dqn",
    "round_robin",
    "aoi",
    "random",
)


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def policy_label(policy: str) -> str:
    return POLICY_LABELS.get(str(policy), str(policy))


def timeline_policy_label(policy: str) -> str:
    if policy == "aoi":
        return "Age-of-info"
    return policy_label(policy)


def save_latex_table(df: pd.DataFrame, path: Path, *, float_format: str = "%.4f") -> None:
    path.write_text(df.to_latex(index=True, escape=True, float_format=float_format), encoding="utf-8")


def build_table1(sensor_cfg: Path, out_dir: Path) -> pd.DataFrame:
    import yaml

    data = yaml.safe_load(sensor_cfg.read_text(encoding="utf-8"))
    rows = []
    for item in data.get("sensors", []):
        rows.append(
            {
                "sensor": str(item["sensor_id"]),
                "observed_variables": ", ".join(str(v) for v in item.get("variables", item.get("observed_variables", []))),
                "power_cost": float(item.get("power_cost", 0.0)),
                "startup_peak_power": float(item.get("startup_peak_power", item.get("power_cost", 0.0))),
                "warmup_steps": int(item.get("warmup_steps", 0)),
                "sampling_interval": int(item.get("refresh_interval", item.get("sampling_interval", 1))),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table1_sensor_parameters.csv", index=False)
    (out_dir / "table1_sensor_parameters.tex").write_text(
        table.to_latex(index=False, escape=True, float_format="%.3f"),
        encoding="utf-8",
    )
    return table


def build_table2_copy(table_dir: Path, out_dir: Path) -> pd.DataFrame:
    table = read_csv(table_dir / "table2_main_results.csv", index_col=0)
    table = table.rename(index=policy_label)
    table.to_csv(out_dir / "table2_main_results.csv")
    (out_dir / "table2_main_results.tex").write_text(table.to_latex(escape=True), encoding="utf-8")
    return table


def build_table3_outputs(table_dir: Path, out_dir: Path) -> None:
    by_var = read_csv(table_dir / "table3_by_variable.csv", index_col=0)
    by_var = by_var.rename(index=policy_label)
    by_var.to_csv(out_dir / "table3_by_variable.csv")
    (out_dir / "table3_by_variable.tex").write_text(by_var.to_latex(escape=True, float_format="%.4f"), encoding="utf-8")

    by_cond = read_csv(table_dir / "table3_by_condition.csv", index_col=0)
    by_cond = by_cond.rename(index=policy_label)
    by_cond.to_csv(out_dir / "table3_by_condition.csv")
    (out_dir / "table3_by_condition.tex").write_text(by_cond.to_latex(escape=True), encoding="utf-8")


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, *, width: float = 1.65, height: float = 0.52) -> None:
    x, y = xy
    rect = plt.Rectangle((x, y), width, height, facecolor="#f4f7fb", edgecolor="#2f4057", linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, text: str | None = None) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#2f4057"})
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.08, text, ha="center", fontsize=8)


def figure1_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    boxes = [
        ((0.2, 3.4), "Public weather\ntruth synthesis"),
        ((2.35, 3.4), "Warmup-aware\nsensor FSM"),
        ((4.5, 3.4), "Power projector\nhard constraints"),
        ((6.65, 3.4), "PD-PPO\nmasked policy"),
        ((8.8, 3.4), "Scheduler\nrollout"),
        ((2.35, 1.7), "Observed history\n+ masks"),
        ((4.5, 1.7), "Frozen TCN\nforecast oracle"),
        ((6.65, 1.7), "Forecast loss\nreward"),
        ((8.8, 1.7), "Paper metrics\nMAE / power"),
    ]
    for xy, text in boxes:
        draw_box(ax, xy, text)
    for x in (1.85, 4.0, 6.15, 8.3):
        arrow(ax, (x, 3.66), (x + 0.45, 3.66))
    arrow(ax, (9.62, 3.4), (9.62, 2.22), text="replay")
    arrow(ax, (3.17, 3.4), (3.17, 2.22), text="measure")
    arrow(ax, (4.0, 1.96), (4.5, 1.96))
    arrow(ax, (6.15, 1.96), (6.65, 1.96))
    arrow(ax, (7.45, 2.22), (7.45, 3.4), text="reward")
    arrow(ax, (8.3, 1.96), (8.8, 1.96))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(1.1, 4.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_framework_architecture.svg")
    fig.savefig(out_dir / "figure1_framework_architecture.png", dpi=220)
    plt.close(fig)


def figure2_state_machine(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axis("off")
    positions = {"OFF": (0.8, 2.0), "WARMING": (3.0, 2.0), "ACTIVE": (5.5, 2.0)}
    for name, (x, y) in positions.items():
        circ = plt.Circle((x, y), 0.55, facecolor="#eef6f1", edgecolor="#2d5f45", linewidth=1.5)
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center", fontsize=11, fontweight="bold")
    arrow(ax, (1.35, 2.0), (2.45, 2.0), text="selected")
    arrow(ax, (3.55, 2.0), (4.95, 2.0), text="warmup counter = 0")
    arrow(ax, (5.5, 1.45), (5.5, 0.75), text="deselected")
    arrow(ax, (5.0, 0.75), (1.3, 0.75))
    arrow(ax, (1.0, 0.9), (0.8, 1.45))
    arrow(ax, (3.0, 1.45), (3.0, 0.75), text="abort")
    arrow(ax, (2.55, 0.75), (1.25, 0.75))
    ax.text(3.15, 3.15, "WARMING consumes power but produces no valid observation", ha="center", fontsize=9)
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0.2, 3.5)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_sensor_state_machine.svg")
    fig.savefig(out_dir / "figure2_sensor_state_machine.png", dpi=220)
    plt.close(fig)


def autocorr(x: np.ndarray, max_lag: int = 40) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.full(max_lag + 1, np.nan)
    arr = arr - np.mean(arr)
    denom = float(np.dot(arr, arr))
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    vals = [1.0]
    for lag in range(1, max_lag + 1):
        vals.append(float(np.dot(arr[:-lag], arr[lag:]) / denom))
    return np.asarray(vals, dtype=float)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    sigma = max(float(std), 1e-8)
    return np.exp(-0.5 * ((x - float(mean)) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _first_finite(row: pd.DataFrame, column: str) -> float | None:
    if row.empty or column not in row.columns:
        return None
    value = row[column].iloc[0]
    return float(value) if np.isfinite(value) else None


def figure3_synthetic_statistics(grid_dir: Path, out_dir: Path, *, budget: float, seed: int) -> None:
    run_dir = grid_dir / f"budget{budget_tag(budget)}_seed{seed}"
    truth = read_csv(grid_dir / f"truth_budget{budget_tag(budget)}_seed{seed}.csv")
    validation = read_csv(run_dir / "dataset_validation" / "synthetic_validation.csv")
    variables = ["air_temperature_c", "wind_speed_ms", "snow_mass_flux_kg_m2_s"]
    display_names = {
        "air_temperature_c": "Air temperature",
        "wind_speed_ms": "Wind speed",
        "snow_mass_flux_kg_m2_s": "Snow mass flux",
    }
    fig, axes = plt.subplots(len(variables), 3, figsize=(12, 8.5))
    for row, var in enumerate(variables):
        if var in truth.columns:
            values = truth[var].to_numpy(dtype=float)
        elif var == "snow_mass_flux_kg_m2_s" and "flux_wind_ge_12" in validation["variable"].values:
            values = truth.get(var, pd.Series(np.zeros(len(truth)))).to_numpy(dtype=float)
        else:
            values = np.zeros(len(truth), dtype=float)
        val_row = validation[validation["variable"] == var]
        if val_row.empty and var == "snow_mass_flux_kg_m2_s":
            val_row = validation[validation["variable"] == "flux_wind_ge_12"]
        real_mean = float(val_row["real_mean"].iloc[0]) if not val_row.empty and np.isfinite(val_row["real_mean"].iloc[0]) else float(np.nanmean(values))
        real_std = float(val_row["real_std"].iloc[0]) if not val_row.empty and np.isfinite(val_row["real_std"].iloc[0]) else float(np.nanstd(values))

        ax = axes[row, 0]
        finite = values[np.isfinite(values)]
        ax.hist(finite, bins=50, density=True, alpha=0.65, color="#4C78A8", label="synthetic")
        xs = np.linspace(float(np.nanmin(finite)), float(np.nanmax(finite)), 200) if finite.size else np.linspace(0, 1, 2)
        if np.isfinite(real_mean) and np.isfinite(real_std) and real_std > 0:
            ax.plot(xs, normal_pdf(xs, real_mean, real_std), color="#F58518", lw=1.8, label="AntAWS moment anchor")
        ax.set_title(display_names.get(var, var))
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        acf = autocorr(finite, max_lag=40)
        ax.plot(np.arange(acf.size), acf, color="#54A24B")
        if not val_row.empty and "acf_max_abs_delta_lag1_20" in val_row:
            delta = val_row["acf_max_abs_delta_lag1_20"].iloc[0]
            if np.isfinite(delta):
                ax.axhspan(-float(delta), float(delta), color="#54A24B", alpha=0.12, label="reported max delta")
                ax.legend(fontsize=7)
        ax.set_ylim(-0.2, 1.05)
        ax.set_title("Synthetic ACF")
        ax.set_xlabel("lag")

        ax = axes[row, 2]
        ks_pvalue = _first_finite(val_row, "ks_pvalue")
        acf_delta = _first_finite(val_row, "acf_max_abs_delta_lag1_20")
        if ks_pvalue is not None or acf_delta is not None:
            labels = ["KS p-value", "ACF max delta"]
            vals = [0.0 if ks_pvalue is None else ks_pvalue, 0.0 if acf_delta is None else acf_delta]
            ax.bar(labels, vals, color=["#72B7B2", "#E45756"])
            ax.set_ylim(0.0, max(1.05, max(vals) * 1.15))
            ax.set_title("Validation summary")
        elif var == "snow_mass_flux_kg_m2_s" and not val_row.empty:
            # The blowing-snow flux row is conditionally generated from wind-speed
            # priors, so the locked validation table reports conditional synthetic
            # moments rather than KS/ACF diagnostics against an AntAWS marginal.
            synth_mean = _first_finite(val_row, "synthetic_mean")
            synth_std = _first_finite(val_row, "synthetic_std")
            labels = ["mean\n(u>=12)", "std\n(u>=12)"]
            vals = [0.0 if synth_mean is None else synth_mean, 0.0 if synth_std is None else synth_std]
            ax.bar(labels, vals, color=["#72B7B2", "#E45756"])
            ymax = max(vals) * 1.25 if max(vals) > 0 else 1.0
            ax.set_ylim(0.0, ymax)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.set_title("Conditional flux prior")
        else:
            ax.text(0.5, 0.5, "diagnostics\nnot available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Validation summary")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 3: synthetic truth statistics and AntAWS anchors")
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_synthetic_statistics.png", dpi=220)
    fig.savefig(out_dir / "figure3_synthetic_statistics.svg")
    plt.close(fig)


def figure4_learning_curves(table_dir: Path, out_dir: Path) -> None:
    curve_dir = table_dir / "figure4_learning_curves"
    logs = sorted(curve_dir.glob("seed*_training_log.csv"))
    if not logs:
        return
    frames = []
    for path in logs:
        frame = pd.read_csv(path)
        seed = int(path.stem.split("_")[0].replace("seed", ""))
        frame["seed"] = seed
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    metrics = [
        ("loss", "PPO total loss"),
        ("entropy_mean", "Policy entropy"),
        ("prior_kl_loss", "Prior KL regularizer"),
        ("advantage_std", "Advantage std"),
    ]
    for ax, (col, title) in zip(axes.reshape(-1), metrics, strict=True):
        if col not in df.columns:
            ax.axis("off")
            continue
        grouped = df.groupby("step")[col]
        mean = grouped.mean()
        std = grouped.std(ddof=1).fillna(0.0)
        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        s = std.to_numpy(dtype=float)
        ax.plot(x, y, color="#4C78A8")
        ax.fill_between(x, y - s, y + s, color="#4C78A8", alpha=0.18)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("training steps")
    axes[-1, 1].set_xlabel("training steps")
    fig.suptitle("Figure 4: PD-PPO training diagnostics, budget=1.70")
    fig.tight_layout()
    fig.savefig(out_dir / "figure4_custom_ppo_learning_curves.png", dpi=220)
    fig.savefig(out_dir / "figure4_custom_ppo_learning_curves.svg")
    plt.close(fig)


def figure5_policy_timeline(grid_dir: Path, out_dir: Path, *, budget: float, seed: int, max_steps: int = 240) -> None:
    run_dir = grid_dir / f"budget{budget_tag(budget)}_seed{seed}"
    policies = ["custom_ppo", "feasible_static_projected", "round_robin", "aoi", "random"]
    fig, axes = plt.subplots(len(policies), 1, figsize=(13, 8.2), sharex=True, constrained_layout=True)
    mode_cmap = ListedColormap(["#F3F4F6", "#56B4E9", "#D55E00"])
    mode_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], mode_cmap.N)
    sensor_ids: list[str] | None = None
    im = None
    for ax, policy in zip(axes, policies, strict=True):
        path = run_dir / f"rollout_{policy}.npz"
        if not path.exists():
            ax.axis("off")
            ax.text(0.5, 0.5, f"{policy_label(policy)} unavailable", ha="center", va="center", transform=ax.transAxes)
            continue
        data = np.load(path, allow_pickle=True)
        modes = np.asarray(data["mode_ids"], dtype=float)[:max_steps].T
        sensor_ids = [str(x) for x in data["sensor_ids"].tolist()]
        im = ax.imshow(modes, aspect="auto", interpolation="nearest", cmap=mode_cmap, norm=mode_norm)
        ax.set_yticks(range(len(sensor_ids)))
        ax.set_yticklabels(sensor_ids, fontsize=7)
        ax.set_yticks(np.arange(-0.5, len(sensor_ids), 1.0), minor=True)
        ax.grid(which="minor", axis="y", color="#FFFFFF", linewidth=1.4)
        ax.tick_params(axis="y", which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
        ax.set_title(timeline_policy_label(policy))
        ax.grid(False)
    axes[-1].set_xlabel("time index")
    if sensor_ids is not None and im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes,
            location="right",
            shrink=0.86,
            pad=0.015,
            ticks=[0, 1, 2],
            boundaries=[-0.5, 0.5, 1.5, 2.5],
            spacing="uniform",
        )
        cbar.ax.set_yticklabels(["OFF", "WARMING", "ACTIVE"])
        cbar.set_label("sensor mode")
    fig.savefig(out_dir / "figure5_sensor_timeline.png", dpi=220)
    fig.savefig(out_dir / "figure5_sensor_timeline.svg")
    plt.close(fig)


def figure6_power_error_tradeoff(table_dir: Path, out_dir: Path) -> None:
    df = read_csv(table_dir / "overall_long.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for policy in POLICY_ORDER:
        subset = df[df["policy"] == policy]
        if subset.empty:
            continue
        grouped = subset.groupby("budget").agg(
            error=("forecast_weighted_mae_overall", "mean"),
            power=("power_mean", "mean"),
        )
        ax.plot(grouped["power"], grouped["error"], marker="o", label=policy_label(policy))
    ax.set_xlabel("mean power")
    ax.set_ylabel("forecast weighted MAE")
    ax.set_title("Power-error tradeoff across budgets")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure6_power_error_tradeoff.png", dpi=220)
    fig.savefig(out_dir / "figure6_power_error_tradeoff.svg")
    plt.close(fig)


def write_summary(table_dir: Path, out_dir: Path) -> None:
    overall = read_csv(table_dir / "overall_long.csv")
    focus = overall[overall["budget"] == 1.70]
    means = focus.groupby("policy")["forecast_weighted_mae_overall"].mean()
    lines = [
        "# v2 Paper Asset Summary",
        "",
        "Main metric: forecast_weighted_mae_overall, lower is better.",
        "",
    ]
    if {"custom_ppo", "round_robin", "aoi", "random", "feasible_static_projected", "full_open_unconstrained"}.issubset(means.index):
        ppo = float(means["custom_ppo"])
        rr = float(means["round_robin"])
        aoi = float(means["aoi"])
        static = float(means["feasible_static_projected"])
        full = float(means["full_open_unconstrained"])
        lines.extend(
            [
                f"- budget=1.70 PD-PPO: {ppo:.4f}",
                f"- improvement vs round_robin: {(rr - ppo) / rr * 100.0:.2f}%",
                f"- improvement vs AoI: {(aoi - ppo) / aoi * 100.0:.2f}%",
                f"- gap vs feasible_static_projected: {(ppo - static) / static * 100.0:.2f}%",
                f"- gap vs full_open_unconstrained: {(ppo - full) / full * 100.0:.2f}%",
            ]
        )
    if "dqn" in means.index:
        dqn = float(means["dqn"])
        lines.extend(["", f"- budget=1.70 DQN: {dqn:.4f}", ""])
    else:
        lines.extend(
            [
                "",
                "Note: this asset set does not include a DQN row. "
                "Merge a DQN grid with scripts/27_v2_aggregate_results.py before "
                "building final comparison assets.",
                "",
            ]
        )
    (out_dir / "paper_asset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready v2 tables and figures from locked KL=1.0 results.")
    parser.add_argument("--grid-dir", default="reports/v2_forecast_eval_grid_prior_kl1")
    parser.add_argument("--table-dir", default="reports/v2_paper_tables_prior_kl1")
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--out-dir", default="reports/v2_paper_assets_prior_kl1")
    parser.add_argument("--focus-budget", type=float, default=1.70)
    parser.add_argument("--focus-seed", type=int, default=41)
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    table_dir = Path(args.table_dir)
    sensor_cfg = Path(args.sensor_cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_table1(sensor_cfg, out_dir)
    build_table2_copy(table_dir, out_dir)
    build_table3_outputs(table_dir, out_dir)
    figure1_architecture(out_dir)
    figure2_state_machine(out_dir)
    figure3_synthetic_statistics(grid_dir, out_dir, budget=float(args.focus_budget), seed=int(args.focus_seed))
    figure4_learning_curves(table_dir, out_dir)
    figure5_policy_timeline(grid_dir, out_dir, budget=float(args.focus_budget), seed=int(args.focus_seed))
    figure6_power_error_tradeoff(table_dir, out_dir)
    write_summary(table_dir, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
