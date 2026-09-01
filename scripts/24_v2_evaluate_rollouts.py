#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.evaluation import (  # noqa: E402
    action_score_metrics,
    event_group_metrics,
    load_rollout_npz,
    overall_metrics,
    sensor_usage_metrics,
    subset_rollout_columns,
    variable_metrics,
)
from v2.forecast_eval import forecast_metric_tables, load_oracle_from_metadata  # noqa: E402


DEFAULT_TARGET_COLUMNS = (
    "air_temperature_c",
    "snow_surface_temperature_c",
    "wind_speed_ms",
    "wind_dir_sin",
    "wind_dir_cos",
    "solar_radiation_wm2",
    "snow_mass_flux_kg_m2_s",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
)

DEFAULT_TARGET_WEIGHTS = (
    1.0,
    1.0,
    1.2,
    0.6,
    0.6,
    1.0,
    3.0,
    2.0,
    2.0,
)

DEFAULT_TARGET_SCALES = (
    5.0,  # air_temperature_c
    5.0,  # snow_surface_temperature_c
    5.0,  # wind_speed_ms
    1.0,  # wind_dir_sin
    1.0,  # wind_dir_cos
    100.0,  # solar_radiation_wm2
    1.0e-4,  # snow_mass_flux_kg_m2_s
    0.2,  # snow_particle_mean_diameter_mm
    5.0,  # snow_particle_mean_velocity_ms
)


def plot_main_summary(overall: pd.DataFrame, out_dir: Path) -> None:
    if overall.empty:
        return
    score_col = (
        "forecast_weighted_mae_overall"
        if "forecast_weighted_mae_overall" in overall.columns
        else "obs_reconstruction_mae"
        if "obs_reconstruction_mae" in overall.columns
        else "weighted_normalized_mae"
        if "weighted_normalized_mae" in overall.columns
        else "mae"
    )
    ordered = overall.sort_values(score_col)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ordered.plot.bar(x="policy", y=score_col, ax=axes[0], legend=False)
    axes[0].set_ylabel(score_col)
    axes[0].set_title("Prediction error by policy")
    axes[0].tick_params(axis="x", rotation=35)
    overall.plot.scatter(x="power_mean", y=score_col, ax=axes[1])
    for _, row in overall.iterrows():
        axes[1].annotate(str(row["policy"]), (float(row["power_mean"]), float(row[score_col])))
    axes[1].set_title("Power vs prediction error")
    axes[1].set_xlabel("Mean power")
    axes[1].set_ylabel(score_col)
    fig.tight_layout()
    fig.savefig(out_dir / "v2_eval_main_summary.png", dpi=180)
    plt.close(fig)


def select_primary_metric(overall: pd.DataFrame) -> str:
    """Select the metric used for the environment-level comparison contract."""
    if "oracle_loss_mean" in overall.columns and overall["oracle_loss_mean"].notna().any():
        return "oracle_loss_mean"
    if "obs_reconstruction_mae" in overall.columns:
        return "obs_reconstruction_mae"
    return "weighted_normalized_mae"


def plot_sensor_diagnostics(usage: pd.DataFrame, action_scores: pd.DataFrame, out_dir: Path) -> None:
    if usage.empty:
        return
    policies = sorted(str(x) for x in usage["policy"].unique())
    for policy in policies:
        use = usage[usage["policy"] == policy].copy()
        scores = action_scores[action_scores["policy"] == policy].copy() if not action_scores.empty else pd.DataFrame()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        use.plot.bar(x="sensor", y=["selected_rate", "warming_rate", "active_rate"], ax=axes[0])
        axes[0].set_title(f"{policy}: selection and states")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].tick_params(axis="x", rotation=35)
        if not scores.empty:
            scores.plot.bar(x="sensor", y=["score_mean", "score_std"], ax=axes[1])
            axes[1].set_title(f"{policy}: actor score distribution")
            axes[1].tick_params(axis="x", rotation=35)
        else:
            axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"v2_eval_sensor_diagnostics_{policy}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v2 rollout NPZ files into paper-oriented tables.")
    parser.add_argument("--run-dir", required=True, help="Directory containing rollout_*.npz files.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run-dir>/evaluation.")
    parser.add_argument("--per-step-budget", type=float, default=None)
    parser.add_argument("--startup-peak-budget", type=float, default=None)
    parser.add_argument("--dtw-window", type=int, default=50)
    parser.add_argument("--target-columns", nargs="*", default=list(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--target-weights", nargs="*", type=float, default=list(DEFAULT_TARGET_WEIGHTS))
    parser.add_argument("--target-scales", nargs="*", type=float, default=list(DEFAULT_TARGET_SCALES))
    parser.add_argument("--forecast-oracle-device", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "v2_ppo_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    target_weights_arg = list(args.target_weights)
    target_scales_arg = list(args.target_scales)
    if target_weights_arg == list(DEFAULT_TARGET_WEIGHTS) and metadata.get("target_weights"):
        target_weights_arg = list(metadata["target_weights"])
    if target_scales_arg == list(DEFAULT_TARGET_SCALES) and metadata.get("target_scales"):
        target_scales_arg = list(metadata["target_scales"])
    target_weights = tuple(float(x) for x in target_weights_arg)
    target_scales = tuple(float(x) for x in target_scales_arg)
    if len(target_weights) != len(args.target_columns):
        raise ValueError(
            f"--target-weights must contain one value per target column: "
            f"{len(target_weights)} weights for {len(args.target_columns)} columns"
        )
    if len(target_scales) != len(args.target_columns):
        raise ValueError(
            f"--target-scales must contain one value per target column: "
            f"{len(target_scales)} scales for {len(args.target_columns)} columns"
        )

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_paths = sorted(run_dir.glob("rollout_*.npz"))
    if not rollout_paths:
        raise FileNotFoundError(f"No rollout_*.npz files found in {run_dir}")

    forecast_oracle = None
    forecast_truth = None
    forecast_target_columns = tuple(str(name) for name in metadata.get("reward_target_columns", args.target_columns))
    if metadata.get("oracle_path") and metadata.get("truth_csv"):
        oracle_device = str(args.forecast_oracle_device or metadata.get("oracle_inference_device", "cpu"))
        forecast_oracle = load_oracle_from_metadata(metadata, run_dir=run_dir, device=oracle_device)
        truth_path = Path(str(metadata["truth_csv"]))
        if not truth_path.exists():
            for base in (Path.cwd(), run_dir, *run_dir.parents):
                candidate = base / truth_path
                if candidate.exists():
                    truth_path = candidate
                    break
        forecast_truth = pd.read_csv(truth_path)

    overall_rows = []
    variable_rows = []
    event_rows = []
    usage_rows = []
    action_rows = []
    forecast_variable_rows = []
    forecast_condition_rows = []
    for path in rollout_paths:
        rollout = load_rollout_npz(path)
        forecast_overall = {}
        if forecast_oracle is not None and forecast_truth is not None:
            forecast_overall, forecast_by_variable, forecast_by_condition = forecast_metric_tables(
                rollout,
                truth_df=forecast_truth,
                oracle=forecast_oracle,
                metadata=metadata,
                target_columns=forecast_target_columns,
                target_weights=target_weights,
                target_scales=target_scales,
            )
            forecast_variable_rows.extend(forecast_by_variable)
            forecast_condition_rows.extend(forecast_by_condition)
        metric_rollout = subset_rollout_columns(rollout, args.target_columns)
        overall_row = overall_metrics(
            metric_rollout,
            per_step_budget=args.per_step_budget,
            startup_peak_budget=args.startup_peak_budget,
            dtw_window=int(args.dtw_window),
            target_weights=target_weights,
            target_scales=target_scales,
        )
        overall_row["obs_reconstruction_mae"] = overall_row["weighted_normalized_mae"]
        overall_row.update({key: value for key, value in forecast_overall.items() if key != "policy"})
        overall_rows.append(overall_row)
        variable_rows.extend(
            variable_metrics(
                metric_rollout,
                dtw_window=int(args.dtw_window),
                target_weights=target_weights,
                target_scales=target_scales,
            )
        )
        event_rows.extend(
            event_group_metrics(
                metric_rollout,
                dtw_window=int(args.dtw_window),
                target_weights=target_weights,
                target_scales=target_scales,
            )
        )
        usage_rows.extend(sensor_usage_metrics(rollout))
        action_rows.extend(action_score_metrics(rollout))

    overall = pd.DataFrame(overall_rows)
    by_variable = pd.DataFrame(variable_rows).sort_values(["policy", "variable"])
    by_event = pd.DataFrame(event_rows).sort_values(["policy", "group"])
    if forecast_variable_rows:
        by_variable = by_variable.merge(
            pd.DataFrame(forecast_variable_rows),
            on=["policy", "variable"],
            how="left",
        )
    by_condition = (
        pd.DataFrame(forecast_condition_rows).sort_values(["policy", "condition"])
        if forecast_condition_rows
        else pd.DataFrame()
    )
    sort_col = "forecast_weighted_mae_overall" if "forecast_weighted_mae_overall" in overall.columns else "weighted_normalized_mae"
    overall = overall.sort_values(sort_col)
    usage = pd.DataFrame(usage_rows).sort_values(["policy", "sensor"]) if usage_rows else pd.DataFrame()
    action_scores = pd.DataFrame(action_rows).sort_values(["policy", "sensor"]) if action_rows else pd.DataFrame()

    overall.to_csv(out_dir / "v2_eval_overall.csv", index=False)
    by_variable.to_csv(out_dir / "v2_eval_by_variable.csv", index=False)
    by_event.to_csv(out_dir / "v2_eval_by_event.csv", index=False)
    by_condition.to_csv(out_dir / "v2_eval_by_condition.csv", index=False)
    usage.to_csv(out_dir / "v2_eval_sensor_usage.csv", index=False)
    action_scores.to_csv(out_dir / "v2_eval_action_scores.csv", index=False)
    plot_main_summary(overall, out_dir)
    plot_sensor_diagnostics(usage, action_scores, out_dir)
    primary_metric = select_primary_metric(overall)
    metadata = {
        "run_dir": str(run_dir),
        "rollout_files": [str(path) for path in rollout_paths],
        "per_step_budget": args.per_step_budget,
        "startup_peak_budget": args.startup_peak_budget,
        "dtw_window": int(args.dtw_window),
        "target_columns": [str(name) for name in args.target_columns],
        "target_weights": list(target_weights),
        "target_scales": list(target_scales),
        "forecast_target_columns": [str(name) for name in forecast_target_columns],
        "primary_metric": primary_metric,
        "secondary_forecast_metric": (
            "forecast_weighted_mae_overall"
            if "forecast_weighted_mae_overall" in overall.columns
            else None
        ),
        "metric_contract": {
            "primary": "environment oracle loss with configured normalization, clipping, and context weights",
            "secondary": "independent frozen-oracle forecast weighted MAE without the environment reward contract",
        },
    }
    (out_dir / "v2_eval_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir / "v2_eval_overall.csv")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
