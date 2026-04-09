from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.forecast_metrics import compute_forecast_metrics
from evaluation.sequence_metrics import dtw_distance_1d, pearson_1d
from forecasting.dataset_builder import ForecastDataset, build_window_dataset
from forecasting.series_preparation import extract_context_series, prepare_input_and_targets
from reward.forecast_reward import (
    FrozenForecastRewardEnsemble,
    FrozenForecastRewardOracle,
    load_reward_oracle,
)


FOCUS_FEATURES = (
    "air_temperature_c",
    "snow_surface_temperature_c",
    "wind_speed_ms",
    "snow_mass_flux_kg_m2_s",
    "snow_particle_mean_velocity_ms",
)

BASELINE_SCHEDULERS = (
    "full_open",
    "periodic",
    "round_robin",
    "dqn",
    "cmdp_dqn",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    mode: str
    columns: tuple[str, ...]
    window: int = 5
    alpha: float = 0.3


VARIANT_SPECS = (
    VariantSpec(
        name="full_open_raw",
        mode="identity",
        columns=(),
    ),
    VariantSpec(
        name="full_open_ma5_primary",
        mode="moving_average",
        columns=("air_temperature_c", "snow_surface_temperature_c", "wind_speed_ms"),
        window=5,
    ),
    VariantSpec(
        name="full_open_ma5_snow",
        mode="moving_average",
        columns=("snow_mass_flux_kg_m2_s", "snow_particle_mean_velocity_ms"),
        window=5,
    ),
    VariantSpec(
        name="full_open_ma5_core",
        mode="moving_average",
        columns=(
            "air_temperature_c",
            "snow_surface_temperature_c",
            "wind_speed_ms",
            "snow_mass_flux_kg_m2_s",
            "snow_particle_mean_velocity_ms",
        ),
        window=5,
    ),
    VariantSpec(
        name="full_open_ema03_core",
        mode="ema",
        columns=(
            "air_temperature_c",
            "snow_surface_temperature_c",
            "wind_speed_ms",
            "snow_mass_flux_kg_m2_s",
            "snow_particle_mean_velocity_ms",
        ),
        alpha=0.3,
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--reward-artifact",
        default=None,
        help="Defaults to reports/runs/<run_tag>_reward_model/reward_oracles.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to reports/aggregate/diagnostics_<run_tag>",
    )
    parser.add_argument(
        "--models",
        default="tcn,lstm,transformer",
        help="Comma-separated reward-oracle models to evaluate",
    )
    parser.add_argument(
        "--baselines",
        default=",".join(BASELINE_SCHEDULERS),
        help="Comma-separated existing schedulers to compare against",
    )
    return parser.parse_args()


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if window <= 1:
        return np.array(arr, copy=True)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _ema_1d(values: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.array(arr, copy=True)
    out = np.array(arr, copy=True)
    for idx in range(1, arr.size):
        out[idx] = float(alpha) * arr[idx] + (1.0 - float(alpha)) * out[idx - 1]
    return out


def _high_freq_energy_ratio(values: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 8:
        return float("nan")
    centered = arr - float(np.mean(arr))
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    if power.size <= 1:
        return float("nan")
    freqs = np.fft.rfftfreq(arr.size, d=1.0)
    hi_mask = freqs >= float(cutoff_ratio) * float(np.max(freqs))
    total = float(np.sum(power[1:]))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(power[hi_mask]) / total)


def _rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def _mae_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(diff)))


def _load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_csv_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def _select_oracle(
    reward_oracle: FrozenForecastRewardOracle | FrozenForecastRewardEnsemble,
    model_name: str,
) -> FrozenForecastRewardOracle:
    if isinstance(reward_oracle, FrozenForecastRewardEnsemble):
        names = [str(name) for name in reward_oracle.model_names]
        if model_name not in names:
            raise ValueError(f"Requested model '{model_name}' not found in reward ensemble {names}")
        return reward_oracle.oracles[names.index(model_name)]
    return reward_oracle


def _normalize_with_oracle(oracle: FrozenForecastRewardOracle, ds: ForecastDataset) -> ForecastDataset:
    x_mean = np.asarray(oracle.x_mean, dtype=np.float32)
    x_std = np.asarray(oracle.x_std, dtype=np.float32)
    y_mean = np.asarray(oracle.y_mean, dtype=np.float32)
    y_std = np.asarray(oracle.y_std, dtype=np.float32)
    return ForecastDataset(
        X=((ds.X - x_mean) / x_std).astype(np.float32),
        Y=((ds.Y - y_mean) / y_std).astype(np.float32),
        target_indices=ds.target_indices,
    )


def _build_variant_input(
    input_series: np.ndarray,
    feature_names: list[str],
    spec: VariantSpec,
) -> np.ndarray:
    series = np.array(input_series, dtype=float, copy=True)
    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    if spec.mode == "identity":
        return series
    for column in spec.columns:
        if column not in name_to_idx:
            continue
        idx = name_to_idx[column]
        if spec.mode == "moving_average":
            series[:, idx] = _moving_average_1d(series[:, idx], window=int(spec.window))
        elif spec.mode == "ema":
            series[:, idx] = _ema_1d(series[:, idx], alpha=float(spec.alpha))
        else:
            raise ValueError(f"Unsupported smoothing mode: {spec.mode}")
    return series


def _evaluate_oracle_on_dataset(
    *,
    dataset: dict[str, np.ndarray],
    scheduler_name: str,
    source_npz: str,
    reward_oracle: FrozenForecastRewardOracle | FrozenForecastRewardEnsemble,
    model_name: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    oracle = _select_oracle(reward_oracle, model_name)
    input_series = dataset["input_series"]
    target_series = dataset["target_series"]
    feature_names = [str(name) for name in dataset["feature_names"].tolist()]
    observed_mask = dataset.get("observed_mask")
    time_index = dataset.get("time_index")
    context_series = extract_context_series(dataset)
    input_prepared, input_feature_names, target_prepared, target_feature_names, target_indices = prepare_input_and_targets(
        input_series=np.asarray(input_series, dtype=float),
        target_series=np.asarray(target_series, dtype=float),
        feature_names=feature_names,
        observed_mask=None if observed_mask is None else np.asarray(observed_mask, dtype=float),
        use_observed_mask=bool(oracle.use_observed_mask),
        use_time_delta=bool(oracle.use_time_delta),
        target_columns=list(oracle.target_columns),
        time_index=None if time_index is None else np.asarray(time_index, dtype=int),
        base_freq_s=int(getattr(oracle, "base_freq_s", 1)),
        context_series=context_series,
        context_features=list(getattr(oracle, "context_features", [])),
    )
    ds = build_window_dataset(
        series=np.asarray(input_prepared, dtype=float),
        lookback=int(oracle.lookback),
        horizon=int(oracle.horizon),
        target_series=np.asarray(target_prepared, dtype=float),
        target_indices=target_indices,
    )
    ds_norm = _normalize_with_oracle(oracle, ds)
    y_pred_norm = np.asarray(oracle.predictor.predict(ds_norm), dtype=float)
    y_pred = y_pred_norm * np.asarray(oracle.y_std, dtype=np.float32) + np.asarray(oracle.y_mean, dtype=np.float32)
    metrics = compute_forecast_metrics(ds.Y, y_pred)
    overall = {
        "scheduler": scheduler_name,
        "model": model_name,
        "source_npz": source_npz,
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "mape": float(metrics["mape"]),
        "smape": float(metrics["smape"]),
        "pearson_h1_mean": float(metrics["pearson_h1_mean"]),
        "dtw_h1_mean": float(metrics["dtw_h1_mean"]),
        "n_windows": int(ds.Y.shape[0]),
        "lookback": int(oracle.lookback),
        "horizon": int(oracle.horizon),
        "n_input_features": int(len(input_feature_names)),
        "n_targets": int(len(target_feature_names)),
    }
    target_rows: list[dict[str, object]] = []
    for target_idx, target_name in enumerate(target_feature_names):
        for horizon_idx in range(ds.Y.shape[1]):
            y_true = np.asarray(ds.Y[:, horizon_idx, target_idx], dtype=float)
            y_hat = np.asarray(y_pred[:, horizon_idx, target_idx], dtype=float)
            target_rows.append(
                {
                    "scheduler": scheduler_name,
                    "model": model_name,
                    "target": str(target_name),
                    "horizon": int(horizon_idx + 1),
                    "rmse": _rmse_1d(y_true, y_hat),
                    "mae": _mae_1d(y_true, y_hat),
                    "pearson": float(pearson_1d(y_true, y_hat)),
                    "dtw": float(dtw_distance_1d(y_true, y_hat)),
                }
            )
    return overall, target_rows


def _collect_saved_baseline_rows(
    *,
    run_tag: str,
    schedulers: list[str],
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for scheduler in schedulers:
        for model in models:
            run_dir = ROOT / "reports" / "runs" / f"{run_tag}_{scheduler}_pred_{model}"
            metrics_csv = run_dir / "metrics_forecast.csv"
            preds_npz = run_dir / "forecast_predictions.npz"
            if not metrics_csv.exists() or not preds_npz.exists():
                continue
            metric_row = _load_csv_row(metrics_csv)
            overall_rows.append(
                {
                    "scheduler": scheduler,
                    "model": model,
                    "source_npz": metric_row.get("dataset_npz", ""),
                    "rmse": float(metric_row["rmse"]),
                    "mae": float(metric_row["mae"]),
                    "mape": float(metric_row["mape"]),
                    "smape": float(metric_row["smape"]),
                    "pearson_h1_mean": float(metric_row["pearson_h1_mean"]),
                    "dtw_h1_mean": float(metric_row["dtw_h1_mean"]),
                    "n_windows": int(np.load(preds_npz, allow_pickle=True)["y_true"].shape[0]),
                    "lookback": int(metric_row["lookback"]),
                    "horizon": int(metric_row["horizon"]),
                    "n_input_features": int(metric_row["n_features"]),
                    "n_targets": int(metric_row["n_targets"]),
                }
            )
            pred_data = np.load(preds_npz, allow_pickle=True)
            y_true = np.asarray(pred_data["y_true"], dtype=float)
            y_pred = np.asarray(pred_data["y_pred"], dtype=float)
            target_names = [str(name) for name in pred_data["target_feature_names"].tolist()]
            for target_idx, target_name in enumerate(target_names):
                for horizon_idx in range(y_true.shape[1]):
                    seq_true = y_true[:, horizon_idx, target_idx]
                    seq_pred = y_pred[:, horizon_idx, target_idx]
                    target_rows.append(
                        {
                            "scheduler": scheduler,
                            "model": model,
                            "target": str(target_name),
                            "horizon": int(horizon_idx + 1),
                            "rmse": _rmse_1d(seq_true, seq_pred),
                            "mae": _mae_1d(seq_true, seq_pred),
                            "pearson": float(pearson_1d(seq_true, seq_pred)),
                            "dtw": float(dtw_distance_1d(seq_true, seq_pred)),
                        }
                    )
    return pd.DataFrame(overall_rows), pd.DataFrame(target_rows)


def _compute_roughness_rows(
    *,
    run_tag: str,
    schedulers: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scheduler in schedulers:
        npz_path = ROOT / "data" / "processed" / f"{run_tag}_{scheduler}.npz"
        if not npz_path.exists():
            continue
        dataset = _load_npz(npz_path)
        feature_names = [str(name) for name in dataset["feature_names"].tolist()]
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        input_series = np.asarray(dataset["input_series"], dtype=float)
        target_series = np.asarray(dataset["target_series"], dtype=float)
        for feature in FOCUS_FEATURES:
            if feature not in name_to_idx:
                continue
            idx = name_to_idx[feature]
            est = input_series[:, idx]
            truth = target_series[:, idx]
            err = est - truth
            rows.append(
                {
                    "scheduler": scheduler,
                    "feature": feature,
                    "rmse_to_truth": _rmse_1d(truth, est),
                    "mae_to_truth": _mae_1d(truth, est),
                    "series_diff_std": float(np.std(np.diff(est))),
                    "series_second_diff_std": float(np.std(np.diff(est, n=2))),
                    "series_total_variation": float(np.mean(np.abs(np.diff(est)))),
                    "error_diff_std": float(np.std(np.diff(err))),
                    "error_second_diff_std": float(np.std(np.diff(err, n=2))),
                    "error_total_variation": float(np.mean(np.abs(np.diff(err)))),
                    "high_freq_energy_ratio": _high_freq_energy_ratio(est),
                    "error_high_freq_energy_ratio": _high_freq_energy_ratio(err),
                }
            )
    return pd.DataFrame(rows)


def _attach_relative_to_full_open(df: pd.DataFrame, *, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if df.empty:
        return out
    key_cols = list(group_cols)
    base = (
        df[df["scheduler"] == "full_open"][key_cols + value_cols]
        .rename(columns={col: f"{col}_full_open" for col in value_cols})
    )
    merged = out.merge(base, on=key_cols, how="left")
    for col in value_cols:
        denom = merged[f"{col}_full_open"].replace(0.0, np.nan)
        merged[f"{col}_relative_to_full_open_pct"] = (merged[col] - merged[f"{col}_full_open"]) / denom * 100.0
    drop_cols = [f"{col}_full_open" for col in value_cols]
    return merged.drop(columns=drop_cols)


def _plot_roughness_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    focus = df[df["feature"].isin(FOCUS_FEATURES)].copy()
    if focus.empty:
        return
    pivot = focus.pivot(index="feature", columns="scheduler", values="error_diff_std_relative_to_full_open_pct")
    sched_order = [sched for sched in BASELINE_SCHEDULERS if sched in pivot.columns]
    pivot = pivot.reindex(index=list(FOCUS_FEATURES), columns=sched_order)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Error Roughness vs full_open (%)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_variant_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    order = [row for row in ["full_open_raw", "full_open_ma5_primary", "full_open_ma5_snow", "full_open_ma5_core", "full_open_ema03_core", "periodic", "round_robin", "dqn", "cmdp_dqn"] if row in set(df["scheduler"])]
    plot_df = df.set_index("scheduler").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    positions = np.arange(len(plot_df))
    bars = ax.bar(positions, plot_df["rmse_increase_pct_vs_full_open"], color="#2e6f95")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("RMSE change vs raw full_open (%)")
    ax.set_title("Overall Frozen-Predictor RMSE Change")
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df["scheduler"], rotation=30, ha="right")
    for bar, value in zip(bars, plot_df["rmse_increase_pct_vs_full_open"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_variant_target_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    focus_targets = [name for name in FOCUS_FEATURES if name in set(df["target"])]
    schedulers = [row for row in ["full_open_ma5_primary", "full_open_ma5_snow", "full_open_ma5_core", "full_open_ema03_core", "periodic", "round_robin"] if row in set(df["scheduler"])]
    plot_df = df[(df["horizon"] == 1) & (df["target"].isin(focus_targets)) & (df["scheduler"].isin(schedulers))]
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index="target", columns="scheduler", values="rmse_relative_to_full_open_pct")
    pivot = pivot.reindex(index=focus_targets, columns=schedulers)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("H=1 Target RMSE vs raw full_open (%)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_tag = str(args.run_tag)
    out_dir = ROOT / "reports" / "aggregate" / f"diagnostics_{run_tag}" if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_artifact = (
        ROOT / "reports" / "runs" / f"{run_tag}_reward_model" / "reward_oracles.yaml"
        if args.reward_artifact is None
        else Path(args.reward_artifact)
    )
    models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    baselines = [item.strip() for item in str(args.baselines).split(",") if item.strip()]
    full_open_npz = ROOT / "data" / "processed" / f"{run_tag}_full_open.npz"
    print(f"[diag] run_tag={run_tag}")
    print(f"[diag] reward_artifact={reward_artifact}")
    print(f"[diag] out_dir={out_dir}")
    reward_oracle = load_reward_oracle(reward_artifact)

    roughness_df = _compute_roughness_rows(run_tag=run_tag, schedulers=baselines)
    roughness_rel_df = _attach_relative_to_full_open(
        roughness_df,
        value_cols=[
            "rmse_to_truth",
            "mae_to_truth",
            "series_diff_std",
            "series_second_diff_std",
            "series_total_variation",
            "error_diff_std",
            "error_second_diff_std",
            "error_total_variation",
            "high_freq_energy_ratio",
            "error_high_freq_energy_ratio",
        ],
        group_cols=["feature"],
    )
    roughness_df.to_csv(out_dir / "roughness_metrics.csv", index=False)
    roughness_rel_df.to_csv(out_dir / "roughness_relative_to_full_open.csv", index=False)
    _plot_roughness_heatmap(roughness_rel_df, out_dir / "roughness_error_diff_std_heatmap.png")

    overall_df, target_df = _collect_saved_baseline_rows(run_tag=run_tag, schedulers=baselines, models=models)

    full_open_data = _load_npz(full_open_npz)
    feature_names = [str(name) for name in full_open_data["feature_names"].tolist()]
    variant_overall_rows: list[dict[str, object]] = []
    variant_target_rows: list[dict[str, object]] = []
    for spec in VARIANT_SPECS:
        print(f"[diag] variant={spec.name}")
        variant_input = _build_variant_input(
            input_series=np.asarray(full_open_data["input_series"], dtype=float),
            feature_names=feature_names,
            spec=spec,
        )
        variant_dataset = dict(full_open_data)
        variant_dataset["input_series"] = variant_input
        for model_name in models:
            print(f"[diag]   model={model_name}")
            overall_row, target_rows = _evaluate_oracle_on_dataset(
                dataset=variant_dataset,
                scheduler_name=spec.name,
                source_npz=str(full_open_npz),
                reward_oracle=reward_oracle,
                model_name=model_name,
            )
            variant_overall_rows.append(overall_row)
            variant_target_rows.extend(target_rows)

    variant_overall_df = pd.DataFrame(variant_overall_rows)
    variant_target_df = pd.DataFrame(variant_target_rows)
    all_overall_df = pd.concat([overall_df, variant_overall_df], ignore_index=True)
    all_target_df = pd.concat([target_df, variant_target_df], ignore_index=True)

    overall_rel_df = _attach_relative_to_full_open(
        all_overall_df,
        value_cols=["rmse", "mae", "pearson_h1_mean", "dtw_h1_mean"],
        group_cols=["model"],
    )
    target_rel_df = _attach_relative_to_full_open(
        all_target_df,
        value_cols=["rmse", "mae", "pearson", "dtw"],
        group_cols=["model", "target", "horizon"],
    )
    overall_summary_df = (
        overall_rel_df.groupby("scheduler", as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_increase_pct_vs_full_open=("rmse_relative_to_full_open_pct", "mean"),
            dtw_h1_mean=("dtw_h1_mean", "mean"),
            dtw_h1_increase_pct_vs_full_open=("dtw_h1_mean_relative_to_full_open_pct", "mean"),
            pearson_h1_mean=("pearson_h1_mean", "mean"),
            pearson_h1_delta_vs_full_open=("pearson_h1_mean_relative_to_full_open_pct", "mean"),
        )
    )
    target_summary_df = (
        target_rel_df.groupby(["scheduler", "target", "horizon"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_relative_to_full_open_pct=("rmse_relative_to_full_open_pct", "mean"),
            dtw=("dtw", "mean"),
            dtw_relative_to_full_open_pct=("dtw_relative_to_full_open_pct", "mean"),
            pearson=("pearson", "mean"),
            pearson_relative_to_full_open_pct=("pearson_relative_to_full_open_pct", "mean"),
        )
    )

    all_overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)
    overall_rel_df.to_csv(out_dir / "overall_metrics_relative_to_full_open.csv", index=False)
    overall_summary_df.to_csv(out_dir / "overall_summary.csv", index=False)
    all_target_df.to_csv(out_dir / "target_metrics.csv", index=False)
    target_rel_df.to_csv(out_dir / "target_metrics_relative_to_full_open.csv", index=False)
    target_summary_df.to_csv(out_dir / "target_summary.csv", index=False)

    _plot_variant_summary(overall_summary_df, out_dir / "variant_overall_rmse_vs_full_open.png")
    _plot_variant_target_heatmap(target_summary_df, out_dir / "variant_target_h1_rmse_heatmap.png")

    print("[diag] wrote:")
    print(out_dir / "roughness_metrics.csv")
    print(out_dir / "roughness_relative_to_full_open.csv")
    print(out_dir / "overall_summary.csv")
    print(out_dir / "target_summary.csv")


if __name__ == "__main__":
    main()
