#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_yaml  # noqa: E402


def _discover_scheduler_npz(run_tag: str, scheduler: str | None) -> list[tuple[str, Path]]:
    processed_dir = ROOT / "data" / "processed"
    discovered: list[tuple[str, Path]] = []
    prefix = f"{run_tag}_"
    for path in sorted(processed_dir.glob(f"{run_tag}_*.npz")):
        scheduler_name = path.stem[len(prefix) :]
        if scheduler is not None and scheduler_name != scheduler:
            continue
        discovered.append((scheduler_name, path))
    if discovered:
        return discovered
    exact = processed_dir / f"{run_tag}.npz"
    if exact.exists():
        scheduler_name = scheduler or "scheduler"
        stats_path = ROOT / "reports" / "runs" / run_tag / "dataset_stats.csv"
        if stats_path.exists():
            try:
                import pandas as pd

                stats = pd.read_csv(stats_path)
                if "scheduler_name" in stats.columns and not stats.empty:
                    scheduler_name = str(stats.loc[0, "scheduler_name"])
            except Exception:
                pass
        discovered.append((scheduler_name, exact))
    return discovered


def _default_sensor_names(sensor_cfg: dict) -> list[str]:
    return [str(spec["sensor_id"]) for spec in sensor_cfg.get("sensors", [])]


def _infer_sensor_timelines(
    data: np.lib.npyio.NpzFile,
    feature_names: list[str],
    sensor_cfg: dict,
) -> tuple[list[str], np.ndarray]:
    if {"powered_mask", "warming_mask", "ready_mask"}.issubset(data.files):
        sensor_names = _default_sensor_names(sensor_cfg)
        if "sensor_ids" in data.files:
            sensor_names = [str(name) for name in data["sensor_ids"].tolist()]
        powered_mask = np.asarray(data["powered_mask"], dtype=float)
        warming_mask = np.asarray(data["warming_mask"], dtype=float)
        ready_mask = np.asarray(data["ready_mask"], dtype=float)
        if powered_mask.ndim != 2 or warming_mask.shape != powered_mask.shape or ready_mask.shape != powered_mask.shape:
            raise ValueError("Warmup masks must all be 2D and share the same shape")
        if len(sensor_names) != powered_mask.shape[1]:
            raise ValueError(
                f"sensor_names length {len(sensor_names)} does not match warmup mask width {powered_mask.shape[1]}"
            )
        mode = np.zeros_like(powered_mask, dtype=float)
        mode[warming_mask > 0.5] = 1.0
        mode[ready_mask > 0.5] = 2.0
        return sensor_names, mode.T

    observed_mask = np.asarray(data["observed_mask"], dtype=float)
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    sensor_names: list[str] = []
    activation: list[np.ndarray] = []
    for spec in sensor_cfg.get("sensors", []):
        sensor_id = str(spec["sensor_id"])
        variables = [str(v) for v in spec.get("variables", [])]
        idxs = [feature_to_idx[v] for v in variables if v in feature_to_idx]
        if not idxs:
            continue
        sensor_mask = observed_mask[:, idxs]
        active = 2.0 * np.all(sensor_mask > 0.5, axis=1).astype(float)
        sensor_names.append(sensor_id)
        activation.append(active)
    if not activation:
        raise ValueError("No sensor activation timeline could be inferred from observed_mask")
    return sensor_names, np.vstack(activation)


def _format_axis_label(label: str, width: int = 14) -> str:
    return textwrap.fill(label.replace("_", " "), width=width)


def _plot_timeline(
    scheduler_name: str,
    sensor_names: list[str],
    activation: np.ndarray,
    target_series: np.ndarray,
    target_name: str,
    power: np.ndarray,
    out_path: Path,
    start: int,
    end: int,
    model_label: str | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_count = len(sensor_names)
    fig_h = 2.2 + 0.75 * sensor_count
    fig, axes = plt.subplots(
        2 + sensor_count,
        1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0] + [0.7] * sensor_count},
    )
    x = np.arange(start, end)

    title = f"{scheduler_name} sensor activation | target={target_name} | steps={start}:{end}"
    if model_label:
        title += f" | model={model_label}"

    axes[0].plot(x, target_series[start:end], color="black", linewidth=1.6)
    axes[0].set_ylabel(_format_axis_label(target_name), rotation=0, ha="right", va="center", labelpad=34)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, power[start:end], color="tab:orange", linewidth=1.4)
    axes[1].set_ylabel("power", rotation=0, ha="right", va="center", labelpad=34)
    axes[1].grid(alpha=0.25)

    for row_idx, sensor_name in enumerate(sensor_names):
        ax = axes[2 + row_idx]
        series = activation[row_idx, start:end]
        warm_series = (series > 0.5).astype(float)
        ready_series = (series > 1.5).astype(float) * 2.0
        ax.step(x, series, where="post", linewidth=1.4, color="#334155")
        ax.fill_between(x, 0.0, warm_series, where=warm_series > 0.0, step="post", alpha=0.32, color="#f59e0b")
        ax.fill_between(x, 0.0, ready_series, where=ready_series > 0.0, step="post", alpha=0.28, color="#10b981")
        ax.set_ylim(-0.1, 2.1)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["off", "warm", "ready"])
        duty = float(np.mean(series > 0.5)) if len(series) else float("nan")
        ready_ratio = float(np.mean(series > 1.5)) if len(series) else float("nan")
        ax.set_ylabel(
            _format_axis_label(sensor_name),
            rotation=0,
            ha="right",
            va="center",
            labelpad=34,
        )
        ax.text(
            0.995,
            0.82,
            f"on={duty:.2f} ready={ready_ratio:.2f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.8", "alpha": 0.9},
        )
        ax.grid(alpha=0.2, axis="x")

    axes[-1].set_xlabel("time index")
    fig.tight_layout(rect=(0.16, 0.02, 1.0, 1.0))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-sensor on/off timelines for one scheduling run")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--scheduler", default=None, help="Optional scheduler name; otherwise generate all")
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors.yaml")
    parser.add_argument("--target", default="snow_mass_flux_kg_m2_s")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--model-label", default=None, help="Optional predictor label for the figure title")
    args = parser.parse_args()

    sensor_cfg = load_yaml(args.sensor_cfg)
    discovered = _discover_scheduler_npz(args.run_tag, args.scheduler)
    if not discovered:
        raise FileNotFoundError(f"No processed NPZ found for run_tag={args.run_tag!r}")

    out_root = ROOT / "reports" / "aggregate" / f"posthoc_{args.run_tag}" / "sensor_timelines"

    for scheduler_name, npz_path in discovered:
        data = np.load(npz_path, allow_pickle=True)
        feature_names = [str(x) for x in data["feature_names"].tolist()]
        if args.target not in feature_names:
            raise ValueError(f"target {args.target!r} not found in feature_names of {npz_path.name}")
        target_idx = feature_names.index(args.target)
        target_series = np.asarray(data["target_series"][:, target_idx], dtype=float)
        power = np.asarray(data["power"], dtype=float)
        sensor_names, activation = _infer_sensor_timelines(data, feature_names, sensor_cfg)
        max_end = len(target_series)
        start = max(0, int(args.start))
        end = min(max_end, int(args.end))
        if end <= start:
            raise ValueError(f"Invalid plotting range [{start}, {end}) for length {max_end}")
        model_suffix = f"_{args.model_label}" if args.model_label else ""
        out_path = out_root / scheduler_name / f"{args.target}{model_suffix}_{start}_{end}_activation.png"
        _plot_timeline(
            scheduler_name=scheduler_name,
            sensor_names=sensor_names,
            activation=activation,
            target_series=target_series,
            target_name=args.target,
            power=power,
            out_path=out_path,
            start=start,
            end=end,
            model_label=args.model_label,
        )
        print(out_path)


if __name__ == "__main__":
    main()
