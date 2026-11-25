#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle, OracleConfig, build_supervised_windows  # noqa: E402
from v2.policies import FullOpenUnconstrainedScorePolicy, default_policies  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import rollout_metrics, run_policy_rollout, save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402


STATE_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "wind_dir_sin",
    "wind_dir_cos",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
)

REWARD_TARGET_COLUMNS = (
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

DEFAULT_REQUIRED_SENSOR_IDS: tuple[str, ...] = ()


def ensure_truth(args: argparse.Namespace) -> Path:
    truth = Path(args.truth_csv)
    if truth.exists():
        return truth
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "20_build_public_weather_truth.py"),
        "--antaws-root",
        args.antaws_root,
        "--stations",
        *args.stations,
        "--steps",
        str(args.truth_steps),
        "--freq-s",
        str(args.freq_s),
        "--seed",
        str(args.seed),
        "--out",
        str(truth),
        "--report-dir",
        str(Path(args.out_dir) / "dataset_validation"),
    ]
    subprocess.run(cmd, check=True)
    return truth


def make_env(
    truth: pd.DataFrame,
    sensors: list,
    constraints: PowerConstraintsV2,
    *,
    lookback: int,
    episode_len: int,
    seed: int,
    oracle: LinearFrozenForecastOracle | None = None,
) -> WarmupSchedulingEnv:
    return WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=REWARD_TARGET_COLUMNS,
            lookback=lookback,
            episode_len=episode_len,
            seed=seed,
        ),
        oracle=oracle,
    )


def plot_policy_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics.sort_values("oracle_loss_mean").plot.bar(x="policy", y="oracle_loss_mean", ax=axes[0], legend=False)
    axes[0].set_ylabel("Frozen oracle loss")
    axes[0].set_title("Forecast loss by scheduler")
    metrics.plot.scatter(x="power_mean", y="oracle_loss_mean", ax=axes[1])
    for _, row in metrics.iterrows():
        axes[1].annotate(str(row["policy"]), (float(row["power_mean"]), float(row["oracle_loss_mean"])))
    axes[1].set_title("Power vs forecast loss")
    fig.tight_layout()
    fig.savefig(out_dir / "v2_policy_summary.png", dpi=180)
    plt.close(fig)


def plot_activation(result, sensor_ids: tuple[str, ...], out_dir: Path) -> None:
    if result.mode_ids.size == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [1, 3]})
    axes[0].plot(result.powers, label="steady power")
    axes[0].plot(result.peaks, label="peak power", alpha=0.7)
    axes[0].set_ylabel("power")
    axes[0].legend(loc="upper right")
    im = axes[1].imshow(result.mode_ids.T, aspect="auto", interpolation="nearest", vmin=0, vmax=2, cmap="viridis")
    axes[1].set_yticks(np.arange(len(sensor_ids)))
    axes[1].set_yticklabels(sensor_ids)
    axes[1].set_xlabel("step")
    axes[1].set_title(f"{result.policy_name} sensor states")
    cbar = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["OFF", "WARM", "ACTIVE"])
    fig.tight_layout()
    fig.savefig(out_dir / f"v2_activation_{result.policy_name}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v2 data -> oracle -> policy evaluation pipeline.")
    parser.add_argument("--truth-csv", default="data/generated/v2_public_weather_truth.csv")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Taishan"])
    parser.add_argument("--truth-steps", type=int, default=2048)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--out-dir", default="reports/v2_smoke")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--oracle-rollout-steps", type=int, default=1200)
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-step-budget", type=float, default=2.3)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--required-sensors", nargs="*", default=list(DEFAULT_REQUIRED_SENSOR_IDS))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = ensure_truth(args)
    truth = pd.read_csv(truth_path)
    sensors = load_sensor_specs(args.sensor_cfg)
    constraints = PowerConstraintsV2(
        max_active=int(args.max_active),
        per_step_budget=float(args.per_step_budget),
        startup_peak_budget=float(args.startup_peak_budget),
        required_sensor_ids=tuple(str(sensor_id) for sensor_id in args.required_sensors),
    )

    oracle_policy_specs = [
        (FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)), PowerConstraintsV2()),
        *[(policy, constraints) for policy in default_policies(len(sensors), seed=int(args.seed))],
    ]
    target_indices = [STATE_COLUMNS.index(name) for name in REWARD_TARGET_COLUMNS]
    observed_batches = []
    mask_batches = []
    truth_batches = []
    for idx, (policy, policy_constraints) in enumerate(oracle_policy_specs):
        env = make_env(
            truth,
            sensors,
            policy_constraints,
            lookback=int(args.lookback),
            episode_len=int(args.oracle_rollout_steps),
            seed=int(args.seed) + idx,
        )
        result = run_policy_rollout(env, policy, steps=int(args.oracle_rollout_steps))
        observed_batches.append(result.observations)
        mask_batches.append(result.masks)
        truth_batches.append(result.truth[:, target_indices])

    observed = np.vstack(observed_batches)
    masks = np.vstack(mask_batches)
    targets = np.vstack(truth_batches)
    x_train, y_train = build_supervised_windows(
        observed,
        masks,
        targets,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
    )
    oracle = LinearFrozenForecastOracle(OracleConfig(lookback=int(args.lookback), horizon=int(args.horizon), ridge_alpha=10.0))
    oracle.fit(x_train, y_train)

    policies = default_policies(len(sensors), seed=int(args.seed) + 100)
    results = []
    rows = []

    full_open_env = make_env(
        truth,
        sensors,
        PowerConstraintsV2(),
        lookback=int(args.lookback),
        episode_len=int(args.eval_steps),
        seed=int(args.seed) + 9500,
        oracle=oracle,
    )
    full_open_result = run_policy_rollout(
        full_open_env,
        FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        steps=int(args.eval_steps),
    )
    results.append(full_open_result)
    rows.append(rollout_metrics(full_open_result))
    save_rollout_npz(
        out_dir / "rollout_full_open_unconstrained.npz",
        full_open_result,
        sensor_ids=[s.sensor_id for s in sensors],
        state_columns=STATE_COLUMNS,
    )

    for idx, policy in enumerate(policies):
        env = make_env(
            truth,
            sensors,
            constraints,
            lookback=int(args.lookback),
            episode_len=int(args.eval_steps),
            seed=int(args.seed) + 1000 + idx,
            oracle=oracle,
        )
        result = run_policy_rollout(env, policy, steps=int(args.eval_steps))
        results.append(result)
        rows.append(rollout_metrics(result))
        save_rollout_npz(
            out_dir / f"rollout_{result.policy_name}.npz",
            result,
            sensor_ids=[s.sensor_id for s in sensors],
            state_columns=STATE_COLUMNS,
        )

    metrics = pd.DataFrame(rows).sort_values("oracle_loss_mean")
    metrics.to_csv(out_dir / "v2_metrics.csv", index=False)
    plot_policy_summary(metrics, out_dir)
    best = results[int(np.argmin([row["oracle_loss_mean"] for row in rows]))]
    plot_activation(best, tuple(s.sensor_id for s in sensors), out_dir)

    metadata = {
        "truth_csv": str(truth_path),
        "sensor_cfg": str(args.sensor_cfg),
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "constraints": {
            "max_active": int(args.max_active),
            "per_step_budget": float(args.per_step_budget),
            "startup_peak_budget": float(args.startup_peak_budget),
            "required_sensor_ids": [str(sensor_id) for sensor_id in args.required_sensors],
        },
        "oracle_train_samples": int(x_train.shape[0]),
        "policies": ["full_open_unconstrained", *[str(p.name) for p in policies]],
    }
    (out_dir / "v2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir / "v2_metrics.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
