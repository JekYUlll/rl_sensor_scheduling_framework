#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle  # noqa: E402
from v2.power_projector import PowerConstraintsV2, PowerProjector  # noqa: E402
from v2.rollout import concat_rollout_results, rollout_metrics, run_policy_rollout  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402
from v2.warmup_state import SensorRuntime  # noqa: E402


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

DEFAULT_COVERAGE_GROUPS = (
    ("weather", ("met_station_core", "ultrasonic_anemometer_hd", "shielded_thermo_hygro")),
    ("surface_forcing", ("radiometer_basic", "surface_temp_ir")),
    ("snow_transport", ("snow_particle_counter", "laser_disdrometer", "fc4_flux")),
)


class FixedMaskPolicy:
    def __init__(self, mask: np.ndarray, name: str) -> None:
        self.mask = np.asarray(mask, dtype=bool).reshape(-1)
        self.name = str(name)

    def reset(self) -> None:
        pass

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        del env
        return self.mask


def _load_oracle(path: Path, oracle_type: str, device: str):
    if oracle_type == "tcn":
        return TCNFrozenForecastOracle.load(path, device=device)
    if oracle_type == "linear":
        return LinearFrozenForecastOracle.load(str(path))
    raise ValueError(f"Unsupported oracle_type={oracle_type}")


def _coverage_groups_from_metadata(meta: dict[str, object]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    constraints = meta.get("constraints", {})
    if not isinstance(constraints, dict):
        return DEFAULT_COVERAGE_GROUPS
    groups = constraints.get("coverage_groups")
    if not isinstance(groups, list) or not groups:
        return DEFAULT_COVERAGE_GROUPS
    out = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        name = str(group.get("name", "group"))
        sensor_ids = tuple(str(x) for x in group.get("sensor_ids", []))
        if sensor_ids:
            out.append((name, sensor_ids))
    return tuple(out) if out else DEFAULT_COVERAGE_GROUPS


def build_candidate_masks(
    sensors: list,
    constraints: PowerConstraintsV2,
    *,
    max_candidate_warmup: int | None,
) -> np.ndarray:
    projector = PowerProjector(sensors, constraints)
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    n_sensors = len(sensors)
    allowed = np.ones(n_sensors, dtype=bool)
    if max_candidate_warmup is not None:
        allowed = np.asarray([int(spec.warmup_steps) <= int(max_candidate_warmup) for spec in sensors], dtype=bool)
    masks: dict[tuple[int, ...], np.ndarray] = {}
    for value in range(1 << n_sensors):
        desired = np.asarray([(value >> idx) & 1 for idx in range(n_sensors)], dtype=bool)
        if np.any(desired & ~allowed):
            continue
        try:
            result = projector.project_mask(desired, runtimes)
        except ValueError:
            continue
        if np.any(result.selected_mask & ~allowed):
            continue
        key = tuple(int(x) for x in result.selected_mask.tolist())
        masks[key] = result.selected_mask.astype(bool)
    if not masks:
        raise ValueError("No feasible candidate masks found")
    return np.asarray(list(masks.values()), dtype=bool)


def constraint_binding_rate(sensors: list, constraints: PowerConstraintsV2, *, samples: int, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    projector = PowerProjector(sensors, constraints)
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    changed = 0
    for _ in range(int(samples)):
        desired = rng.random(len(sensors)) < 0.5
        try:
            projected = projector.project_mask(desired, runtimes).selected_mask
        except ValueError:
            changed += 1
            continue
        changed += int(not np.array_equal(desired, projected))
    return float(changed) / float(max(1, int(samples)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether v2 has a time-varying action landscape.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-rollouts", type=int, default=4)
    parser.add_argument("--max-candidate-warmup", type=int, default=-1)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--binding-samples", type=int, default=512)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "action_landscape"
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(meta["truth_csv"])
    sensors = load_sensor_specs(meta["sensor_cfg"])
    oracle_type = str(meta.get("oracle_type", "tcn"))
    oracle = _load_oracle(Path(meta["oracle_path"]), oracle_type=oracle_type, device=str(args.oracle_device))
    constraints_meta = meta.get("constraints", {})
    constraints = PowerConstraintsV2(
        max_active=int(constraints_meta.get("max_active", 4)),
        per_step_budget=float(constraints_meta.get("per_step_budget", 1.7)),
        startup_peak_budget=float(constraints_meta.get("startup_peak_budget", 3.2)),
        required_sensor_ids=tuple(str(x) for x in constraints_meta.get("required_sensor_ids", [])),
        coverage_groups=_coverage_groups_from_metadata(meta),
    )
    max_candidate_warmup = None if int(args.max_candidate_warmup) < 0 else int(args.max_candidate_warmup)
    candidate_masks = build_candidate_masks(sensors, constraints, max_candidate_warmup=max_candidate_warmup)
    starts = [int(x) for x in meta.get("eval_start_indices", [0])][: max(1, int(args.max_rollouts))]
    base_freq_s = int(meta.get("freq_s", 10800))

    rows = []
    oracle_loss_rows = []
    for idx, mask in enumerate(candidate_masks):
        policy = FixedMaskPolicy(mask, name=f"candidate_{idx:03d}")
        rollouts = []
        for offset, start_idx in enumerate(starts):
            cfg = WarmupEnvConfig(
                state_columns=STATE_COLUMNS,
                reward_target_columns=REWARD_TARGET_COLUMNS,
                lookback=int(meta.get("lookback", 20)),
                episode_len=int(args.steps),
                seed=int(meta.get("seed", 42)) + 1000 + offset,
                base_freq_s=base_freq_s,
                lambda_warmup_abort=float(meta.get("reward_shaping", {}).get("lambda_warmup_abort", 0.08)),
                lambda_switch=float(meta.get("reward_shaping", {}).get("lambda_switch", 0.002)),
            )
            env = WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
            rollouts.append(run_policy_rollout(env, policy, steps=int(args.steps), start_idx=int(start_idx)))
        result = concat_rollout_results(rollouts, policy_name=policy.name)
        metrics = rollout_metrics(result)
        rows.append(
            {
                "candidate": policy.name,
                "oracle_loss_mean": metrics["oracle_loss_mean"],
                "instant_mae": metrics["instant_mae"],
                "power_mean": metrics["power_mean"],
                "warmup_abort_count": metrics["warmup_abort_count"],
                "selected_sensor_ids": ";".join(
                    spec.sensor_id for spec, selected in zip(sensors, mask, strict=True) if bool(selected)
                ),
            }
        )
        oracle_loss_rows.append(result.oracle_losses)

    losses = np.vstack(oracle_loss_rows)
    finite_losses = np.where(np.isfinite(losses), losses, np.inf)
    best_idx = np.argmin(finite_losses, axis=0)
    values, counts = np.unique(best_idx, return_counts=True)
    dominant_rate = float(np.max(counts) / max(1, best_idx.size)) if counts.size else float("nan")
    switch_rate = float(np.mean(best_idx[1:] != best_idx[:-1])) if best_idx.size > 1 else 0.0

    candidate_summary = pd.DataFrame(rows).sort_values("oracle_loss_mean")
    candidate_summary.to_csv(out_dir / "candidate_summary.csv", index=False)
    truth_event = truth["event_flag"].astype(bool).to_numpy() if "event_flag" in truth.columns else np.zeros(len(truth), dtype=bool)
    summary = {
        "candidate_count": int(candidate_masks.shape[0]),
        "event_flag_rate": float(np.mean(truth_event)),
        "wind_ge_8_rate": float(np.mean(truth["wind_speed_ms"].to_numpy(dtype=float) >= 8.0)),
        "constraint_binding_rate_random_masks": constraint_binding_rate(
            sensors,
            constraints,
            samples=int(args.binding_samples),
            seed=int(meta.get("seed", 42)) + 99,
        ),
        "best_action_dominant_rate": dominant_rate,
        "best_action_switch_rate": switch_rate,
        "best_action_unique_count": int(values.size),
        "best_candidate": str(candidate_summary.iloc[0]["candidate"]) if not candidate_summary.empty else None,
        "best_candidate_sensors": str(candidate_summary.iloc[0]["selected_sensor_ids"]) if not candidate_summary.empty else None,
    }
    (out_dir / "action_landscape_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(out_dir / "candidate_summary.csv")


if __name__ == "__main__":
    main()
