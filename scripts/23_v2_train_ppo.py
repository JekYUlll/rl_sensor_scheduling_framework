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
from v2.oracle import (  # noqa: E402
    LinearFrozenForecastOracle,
    OracleConfig,
    build_supervised_windows_with_context,
)
from v2.policies import FullOpenUnconstrainedScorePolicy, StaticMaskPolicy, default_policies  # noqa: E402
from v2.power_projector import PowerConstraintsV2, PowerProjector  # noqa: E402
from v2.rollout import concat_rollout_results, rollout_metrics, run_policy_rollout, save_rollout_npz  # noqa: E402
from v2.sb3_ppo import evaluate_sb3_model, save_model, train_projected_ppo  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle, TCNOracleConfig  # noqa: E402
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
    "event_subtype_particle_latent",
    "event_subtype_flux_latent",
    "event_subtype_thermal_latent",
)

OPTIONAL_STATE_COLUMNS = (
    "event_subtype_particle_latent",
    "event_subtype_flux_latent",
    "event_subtype_thermal_latent",
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

DEFAULT_TARGET_WEIGHTS = (
    1.0,  # air_temperature_c
    1.0,  # snow_surface_temperature_c
    1.2,  # wind_speed_ms
    0.6,  # wind_dir_sin
    0.6,  # wind_dir_cos
    1.0,  # solar_radiation_wm2
    3.0,  # snow_mass_flux_kg_m2_s
    2.0,  # snow_particle_mean_diameter_mm
    2.0,  # snow_particle_mean_velocity_ms
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

DEFAULT_REQUIRED_SENSOR_IDS: tuple[str, ...] = ()

DEFAULT_COVERAGE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weather", ("met_station_core", "ultrasonic_anemometer_hd", "shielded_thermo_hygro")),
    ("surface_forcing", ("radiometer_basic", "surface_temp_ir")),
    ("snow_transport", ("snow_particle_counter", "laser_disdrometer", "fc4_flux")),
)

EVENT_PRIOR_WEIGHTS = {
    "snow_mass_flux_kg_m2_s": 12.0,
    "snow_particle_mean_diameter_mm": 4.0,
    "snow_particle_mean_velocity_ms": 4.0,
}

NON_EVENT_PRIOR_WEIGHTS = {
    "air_temperature_c": 1.0,
    "snow_surface_temperature_c": 1.0,
    "wind_speed_ms": 1.2,
    "wind_dir_sin": 0.6,
    "wind_dir_cos": 0.6,
    "solar_radiation_wm2": 1.0,
}


def optional_target_weights(values: list[float] | tuple[float, ...] | None, *, name: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    weights = tuple(float(x) for x in values)
    if len(weights) != len(REWARD_TARGET_COLUMNS):
        raise ValueError(f"{name} must contain {len(REWARD_TARGET_COLUMNS)} values, got {len(weights)}")
    return weights


def ensure_state_columns(truth: pd.DataFrame) -> pd.DataFrame:
    out = truth.copy()
    for column in OPTIONAL_STATE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    return out


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
        "--blowing-snow-event-coverage",
        str(args.blowing_snow_event_coverage),
        "--blowing-snow-event-model",
        str(getattr(args, "blowing_snow_event_model", "clustered")),
        "--blowing-snow-min-duration-steps",
        str(args.blowing_snow_min_duration_steps),
        "--blowing-snow-max-duration-steps",
        str(args.blowing_snow_max_duration_steps),
        "--blowing-snow-min-gap-steps",
        str(getattr(args, "blowing_snow_min_gap_steps", 6)),
        "--blowing-snow-lead-steps",
        str(args.blowing_snow_lead_steps),
        "--blowing-snow-wind-margin-ms",
        str(args.blowing_snow_wind_margin_ms),
        "--cred-hysteresis-on",
        str(getattr(args, "cred_hysteresis_on", 0.6)),
        "--cred-hysteresis-off",
        str(getattr(args, "cred_hysteresis_off", 0.3)),
        "--flux-wind-exponent",
        str(getattr(args, "flux_wind_exponent", 3.6)),
        "--event-microstructure-sigma",
        str(getattr(args, "event_microstructure_sigma", 0.0)),
        "--event-microstructure-alpha",
        str(getattr(args, "event_microstructure_alpha", 0.18)),
        "--event-microstructure-diameter-scale",
        str(getattr(args, "event_microstructure_diameter_scale", 0.0)),
        "--event-microstructure-velocity-scale",
        str(getattr(args, "event_microstructure_velocity_scale", 0.0)),
        "--event-particle-microstructure-correlation",
        str(getattr(args, "event_particle_microstructure_correlation", 1.0)),
        "--event-subtype-assignment",
        str(getattr(args, "event_subtype_assignment", "random")),
        "--event-subtype-particle-min-parsivel-availability",
        str(getattr(args, "event_subtype_particle_min_parsivel_availability", 0.0)),
        "--event-subtype-particle-prob",
        str(getattr(args, "event_subtype_particle_prob", 0.34)),
        "--event-subtype-flux-prob",
        str(getattr(args, "event_subtype_flux_prob", 0.33)),
        "--event-subtype-thermal-prob",
        str(getattr(args, "event_subtype_thermal_prob", 0.33)),
        "--event-subtype-particle-flux-multiplier",
        str(getattr(args, "event_subtype_particle_flux_multiplier", 0.72)),
        "--event-subtype-flux-multiplier",
        str(getattr(args, "event_subtype_flux_multiplier", 2.4)),
        "--event-subtype-thermal-flux-multiplier",
        str(getattr(args, "event_subtype_thermal_flux_multiplier", 0.55)),
        "--event-subtype-particle-diameter-shift-mm",
        str(getattr(args, "event_subtype_particle_diameter_shift_mm", 0.10)),
        "--event-subtype-particle-velocity-boost-ms",
        str(getattr(args, "event_subtype_particle_velocity_boost_ms", 1.3)),
        "--event-subtype-flux-diameter-shift-mm",
        str(getattr(args, "event_subtype_flux_diameter_shift_mm", -0.04)),
        "--event-subtype-flux-velocity-boost-ms",
        str(getattr(args, "event_subtype_flux_velocity_boost_ms", 0.7)),
        "--event-subtype-thermal-surface-drop-c",
        str(getattr(args, "event_subtype_thermal_surface_drop_c", 2.0)),
        "--event-subtype-particle-humidity-boost-pct",
        str(getattr(args, "event_subtype_particle_humidity_boost_pct", 0.0)),
        "--event-subtype-flux-wind-boost-ms",
        str(getattr(args, "event_subtype_flux_wind_boost_ms", 0.0)),
        "--event-subtype-thermal-air-temp-drop-c",
        str(getattr(args, "event_subtype_thermal_air_temp_drop_c", 0.0)),
        "--event-subtype-latent-alpha",
        str(getattr(args, "event_subtype_latent_alpha", 0.18)),
        "--event-subtype-particle-latent-diameter-scale-mm",
        str(getattr(args, "event_subtype_particle_latent_diameter_scale_mm", 0.0)),
        "--event-subtype-particle-latent-velocity-scale-ms",
        str(getattr(args, "event_subtype_particle_latent_velocity_scale_ms", 0.0)),
        "--event-subtype-flux-latent-sigma",
        str(getattr(args, "event_subtype_flux_latent_sigma", 0.0)),
        "--event-subtype-thermal-latent-surface-scale-c",
        str(getattr(args, "event_subtype_thermal_latent_surface_scale_c", 0.0)),
        "--event-subtype-latent-target-lag-steps",
        str(getattr(args, "event_subtype_latent_target_lag_steps", 0)),
        "--event-subtype-context-lead-steps",
        str(getattr(args, "event_subtype_context_lead_steps", 0)),
        "--event-subtype-context-noise-std",
        str(getattr(args, "event_subtype_context_noise_std", 0.08)),
        "--event-subtype-context-latent-strength",
        str(getattr(args, "event_subtype_context_latent_strength", 0.0)),
        "--out",
        str(truth),
        "--report-dir",
        str(Path(args.out_dir) / "dataset_validation"),
    ]
    if bool(getattr(args, "event_subtypes_enabled", False)):
        cmd.append("--event-subtypes-enabled")
    subprocess.run(cmd, check=True)
    return truth


def build_oracle_policy_specs(
    n_sensors: int,
    constraints: PowerConstraintsV2,
    *,
    seed: int,
    full_open_repeat: int,
    candidate_masks: np.ndarray | None = None,
    candidate_mask_repeat: int = 0,
    candidate_mask_limit: int = 0,
) -> list[tuple[object, PowerConstraintsV2]]:
    """Build the rollout mixture used to train the frozen forecast oracle.

    Full-open is deliberately oversampled because it is the semantic upper
    bound for forecast quality. If the oracle underfits that input regime, the
    RL reward can incorrectly prefer a fixed partial-observation pattern.
    """
    full_open_count = max(1, int(full_open_repeat))
    full_open_specs = [
        (FullOpenUnconstrainedScorePolicy(n_sensors=int(n_sensors)), PowerConstraintsV2())
        for _ in range(full_open_count)
    ]
    baseline_specs = [
        (policy, constraints)
        for policy in default_policies(int(n_sensors), seed=int(seed))
    ]
    candidate_specs: list[tuple[object, PowerConstraintsV2]] = []
    repeats = max(0, int(candidate_mask_repeat))
    if candidate_masks is not None and repeats > 0:
        masks = np.asarray(candidate_masks, dtype=bool).reshape(-1, int(n_sensors))
        limit = int(candidate_mask_limit)
        if limit > 0:
            masks = masks[:limit]
        for repeat_idx in range(repeats):
            for action_idx, mask in enumerate(masks):
                policy = StaticMaskPolicy(
                    mask=tuple(bool(x) for x in mask),
                    name=f"oracle_candidate_r{repeat_idx}_a{action_idx}",
                )
                candidate_specs.append((policy, constraints))
    return [*full_open_specs, *baseline_specs, *candidate_specs]


class OracleSubtypeMaskPolicy:
    """Subtype-conditioned policy used only to diversify oracle pretraining."""

    def __init__(
        self,
        *,
        name: str,
        subtype_ids: np.ndarray,
        calm_mask: np.ndarray,
        particle_mask: np.ndarray,
        flux_mask: np.ndarray,
        thermal_mask: np.ndarray,
        lookahead_steps: int,
    ) -> None:
        self.name = str(name)
        self.subtype_ids = np.asarray(subtype_ids, dtype=int).reshape(-1)
        self.calm_mask = np.asarray(calm_mask, dtype=bool).reshape(-1)
        self.subtype_masks = {
            1: np.asarray(particle_mask, dtype=bool).reshape(-1),
            2: np.asarray(flux_mask, dtype=bool).reshape(-1),
            3: np.asarray(thermal_mask, dtype=bool).reshape(-1),
        }
        self.lookahead_steps = int(max(0, int(lookahead_steps)))

    def reset(self) -> None:
        return None

    def act_mask(self, env: object) -> np.ndarray:
        current_idx = int(getattr(env, "current_idx"))
        end_idx = min(len(self.subtype_ids), current_idx + self.lookahead_steps + 1)
        window = self.subtype_ids[current_idx:end_idx]
        active = window[window > 0]
        subtype_id = int(active[0]) if active.size else 0
        return self.subtype_masks.get(subtype_id, self.calm_mask).copy()

    def act_scores(self, env: object) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


def summarize_oracle_policy_specs(
    policy_specs: list[tuple[object, PowerConstraintsV2]],
    *,
    rollouts_per_policy: int,
    per_rollout_steps: int,
) -> dict[str, object]:
    count_by_policy: dict[str, int] = {}
    rollouts_by_policy: dict[str, int] = {}
    steps_by_policy: dict[str, int] = {}
    for policy, _ in policy_specs:
        policy_name = str(getattr(policy, "name", policy.__class__.__name__))
        count_by_policy[policy_name] = count_by_policy.get(policy_name, 0) + 1
        rollouts_by_policy[policy_name] = rollouts_by_policy.get(policy_name, 0) + int(rollouts_per_policy)
        steps_by_policy[policy_name] = (
            steps_by_policy.get(policy_name, 0) + int(rollouts_per_policy) * int(per_rollout_steps)
        )
    total_steps = float(sum(steps_by_policy.values()))
    step_fraction_by_policy = {
        policy_name: (float(steps) / total_steps if total_steps > 0 else 0.0)
        for policy_name, steps in steps_by_policy.items()
    }
    return {
        "spec_count_by_policy": count_by_policy,
        "rollouts_by_policy": rollouts_by_policy,
        "steps_by_policy": steps_by_policy,
        "step_fraction_by_policy": step_fraction_by_policy,
        "total_specs": int(len(policy_specs)),
        "total_rollouts": int(sum(rollouts_by_policy.values())),
        "total_steps": int(sum(steps_by_policy.values())),
    }


def oracle_per_rollout_steps(*, lookback: int, horizon: int, rollout_steps: int, rollouts_per_policy: int) -> int:
    return max(
        int(lookback) + int(horizon) + 2,
        int(np.ceil(float(rollout_steps) / float(max(1, int(rollouts_per_policy))))),
    )


def train_oracle(
    truth: pd.DataFrame,
    sensors: list,
    constraints: PowerConstraintsV2,
    *,
    oracle_type: str,
    lookback: int,
    horizon: int,
    rollout_steps: int,
    tcn_epochs: int,
    tcn_batch_size: int,
    tcn_lr: float,
    tcn_channels: int,
    tcn_levels: int,
    tcn_device: str,
    tcn_loss_clip: float,
    tcn_use_mask_channels: bool,
    target_weights: tuple[float, ...],
    target_scales: tuple[float, ...],
    subtype_loss_weighting: bool = False,
    subtype_particle_target_weights: tuple[float, ...] | None = None,
    subtype_flux_target_weights: tuple[float, ...] | None = None,
    subtype_thermal_target_weights: tuple[float, ...] | None = None,
    rollouts_per_policy: int,
    event_fraction: float,
    full_open_repeat: int,
    candidate_masks: np.ndarray | None = None,
    candidate_mask_repeat: int = 0,
    candidate_mask_limit: int = 0,
    subtype_teacher_repeat: int = 0,
    subtype_teacher_lookahead_steps: int = 0,
    subtype_teacher_calm_mask: np.ndarray | None = None,
    subtype_teacher_particle_mask: np.ndarray | None = None,
    subtype_teacher_flux_mask: np.ndarray | None = None,
    subtype_teacher_thermal_mask: np.ndarray | None = None,
    base_freq_s: int,
    seed: int,
) -> LinearFrozenForecastOracle | TCNFrozenForecastOracle:
    policy_specs = build_oracle_policy_specs(
        len(sensors),
        constraints,
        seed=int(seed),
        full_open_repeat=int(full_open_repeat),
        candidate_masks=candidate_masks,
        candidate_mask_repeat=int(candidate_mask_repeat),
        candidate_mask_limit=int(candidate_mask_limit),
    )
    if int(subtype_teacher_repeat) > 0:
        required_masks = {
            "calm": subtype_teacher_calm_mask,
            "particle": subtype_teacher_particle_mask,
            "flux": subtype_teacher_flux_mask,
        }
        missing = [name for name, mask in required_masks.items() if mask is None]
        if missing:
            raise ValueError(f"oracle subtype teacher masks are missing: {missing}")
        thermal_mask = (
            subtype_teacher_thermal_mask
            if subtype_teacher_thermal_mask is not None
            else subtype_teacher_calm_mask
        )
        if "event_subtype_id" not in truth.columns:
            raise ValueError("oracle subtype teacher requires truth column event_subtype_id")
        subtype_ids_for_policy = truth["event_subtype_id"].astype(int).to_numpy()
        for repeat_idx in range(int(subtype_teacher_repeat)):
            policy_specs.append(
                (
                    OracleSubtypeMaskPolicy(
                        name=f"oracle_subtype_teacher_r{int(repeat_idx)}",
                        subtype_ids=subtype_ids_for_policy,
                        calm_mask=np.asarray(subtype_teacher_calm_mask, dtype=bool),
                        particle_mask=np.asarray(subtype_teacher_particle_mask, dtype=bool),
                        flux_mask=np.asarray(subtype_teacher_flux_mask, dtype=bool),
                        thermal_mask=np.asarray(thermal_mask, dtype=bool),
                        lookahead_steps=int(subtype_teacher_lookahead_steps),
                    ),
                    constraints,
                )
            )
    target_indices = [STATE_COLUMNS.index(name) for name in REWARD_TARGET_COLUMNS]
    x_batches = []
    y_batches = []
    subtype_batches = []
    subtype_values = (
        truth["event_subtype_id"].astype(int).to_numpy()
        if "event_subtype_id" in truth.columns
        else np.zeros(len(truth), dtype=int)
    )
    per_rollout_steps = oracle_per_rollout_steps(
        lookback=int(lookback),
        horizon=int(horizon),
        rollout_steps=int(rollout_steps),
        rollouts_per_policy=int(rollouts_per_policy),
    )
    for idx, (policy, policy_constraints) in enumerate(policy_specs):
        start_indices = select_eval_start_indices(
            truth,
            steps=int(per_rollout_steps),
            horizon=int(horizon),
            n_rollouts=max(1, int(rollouts_per_policy)),
            event_fraction=float(event_fraction),
            seed=int(seed) + 10_000 + idx,
        )
        for offset, start_idx in enumerate(start_indices):
            env = WarmupSchedulingEnv(
                truth,
                sensors,
                policy_constraints,
                WarmupEnvConfig(
                    state_columns=STATE_COLUMNS,
                    reward_target_columns=REWARD_TARGET_COLUMNS,
                    lookback=int(lookback),
                    episode_len=int(per_rollout_steps),
                    seed=int(seed) + idx * 100 + offset,
                    base_freq_s=int(base_freq_s),
                ),
            )
            result = run_policy_rollout(env, policy, steps=int(per_rollout_steps), start_idx=int(start_idx))
            step_indices = np.asarray(result.step_indices, dtype=int)
            rollout_subtypes = np.zeros(result.observations.shape[0], dtype=int)
            valid_steps = (step_indices >= 0) & (step_indices < subtype_values.shape[0])
            rollout_subtypes[valid_steps] = subtype_values[step_indices[valid_steps]]
            x_part, y_part, subtype_part = build_supervised_windows_with_context(
                result.observations,
                result.masks,
                result.truth[:, target_indices],
                context_series=rollout_subtypes,
                lookback=int(lookback),
                horizon=int(horizon),
            )
            x_batches.append(x_part)
            y_batches.append(y_part)
            subtype_batches.append(subtype_part)
    if not x_batches:
        raise ValueError("No oracle training windows were built")
    x_train = np.vstack(x_batches)
    y_train = np.vstack(y_batches)
    subtype_train = np.concatenate(subtype_batches, axis=0).astype(int)
    if oracle_type == "linear":
        oracle = LinearFrozenForecastOracle(
            OracleConfig(
                lookback=int(lookback),
                horizon=int(horizon),
                ridge_alpha=10.0,
                target_weights=tuple(float(x) for x in target_weights),
                target_scales=tuple(float(x) for x in target_scales),
                subtype_loss_weighting=bool(subtype_loss_weighting),
                subtype_particle_target_weights=subtype_particle_target_weights,
                subtype_flux_target_weights=subtype_flux_target_weights,
                subtype_thermal_target_weights=subtype_thermal_target_weights,
            )
        )
    elif oracle_type == "tcn":
        oracle = TCNFrozenForecastOracle(
            TCNOracleConfig(
                lookback=int(lookback),
                horizon=int(horizon),
                channels=int(tcn_channels),
                levels=int(tcn_levels),
                epochs=int(tcn_epochs),
                batch_size=int(tcn_batch_size),
                learning_rate=float(tcn_lr),
                seed=int(seed),
                device=str(tcn_device),
                loss_clip=float(tcn_loss_clip),
                use_mask_channels=bool(tcn_use_mask_channels),
                target_weights=tuple(float(x) for x in target_weights),
                target_scales=tuple(float(x) for x in target_scales),
                subtype_loss_weighting=bool(subtype_loss_weighting),
                subtype_particle_target_weights=subtype_particle_target_weights,
                subtype_flux_target_weights=subtype_flux_target_weights,
                subtype_thermal_target_weights=subtype_thermal_target_weights,
            )
        )
    else:
        raise ValueError(f"Unsupported oracle type: {oracle_type}")
    if oracle_type == "tcn":
        oracle.fit(x_train, y_train, sample_contexts=subtype_train)
    else:
        oracle.fit(x_train, y_train)
    return oracle


def _sensor_utility(spec, variable_weights: dict[str, float]) -> float:
    observed = {str(variable) for variable in spec.observed_variables}
    utility = 0.0
    for variable, weight in variable_weights.items():
        if variable in observed:
            utility += float(weight)
        elif variable in {"wind_dir_sin", "wind_dir_cos"} and "wind_direction_deg" in observed:
            utility += float(weight)
    return utility


def make_weighted_sensor_score_prior(
    sensors: list,
    *,
    scale: float,
    variable_weights: dict[str, float],
) -> np.ndarray:
    if float(scale) <= 0:
        return np.zeros(len(sensors), dtype=float)
    values = []
    for spec in sensors:
        utility = _sensor_utility(spec, variable_weights)
        values.append(utility / max(float(spec.power_cost), 1e-6))
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or float(np.max(arr)) <= 0:
        return np.zeros(len(sensors), dtype=float)
    return float(scale) * (arr / float(np.max(arr)))


def make_sensor_score_prior(sensors: list, *, scale: float, target_weights: tuple[float, ...]) -> np.ndarray:
    variable_weights = {
        str(name): float(weight)
        for name, weight in zip(REWARD_TARGET_COLUMNS, target_weights, strict=True)
    }
    return make_weighted_sensor_score_prior(sensors, scale=float(scale), variable_weights=variable_weights)


def make_static_anchor_prior(sensors: list, *, scale: float) -> np.ndarray:
    if float(scale) <= 0 or not sensors:
        return np.zeros(len(sensors), dtype=float)
    return float(scale) * np.linspace(1.0, 0.0, len(sensors), dtype=float)


def build_projected_candidate_masks(
    sensors: list,
    constraints: PowerConstraintsV2,
    *,
    max_candidate_warmup: int | None = None,
) -> np.ndarray:
    projector = PowerProjector(sensors, constraints)
    runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in sensors}
    masks: dict[tuple[int, ...], np.ndarray] = {}
    n_sensors = len(sensors)
    allowed = np.ones(n_sensors, dtype=bool)
    if max_candidate_warmup is not None:
        allowed = np.asarray([int(spec.warmup_steps) <= int(max_candidate_warmup) for spec in sensors], dtype=bool)
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
        raise ValueError("No feasible PPO candidate masks were generated for the current constraints")
    return np.asarray(list(masks.values()), dtype=bool)


def plot_training_eval(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics.sort_values("oracle_loss_mean").plot.bar(x="policy", y="oracle_loss_mean", ax=ax, legend=False)
    ax.set_ylabel("Frozen oracle loss")
    ax.set_title("v2 PPO mainline evaluation")
    fig.tight_layout()
    fig.savefig(out_dir / "v2_ppo_eval.png", dpi=180)
    plt.close(fig)


def select_eval_start_indices(
    truth: pd.DataFrame,
    *,
    steps: int,
    horizon: int,
    n_rollouts: int,
    event_fraction: float,
    seed: int,
    event_column: str = "event_flag",
) -> tuple[int, ...]:
    max_start = max(0, len(truth) - int(steps) - int(horizon) - 1)
    if max_start <= 0 or int(n_rollouts) <= 1:
        return (0,)
    rng = np.random.default_rng(int(seed))
    starts: list[int] = []
    n_event = int(round(float(np.clip(event_fraction, 0.0, 1.0)) * int(n_rollouts)))
    event_flags = (
        truth[event_column].astype(bool).to_numpy()
        if event_column in truth.columns
        else np.zeros(len(truth), dtype=bool)
    )
    event_indices = np.flatnonzero(event_flags[: max_start + int(steps)])
    for _ in range(min(n_event, int(n_rollouts))):
        if event_indices.size == 0:
            break
        event_idx = int(rng.choice(event_indices))
        starts.append(int(np.clip(event_idx - int(steps) // 3, 0, max_start)))
    while len(starts) < int(n_rollouts):
        starts.append(int(rng.integers(0, max_start + 1)))
    return tuple(starts)


def evaluate_score_policy_over_starts(
    *,
    truth: pd.DataFrame,
    sensors: list,
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: LinearFrozenForecastOracle | TCNFrozenForecastOracle,
    policy,
    steps: int,
    start_indices: tuple[int, ...],
) -> tuple[object, dict[str, float | str | int]]:
    rollouts = []
    for offset, start_idx in enumerate(start_indices):
        env = WarmupSchedulingEnv(
            truth,
            sensors,
            constraints,
            WarmupEnvConfig(
                state_columns=cfg.state_columns,
                reward_target_columns=cfg.reward_target_columns,
                lookback=cfg.lookback,
                episode_len=cfg.episode_len,
                seed=int(cfg.seed) + int(offset),
                base_freq_s=cfg.base_freq_s,
                event_column=cfg.event_column,
                normalize_agent_state=cfg.normalize_agent_state,
                normalization_mean=cfg.normalization_mean,
                normalization_std=cfg.normalization_std,
                lambda_warmup_abort=cfg.lambda_warmup_abort,
                lambda_switch=cfg.lambda_switch,
                event_reward_multiplier=cfg.event_reward_multiplier,
                energy_account_enabled=cfg.energy_account_enabled,
                energy_capacity=cfg.energy_capacity,
                initial_energy=cfg.initial_energy,
                harvest_per_step=cfg.harvest_per_step,
                reserve_energy=cfg.reserve_energy,
                lambda_energy_deficit=cfg.lambda_energy_deficit,
                soc_soft_penalty_buffer=cfg.soc_soft_penalty_buffer,
                lambda_soc_soft_penalty=cfg.lambda_soc_soft_penalty,
                lambda_duty_balance=cfg.lambda_duty_balance,
                duty_balance_low=cfg.duty_balance_low,
                duty_balance_high=cfg.duty_balance_high,
                duty_balance_grace_steps=cfg.duty_balance_grace_steps,
                duty_score_feedback=cfg.duty_score_feedback,
                duty_score_target=cfg.duty_score_target,
                duty_hard_guard=cfg.duty_hard_guard,
                duty_hard_low=cfg.duty_hard_low,
                duty_hard_high=cfg.duty_hard_high,
                duty_hard_score=cfg.duty_hard_score,
                min_dwell_steps=cfg.min_dwell_steps,
                include_agent_cycle_phase=cfg.include_agent_cycle_phase,
                agent_cycle_period_steps=cfg.agent_cycle_period_steps,
                agent_cycle_dwell_steps=cfg.agent_cycle_dwell_steps,
                include_observable_regime_belief=cfg.include_observable_regime_belief,
                regime_belief_lookback=cfg.regime_belief_lookback,
                agent_context_columns=cfg.agent_context_columns,
            ),
            oracle=oracle,
        )
        rollouts.append(run_policy_rollout(env, policy, steps=int(steps), start_idx=int(start_idx)))
    result = rollouts[0] if len(rollouts) == 1 else concat_rollout_results(rollouts, policy_name=policy.name)
    return result, rollout_metrics(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v2 projected-score PPO with Stable-Baselines3.")
    parser.add_argument("--truth-csv", default="data/generated/v2_public_weather_truth_ppo.csv")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Taishan"])
    parser.add_argument("--truth-steps", type=int, default=4096)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--blowing-snow-event-coverage", type=float, default=0.30)
    parser.add_argument("--blowing-snow-event-model", default="clustered")
    parser.add_argument("--blowing-snow-min-duration-steps", type=int, default=10)
    parser.add_argument("--blowing-snow-max-duration-steps", type=int, default=30)
    parser.add_argument("--blowing-snow-min-gap-steps", type=int, default=6)
    parser.add_argument("--blowing-snow-lead-steps", type=int, default=5)
    parser.add_argument("--blowing-snow-wind-margin-ms", type=float, default=1.5)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.6)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.3)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.6)
    parser.add_argument("--event-microstructure-sigma", type=float, default=0.0)
    parser.add_argument("--event-microstructure-alpha", type=float, default=0.18)
    parser.add_argument("--event-microstructure-diameter-scale", type=float, default=0.0)
    parser.add_argument("--event-microstructure-velocity-scale", type=float, default=0.0)
    parser.add_argument("--event-particle-microstructure-correlation", type=float, default=1.0)
    parser.add_argument("--event-subtypes-enabled", action="store_true")
    parser.add_argument("--event-subtype-particle-prob", type=float, default=0.34)
    parser.add_argument("--event-subtype-assignment", choices=["random", "stratified"], default="random")
    parser.add_argument("--event-subtype-particle-min-parsivel-availability", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-prob", type=float, default=0.33)
    parser.add_argument("--event-subtype-thermal-prob", type=float, default=0.33)
    parser.add_argument("--event-subtype-particle-flux-multiplier", type=float, default=0.72)
    parser.add_argument("--event-subtype-flux-multiplier", type=float, default=2.4)
    parser.add_argument("--event-subtype-thermal-flux-multiplier", type=float, default=0.55)
    parser.add_argument("--event-subtype-particle-diameter-shift-mm", type=float, default=0.10)
    parser.add_argument("--event-subtype-particle-velocity-boost-ms", type=float, default=1.3)
    parser.add_argument("--event-subtype-flux-diameter-shift-mm", type=float, default=-0.04)
    parser.add_argument("--event-subtype-flux-velocity-boost-ms", type=float, default=0.7)
    parser.add_argument("--event-subtype-thermal-surface-drop-c", type=float, default=2.0)
    parser.add_argument("--event-subtype-particle-humidity-boost-pct", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-wind-boost-ms", type=float, default=0.0)
    parser.add_argument("--event-subtype-thermal-air-temp-drop-c", type=float, default=0.0)
    parser.add_argument("--event-subtype-latent-alpha", type=float, default=0.18)
    parser.add_argument("--event-subtype-particle-latent-diameter-scale-mm", type=float, default=0.0)
    parser.add_argument("--event-subtype-particle-latent-velocity-scale-ms", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-latent-sigma", type=float, default=0.0)
    parser.add_argument("--event-subtype-thermal-latent-surface-scale-c", type=float, default=0.0)
    parser.add_argument("--event-subtype-latent-target-lag-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-lead-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-noise-std", type=float, default=0.08)
    parser.add_argument("--event-subtype-context-latent-strength", type=float, default=0.0)
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--out-dir", default="reports/v2_ppo")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--oracle-type", choices=["linear", "tcn"], default="tcn")
    parser.add_argument("--oracle-rollout-steps", type=int, default=1600)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-event-fraction", type=float, default=0.50)
    parser.add_argument(
        "--oracle-full-open-repeat",
        type=int,
        default=3,
        help=(
            "Number of full_open_unconstrained rollout groups in the frozen-oracle "
            "pretraining mixture. Values around 3 give full-open roughly 40% of "
            "the oracle rollout steps with the default baseline set."
        ),
    )
    parser.add_argument("--oracle-epochs", type=int, default=12)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-learning-rate", type=float, default=1e-3)
    parser.add_argument("--oracle-channels", type=int, default=64)
    parser.add_argument("--oracle-levels", type=int, default=3)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--oracle-loss-clip", type=float, default=10.0)
    parser.add_argument("--oracle-candidate-mask-repeat", type=int, default=0)
    parser.add_argument("--oracle-candidate-mask-limit", type=int, default=0)
    parser.add_argument(
        "--oracle-disable-mask-channels",
        action="store_true",
        help="Train the TCN oracle on mean-filled observations only instead of concatenating mask channels.",
    )
    parser.add_argument("--target-weights", nargs="*", type=float, default=list(DEFAULT_TARGET_WEIGHTS))
    parser.add_argument("--target-scales", nargs="*", type=float, default=list(DEFAULT_TARGET_SCALES))
    parser.add_argument("--subtype-loss-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-particle-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-flux-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-thermal-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--train-episode-len", type=int, default=512)
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--eval-event-fraction", type=float, default=0.5)
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vec-type", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--diagnostic-freq", type=int, default=0)
    parser.add_argument("--diagnostic-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-step-budget", type=float, default=1.7)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--required-sensors", nargs="*", default=list(DEFAULT_REQUIRED_SENSOR_IDS))
    parser.add_argument("--disable-coverage-groups", action="store_true")
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--ppo-active-score-bonus", type=float, default=0.08)
    parser.add_argument("--ppo-warming-score-bonus", type=float, default=0.80)
    parser.add_argument("--ppo-prior-scale", type=float, default=0.05)
    parser.add_argument("--ppo-static-anchor-scale", type=float, default=0.35)
    parser.add_argument("--ppo-event-prior-scale", type=float, default=0.0)
    parser.add_argument("--ppo-non-event-prior-scale", type=float, default=0.15)
    parser.add_argument("--ppo-action-scale", type=float, default=0.15)
    parser.add_argument("--ppo-action-mode", choices=["discrete_subset", "score"], default="score")
    parser.add_argument("--ppo-max-candidate-warmup", type=int, default=2)
    parser.add_argument("--ppo-bc-warmstart-steps", type=int, default=0)
    parser.add_argument("--ppo-bc-dataset-steps", type=int, default=2048)
    parser.add_argument("--ppo-bc-rollouts", type=int, default=4)
    parser.add_argument("--ppo-bc-batch-size", type=int, default=256)
    parser.add_argument("--ppo-bc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-bc-event-fraction", type=float, default=0.67)
    parser.add_argument("--ppo-bc-greedy-lookahead-steps", type=int, default=4)
    parser.add_argument("--event-start-prob", type=float, default=0.6)
    args = parser.parse_args()
    target_weights = tuple(float(x) for x in args.target_weights)
    target_scales = tuple(float(x) for x in args.target_scales)
    if len(target_weights) != len(REWARD_TARGET_COLUMNS):
        raise ValueError(
            f"--target-weights must contain {len(REWARD_TARGET_COLUMNS)} values "
            f"matching REWARD_TARGET_COLUMNS, got {len(target_weights)}"
        )
    if len(target_scales) != len(REWARD_TARGET_COLUMNS):
        raise ValueError(
            f"--target-scales must contain {len(REWARD_TARGET_COLUMNS)} values "
            f"matching REWARD_TARGET_COLUMNS, got {len(target_scales)}"
        )
    subtype_particle_target_weights = optional_target_weights(
        args.subtype_particle_target_weights,
        name="--subtype-particle-target-weights",
    )
    subtype_flux_target_weights = optional_target_weights(
        args.subtype_flux_target_weights,
        name="--subtype-flux-target-weights",
    )
    subtype_thermal_target_weights = optional_target_weights(
        args.subtype_thermal_target_weights,
        name="--subtype-thermal-target-weights",
    )
    if int(args.torch_num_threads) > 0:
        try:
            import torch

            torch.set_num_threads(int(args.torch_num_threads))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = ensure_truth(args)
    truth = ensure_state_columns(pd.read_csv(truth_path))
    sensors = load_sensor_specs(args.sensor_cfg)
    target_sensor_score_prior = make_sensor_score_prior(
        sensors,
        scale=float(args.ppo_prior_scale),
        target_weights=target_weights,
    )
    static_anchor_prior = make_static_anchor_prior(
        sensors,
        scale=float(args.ppo_static_anchor_scale),
    )
    sensor_score_prior = target_sensor_score_prior + static_anchor_prior
    event_sensor_score_prior = make_weighted_sensor_score_prior(
        sensors,
        scale=float(args.ppo_event_prior_scale),
        variable_weights=EVENT_PRIOR_WEIGHTS,
    )
    non_event_sensor_score_prior = make_weighted_sensor_score_prior(
        sensors,
        scale=float(args.ppo_non_event_prior_scale),
        variable_weights=NON_EVENT_PRIOR_WEIGHTS,
    )
    coverage_groups = () if bool(args.disable_coverage_groups) else DEFAULT_COVERAGE_GROUPS
    constraints = PowerConstraintsV2(
        max_active=int(args.max_active),
        per_step_budget=float(args.per_step_budget),
        startup_peak_budget=float(args.startup_peak_budget),
        required_sensor_ids=tuple(str(sensor_id) for sensor_id in args.required_sensors),
        coverage_groups=coverage_groups,
    )
    candidate_masks = (
        build_projected_candidate_masks(
            sensors,
            constraints,
            max_candidate_warmup=None
            if int(args.ppo_max_candidate_warmup) < 0
            else int(args.ppo_max_candidate_warmup),
        )
        if str(args.ppo_action_mode) == "discrete_subset"
        else None
    )
    oracle = train_oracle(
        truth,
        sensors,
        constraints,
        oracle_type=str(args.oracle_type),
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        rollout_steps=int(args.oracle_rollout_steps),
        tcn_epochs=int(args.oracle_epochs),
        tcn_batch_size=int(args.oracle_batch_size),
        tcn_lr=float(args.oracle_learning_rate),
        tcn_channels=int(args.oracle_channels),
        tcn_levels=int(args.oracle_levels),
        tcn_device=str(args.oracle_device),
        tcn_loss_clip=float(args.oracle_loss_clip),
        tcn_use_mask_channels=not bool(args.oracle_disable_mask_channels),
        target_weights=target_weights,
        target_scales=target_scales,
        subtype_loss_weighting=bool(args.subtype_loss_weighting),
        subtype_particle_target_weights=subtype_particle_target_weights,
        subtype_flux_target_weights=subtype_flux_target_weights,
        subtype_thermal_target_weights=subtype_thermal_target_weights,
        rollouts_per_policy=int(args.oracle_rollouts_per_policy),
        event_fraction=float(args.oracle_event_fraction),
        full_open_repeat=int(args.oracle_full_open_repeat),
        candidate_masks=candidate_masks,
        candidate_mask_repeat=int(args.oracle_candidate_mask_repeat),
        candidate_mask_limit=int(args.oracle_candidate_mask_limit),
        base_freq_s=int(args.freq_s),
        seed=int(args.seed),
    )
    oracle_path = out_dir / ("v2_tcn_oracle.pt" if args.oracle_type == "tcn" else "v2_linear_oracle.npz")
    oracle.save(str(oracle_path))
    if args.oracle_type == "tcn":
        oracle.to_device(str(args.oracle_inference_device))

    train_cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        reward_target_columns=REWARD_TARGET_COLUMNS,
        lookback=int(args.lookback),
        episode_len=int(args.train_episode_len),
        seed=int(args.seed),
        base_freq_s=int(args.freq_s),
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
    )
    model = train_projected_ppo(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=train_cfg,
        oracle=oracle,
        total_timesteps=int(args.total_timesteps),
        n_envs=int(args.n_envs),
        seed=int(args.seed),
        device=str(args.device),
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ent_coef=float(args.ent_coef),
        clip_range=float(args.clip_range),
        vec_type=str(args.vec_type),
        tensorboard_log=str(out_dir / "tb"),
        diagnostic_csv=str(out_dir / "ppo_training_diagnostics.csv"),
        diagnostic_freq=int(args.diagnostic_freq),
        diagnostic_steps=int(args.diagnostic_steps),
        best_model_path=str(out_dir / "projected_ppo_best_model"),
        active_score_bonus=float(args.ppo_active_score_bonus),
        warming_score_bonus=float(args.ppo_warming_score_bonus),
        event_start_prob=float(args.event_start_prob),
        sensor_score_prior=sensor_score_prior,
        event_sensor_score_prior=event_sensor_score_prior,
        non_event_sensor_score_prior=non_event_sensor_score_prior,
        action_scale=float(args.ppo_action_scale),
        candidate_masks=candidate_masks,
        bc_warmstart_steps=int(args.ppo_bc_warmstart_steps),
        bc_dataset_steps=int(args.ppo_bc_dataset_steps),
        bc_rollouts=int(args.ppo_bc_rollouts),
        bc_batch_size=int(args.ppo_bc_batch_size),
        bc_learning_rate=float(args.ppo_bc_learning_rate),
        bc_event_fraction=float(args.ppo_bc_event_fraction),
        bc_greedy_lookahead_steps=int(args.ppo_bc_greedy_lookahead_steps),
        bc_log_path=str(out_dir / "ppo_bc_warmstart.json"),
    )
    model_path = out_dir / "projected_ppo_model"
    save_model(model, model_path)

    rows = []
    eval_start_indices = select_eval_start_indices(
        truth,
        steps=int(args.eval_steps),
        horizon=int(args.horizon),
        n_rollouts=int(args.eval_rollouts),
        event_fraction=float(args.eval_event_fraction),
        seed=int(args.seed) + 1777,
    )
    eval_cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        reward_target_columns=REWARD_TARGET_COLUMNS,
        lookback=int(args.lookback),
        episode_len=int(args.eval_steps),
        seed=int(args.seed) + 9000,
        base_freq_s=int(args.freq_s),
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
    )
    ppo_result, ppo_metrics = evaluate_sb3_model(
        model=model,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=int(args.eval_steps),
        start_indices=eval_start_indices,
        active_score_bonus=float(args.ppo_active_score_bonus),
        warming_score_bonus=float(args.ppo_warming_score_bonus),
        sensor_score_prior=sensor_score_prior,
        event_sensor_score_prior=event_sensor_score_prior,
        non_event_sensor_score_prior=non_event_sensor_score_prior,
        action_scale=float(args.ppo_action_scale),
        candidate_masks=candidate_masks,
    )
    rows.append(ppo_metrics)
    save_rollout_npz(
        out_dir / "rollout_ppo.npz",
        ppo_result,
        sensor_ids=[s.sensor_id for s in sensors],
        state_columns=STATE_COLUMNS,
    )

    full_open_result, full_open_metrics = evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=PowerConstraintsV2(),
        cfg=eval_cfg,
        oracle=oracle,
        policy=FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        steps=int(args.eval_steps),
        start_indices=eval_start_indices,
    )
    rows.append(full_open_metrics)
    save_rollout_npz(
        out_dir / "rollout_full_open_unconstrained.npz",
        full_open_result,
        sensor_ids=[s.sensor_id for s in sensors],
        state_columns=STATE_COLUMNS,
    )

    for idx, policy in enumerate(default_policies(len(sensors), seed=int(args.seed) + 100)):
        result, metrics = evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(args.eval_steps),
            start_indices=eval_start_indices,
        )
        rows.append(metrics)
        save_rollout_npz(
            out_dir / f"rollout_{result.policy_name}.npz",
            result,
            sensor_ids=[s.sensor_id for s in sensors],
            state_columns=STATE_COLUMNS,
        )
    metrics = pd.DataFrame(rows).sort_values("oracle_loss_mean")
    metrics.to_csv(out_dir / "v2_ppo_metrics.csv", index=False)
    plot_training_eval(metrics, out_dir)
    oracle_policy_specs = build_oracle_policy_specs(
        len(sensors),
        constraints,
        seed=int(args.seed),
        full_open_repeat=int(args.oracle_full_open_repeat),
        candidate_masks=candidate_masks,
        candidate_mask_repeat=int(args.oracle_candidate_mask_repeat),
        candidate_mask_limit=int(args.oracle_candidate_mask_limit),
    )
    oracle_rollout_summary = summarize_oracle_policy_specs(
        oracle_policy_specs,
        rollouts_per_policy=int(args.oracle_rollouts_per_policy),
        per_rollout_steps=oracle_per_rollout_steps(
            lookback=int(args.lookback),
            horizon=int(args.horizon),
            rollout_steps=int(args.oracle_rollout_steps),
            rollouts_per_policy=int(args.oracle_rollouts_per_policy),
        ),
    )
    metadata = {
        "truth_csv": str(truth_path),
        "sensor_cfg": str(args.sensor_cfg),
        "oracle_path": str(oracle_path),
        "oracle_type": str(args.oracle_type),
        "oracle_rollout_steps": int(args.oracle_rollout_steps),
        "oracle_rollouts_per_policy": int(args.oracle_rollouts_per_policy),
        "oracle_event_fraction": float(args.oracle_event_fraction),
        "oracle_full_open_repeat": int(args.oracle_full_open_repeat),
        "oracle_pretrain_rollout_summary": oracle_rollout_summary,
        "reward_target_columns": list(REWARD_TARGET_COLUMNS),
        "target_weights": list(target_weights),
        "target_scales": list(target_scales),
        "subtype_loss_weighting": bool(args.subtype_loss_weighting),
        "subtype_particle_target_weights": None if subtype_particle_target_weights is None else list(subtype_particle_target_weights),
        "subtype_flux_target_weights": None if subtype_flux_target_weights is None else list(subtype_flux_target_weights),
        "subtype_thermal_target_weights": None if subtype_thermal_target_weights is None else list(subtype_thermal_target_weights),
        "oracle_inference_device": str(args.oracle_inference_device),
        "oracle_use_mask_channels": not bool(args.oracle_disable_mask_channels),
        "model_path": str(model_path) + ".zip",
        "eval_policies": [
            "ppo",
            "full_open_unconstrained",
            *[str(policy.name) for policy in default_policies(len(sensors), seed=int(args.seed) + 100)],
        ],
        "device": str(args.device),
        "n_envs": int(args.n_envs),
        "total_timesteps": int(args.total_timesteps),
        "seed": int(args.seed),
        "freq_s": int(args.freq_s),
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "torch_num_threads": int(args.torch_num_threads),
        "diagnostic_freq": int(args.diagnostic_freq),
        "diagnostic_steps": int(args.diagnostic_steps),
        "eval_rollouts": int(args.eval_rollouts),
        "eval_event_fraction": float(args.eval_event_fraction),
        "eval_start_indices": [int(x) for x in eval_start_indices],
        "truth_event_design": {
            "blowing_snow_event_coverage": float(args.blowing_snow_event_coverage),
            "blowing_snow_event_model": str(args.blowing_snow_event_model),
            "blowing_snow_min_duration_steps": int(args.blowing_snow_min_duration_steps),
            "blowing_snow_max_duration_steps": int(args.blowing_snow_max_duration_steps),
            "blowing_snow_min_gap_steps": int(args.blowing_snow_min_gap_steps),
            "blowing_snow_lead_steps": int(args.blowing_snow_lead_steps),
            "blowing_snow_wind_margin_ms": float(args.blowing_snow_wind_margin_ms),
            "cred_hysteresis_on": float(args.cred_hysteresis_on),
            "cred_hysteresis_off": float(args.cred_hysteresis_off),
            "flux_wind_exponent": float(args.flux_wind_exponent),
            "event_microstructure_sigma": float(args.event_microstructure_sigma),
            "event_microstructure_alpha": float(args.event_microstructure_alpha),
            "event_microstructure_diameter_scale": float(args.event_microstructure_diameter_scale),
            "event_microstructure_velocity_scale": float(args.event_microstructure_velocity_scale),
            "event_particle_microstructure_correlation": float(args.event_particle_microstructure_correlation),
        },
        "ppo": {
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "ent_coef": float(args.ent_coef),
            "clip_range": float(args.clip_range),
            "action_mode": str(args.ppo_action_mode),
            "candidate_count": int(candidate_masks.shape[0]) if candidate_masks is not None else None,
            "max_candidate_warmup": int(args.ppo_max_candidate_warmup),
            "bc_warmstart": getattr(model, "bc_warmstart_info", {"enabled": 0}),
        },
        "reward_shaping": {
            "lambda_warmup_abort": float(args.lambda_warmup_abort),
            "lambda_switch": float(args.lambda_switch),
            "ppo_active_score_bonus": float(args.ppo_active_score_bonus),
            "ppo_warming_score_bonus": float(args.ppo_warming_score_bonus),
            "ppo_prior_scale": float(args.ppo_prior_scale),
            "ppo_static_anchor_scale": float(args.ppo_static_anchor_scale),
            "ppo_event_prior_scale": float(args.ppo_event_prior_scale),
            "ppo_non_event_prior_scale": float(args.ppo_non_event_prior_scale),
            "ppo_action_scale": float(args.ppo_action_scale),
            "target_sensor_score_prior": {
                str(spec.sensor_id): float(target_sensor_score_prior[idx])
                for idx, spec in enumerate(sensors)
            },
            "static_anchor_prior": {
                str(spec.sensor_id): float(static_anchor_prior[idx])
                for idx, spec in enumerate(sensors)
            },
            "sensor_score_prior": {
                str(spec.sensor_id): float(sensor_score_prior[idx])
                for idx, spec in enumerate(sensors)
            },
            "event_sensor_score_prior": {
                str(spec.sensor_id): float(event_sensor_score_prior[idx])
                for idx, spec in enumerate(sensors)
            },
            "non_event_sensor_score_prior": {
                str(spec.sensor_id): float(non_event_sensor_score_prior[idx])
                for idx, spec in enumerate(sensors)
            },
            "event_start_prob": float(args.event_start_prob),
        },
        "constraints": {
            "max_active": int(args.max_active),
            "per_step_budget": float(args.per_step_budget),
            "startup_peak_budget": float(args.startup_peak_budget),
            "required_sensor_ids": [str(sensor_id) for sensor_id in args.required_sensors],
            "coverage_groups": [
                {"name": str(name), "sensor_ids": [str(sensor_id) for sensor_id in sensor_ids]}
                for name, sensor_ids in coverage_groups
            ],
        },
    }
    (out_dir / "v2_ppo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir / "v2_ppo_metrics.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
