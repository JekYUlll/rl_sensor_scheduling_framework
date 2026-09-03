#!/usr/bin/env python3
"""Audit downstream forecast-loss geometry for every feasible fixed subset.

This is a frozen diagnostic. It does not train a policy and does not use
policy-selected or sealed test feedback to change the scene.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v2.env import WarmupEnvConfig
from v2.oracle import LinearFrozenForecastOracle
from v2.rollout import concat_rollout_results, run_policy_rollout
from v2.sensor_spec import load_sensor_specs
from v2.tcn_oracle import TCNFrozenForecastOracle
from v2.power_projector import PowerConstraintsV2
from v2.warmup_state import SensorRuntime


ROOT = Path(__file__).resolve().parents[1]
STATE_COLUMNS = (
    "wind_speed_ms", "wind_direction_deg", "wind_dir_sin", "wind_dir_cos",
    "air_temperature_c", "relative_humidity", "air_pressure_pa",
    "solar_radiation_wm2", "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm", "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
)
REWARD_TARGET_COLUMNS = (
    "air_temperature_c", "snow_surface_temperature_c", "wind_speed_ms",
    "wind_dir_sin", "wind_dir_cos", "solar_radiation_wm2",
    "snow_mass_flux_kg_m2_s", "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
)
SUBTYPE_NAMES = {0: "calm", 1: "particle", 2: "flux", 3: "thermal"}
OPERATING_STATE_GROUPS = (
    (
        "generator_flux_demand_state",
        "generator_particle_demand_state",
        "generator_thermal_demand_state",
    ),
    (
        "generator_flux_exposure_state",
        "generator_particle_exposure_state",
        "generator_thermal_exposure_state",
    ),
    ("generator_exposure_transport_state", "generator_exposure_frost_state"),
    ("generator_operating_transport_state", "generator_operating_thermal_state"),
)


def load_diagnostic_module():
    path = Path(__file__).with_name("27_v2_diagnose_action_landscape.py")
    spec = importlib.util.spec_from_file_location("action_landscape_diag", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oracle(path: Path, oracle_type: str):
    if oracle_type == "tcn":
        return TCNFrozenForecastOracle.load(path, device="cpu")
    if oracle_type == "linear":
        return LinearFrozenForecastOracle.load(str(path))
    raise ValueError(f"unsupported oracle type: {oracle_type}")


def constraints_from_metadata(
    meta: dict,
    sensors: list,
    *,
    steady_budget: float | None = None,
    startup_budget: float | None = None,
) -> PowerConstraintsV2:
    constraints_meta = meta.get("constraints", {})
    max_active_value = constraints_meta.get("max_active")
    max_active = len(sensors) if max_active_value is None else int(max_active_value)
    diag = load_diagnostic_module()
    return PowerConstraintsV2(
        max_active=max_active,
        per_step_budget=(
            float(steady_budget)
            if steady_budget is not None
            else float(constraints_meta.get("per_step_budget", 1.75))
        ),
        startup_peak_budget=(
            float(startup_budget)
            if startup_budget is not None
            else float(constraints_meta.get("startup_peak_budget", 2.15))
        ),
        required_sensor_ids=tuple(str(x) for x in constraints_meta.get("required_sensor_ids", [])),
        coverage_groups=diag._coverage_groups_from_metadata(meta),
    )


def env_config_from_metadata(
    meta: dict,
    truth: pd.DataFrame,
    *,
    seed: int,
    episode_len: int,
    extra_context_columns: tuple[str, ...] = (),
) -> WarmupEnvConfig:
    """Reconstruct the evaluated environment from frozen run metadata."""
    reward = dict(meta.get("reward_shaping", {}))
    partition = dict(meta.get("partition_protocol", {}))
    quality = dict(meta.get("sensor_quality", {}))
    alert = dict(meta.get("agent_alert_context", {}))
    uncertainty = dict(meta.get("uncertainty_proxy", {}))
    cycle = dict(meta.get("agent_cycle_phase", {}))
    regime = dict(meta.get("observable_regime_belief", {}))
    energy = dict(meta.get("energy_account", {}))

    state_columns = tuple(str(x) for x in meta.get("state_columns", STATE_COLUMNS))
    target_columns = tuple(str(x) for x in meta.get("reward_target_columns", REWARD_TARGET_COLUMNS))
    norm_start = int(partition.get("normalization_start_idx", 0))
    norm_end = int(partition.get("normalization_end_idx", len(truth)))
    if norm_start < 0 or norm_end <= norm_start or norm_end > len(truth):
        raise ValueError(f"invalid normalization interval [{norm_start}, {norm_end})")
    state_values = truth.iloc[norm_start:norm_end][list(state_columns)].to_numpy(dtype=float)
    normalization_mean = tuple(float(x) for x in np.mean(state_values, axis=0))
    normalization_std = tuple(float(max(x, 1.0e-6)) for x in np.std(state_values, axis=0))

    context_columns = tuple(str(x) for x in meta.get("agent_context_columns", ()))
    for column in extra_context_columns:
        if str(column) not in context_columns:
            context_columns += (str(column),)
    context_meta = dict(meta.get("agent_context_normalization", {}))
    stored_columns = tuple(str(x) for x in context_meta.get("columns", ()))
    if context_columns and stored_columns == context_columns:
        context_mean = tuple(float(x) for x in context_meta.get("mean", ()))
        context_std = tuple(float(x) for x in context_meta.get("std", ()))
    elif context_columns:
        context_values = truth.iloc[norm_start:norm_end][list(context_columns)].to_numpy(dtype=float)
        context_mean = tuple(float(x) for x in np.mean(context_values, axis=0))
        context_std = tuple(float(max(x, 1.0e-6)) for x in np.std(context_values, axis=0))
    else:
        context_mean = None
        context_std = None

    return WarmupEnvConfig(
        state_columns=state_columns,
        reward_target_columns=target_columns,
        reward_proxy_mode=str(reward.get("reward_proxy_mode", "forecast")),
        lookback=int(meta.get("lookback", 20)),
        episode_len=int(episode_len),
        seed=int(seed),
        base_freq_s=int(meta.get("freq_s", 3600)),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        lambda_warmup_abort=float(reward.get("lambda_warmup_abort", 0.08)),
        lambda_switch=float(reward.get("lambda_switch", 0.002)),
        event_reward_multiplier=float(meta.get("event_reward_multiplier", 1.0)),
        event_subtype_particle_reward_multiplier=float(reward.get("event_subtype_particle_reward_multiplier", 1.0)),
        event_subtype_flux_reward_multiplier=float(reward.get("event_subtype_flux_reward_multiplier", 1.0)),
        event_subtype_thermal_reward_multiplier=float(reward.get("event_subtype_thermal_reward_multiplier", 1.0)),
        oracle_loss_reward_normalizers=(
            None
            if reward.get("reward_loss_normalizers") is None
            else tuple(float(x) for x in reward["reward_loss_normalizers"])
        ),
        oracle_loss_reward_default_normalizer=float(reward.get("reward_loss_default_normalizer", 1.0)),
        energy_account_enabled=bool(energy.get("enabled", False)),
        energy_capacity=float(energy.get("energy_capacity", 0.0)),
        initial_energy=float(energy.get("initial_energy", 0.0)),
        harvest_per_step=float(energy.get("harvest_per_step", 0.0)),
        reserve_energy=float(energy.get("reserve_energy", 0.0)),
        lambda_energy_deficit=float(energy.get("lambda_energy_deficit", 1.0)),
        soc_soft_penalty_buffer=float(energy.get("soc_soft_penalty_buffer", 0.0)),
        lambda_soc_soft_penalty=float(energy.get("lambda_soc_soft_penalty", 0.0)),
        lambda_duty_balance=float(reward.get("lambda_duty_balance", 0.0)),
        duty_balance_low=float(reward.get("duty_balance_low", 0.05)),
        duty_balance_high=float(reward.get("duty_balance_high", 0.95)),
        duty_balance_grace_steps=int(reward.get("duty_balance_grace_steps", 64)),
        duty_score_feedback=float(reward.get("duty_score_feedback", 0.0)),
        duty_score_target=float(reward.get("duty_score_target", 0.4)),
        duty_hard_guard=bool(reward.get("duty_hard_guard", False)),
        duty_hard_low=float(reward.get("duty_hard_low", 0.08)),
        duty_hard_high=float(reward.get("duty_hard_high", 0.92)),
        duty_hard_score=float(reward.get("duty_hard_score", 8.0)),
        min_dwell_steps=max(1, int(reward.get("min_dwell_steps", 1))),
        common_random_numbers=bool(reward.get("common_random_numbers", False)),
        include_agent_cycle_phase=bool(cycle.get("enabled", False)),
        agent_cycle_period_steps=int(cycle.get("period_steps", 0)),
        agent_cycle_dwell_steps=max(1, int(cycle.get("dwell_steps", 1))),
        include_observable_regime_belief=bool(regime.get("enabled", False)),
        regime_belief_lookback=max(1, int(regime.get("lookback", 6))),
        agent_context_columns=context_columns,
        agent_context_normalization_mean=context_mean,
        agent_context_normalization_std=context_std,
        include_event_flag_in_state=bool(alert.get("include_event_flag_in_state", True)),
        include_alert_context_features=bool(alert.get("include_alert_context_features", False)),
        alert_context_columns=tuple(str(x) for x in alert.get("columns", WarmupEnvConfig.alert_context_columns)),
        alert_context_threshold=float(alert.get("threshold", 0.5)),
        alert_context_trend_lookback=max(1, int(alert.get("trend_lookback", 6))),
        uncertainty_process_variance=(
            None
            if uncertainty.get("process_variance") is None
            else tuple(float(x) for x in uncertainty["process_variance"])
        ),
        uncertainty_initial_variance=float(uncertainty.get("initial_variance", 1.0)),
        uncertainty_max_variance=float(uncertainty.get("max_variance", 25.0)),
        measurement_update_mode=str(uncertainty.get("measurement_update_mode", "direct")),
        sensor_quality_columns=tuple(str(x) for x in quality.get("columns", ())),
        sensor_quality_max_noise_multiplier=float(quality.get("max_noise_multiplier", 1.0)),
        sensor_quality_availability_floor=float(quality.get("availability_floor", 1.0)),
    )


def operating_condition_labels(
    truth: pd.DataFrame,
    meta: dict,
    *,
    activity_aligned_transport_demand: bool = False,
) -> tuple[np.ndarray, tuple[str, ...] | None, dict[str, float]]:
    """Return disjoint operating-state bins fixed on the training partition."""
    state_columns = next(
        (group for group in OPERATING_STATE_GROUPS if all(column in truth for column in group)),
        None,
    )
    if state_columns is None:
        return np.full(len(truth), "unavailable", dtype=object), None, {}
    partition = dict(meta.get("partition_protocol", {}))
    start = int(partition.get("normalization_start_idx", 0))
    end = int(partition.get("normalization_end_idx", len(truth)))
    if start < 0 or end <= start or end > len(truth):
        raise ValueError(f"invalid operating-state calibration interval [{start}, {end})")
    calibration = truth.iloc[start:end]
    calibration_active = (
        calibration["blowing_snow_active"].to_numpy(dtype=bool)
        if activity_aligned_transport_demand
        else np.ones(len(calibration), dtype=bool)
    )
    thresholds = {}
    for index, column in enumerate(state_columns):
        values = calibration[column].to_numpy(dtype=float)
        threshold_values = (
            values[calibration_active]
            if activity_aligned_transport_demand and index < 2 and len(state_columns) == 3
            else values
        )
        if len(threshold_values) == 0:
            raise ValueError(f"no active calibration samples for {column}")
        thresholds[column] = float(np.median(threshold_values))
    high = {
        column: truth[column].to_numpy(dtype=float) >= thresholds[column]
        for column in state_columns
    }
    if len(state_columns) == 3:
        labels = np.array(
            [
                f"flux_{int(f)}_particle_{int(p)}_thermal_{int(t)}"
                for f, p, t in zip(
                    high[state_columns[0]], high[state_columns[1]], high[state_columns[2]], strict=True
                )
            ],
            dtype=object,
        )
        if activity_aligned_transport_demand:
            if "blowing_snow_active" not in truth:
                raise ValueError("activity-aligned operating bins require blowing_snow_active")
            labels[~truth["blowing_snow_active"].to_numpy(dtype=bool)] = "unavailable"
        return labels, state_columns, thresholds
    transport_high = high[state_columns[0]]
    secondary_high = high[state_columns[1]]
    labels = np.select(
        (
            ~transport_high & ~secondary_high,
            transport_high & ~secondary_high,
            ~transport_high & secondary_high,
            transport_high & secondary_high,
        ),
        ("recovered", "transport_loaded", "secondary_loaded", "combined_exposure"),
        default="unclassified",
    )
    return labels.astype(object), state_columns, thresholds


def summarize_condition_geometry(
    frame: pd.DataFrame,
    *,
    condition_column: str,
    epsilons: list[float],
) -> dict[str, object]:
    frame = frame[frame[condition_column] != "unavailable"].copy()
    if frame.empty:
        return {
            "conditions": [],
            "sample_counts": {},
            "best_candidates": {},
            "best_static_candidate": None,
            "best_static_loss": float("nan"),
            "weighted_best_loss": float("nan"),
            "opportunity_gap_best_static_minus_conditionwise": float("nan"),
            "near_optimal_intersections": {str(epsilon): [] for epsilon in epsilons},
        }
    means = frame.groupby(
        ["candidate", "selected_sensor_ids", "steady_cost", condition_column],
        as_index=False,
    ).agg(oracle_loss=("oracle_loss", "mean"), samples=("oracle_loss", "size"))
    pivot = means.pivot(index="candidate", columns=condition_column, values="oracle_loss")
    conditions = [str(column) for column in pivot.columns if str(column) != "unavailable"]
    condition_best = {condition: str(pivot[condition].idxmin()) for condition in conditions}
    weights = frame[condition_column].value_counts(normalize=True)
    weighted_candidate_losses = {
        str(candidate): float(sum(
            float(pivot.loc[candidate, condition]) * float(weights.get(condition, 0.0))
            for condition in conditions
        ))
        for candidate in pivot.index
    }
    best_static_candidate = min(weighted_candidate_losses, key=weighted_candidate_losses.get)
    best_static_loss = float(weighted_candidate_losses[best_static_candidate])
    weighted_best = float(
        sum(float(pivot.loc[condition_best[c], c]) * float(weights.get(c, 0.0)) for c in conditions)
    )
    intersections: dict[str, list[str]] = {}
    for epsilon in epsilons:
        near_sets = []
        for condition in conditions:
            best = float(pivot[condition].min())
            near_sets.append(set(pivot.index[pivot[condition] <= best + float(epsilon)]))
        intersections[str(epsilon)] = sorted(set.intersection(*near_sets)) if near_sets else []
    return {
        "conditions": conditions,
        "sample_counts": {c: int((frame[condition_column] == c).sum()) for c in conditions},
        "best_candidates": condition_best,
        "best_static_candidate": str(best_static_candidate),
        "best_static_loss": best_static_loss,
        "weighted_best_loss": weighted_best,
        "opportunity_gap_best_static_minus_conditionwise": float(best_static_loss - weighted_best),
        "near_optimal_intersections": intersections,
    }


def audit_run(
    run_dir: Path,
    out_dir: Path,
    steps: int,
    max_rollouts: int,
    epsilons: list[float],
    *,
    steady_budget: float | None = None,
    startup_budget: float | None = None,
    activity_aligned_transport_demand: bool = False,
) -> dict:
    diag = load_diagnostic_module()
    meta = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(meta["truth_csv"])
    sensors = load_sensor_specs(meta["sensor_cfg"])
    constraints = constraints_from_metadata(
        meta,
        sensors,
        steady_budget=steady_budget,
        startup_budget=startup_budget,
    )
    masks = diag.build_candidate_masks(sensors, constraints, max_candidate_warmup=None)
    oracle = load_oracle(Path(meta["oracle_path"]), str(meta.get("oracle_type", "tcn")))
    starts = [int(x) for x in meta.get("eval_start_indices", [0])][: max(1, max_rollouts)]
    operating_labels, operating_state_columns, operating_thresholds = operating_condition_labels(
        truth,
        meta,
        activity_aligned_transport_demand=activity_aligned_transport_demand,
    )
    records: list[dict] = []

    for idx, mask in enumerate(masks):
        policy = diag.FixedMaskPolicy(mask, name=f"candidate_{idx:03d}")
        rollouts = []
        for offset, start_idx in enumerate(starts):
            cfg = env_config_from_metadata(
                meta,
                truth,
                seed=int(meta.get("seed", 42)) + 1000 + offset,
                episode_len=int(steps),
            )
            env = diag.WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
            rollouts.append(run_policy_rollout(env, policy, steps=int(steps), start_idx=start_idx))
        result = concat_rollout_results(rollouts, policy_name=policy.name)
        subtype = truth.iloc[result.step_indices]["event_subtype_id"].to_numpy(dtype=int)
        operating_condition = operating_labels[result.step_indices]
        sensor_ids = ";".join(spec.sensor_id for spec, selected in zip(sensors, mask, strict=True) if bool(selected))
        steady_cost = float(sum(spec.power_cost for spec, selected in zip(sensors, mask, strict=True) if bool(selected)))
        for loss, subtype_id, operating_label in zip(
            result.oracle_losses,
            subtype,
            operating_condition,
            strict=True,
        ):
            records.append({
                "seed": int(meta.get("seed", -1)),
                "candidate": policy.name,
                "selected_sensor_ids": sensor_ids,
                "steady_cost": steady_cost,
                "condition": SUBTYPE_NAMES.get(int(subtype_id), "unknown"),
                "operating_condition": str(operating_label),
                "event_subtype_id": int(subtype_id),
                "oracle_loss": float(loss),
            })

    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / f"subset_condition_losses_seed{int(meta.get('seed', -1))}.csv", index=False)
    event_geometry = summarize_condition_geometry(
        frame,
        condition_column="condition",
        epsilons=epsilons,
    )
    operating_geometry = summarize_condition_geometry(
        frame,
        condition_column="operating_condition",
        epsilons=epsilons,
    )
    overall = frame.groupby(["candidate", "selected_sensor_ids", "steady_cost"], as_index=False).oracle_loss.mean()
    best_overall = overall.sort_values("oracle_loss").iloc[0]

    sensor_ids = [spec.sensor_id for spec in sensors]
    power_rows = []
    for mask in masks:
        power_rows.append({
            "candidate": f"candidate_{len(power_rows):03d}",
            "selected_sensor_ids": ";".join(s for s, selected in zip(sensor_ids, mask, strict=True) if bool(selected)),
            "steady_cost": float(sum(spec.power_cost for spec, selected in zip(sensors, mask, strict=True) if bool(selected))),
            "startup_cost": float(sum(spec.startup_peak_power for spec, selected in zip(sensors, mask, strict=True) if bool(selected))),
        })
    specialist = {"surface_temp_ir", "laser_disdrometer", "fc4_flux"}
    specialist_union_feasible = any(
        set(row["selected_sensor_ids"].split(";")) == specialist for row in power_rows
    )
    summary = {
        "seed": int(meta.get("seed", -1)),
        "steady_budget": float(constraints.per_step_budget),
        "startup_budget": float(constraints.startup_peak_budget),
        "candidate_count": int(len(masks)),
        "conditions": event_geometry["conditions"],
        "condition_sample_counts": event_geometry["sample_counts"],
        "best_overall_candidate": str(best_overall.candidate),
        "best_overall_sensors": str(best_overall.selected_sensor_ids),
        "best_overall_loss": float(best_overall.oracle_loss),
        "condition_best_candidates": event_geometry["best_candidates"],
        "weighted_condition_best_loss": event_geometry["weighted_best_loss"],
        "condition_domain_best_static_candidate": event_geometry["best_static_candidate"],
        "condition_domain_best_static_loss": event_geometry["best_static_loss"],
        "opportunity_gap_best_static_minus_conditionwise": event_geometry[
            "opportunity_gap_best_static_minus_conditionwise"
        ],
        "near_optimal_intersections": event_geometry["near_optimal_intersections"],
        "operating_state_columns": list(operating_state_columns or ()),
        "operating_state_thresholds": operating_thresholds,
        "operating_conditions": operating_geometry["conditions"],
        "operating_condition_sample_counts": operating_geometry["sample_counts"],
        "operating_condition_best_candidates": operating_geometry["best_candidates"],
        "operating_weighted_condition_best_loss": operating_geometry["weighted_best_loss"],
        "operating_domain_best_static_candidate": operating_geometry["best_static_candidate"],
        "operating_domain_best_static_loss": operating_geometry["best_static_loss"],
        "opportunity_gap_best_static_minus_operating_conditionwise": operating_geometry[
            "opportunity_gap_best_static_minus_conditionwise"
        ],
        "operating_near_optimal_intersections": operating_geometry["near_optimal_intersections"],
        "specialist_union_feasible": specialist_union_feasible,
        "power_rows": power_rows,
    }
    (out_dir / f"subset_forecast_geometry_seed{int(meta.get('seed', -1))}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-rollouts", type=int, default=4)
    parser.add_argument("--epsilon", action="append", type=float, default=[0.01, 0.05])
    parser.add_argument("--steady-budget", type=float)
    parser.add_argument("--startup-budget", type=float)
    parser.add_argument("--activity-aligned-transport-demand", action="store_true")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Bound PyTorch CPU thread pools for this small frozen audit.",
    )
    args = parser.parse_args()
    thread_count = max(1, int(args.torch_threads))
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError:
        pass
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        audit_run(
            path,
            args.out_dir,
            args.steps,
            args.max_rollouts,
            args.epsilon,
            steady_budget=args.steady_budget,
            startup_budget=args.startup_budget,
            activity_aligned_transport_demand=bool(args.activity_aligned_transport_demand),
        )
        for path in args.run_dir
    ]
    all_frames = [pd.read_csv(args.out_dir / f"subset_condition_losses_seed{row['seed']}.csv") for row in summaries]
    pd.concat(all_frames, ignore_index=True).to_csv(args.out_dir / "subset_condition_losses.csv", index=False)
    (args.out_dir / "subset_forecast_geometry_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
