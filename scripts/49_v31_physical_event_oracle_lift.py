#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.custom_ppo import oracle_greedy_candidate_costs  # noqa: E402
from v2.oracle import make_oracle_feature  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


MID_DUTY_LOW = 0.05
MID_DUTY_HIGH = 0.95
ALWAYS_OFF_DUTY = 0.01
ALWAYS_ON_DUTY = 0.99


def load_train_helpers():
    path = ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("_v2_train_ppo_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_sensor_cfg(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def evaluate_candidate_masks(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: PowerConstraintsV2,
    candidate_masks: np.ndarray,
    oracle: object,
    cfg: WarmupEnvConfig,
    start_indices: tuple[int, ...],
    steps: int,
    target_diagnostics: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    masks = np.asarray(candidate_masks, dtype=bool)
    subtype_ids = (
        truth["event_subtype_id"].to_numpy(dtype=int)
        if "event_subtype_id" in truth.columns
        else None
    )
    for action_idx, mask in enumerate(masks):
        losses: list[float] = []
        event_losses: list[float] = []
        non_event_losses: list[float] = []
        subtype_losses: dict[int, list[float]] = {1: [], 2: [], 3: []}
        powers: list[float] = []
        peaks: list[float] = []
        soc_values: list[float] = []
        deficits: list[float] = []
        guard_drops = 0
        event_target_errors: list[np.ndarray] = []
        non_event_target_errors: list[np.ndarray] = []
        aborts = 0
        for offset, start_idx in enumerate(start_indices):
            env = WarmupSchedulingEnv(
                truth,
                sensors,
                constraints,
                WarmupEnvConfig(
                    state_columns=cfg.state_columns,
                    reward_target_columns=cfg.reward_target_columns,
                    lookback=cfg.lookback,
                    episode_len=int(steps),
                    seed=int(cfg.seed) + int(offset) + 51_000,
                    base_freq_s=cfg.base_freq_s,
                    event_column=cfg.event_column,
                    normalize_agent_state=cfg.normalize_agent_state,
                    lambda_warmup_abort=cfg.lambda_warmup_abort,
                    lambda_switch=cfg.lambda_switch,
                    energy_account_enabled=cfg.energy_account_enabled,
                    energy_capacity=cfg.energy_capacity,
                    initial_energy=cfg.initial_energy,
                    harvest_per_step=cfg.harvest_per_step,
                    reserve_energy=cfg.reserve_energy,
                    lambda_energy_deficit=cfg.lambda_energy_deficit,
                ),
                oracle=oracle,
            )
            env.reset(start_idx=int(start_idx))
            for _ in range(int(steps)):
                _, _, done, info = env.step_mask(mask)
                loss = float(info.get("oracle_loss", float("nan")))
                if np.isfinite(loss):
                    losses.append(loss)
                    is_event = bool(info.get("event", False))
                    if subtype_ids is not None:
                        step_idx = int(env.current_idx)
                        if 0 <= step_idx < len(subtype_ids):
                            subtype_id = int(subtype_ids[step_idx])
                            if subtype_id in subtype_losses:
                                subtype_losses[subtype_id].append(loss)
                    if target_diagnostics:
                        start = int(env.current_idx) + 1
                        end = start + int(oracle.cfg.horizon)
                        if end <= len(env.truth_values):
                            feature = make_oracle_feature(env.history, env.mask_history)
                            pred = np.asarray(oracle.predict(feature), dtype=float).reshape(
                                int(oracle.cfg.horizon),
                                len(cfg.reward_target_columns),
                            )
                            target = env.truth_values[start:end][:, env.reward_target_indices]
                            err = np.mean(np.abs(pred - target), axis=0)
                            if is_event:
                                event_target_errors.append(err)
                            else:
                                non_event_target_errors.append(err)
                    if is_event:
                        event_losses.append(loss)
                    else:
                        non_event_losses.append(loss)
                powers.append(float(info.get("power", 0.0)))
                peaks.append(float(info.get("peak_power", 0.0)))
                soc_values.append(float(info.get("soc", float("nan"))))
                deficits.append(float(info.get("energy_deficit", 0.0)))
                guard_drops += int(info.get("energy_guard_dropped", 0))
                aborts += int(info.get("warmup_abort_delta", 0))
                if done:
                    break
        selected_ids = tuple(str(sensors[i].sensor_id) for i in np.flatnonzero(mask))
        row = {
            "action_idx": int(action_idx),
            "sensor_ids": "|".join(selected_ids),
            "sensor_count": int(len(selected_ids)),
            "has_laser": bool("laser_disdrometer" in selected_ids),
            "has_fc4": bool("fc4_flux" in selected_ids),
            "has_snow_particle_counter": bool("snow_particle_counter" in selected_ids),
            "has_radiometer": bool("radiometer_basic" in selected_ids),
            "oracle_loss_mean": float(np.mean(losses)) if losses else float("inf"),
            "oracle_loss_event": float(np.mean(event_losses)) if event_losses else float("inf"),
            "oracle_loss_non_event": float(np.mean(non_event_losses)) if non_event_losses else float("inf"),
            "oracle_loss_subtype_particle": float(np.mean(subtype_losses[1])) if subtype_losses[1] else float("inf"),
            "oracle_loss_subtype_flux": float(np.mean(subtype_losses[2])) if subtype_losses[2] else float("inf"),
            "oracle_loss_subtype_thermal": float(np.mean(subtype_losses[3])) if subtype_losses[3] else float("inf"),
            "steps_subtype_particle": int(len(subtype_losses[1])),
            "steps_subtype_flux": int(len(subtype_losses[2])),
            "steps_subtype_thermal": int(len(subtype_losses[3])),
            "power_mean": float(np.mean(powers)) if powers else 0.0,
            "peak_max": float(np.max(peaks)) if peaks else 0.0,
            "soc_min": float(np.nanmin(soc_values)) if soc_values else float("nan"),
            "energy_deficit_total": float(np.sum(deficits)) if deficits else 0.0,
            "energy_deficit_steps": int(np.sum(np.asarray(deficits, dtype=float) > 1e-12)) if deficits else 0,
            "energy_guard_dropped": int(guard_drops),
            "warmup_abort_count": int(aborts),
            "always_on_sensor_count": int(np.sum(mask.astype(float) >= ALWAYS_ON_DUTY)),
            "always_off_sensor_count": int(np.sum(mask.astype(float) <= ALWAYS_OFF_DUTY)),
            "mid_duty_sensor_count": int(np.sum((mask.astype(float) >= MID_DUTY_LOW) & (mask.astype(float) <= MID_DUTY_HIGH))),
            "duty_entropy": 0.0,
            "switches_per_step": 0.0,
        }
        for idx, spec in enumerate(sensors):
            safe_id = str(spec.sensor_id).replace("/", "_")
            row[f"duty__{safe_id}"] = float(mask[idx])
        if target_diagnostics:
            event_arr = np.vstack(event_target_errors) if event_target_errors else np.full((1, len(cfg.reward_target_columns)), np.nan)
            non_event_arr = (
                np.vstack(non_event_target_errors)
                if non_event_target_errors
                else np.full((1, len(cfg.reward_target_columns)), np.nan)
            )
            for idx, name in enumerate(cfg.reward_target_columns):
                safe_name = str(name).replace("/", "_")
                row[f"target_abs_error_event__{safe_name}"] = float(np.nanmean(event_arr[:, idx]))
                row[f"target_abs_error_non_event__{safe_name}"] = float(np.nanmean(non_event_arr[:, idx]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("oracle_loss_mean").reset_index(drop=True)


def _mask_for_sensor_ids(sensors: list[object], sensor_ids: tuple[str, ...]) -> np.ndarray:
    wanted = {str(x) for x in sensor_ids}
    return np.asarray([str(spec.sensor_id) in wanted for spec in sensors], dtype=bool)


def _sensor_ids_from_pipe(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    text = str(value)
    if not text:
        return ()
    if text.startswith("dynamic:"):
        return ()
    return tuple(part for part in text.split("|") if part)


def _auto_pair_schedule_specs(
    table: pd.DataFrame,
    sensors: list[object],
    *,
    top_k: int,
    lead_steps: int,
) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    static = table[table["action_idx"].astype(int) >= 0].copy()
    schedules: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    if static.empty:
        return schedules
    event_top = static[np.isfinite(static["oracle_loss_event"].to_numpy(dtype=float))].sort_values("oracle_loss_event").head(
        int(top_k)
    )
    non_event_top = static[
        np.isfinite(static["oracle_loss_non_event"].to_numpy(dtype=float))
    ].sort_values("oracle_loss_non_event").head(int(top_k))
    seen: set[tuple[tuple[str, ...], tuple[str, ...], int]] = set()
    for lead in (0, int(lead_steps)):
        for _, calm_row in non_event_top.iterrows():
            calm_ids = _sensor_ids_from_pipe(calm_row["sensor_ids"])
            if not calm_ids:
                continue
            for _, event_row in event_top.iterrows():
                event_ids = _sensor_ids_from_pipe(event_row["sensor_ids"])
                if not event_ids or calm_ids == event_ids:
                    continue
                key = (calm_ids, event_ids, int(lead))
                if key in seen:
                    continue
                seen.add(key)
                schedules.append(
                    (
                        f"auto_non{int(calm_row['action_idx'])}_event{int(event_row['action_idx'])}_lead{int(lead)}",
                        _mask_for_sensor_ids(sensors, calm_ids),
                        _mask_for_sensor_ids(sensors, event_ids),
                        int(lead),
                    )
                )
    return schedules


def _unique_top_masks(
    table: pd.DataFrame,
    sensors: list[object],
    *,
    score_col: str,
    top_k: int,
) -> list[np.ndarray]:
    static = table[table["action_idx"].astype(int) >= 0].copy()
    finite = static[np.isfinite(static[score_col].to_numpy(dtype=float))].sort_values(score_col)
    masks: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for _, row in finite.iterrows():
        ids = _sensor_ids_from_pipe(row["sensor_ids"])
        if not ids:
            continue
        mask = _mask_for_sensor_ids(sensors, ids)
        key = tuple(int(x) for x in mask.astype(int).tolist())
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
        if len(masks) >= int(top_k):
            break
    return masks


def _schedule_label_for_masks(sensors: list[object], masks: list[np.ndarray]) -> str:
    labels: list[str] = []
    for mask in masks:
        ids = [str(sensors[idx].sensor_id) for idx in np.flatnonzero(mask)]
        labels.append("+".join(ids) if ids else "none")
    return ";".join(labels)


def _cyclic_schedule_rows(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: PowerConstraintsV2,
    oracle: object,
    cfg: WarmupEnvConfig,
    start_indices: tuple[int, ...],
    steps: int,
    schedules: list[tuple[str, list[np.ndarray], list[np.ndarray], int, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for schedule_idx, (name, calm_masks, event_masks, lead_steps, dwell_steps) in enumerate(schedules):
        if not calm_masks or not event_masks:
            continue
        losses: list[float] = []
        event_losses: list[float] = []
        non_event_losses: list[float] = []
        powers: list[float] = []
        peaks: list[float] = []
        soc_values: list[float] = []
        deficits: list[float] = []
        guard_drops = 0
        switches: list[float] = []
        aborts = 0
        selected_counts = np.zeros(len(sensors), dtype=float)
        total_steps = 0
        for offset, start_idx in enumerate(start_indices):
            env = WarmupSchedulingEnv(
                truth,
                sensors,
                constraints,
                WarmupEnvConfig(
                    state_columns=cfg.state_columns,
                    reward_target_columns=cfg.reward_target_columns,
                    lookback=cfg.lookback,
                    episode_len=int(steps),
                    seed=int(cfg.seed) + int(offset) + 81_000,
                    base_freq_s=cfg.base_freq_s,
                    event_column=cfg.event_column,
                    normalize_agent_state=cfg.normalize_agent_state,
                    lambda_warmup_abort=cfg.lambda_warmup_abort,
                    lambda_switch=cfg.lambda_switch,
                    energy_account_enabled=cfg.energy_account_enabled,
                    energy_capacity=cfg.energy_capacity,
                    initial_energy=cfg.initial_energy,
                    harvest_per_step=cfg.harvest_per_step,
                    reserve_energy=cfg.reserve_energy,
                    lambda_energy_deficit=cfg.lambda_energy_deficit,
                ),
                oracle=oracle,
            )
            env.reset(start_idx=int(start_idx))
            for _ in range(int(steps)):
                lookahead_end = min(len(env.event_flags), int(env.current_idx) + max(0, int(lead_steps)) + 1)
                trigger = bool(np.any(env.event_flags[int(env.current_idx) : lookahead_end]))
                phase = max(0, int(env.current_idx) - int(start_idx)) // max(1, int(dwell_steps))
                mask_pool = event_masks if trigger else calm_masks
                mask = mask_pool[int(phase) % len(mask_pool)]
                _, _, done, info = env.step_mask(mask)
                loss = float(info.get("oracle_loss", float("nan")))
                if np.isfinite(loss):
                    losses.append(loss)
                    if bool(info.get("event", False)):
                        event_losses.append(loss)
                    else:
                        non_event_losses.append(loss)
                powers.append(float(info.get("power", 0.0)))
                peaks.append(float(info.get("peak_power", 0.0)))
                soc_values.append(float(info.get("soc", float("nan"))))
                deficits.append(float(info.get("energy_deficit", 0.0)))
                guard_drops += int(info.get("energy_guard_dropped", 0))
                switches.append(float(info.get("switch_rate", 0.0)))
                aborts += int(info.get("warmup_abort_delta", 0))
                selected_counts += np.asarray(info.get("selected_mask", [0] * len(sensors)), dtype=float)
                total_steps += 1
                if done:
                    break
        duties = selected_counts / max(1, int(total_steps))
        selected_names = [
            str(spec.sensor_id)
            for idx, spec in enumerate(sensors)
            if duties[idx] > ALWAYS_OFF_DUTY
        ]
        duty_entropy = float(
            -np.mean(
                duties * np.log(np.clip(duties, 1e-9, 1.0))
                + (1.0 - duties) * np.log(np.clip(1.0 - duties, 1e-9, 1.0))
            )
            / np.log(2.0)
        )
        row: dict[str, object] = {
            "action_idx": int(-2000 - schedule_idx),
            "sensor_ids": f"dynamic:{name}",
            "sensor_count": int(len(selected_names)),
            "has_laser": bool("laser_disdrometer" in selected_names),
            "has_fc4": bool("fc4_flux" in selected_names),
            "has_snow_particle_counter": bool("snow_particle_counter" in selected_names),
            "has_radiometer": bool("radiometer_basic" in selected_names),
            "oracle_loss_mean": float(np.mean(losses)) if losses else float("inf"),
            "oracle_loss_event": float(np.mean(event_losses)) if event_losses else float("inf"),
            "oracle_loss_non_event": float(np.mean(non_event_losses)) if non_event_losses else float("inf"),
            "power_mean": float(np.mean(powers)) if powers else 0.0,
            "peak_max": float(np.max(peaks)) if peaks else 0.0,
            "soc_min": float(np.nanmin(soc_values)) if soc_values else float("nan"),
            "energy_deficit_total": float(np.sum(deficits)) if deficits else 0.0,
            "energy_deficit_steps": int(np.sum(np.asarray(deficits, dtype=float) > 1e-12)) if deficits else 0,
            "energy_guard_dropped": int(guard_drops),
            "warmup_abort_count": int(aborts),
            "switches_per_step": float(np.mean(switches)) if switches else 0.0,
            "always_on_sensor_count": int(np.sum(duties >= ALWAYS_ON_DUTY)),
            "always_off_sensor_count": int(np.sum(duties <= ALWAYS_OFF_DUTY)),
            "mid_duty_sensor_count": int(np.sum((duties >= MID_DUTY_LOW) & (duties <= MID_DUTY_HIGH))),
            "duty_entropy": duty_entropy,
            "calm_mask_family": _schedule_label_for_masks(sensors, calm_masks),
            "event_mask_family": _schedule_label_for_masks(sensors, event_masks),
        }
        for idx, spec in enumerate(sensors):
            safe_id = str(spec.sensor_id).replace("/", "_")
            row[f"duty__{safe_id}"] = float(duties[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _auto_diverse_schedule_specs(
    table: pd.DataFrame,
    sensors: list[object],
    *,
    top_k: int,
    lead_steps: int,
    dwell_steps: int,
) -> list[tuple[str, list[np.ndarray], list[np.ndarray], int, int]]:
    event_masks = _unique_top_masks(table, sensors, score_col="oracle_loss_event", top_k=int(top_k))
    non_event_masks = _unique_top_masks(table, sensors, score_col="oracle_loss_non_event", top_k=int(top_k))
    if not event_masks or not non_event_masks:
        return []
    schedules: list[tuple[str, list[np.ndarray], list[np.ndarray], int, int]] = []
    for lead in (0, int(lead_steps)):
        for width in range(2, min(int(top_k), len(event_masks), len(non_event_masks)) + 1):
            schedules.append(
                (
                    f"diverse_top{width}_lead{int(lead)}_dwell{int(dwell_steps)}",
                    non_event_masks[:width],
                    event_masks[:width],
                    int(lead),
                    int(dwell_steps),
                )
            )
    return schedules


def _deployable_static_rows(
    *,
    helpers: object,
    table: pd.DataFrame,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: PowerConstraintsV2,
    oracle: object,
    cfg: WarmupEnvConfig,
    start_indices: tuple[int, ...],
    steps: int,
    top_k: int,
    duty_hard_low: float,
    duty_hard_high: float,
    duty_hard_score: float,
    duty_score_feedback: float,
) -> pd.DataFrame:
    mask_specs: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for score_col in ("oracle_loss_mean", "oracle_loss_event", "oracle_loss_non_event"):
        for mask in _unique_top_masks(table, sensors, score_col=score_col, top_k=int(top_k)):
            key = tuple(int(x) for x in np.asarray(mask, dtype=bool).astype(int).tolist())
            if key in seen:
                continue
            seen.add(key)
            ids = [str(sensors[idx].sensor_id) for idx in np.flatnonzero(mask)]
            mask_specs.append(("|".join(ids), np.asarray(mask, dtype=bool)))

    if not mask_specs:
        return pd.DataFrame()

    deployable_cfg = replace(
        cfg,
        duty_score_feedback=float(duty_score_feedback),
        duty_hard_guard=True,
        duty_hard_low=float(duty_hard_low),
        duty_hard_high=float(duty_hard_high),
        duty_hard_score=float(duty_hard_score),
    )
    rows: list[dict[str, object]] = []
    for rank, (source_ids, mask) in enumerate(mask_specs, start=1):
        policy = StaticMaskPolicy(
            mask=tuple(bool(x) for x in np.asarray(mask, dtype=bool).tolist()),
            name=f"deployable_static_top{rank}",
        )
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=deployable_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(steps),
            start_indices=tuple(int(x) for x in start_indices),
        )
        losses = np.asarray(result.oracle_losses, dtype=float)
        events = np.asarray(result.event_flags, dtype=float).reshape(-1) > 0.5
        finite = np.isfinite(losses)
        event_losses = losses[finite & events]
        non_event_losses = losses[finite & ~events]
        selected = np.asarray(result.selected_masks, dtype=float)
        duties = np.mean(selected, axis=0) if selected.size else np.zeros(len(sensors), dtype=float)
        selected_names = [
            str(spec.sensor_id)
            for idx, spec in enumerate(sensors)
            if duties[idx] > ALWAYS_OFF_DUTY
        ]
        row: dict[str, object] = {
            "action_idx": int(10000 + rank),
            "sensor_ids": f"deployable_static:{source_ids}",
            "source_static_sensor_ids": str(source_ids),
            "sensor_count": int(len(selected_names)),
            "has_laser": bool("laser_disdrometer" in selected_names),
            "has_fc4": bool("fc4_flux" in selected_names),
            "has_snow_particle_counter": bool("snow_particle_counter" in selected_names),
            "has_radiometer": bool("radiometer_basic" in selected_names),
            "oracle_loss_mean": float(metrics.get("oracle_loss_mean", float("nan"))),
            "oracle_loss_event": float(np.mean(event_losses)) if event_losses.size else float("inf"),
            "oracle_loss_non_event": float(np.mean(non_event_losses)) if non_event_losses.size else float("inf"),
            "power_mean": float(metrics.get("power_mean", float("nan"))),
            "peak_max": float(metrics.get("peak_power_max", float("nan"))),
            "soc_min": float(np.nanmin(result.soc)) if np.asarray(result.soc).size else float("nan"),
            "energy_deficit_total": float("nan"),
            "energy_deficit_steps": int(-1),
            "energy_guard_dropped": int(np.sum(result.energy_guard_dropped)) if np.asarray(result.energy_guard_dropped).size else 0,
            "warmup_abort_count": int(metrics.get("warmup_abort_count", 0)),
            "switches_per_step": float(metrics.get("switches_per_step", float("nan"))),
            "always_on_sensor_count": int(metrics.get("always_on_sensor_count", -1)),
            "always_off_sensor_count": int(metrics.get("always_off_sensor_count", -1)),
            "mid_duty_sensor_count": int(metrics.get("mid_duty_sensor_count", -1)),
            "duty_entropy": float(metrics.get("duty_entropy", float("nan"))),
            "is_deployable_static": True,
        }
        for idx, spec in enumerate(sensors):
            safe_id = str(spec.sensor_id).replace("/", "_")
            row[f"duty__{safe_id}"] = float(duties[idx]) if idx < duties.size else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _legacy_schedule_specs(sensors: list[object], lead_steps: int) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    calm_core = _mask_for_sensor_ids(sensors, ("met_station_core", "radiometer_basic", "surface_temp_ir"))
    snow_core = _mask_for_sensor_ids(
        sensors,
        ("met_station_core", "radiometer_basic", "surface_temp_ir", "snow_particle_counter"),
    )
    return [
        (
            "calm_core__event_laser_surface",
            calm_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "surface_temp_ir", "laser_disdrometer")),
            0,
        ),
        (
            "calm_core__event_laser_fc4",
            calm_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "laser_disdrometer", "fc4_flux")),
            0,
        ),
        (
            "calm_core__event_fc4_surface",
            calm_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "radiometer_basic", "surface_temp_ir", "fc4_flux")),
            0,
        ),
        (
            "calm_core__event_snow_counter_fc4",
            calm_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "surface_temp_ir", "snow_particle_counter", "fc4_flux")),
            0,
        ),
        (
            "snow_core__event_laser_surface",
            snow_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "surface_temp_ir", "laser_disdrometer")),
            0,
        ),
        (
            "snow_core__event_laser_fc4",
            snow_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "laser_disdrometer", "fc4_flux")),
            0,
        ),
        (
            f"calm_core__lead{int(lead_steps)}_laser_surface",
            calm_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "surface_temp_ir", "laser_disdrometer")),
            int(lead_steps),
        ),
        (
            f"calm_core__lead{int(lead_steps)}_laser_fc4",
            calm_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "laser_disdrometer", "fc4_flux")),
            int(lead_steps),
        ),
        (
            f"snow_core__lead{int(lead_steps)}_laser_surface",
            snow_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "surface_temp_ir", "laser_disdrometer")),
            int(lead_steps),
        ),
        (
            f"snow_core__lead{int(lead_steps)}_laser_fc4",
            snow_core,
            _mask_for_sensor_ids(sensors, ("met_station_core", "laser_disdrometer", "fc4_flux")),
            int(lead_steps),
        ),
        (
            f"snow_core__lead{int(lead_steps)}_snow_laser_surface",
            snow_core,
            _mask_for_sensor_ids(
                sensors,
                ("met_station_core", "surface_temp_ir", "snow_particle_counter", "laser_disdrometer"),
            ),
            int(lead_steps),
        ),
        (
            f"snow_core__lead{int(lead_steps)}_snow_laser_fc4",
            snow_core,
            _mask_for_sensor_ids(
                sensors,
                ("met_station_core", "snow_particle_counter", "laser_disdrometer", "fc4_flux"),
            ),
            int(lead_steps),
        ),
    ]


def _v6_static_break_schedule_specs(
    sensors: list[object],
    lead_steps: int,
) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    context_masks = {
        "met_context": ("met_station_core", "radiometer_basic", "ultrasonic_anemometer_hd", "shielded_thermo_hygro"),
        "thermal_context": ("met_station_core", "radiometer_basic", "surface_temp_ir", "shielded_thermo_hygro"),
        "wind_surface_context": ("met_station_core", "radiometer_basic", "surface_temp_ir", "ultrasonic_anemometer_hd"),
    }
    event_masks = {
        "snow_transport": ("met_station_core", "surface_temp_ir", "snow_particle_counter", "fc4_flux"),
        "particle_flux": ("met_station_core", "ultrasonic_anemometer_hd", "snow_particle_counter", "fc4_flux"),
        "thermal_flux": ("met_station_core", "surface_temp_ir", "shielded_thermo_hygro", "fc4_flux"),
        "laser_only": ("laser_disdrometer",),
    }
    specs: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    for lead in (0, int(lead_steps)):
        for calm_name, calm_ids in context_masks.items():
            for event_name, event_ids in event_masks.items():
                if event_name == "laser_only" and calm_name != "met_context":
                    continue
                prefix = f"{calm_name}__event_{event_name}"
                name = prefix if lead == 0 else f"{prefix}__lead{lead}"
                specs.append(
                    (
                        name,
                        _mask_for_sensor_ids(sensors, calm_ids),
                        _mask_for_sensor_ids(sensors, event_ids),
                        int(lead),
                    )
                )
    return specs


def _subtype_static_break_schedule_specs(
    sensors: list[object],
    lead_steps: int,
) -> list[tuple[str, np.ndarray, dict[int, np.ndarray], int]]:
    calm_context = _mask_for_sensor_ids(
        sensors,
        ("met_station_core", "radiometer_basic", "surface_temp_ir", "shielded_thermo_hygro"),
    )
    subtype_masks = {
        1: _mask_for_sensor_ids(sensors, ("surface_temp_ir", "shielded_thermo_hygro", "laser_disdrometer")),
        2: _mask_for_sensor_ids(sensors, ("ultrasonic_anemometer_hd", "shielded_thermo_hygro", "fc4_flux")),
        3: _mask_for_sensor_ids(
            sensors,
            ("met_station_core", "radiometer_basic", "surface_temp_ir", "shielded_thermo_hygro"),
        ),
    }
    subtype_masks_with_counter = {
        1: _mask_for_sensor_ids(sensors, ("surface_temp_ir", "snow_particle_counter", "laser_disdrometer")),
        2: _mask_for_sensor_ids(sensors, ("ultrasonic_anemometer_hd", "snow_particle_counter", "fc4_flux")),
        3: _mask_for_sensor_ids(
            sensors,
            ("met_station_core", "radiometer_basic", "surface_temp_ir", "shielded_thermo_hygro"),
        ),
    }
    return [
        ("subtype_laser_fc4_thermal", calm_context, subtype_masks, 0),
        (f"subtype_laser_fc4_thermal_lead{int(lead_steps)}", calm_context, subtype_masks, int(lead_steps)),
        ("subtype_particle_counter_mix", calm_context, subtype_masks_with_counter, 0),
        (
            f"subtype_particle_counter_mix_lead{int(lead_steps)}",
            calm_context,
            subtype_masks_with_counter,
            int(lead_steps),
        ),
    ]


def _auto_subtype_schedule_specs(
    table: pd.DataFrame,
    sensors: list[object],
    *,
    top_k: int,
    lead_steps: int,
) -> list[tuple[str, np.ndarray, dict[int, np.ndarray], int]]:
    required_cols = (
        "oracle_loss_non_event",
        "oracle_loss_subtype_particle",
        "oracle_loss_subtype_flux",
        "oracle_loss_subtype_thermal",
    )
    if any(col not in table.columns for col in required_cols):
        return []
    # Cap the Cartesian product so this diagnostic stays cheap enough to run
    # inside calibration scans. A pass here is a structural signal, not a final
    # policy.
    top_n = max(1, min(2, int(top_k)))
    calm_masks = _unique_top_masks(table, sensors, score_col="oracle_loss_non_event", top_k=top_n)
    particle_masks = _unique_top_masks(table, sensors, score_col="oracle_loss_subtype_particle", top_k=top_n)
    flux_masks = _unique_top_masks(table, sensors, score_col="oracle_loss_subtype_flux", top_k=top_n)
    thermal_masks = _unique_top_masks(table, sensors, score_col="oracle_loss_subtype_thermal", top_k=top_n)
    if not calm_masks:
        return []
    if not particle_masks:
        particle_masks = calm_masks[:1]
    if not flux_masks:
        flux_masks = calm_masks[:1]
    if not thermal_masks:
        thermal_masks = calm_masks[:1]

    schedules: list[tuple[str, np.ndarray, dict[int, np.ndarray], int]] = []
    seen: set[tuple[int, ...]] = set()
    for lead in (0, int(lead_steps)):
        for calm_idx, calm_mask in enumerate(calm_masks):
            for particle_idx, particle_mask in enumerate(particle_masks):
                for flux_idx, flux_mask in enumerate(flux_masks):
                    for thermal_idx, thermal_mask in enumerate(thermal_masks):
                        key = tuple(
                            int(x)
                            for mask in (calm_mask, particle_mask, flux_mask, thermal_mask)
                            for x in np.asarray(mask, dtype=bool).astype(int).tolist()
                        ) + (int(lead),)
                        if key in seen:
                            continue
                        seen.add(key)
                        schedules.append(
                            (
                                (
                                    "subtype_auto_"
                                    f"c{calm_idx}_p{particle_idx}_f{flux_idx}_t{thermal_idx}_lead{int(lead)}"
                                ),
                                np.asarray(calm_mask, dtype=bool),
                                {
                                    1: np.asarray(particle_mask, dtype=bool),
                                    2: np.asarray(flux_mask, dtype=bool),
                                    3: np.asarray(thermal_mask, dtype=bool),
                                },
                                int(lead),
                            )
                        )
    return schedules


def _subtype_schedule_rows(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: PowerConstraintsV2,
    oracle: object,
    cfg: WarmupEnvConfig,
    start_indices: tuple[int, ...],
    steps: int,
    schedules: list[tuple[str, np.ndarray, dict[int, np.ndarray], int]],
) -> pd.DataFrame:
    if "event_subtype_id" not in truth.columns:
        return pd.DataFrame()
    subtype_ids = truth["event_subtype_id"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    for schedule_idx, (name, calm_mask, subtype_masks, lead_steps) in enumerate(schedules):
        losses: list[float] = []
        event_losses: list[float] = []
        non_event_losses: list[float] = []
        powers: list[float] = []
        peaks: list[float] = []
        soc_values: list[float] = []
        deficits: list[float] = []
        guard_drops = 0
        switches: list[float] = []
        aborts = 0
        selected_counts = np.zeros(len(sensors), dtype=float)
        subtype_counts = {1: 0, 2: 0, 3: 0}
        subtype_losses: dict[int, list[float]] = {1: [], 2: [], 3: []}
        total_steps = 0
        for offset, start_idx in enumerate(start_indices):
            env = WarmupSchedulingEnv(
                truth,
                sensors,
                constraints,
                WarmupEnvConfig(
                    state_columns=cfg.state_columns,
                    reward_target_columns=cfg.reward_target_columns,
                    lookback=cfg.lookback,
                    episode_len=int(steps),
                    seed=int(cfg.seed) + int(offset) + 91_000,
                    base_freq_s=cfg.base_freq_s,
                    event_column=cfg.event_column,
                    normalize_agent_state=cfg.normalize_agent_state,
                    lambda_warmup_abort=cfg.lambda_warmup_abort,
                    lambda_switch=cfg.lambda_switch,
                    energy_account_enabled=cfg.energy_account_enabled,
                    energy_capacity=cfg.energy_capacity,
                    initial_energy=cfg.initial_energy,
                    harvest_per_step=cfg.harvest_per_step,
                    reserve_energy=cfg.reserve_energy,
                    lambda_energy_deficit=cfg.lambda_energy_deficit,
                ),
                oracle=oracle,
            )
            env.reset(start_idx=int(start_idx))
            for _ in range(int(steps)):
                current_idx = int(env.current_idx)
                lookahead_end = min(len(subtype_ids), current_idx + max(0, int(lead_steps)) + 1)
                window = subtype_ids[current_idx:lookahead_end]
                active_subtypes = window[window > 0]
                subtype_id = int(active_subtypes[0]) if active_subtypes.size else 0
                mask = subtype_masks.get(subtype_id, calm_mask)
                if subtype_id in subtype_counts:
                    subtype_counts[subtype_id] += 1
                _, _, done, info = env.step_mask(mask)
                loss = float(info.get("oracle_loss", float("nan")))
                if np.isfinite(loss):
                    losses.append(loss)
                    if subtype_id in subtype_losses:
                        subtype_losses[subtype_id].append(loss)
                    if bool(info.get("event", False)):
                        event_losses.append(loss)
                    else:
                        non_event_losses.append(loss)
                powers.append(float(info.get("power", 0.0)))
                peaks.append(float(info.get("peak_power", 0.0)))
                soc_values.append(float(info.get("soc", float("nan"))))
                deficits.append(float(info.get("energy_deficit", 0.0)))
                guard_drops += int(info.get("energy_guard_dropped", 0))
                switches.append(float(info.get("switch_rate", 0.0)))
                aborts += int(info.get("warmup_abort_delta", 0))
                selected_counts += np.asarray(info.get("selected_mask", [0] * len(sensors)), dtype=float)
                total_steps += 1
                if done:
                    break
        duties = selected_counts / max(1, int(total_steps))
        selected_names = [
            str(spec.sensor_id)
            for idx, spec in enumerate(sensors)
            if duties[idx] > ALWAYS_OFF_DUTY
        ]
        duty_entropy = float(
            -np.mean(
                duties * np.log(np.clip(duties, 1e-9, 1.0))
                + (1.0 - duties) * np.log(np.clip(1.0 - duties, 1e-9, 1.0))
            )
            / np.log(2.0)
        )
        row: dict[str, object] = {
            "action_idx": int(-3000 - schedule_idx),
            "sensor_ids": f"dynamic:{name}",
            "sensor_count": int(len(selected_names)),
            "has_laser": bool("laser_disdrometer" in selected_names),
            "has_fc4": bool("fc4_flux" in selected_names),
            "has_snow_particle_counter": bool("snow_particle_counter" in selected_names),
            "has_radiometer": bool("radiometer_basic" in selected_names),
            "oracle_loss_mean": float(np.mean(losses)) if losses else float("inf"),
            "oracle_loss_event": float(np.mean(event_losses)) if event_losses else float("inf"),
            "oracle_loss_non_event": float(np.mean(non_event_losses)) if non_event_losses else float("inf"),
            "oracle_loss_subtype_particle": float(np.mean(subtype_losses[1])) if subtype_losses[1] else float("inf"),
            "oracle_loss_subtype_flux": float(np.mean(subtype_losses[2])) if subtype_losses[2] else float("inf"),
            "oracle_loss_subtype_thermal": float(np.mean(subtype_losses[3])) if subtype_losses[3] else float("inf"),
            "power_mean": float(np.mean(powers)) if powers else 0.0,
            "peak_max": float(np.max(peaks)) if peaks else 0.0,
            "soc_min": float(np.nanmin(soc_values)) if soc_values else float("nan"),
            "energy_deficit_total": float(np.sum(deficits)) if deficits else 0.0,
            "energy_deficit_steps": int(np.sum(np.asarray(deficits, dtype=float) > 1e-12)) if deficits else 0,
            "energy_guard_dropped": int(guard_drops),
            "warmup_abort_count": int(aborts),
            "switches_per_step": float(np.mean(switches)) if switches else 0.0,
            "always_on_sensor_count": int(np.sum(duties >= ALWAYS_ON_DUTY)),
            "always_off_sensor_count": int(np.sum(duties <= ALWAYS_OFF_DUTY)),
            "mid_duty_sensor_count": int(np.sum((duties >= MID_DUTY_LOW) & (duties <= MID_DUTY_HIGH))),
            "duty_entropy": duty_entropy,
            "subtype_particle_steps": int(subtype_counts[1]),
            "subtype_flux_steps": int(subtype_counts[2]),
            "subtype_thermal_steps": int(subtype_counts[3]),
        }
        for idx, spec in enumerate(sensors):
            safe_id = str(spec.sensor_id).replace("/", "_")
            row[f"duty__{safe_id}"] = float(duties[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _receding_oracle_rows(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: PowerConstraintsV2,
    candidate_masks: np.ndarray,
    oracle: object,
    cfg: WarmupEnvConfig,
    start_indices: tuple[int, ...],
    steps: int,
    lookahead_steps: int,
    trace_path: Path | None = None,
) -> pd.DataFrame:
    """Evaluate a privileged all-action receding-horizon structural diagnostic."""
    losses: list[float] = []
    event_losses: list[float] = []
    non_event_losses: list[float] = []
    subtype_losses: dict[int, list[float]] = {1: [], 2: [], 3: []}
    powers: list[float] = []
    peaks: list[float] = []
    switches: list[float] = []
    selected_counts = np.zeros(len(sensors), dtype=float)
    action_counts = np.zeros(len(candidate_masks), dtype=int)
    aborts = 0
    total_steps = 0
    trace_rows: list[dict[str, object]] = []
    subtype_ids = (
        truth["event_subtype_id"].to_numpy(dtype=int)
        if "event_subtype_id" in truth.columns
        else np.zeros(len(truth), dtype=int)
    )
    for offset, start_idx in enumerate(start_indices):
        env = WarmupSchedulingEnv(
            truth,
            sensors,
            constraints,
            replace(cfg, seed=int(cfg.seed) + int(offset) + 111_000),
            oracle=oracle,
        )
        env.reset(start_idx=int(start_idx))
        for rollout_step in range(int(steps)):
            step_idx = int(env.current_idx)
            dwell_hold_before_action = int(env.dwell_hold_remaining)
            online_state = np.asarray(env._state(), dtype=float).reshape(-1)
            alert_features = np.asarray(env._alert_context_features(), dtype=float).reshape(-1)
            action_costs = oracle_greedy_candidate_costs(
                env,
                candidate_masks,
                lookahead_steps=max(1, int(lookahead_steps)),
            )
            action_idx = int(np.argmin(action_costs))
            action_counts[action_idx] += 1
            _, _, done, info = env.step_mask(candidate_masks[action_idx])
            loss = float(info.get("oracle_loss", float("nan")))
            if np.isfinite(loss):
                losses.append(loss)
                if bool(info.get("event", False)):
                    event_losses.append(loss)
                else:
                    non_event_losses.append(loss)
                subtype_id = int(subtype_ids[min(int(env.current_idx), len(subtype_ids) - 1)])
                if subtype_id in subtype_losses:
                    subtype_losses[subtype_id].append(loss)
            powers.append(float(info.get("power", 0.0)))
            peaks.append(float(info.get("peak_power", 0.0)))
            switches.append(float(info.get("switch_rate", 0.0)))
            aborts += int(info.get("warmup_abort_delta", 0))
            selected_counts += np.asarray(info.get("selected_mask", [0] * len(sensors)), dtype=float)
            trace_row: dict[str, object] = {
                "rollout_idx": int(offset),
                "rollout_step": int(rollout_step),
                "truth_step_idx": int(step_idx),
                "selected_action_idx": int(action_idx),
                "selected_action_cost": float(action_costs[action_idx]),
                "second_best_action_cost": float(np.partition(action_costs, 1)[1]),
                "action_cost_gap": float(np.partition(action_costs, 1)[1] - action_costs[action_idx]),
                "event_subtype_id": int(subtype_ids[min(step_idx, len(subtype_ids) - 1)]),
                "executed_oracle_loss": float(info.get("oracle_loss", float("nan"))),
                "dwell_hold_remaining_before_action": dwell_hold_before_action,
            }
            for feature_idx, value in enumerate(online_state):
                trace_row[f"online_state_{feature_idx}"] = float(value)
            for feature_idx, value in enumerate(alert_features):
                trace_row[f"alert_feature_{feature_idx}"] = float(value)
            for candidate_idx, value in enumerate(action_costs):
                trace_row[f"candidate_cost_{candidate_idx}"] = float(value)
            for sensor_idx, value in enumerate(np.asarray(info.get("selected_mask", []), dtype=int)):
                trace_row[f"executed_sensor_{sensor_idx}"] = int(value)
            trace_rows.append(trace_row)
            total_steps += 1
            if done:
                break
    duties = selected_counts / max(1, int(total_steps))
    selected_names = [
        str(spec.sensor_id)
        for idx, spec in enumerate(sensors)
        if duties[idx] > ALWAYS_OFF_DUTY
    ]
    duty_entropy = float(
        -np.mean(
            duties * np.log(np.clip(duties, 1e-9, 1.0))
            + (1.0 - duties) * np.log(np.clip(1.0 - duties, 1e-9, 1.0))
        )
        / np.log(2.0)
    )
    row: dict[str, object] = {
        "action_idx": -4000,
        "sensor_ids": f"dynamic:receding_oracle_l{int(lookahead_steps)}",
        "sensor_count": len(selected_names),
        "has_laser": "laser_disdrometer" in selected_names,
        "has_fc4": "fc4_flux" in selected_names,
        "has_snow_particle_counter": "snow_particle_counter" in selected_names,
        "has_radiometer": "radiometer_basic" in selected_names,
        "oracle_loss_mean": float(np.mean(losses)) if losses else float("inf"),
        "oracle_loss_event": float(np.mean(event_losses)) if event_losses else float("inf"),
        "oracle_loss_non_event": float(np.mean(non_event_losses)) if non_event_losses else float("inf"),
        "oracle_loss_subtype_particle": float(np.mean(subtype_losses[1])) if subtype_losses[1] else float("inf"),
        "oracle_loss_subtype_flux": float(np.mean(subtype_losses[2])) if subtype_losses[2] else float("inf"),
        "oracle_loss_subtype_thermal": float(np.mean(subtype_losses[3])) if subtype_losses[3] else float("inf"),
        "steps_subtype_particle": len(subtype_losses[1]),
        "steps_subtype_flux": len(subtype_losses[2]),
        "steps_subtype_thermal": len(subtype_losses[3]),
        "power_mean": float(np.mean(powers)) if powers else 0.0,
        "peak_max": float(np.max(peaks)) if peaks else 0.0,
        "warmup_abort_count": int(aborts),
        "switches_per_step": float(np.mean(switches)) if switches else 0.0,
        "always_on_sensor_count": int(np.sum(duties >= ALWAYS_ON_DUTY)),
        "always_off_sensor_count": int(np.sum(duties <= ALWAYS_OFF_DUTY)),
        "mid_duty_sensor_count": int(np.sum((duties >= MID_DUTY_LOW) & (duties <= MID_DUTY_HIGH))),
        "duty_entropy": duty_entropy,
        "receding_action_coverage": int(np.sum(action_counts > 0)),
    }
    for idx, spec in enumerate(sensors):
        row[f"duty__{str(spec.sensor_id).replace('/', '_')}"] = float(duties[idx])
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
    return pd.DataFrame([row])


def _schedule_rows(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: PowerConstraintsV2,
    oracle: object,
    cfg: WarmupEnvConfig,
    start_indices: tuple[int, ...],
    steps: int,
    schedules: list[tuple[str, np.ndarray, np.ndarray, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for schedule_idx, (name, calm_mask, event_mask, lead_steps) in enumerate(schedules):
        losses: list[float] = []
        event_losses: list[float] = []
        non_event_losses: list[float] = []
        powers: list[float] = []
        peaks: list[float] = []
        soc_values: list[float] = []
        deficits: list[float] = []
        guard_drops = 0
        switches: list[float] = []
        aborts = 0
        selected_counts = np.zeros(len(sensors), dtype=float)
        total_steps = 0
        for offset, start_idx in enumerate(start_indices):
            env = WarmupSchedulingEnv(
                truth,
                sensors,
                constraints,
                WarmupEnvConfig(
                    state_columns=cfg.state_columns,
                    reward_target_columns=cfg.reward_target_columns,
                    lookback=cfg.lookback,
                    episode_len=int(steps),
                    seed=int(cfg.seed) + int(offset) + 71_000,
                    base_freq_s=cfg.base_freq_s,
                    event_column=cfg.event_column,
                    normalize_agent_state=cfg.normalize_agent_state,
                    lambda_warmup_abort=cfg.lambda_warmup_abort,
                    lambda_switch=cfg.lambda_switch,
                    energy_account_enabled=cfg.energy_account_enabled,
                    energy_capacity=cfg.energy_capacity,
                    initial_energy=cfg.initial_energy,
                    harvest_per_step=cfg.harvest_per_step,
                    reserve_energy=cfg.reserve_energy,
                    lambda_energy_deficit=cfg.lambda_energy_deficit,
                ),
                oracle=oracle,
            )
            env.reset(start_idx=int(start_idx))
            for _ in range(int(steps)):
                lookahead_end = min(len(env.event_flags), int(env.current_idx) + max(0, int(lead_steps)) + 1)
                trigger = bool(np.any(env.event_flags[int(env.current_idx) : lookahead_end]))
                mask = event_mask if trigger else calm_mask
                _, _, done, info = env.step_mask(mask)
                loss = float(info.get("oracle_loss", float("nan")))
                if np.isfinite(loss):
                    losses.append(loss)
                    if bool(info.get("event", False)):
                        event_losses.append(loss)
                    else:
                        non_event_losses.append(loss)
                powers.append(float(info.get("power", 0.0)))
                peaks.append(float(info.get("peak_power", 0.0)))
                soc_values.append(float(info.get("soc", float("nan"))))
                deficits.append(float(info.get("energy_deficit", 0.0)))
                guard_drops += int(info.get("energy_guard_dropped", 0))
                switches.append(float(info.get("switch_rate", 0.0)))
                aborts += int(info.get("warmup_abort_delta", 0))
                selected_counts += np.asarray(info.get("selected_mask", [0] * len(sensors)), dtype=float)
                total_steps += 1
                if done:
                    break
        duties = selected_counts / max(1, int(total_steps))
        duty_entropy = float(
            -np.mean(
                duties * np.log(np.clip(duties, 1e-9, 1.0))
                + (1.0 - duties) * np.log(np.clip(1.0 - duties, 1e-9, 1.0))
            )
            / np.log(2.0)
        )
        selected_names = [
            str(spec.sensor_id)
            for idx, spec in enumerate(sensors)
            if duties[idx] > ALWAYS_OFF_DUTY
        ]
        row: dict[str, object] = {
            "action_idx": int(-1000 - schedule_idx),
            "sensor_ids": f"dynamic:{name}",
            "sensor_count": int(len(selected_names)),
            "has_laser": bool("laser_disdrometer" in selected_names),
            "has_fc4": bool("fc4_flux" in selected_names),
            "has_snow_particle_counter": bool("snow_particle_counter" in selected_names),
            "has_radiometer": bool("radiometer_basic" in selected_names),
            "oracle_loss_mean": float(np.mean(losses)) if losses else float("inf"),
            "oracle_loss_event": float(np.mean(event_losses)) if event_losses else float("inf"),
            "oracle_loss_non_event": float(np.mean(non_event_losses)) if non_event_losses else float("inf"),
            "power_mean": float(np.mean(powers)) if powers else 0.0,
            "peak_max": float(np.max(peaks)) if peaks else 0.0,
            "soc_min": float(np.nanmin(soc_values)) if soc_values else float("nan"),
            "energy_deficit_total": float(np.sum(deficits)) if deficits else 0.0,
            "energy_deficit_steps": int(np.sum(np.asarray(deficits, dtype=float) > 1e-12)) if deficits else 0,
            "energy_guard_dropped": int(guard_drops),
            "warmup_abort_count": int(aborts),
            "switches_per_step": float(np.mean(switches)) if switches else 0.0,
            "always_on_sensor_count": int(np.sum(duties >= ALWAYS_ON_DUTY)),
            "always_off_sensor_count": int(np.sum(duties <= ALWAYS_OFF_DUTY)),
            "mid_duty_sensor_count": int(np.sum((duties >= MID_DUTY_LOW) & (duties <= MID_DUTY_HIGH))),
            "duty_entropy": duty_entropy,
        }
        for idx, spec in enumerate(sensors):
            safe_id = str(spec.sensor_id).replace("/", "_")
            row[f"duty__{safe_id}"] = float(duties[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _best_row(table: pd.DataFrame, mask: pd.Series, score_col: str) -> dict[str, object] | None:
    sub = table[mask & np.isfinite(table[score_col].to_numpy(dtype=float))].copy()
    if sub.empty:
        return None
    row = sub.sort_values(score_col).iloc[0]
    return {
        "action_idx": int(row["action_idx"]),
        "sensor_ids": str(row["sensor_ids"]),
        score_col: float(row[score_col]),
        "power_mean": float(row["power_mean"]),
        "peak_max": float(row["peak_max"]),
    }


def build_summary(table: pd.DataFrame, *, budget: float, truth: pd.DataFrame, start_indices: tuple[int, ...]) -> dict[str, object]:
    finite_event = table["oracle_loss_event"].replace([np.inf, -np.inf], np.nan)
    finite_non_event = table["oracle_loss_non_event"].replace([np.inf, -np.inf], np.nan)
    best_event_laser = _best_row(table, table["has_laser"].astype(bool), "oracle_loss_event")
    best_event_no_laser = _best_row(table, ~table["has_laser"].astype(bool), "oracle_loss_event")
    best_event_fc4 = _best_row(table, table["has_fc4"].astype(bool), "oracle_loss_event")
    best_event_no_fc4 = _best_row(table, ~table["has_fc4"].astype(bool), "oracle_loss_event")
    best_non_event_laser = _best_row(table, table["has_laser"].astype(bool), "oracle_loss_non_event")
    best_non_event_no_laser = _best_row(table, ~table["has_laser"].astype(bool), "oracle_loss_non_event")

    def lift(with_sensor: dict[str, object] | None, without_sensor: dict[str, object] | None, col: str) -> float:
        if with_sensor is None or without_sensor is None:
            return float("nan")
        return float(without_sensor[col]) - float(with_sensor[col])

    return {
        "budget": float(budget),
        "candidate_count": int(len(table)),
        "truth_event_rate": float(truth["event_flag"].mean()) if "event_flag" in truth.columns else float("nan"),
        "truth_event_subtype_particle_rate": float(truth["event_subtype_particle"].mean())
        if "event_subtype_particle" in truth.columns
        else float("nan"),
        "truth_event_subtype_flux_rate": float(truth["event_subtype_flux"].mean())
        if "event_subtype_flux" in truth.columns
        else float("nan"),
        "truth_event_subtype_thermal_rate": float(truth["event_subtype_thermal"].mean())
        if "event_subtype_thermal" in truth.columns
        else float("nan"),
        "eval_start_indices": [int(x) for x in start_indices],
        "event_loss_min": float(finite_event.min()),
        "non_event_loss_min": float(finite_non_event.min()),
        "best_overall": _best_row(table, pd.Series(True, index=table.index), "oracle_loss_mean"),
        "best_event_laser": best_event_laser,
        "best_event_no_laser": best_event_no_laser,
        "best_event_fc4": best_event_fc4,
        "best_event_no_fc4": best_event_no_fc4,
        "best_non_event_laser": best_non_event_laser,
        "best_non_event_no_laser": best_non_event_no_laser,
        "laser_event_lift": lift(best_event_laser, best_event_no_laser, "oracle_loss_event"),
        "fc4_event_lift": lift(best_event_fc4, best_event_no_fc4, "oracle_loss_event"),
        "laser_non_event_lift": lift(best_non_event_laser, best_non_event_no_laser, "oracle_loss_non_event"),
    }


def select_rich_start_indices(
    truth: pd.DataFrame,
    *,
    steps: int,
    horizon: int,
    n_rollouts: int,
    selection: str,
    stride: int,
    seed: int,
    event_column: str = "event_flag",
) -> tuple[int, ...]:
    if str(selection) == "event_fraction":
        raise ValueError("event_fraction selection is handled by the caller")
    max_start = int(len(truth) - int(steps) - int(horizon) - 1)
    if max_start <= 0 or int(n_rollouts) <= 1:
        return (0,)
    event_flags = (
        truth[event_column].astype(bool).to_numpy()
        if event_column in truth.columns
        else np.zeros(len(truth), dtype=bool)
    )
    flux = (
        np.asarray(truth["snow_mass_flux_kg_m2_s"], dtype=float)
        if "snow_mass_flux_kg_m2_s" in truth.columns
        else np.zeros(len(truth), dtype=float)
    )
    flux = np.maximum(0.0, np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0))
    starts = np.arange(0, max_start + 1, max(1, int(stride)), dtype=int)
    rows: list[tuple[float, int]] = []
    rng = np.random.default_rng(int(seed))
    flux_scale = max(float(np.percentile(flux, 95)) if flux.size else 0.0, 1e-12)
    for start in starts:
        end = int(start) + int(steps)
        event_rate = float(np.mean(event_flags[int(start) : end])) if end > start else 0.0
        if str(selection) == "event_rich":
            score = event_rate
        elif str(selection) == "event_transport_rich":
            window_flux = float(np.mean(flux[int(start) : end])) if end > start else 0.0
            flux_score = float(np.clip(window_flux / flux_scale, 0.0, 4.0))
            score = event_rate * (1.0 + flux_score)
        else:
            raise ValueError(f"Unknown eval start selection: {selection}")
        rows.append((float(score + 1e-9 * rng.random()), int(start)))
    selected: list[int] = []
    for _, start in sorted(rows, reverse=True):
        if all(abs(int(start) - int(prev)) >= int(steps) for prev in selected):
            selected.append(int(start))
            if len(selected) >= int(n_rollouts):
                break
    if len(selected) < int(n_rollouts):
        for _, start in sorted(rows, reverse=True):
            if int(start) not in selected:
                selected.append(int(start))
                if len(selected) >= int(n_rollouts):
                    break
    return tuple(sorted(int(x) for x in selected))


def main() -> None:
    helpers = load_train_helpers()
    parser = argparse.ArgumentParser(description="Oracle-lift diagnostics for the V3.1 physical-event scenario.")
    parser.add_argument("--truth-csv", default="reports/physical_event_v2_oracle_lift/truth_v31.csv")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--truth-steps", type=int, default=30000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--blowing-snow-event-coverage", type=float, default=0.28)
    parser.add_argument("--blowing-snow-event-model", default="semi_markov")
    parser.add_argument("--blowing-snow-min-duration-steps", type=int, default=12)
    parser.add_argument("--blowing-snow-max-duration-steps", type=int, default=24)
    parser.add_argument("--blowing-snow-min-gap-steps", type=int, default=4)
    parser.add_argument("--blowing-snow-lead-steps", type=int, default=6)
    parser.add_argument("--blowing-snow-wind-margin-ms", type=float, default=1.2)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.6)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.3)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.0)
    parser.add_argument("--event-microstructure-sigma", type=float, default=0.0)
    parser.add_argument("--event-microstructure-alpha", type=float, default=0.18)
    parser.add_argument("--event-microstructure-diameter-scale", type=float, default=0.0)
    parser.add_argument("--event-microstructure-velocity-scale", type=float, default=0.0)
    parser.add_argument("--event-particle-microstructure-correlation", type=float, default=1.0)
    parser.add_argument("--event-subtypes-enabled", action="store_true")
    parser.add_argument("--event-subtype-particle-prob", type=float, default=0.34)
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
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v2.yaml")
    parser.add_argument("--out-dir", default="reports/physical_event_v2_oracle_lift")
    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--required-sensors", nargs="*", default=["met_station_core"])
    parser.add_argument("--coverage-groups", action="store_true")
    parser.add_argument("--target-weights", nargs="*", type=float, default=[0.8, 0.8, 1.2, 0.4, 0.4, 0.55, 4.0, 2.5, 2.5])
    parser.add_argument("--target-scales", nargs="*", type=float, default=list(helpers.DEFAULT_TARGET_SCALES))
    parser.add_argument("--state-columns", nargs="*", default=list(helpers.STATE_COLUMNS))
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--oracle-type", choices=["linear", "tcn"], default="linear")
    parser.add_argument("--oracle-path", default="")
    parser.add_argument("--oracle-rollout-steps", type=int, default=1200)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=3)
    parser.add_argument("--oracle-event-fraction", type=float, default=0.50)
    parser.add_argument("--oracle-full-open-repeat", type=int, default=2)
    parser.add_argument("--oracle-epochs", type=int, default=8)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-learning-rate", type=float, default=1e-3)
    parser.add_argument("--oracle-channels", type=int, default=64)
    parser.add_argument("--oracle-levels", type=int, default=3)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--oracle-loss-clip", type=float, default=10.0)
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--env-min-dwell-steps", type=int, default=1)
    parser.add_argument(
        "--eval-start-selection",
        choices=["event_fraction", "event_rich", "event_transport_rich"],
        default="event_fraction",
    )
    parser.add_argument("--eval-selection-stride", type=int, default=64)
    parser.add_argument("--eval-start-indices", nargs="*", type=int, default=None)
    parser.add_argument("--target-diagnostics", action="store_true")
    parser.add_argument("--schedule-diagnostics", action="store_true")
    parser.add_argument(
        "--schedule-family",
        choices=[
            "legacy",
            "v6_static_break",
            "auto_pairs",
            "diverse_auto",
            "subtype_static_break",
            "subtype_auto",
            "receding_oracle",
            "all",
        ],
        default="legacy",
    )
    parser.add_argument("--schedule-lead-steps", type=int, default=4)
    parser.add_argument("--auto-schedule-top-k", type=int, default=4)
    parser.add_argument("--diverse-schedule-dwell-steps", type=int, default=16)
    parser.add_argument("--receding-oracle-lookahead-steps", type=int, default=8)
    parser.add_argument("--deployable-static-diagnostics", action="store_true")
    parser.add_argument("--deployable-static-top-k", type=int, default=6)
    parser.add_argument("--deployable-static-duty-low", type=float, default=0.12)
    parser.add_argument("--deployable-static-duty-high", type=float, default=0.75)
    parser.add_argument("--deployable-static-duty-score", type=float, default=12.0)
    parser.add_argument("--deployable-static-duty-feedback", type=float, default=2.5)
    parser.add_argument("--energy-account", action="store_true")
    parser.add_argument("--energy-capacity", type=float, default=24.0)
    parser.add_argument("--initial-energy", type=float, default=12.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.65)
    parser.add_argument("--reserve-energy", type=float, default=2.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--force-truth", action="store_true")
    args = parser.parse_args()
    helpers.STATE_COLUMNS = tuple(str(name) for name in args.state_columns)

    if len(args.target_weights) != len(helpers.REWARD_TARGET_COLUMNS):
        raise ValueError(f"--target-weights must contain {len(helpers.REWARD_TARGET_COLUMNS)} values")
    if len(args.target_scales) != len(helpers.REWARD_TARGET_COLUMNS):
        raise ValueError(f"--target-scales must contain {len(helpers.REWARD_TARGET_COLUMNS)} values")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = Path(args.truth_csv)
    if bool(args.force_truth) and truth_path.exists():
        truth_path.unlink()
    truth_path = helpers.ensure_truth(args)
    truth = pd.read_csv(truth_path)
    sensors = load_sensor_specs(resolve_sensor_cfg(str(args.sensor_cfg)))
    constraints = PowerConstraintsV2(
        max_active=int(args.max_active),
        per_step_budget=float(args.budget),
        startup_peak_budget=float(args.startup_peak_budget),
        required_sensor_ids=tuple(str(x) for x in args.required_sensors),
        coverage_groups=helpers.DEFAULT_COVERAGE_GROUPS if bool(args.coverage_groups) else (),
    )
    target_weights = tuple(float(x) for x in args.target_weights)
    target_scales = tuple(float(x) for x in args.target_scales)
    if str(args.oracle_path):
        oracle_path = Path(args.oracle_path)
        if str(args.oracle_type) == "tcn":
            oracle = TCNFrozenForecastOracle.load(oracle_path, device=str(args.oracle_inference_device))
        else:
            oracle = LinearFrozenForecastOracle.load(str(oracle_path))
    else:
        oracle = helpers.train_oracle(
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
            tcn_use_mask_channels=True,
            target_weights=target_weights,
            target_scales=target_scales,
            rollouts_per_policy=int(args.oracle_rollouts_per_policy),
            event_fraction=float(args.oracle_event_fraction),
            full_open_repeat=int(args.oracle_full_open_repeat),
            base_freq_s=int(args.freq_s),
            seed=int(args.seed),
        )
    if str(args.oracle_type) == "tcn":
        oracle.to_device(str(args.oracle_inference_device))
    candidate_masks = helpers.build_projected_candidate_masks(sensors, constraints, max_candidate_warmup=None)
    if args.eval_start_indices:
        eval_start_indices = tuple(int(x) for x in args.eval_start_indices)
    else:
        if str(args.eval_start_selection) == "event_fraction":
            eval_start_indices = helpers.select_eval_start_indices(
                truth,
                steps=int(args.eval_steps),
                horizon=int(args.horizon),
                n_rollouts=int(args.eval_rollouts),
                event_fraction=float(args.eval_event_fraction),
                seed=int(args.seed) + 991,
            )
        else:
            eval_start_indices = select_rich_start_indices(
                truth,
                steps=int(args.eval_steps),
                horizon=int(args.horizon),
                n_rollouts=int(args.eval_rollouts),
                selection=str(args.eval_start_selection),
                stride=int(args.eval_selection_stride),
                seed=int(args.seed) + 991,
            )
    eval_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=int(args.lookback),
        episode_len=int(args.eval_steps),
        seed=int(args.seed) + 9000,
        base_freq_s=int(args.freq_s),
        energy_account_enabled=bool(args.energy_account),
        energy_capacity=float(args.energy_capacity),
        initial_energy=float(args.initial_energy),
        harvest_per_step=float(args.harvest_per_step),
        reserve_energy=float(args.reserve_energy),
        lambda_energy_deficit=float(args.lambda_energy_deficit),
        min_dwell_steps=int(max(1, int(args.env_min_dwell_steps))),
    )
    table = evaluate_candidate_masks(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        candidate_masks=candidate_masks,
        oracle=oracle,
        cfg=eval_cfg,
        start_indices=tuple(int(x) for x in eval_start_indices),
        steps=int(args.eval_steps),
        target_diagnostics=bool(args.target_diagnostics),
    )
    if bool(args.schedule_diagnostics):
        schedules: list[tuple[str, np.ndarray, np.ndarray, int]] = []
        if str(args.schedule_family) in {"legacy", "all"}:
            schedules.extend(_legacy_schedule_specs(sensors, int(args.schedule_lead_steps)))
        if str(args.schedule_family) in {"v6_static_break", "all"}:
            schedules.extend(_v6_static_break_schedule_specs(sensors, int(args.schedule_lead_steps)))
        if str(args.schedule_family) in {"auto_pairs", "all"}:
            schedules.extend(
                _auto_pair_schedule_specs(
                    table,
                    sensors,
                    top_k=int(args.auto_schedule_top_k),
                    lead_steps=int(args.schedule_lead_steps),
                )
            )
        tables: list[pd.DataFrame] = []
        if schedules:
            tables.append(
                _schedule_rows(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    oracle=oracle,
                    cfg=eval_cfg,
                    start_indices=tuple(int(x) for x in eval_start_indices),
                    steps=int(args.eval_steps),
                    schedules=schedules,
                )
            )
        if str(args.schedule_family) in {"diverse_auto", "all"}:
            diverse_schedules = _auto_diverse_schedule_specs(
                table,
                sensors,
                top_k=int(args.auto_schedule_top_k),
                lead_steps=int(args.schedule_lead_steps),
                dwell_steps=int(args.diverse_schedule_dwell_steps),
            )
            if diverse_schedules:
                tables.append(
                    _cyclic_schedule_rows(
                        truth=truth,
                        sensors=sensors,
                        constraints=constraints,
                        oracle=oracle,
                        cfg=eval_cfg,
                        start_indices=tuple(int(x) for x in eval_start_indices),
                        steps=int(args.eval_steps),
                        schedules=diverse_schedules,
                    )
                )
        if str(args.schedule_family) in {"subtype_static_break", "all"}:
            subtype_schedules = _subtype_static_break_schedule_specs(
                sensors,
                int(args.schedule_lead_steps),
            )
            if subtype_schedules:
                subtype_rows = _subtype_schedule_rows(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    oracle=oracle,
                    cfg=eval_cfg,
                    start_indices=tuple(int(x) for x in eval_start_indices),
                    steps=int(args.eval_steps),
                    schedules=subtype_schedules,
                )
                if not subtype_rows.empty:
                    tables.append(subtype_rows)
        if str(args.schedule_family) == "subtype_auto":
            subtype_auto_schedules = _auto_subtype_schedule_specs(
                table,
                sensors,
                top_k=int(args.auto_schedule_top_k),
                lead_steps=int(args.schedule_lead_steps),
            )
            if subtype_auto_schedules:
                subtype_auto_rows = _subtype_schedule_rows(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    oracle=oracle,
                    cfg=eval_cfg,
                    start_indices=tuple(int(x) for x in eval_start_indices),
                    steps=int(args.eval_steps),
                    schedules=subtype_auto_schedules,
                )
                if not subtype_auto_rows.empty:
                    tables.append(subtype_auto_rows)
        if str(args.schedule_family) in {"receding_oracle", "all"}:
            tables.append(
                _receding_oracle_rows(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    candidate_masks=candidate_masks,
                    oracle=oracle,
                    cfg=eval_cfg,
                    start_indices=tuple(int(x) for x in eval_start_indices),
                    steps=int(args.eval_steps),
                    lookahead_steps=int(args.receding_oracle_lookahead_steps),
                    trace_path=out_dir / "receding_oracle_trace.csv",
                )
            )
        if tables:
            table = pd.concat([table, *tables], ignore_index=True).sort_values("oracle_loss_mean").reset_index(drop=True)
    if bool(args.deployable_static_diagnostics):
        deployable_static = _deployable_static_rows(
            helpers=helpers,
            table=table[table["action_idx"].astype(int) >= 0].copy(),
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            oracle=oracle,
            cfg=eval_cfg,
            start_indices=tuple(int(x) for x in eval_start_indices),
            steps=int(args.eval_steps),
            top_k=int(args.deployable_static_top_k),
            duty_hard_low=float(args.deployable_static_duty_low),
            duty_hard_high=float(args.deployable_static_duty_high),
            duty_hard_score=float(args.deployable_static_duty_score),
            duty_score_feedback=float(args.deployable_static_duty_feedback),
        )
        if not deployable_static.empty:
            table = pd.concat([table, deployable_static], ignore_index=True).sort_values("oracle_loss_mean").reset_index(drop=True)
    table.to_csv(out_dir / "oracle_lift_candidate_table.csv", index=False)
    summary = build_summary(
        table,
        budget=float(args.budget),
        truth=truth,
        start_indices=tuple(int(x) for x in eval_start_indices),
    )
    summary.update(
        {
            "sensor_cfg": str(resolve_sensor_cfg(str(args.sensor_cfg))),
            "truth_csv": str(truth_path),
            "oracle_type": str(args.oracle_type),
            "target_weights": list(target_weights),
            "target_scales": list(target_scales),
            "constraints": {
                "max_active": int(args.max_active),
                "per_step_budget": float(args.budget),
                "startup_peak_budget": float(args.startup_peak_budget),
                "required_sensor_ids": [str(x) for x in args.required_sensors],
                "coverage_groups": [
                    {"name": str(name), "sensor_ids": [str(sensor_id) for sensor_id in sensor_ids]}
                    for name, sensor_ids in constraints.coverage_groups
                ],
            },
            "energy_account": {
                "enabled": bool(args.energy_account),
                "energy_capacity": float(args.energy_capacity),
                "initial_energy": float(args.initial_energy),
                "harvest_per_step": float(args.harvest_per_step),
                "reserve_energy": float(args.reserve_energy),
                "lambda_energy_deficit": float(args.lambda_energy_deficit),
            },
            "env_min_dwell_steps": int(max(1, int(args.env_min_dwell_steps))),
            "eval_start_selection": str(args.eval_start_selection),
            "eval_selection_stride": int(args.eval_selection_stride),
            "deployable_static_diagnostics": {
                "enabled": bool(args.deployable_static_diagnostics),
                "top_k": int(args.deployable_static_top_k),
                "duty_low": float(args.deployable_static_duty_low),
                "duty_high": float(args.deployable_static_duty_high),
                "duty_score": float(args.deployable_static_duty_score),
                "duty_feedback": float(args.deployable_static_duty_feedback),
            },
        }
    )
    (out_dir / "oracle_lift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(out_dir / "oracle_lift_candidate_table.csv")


if __name__ == "__main__":
    main()
