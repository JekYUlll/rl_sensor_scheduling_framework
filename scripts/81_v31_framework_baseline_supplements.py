#!/usr/bin/env python
"""Supplementary framework baselines for the final V3.1 benchmark.

This script is intentionally post-hoc and replay-based: it reuses the frozen
forecast evaluator, truth sequence, candidate masks, and final-test starts from
completed PD-PPO runs. The goal is to add high-value framework evidence without
changing the trained PD-PPO controller.
"""
from __future__ import annotations

import argparse
import copy
import glob
import importlib.util
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SUBTYPE_LABELS = {
    1: "particle",
    2: "flux",
    3: "thermal",
}
SUBTYPE_LOSS_COLUMNS = tuple(f"oracle_loss_subtype_{label}" for label in SUBTYPE_LABELS.values())
MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event"
STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event_staticnorm"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def parse_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def resolve_path(value: str | Path, *, run_dir: Path) -> Path:
    path = Path(value)
    candidates = [path, run_dir / path.name, ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve {value!r} from {run_dir}")


def sensor_ids_for_mask(sensors: list[Any], mask: np.ndarray) -> str:
    return "|".join(str(sensors[idx].sensor_id) for idx in np.flatnonzero(np.asarray(mask, dtype=bool)))


def append_subtype_loss_split(row: dict[str, Any], result: Any, truth: pd.DataFrame) -> dict[str, Any]:
    losses = np.asarray(result.oracle_losses, dtype=float).reshape(-1)
    step_indices = np.asarray(getattr(result, "step_indices", np.asarray([], dtype=int)), dtype=int).reshape(-1)
    if "event_subtype_id" not in truth.columns or step_indices.size != losses.size:
        for label in SUBTYPE_LABELS.values():
            row[f"oracle_loss_subtype_{label}"] = float("nan")
            row[f"steps_subtype_{label}"] = 0
        row[MACRO_SUBTYPE_LOSS_COLUMN] = float("nan")
        row["macro_subtype_event_count"] = 0
        return row

    valid = (step_indices >= 0) & (step_indices < len(truth))
    subtype_values = np.zeros_like(step_indices, dtype=int)
    subtype_values[valid] = truth["event_subtype_id"].to_numpy(dtype=int)[step_indices[valid]]
    finite = np.isfinite(losses)
    subtype_losses: list[float] = []
    for subtype_id, label in SUBTYPE_LABELS.items():
        mask = (subtype_values == int(subtype_id)) & finite
        subtype_loss = float(np.mean(losses[mask])) if np.any(mask) else float("nan")
        row[f"oracle_loss_subtype_{label}"] = subtype_loss
        row[f"steps_subtype_{label}"] = int(np.sum(subtype_values == int(subtype_id)))
        if np.isfinite(subtype_loss):
            subtype_losses.append(subtype_loss)
    row[MACRO_SUBTYPE_LOSS_COLUMN] = finite_mean(subtype_losses)
    row["macro_subtype_event_count"] = int(len(subtype_losses))
    return row


def subtype_static_normalizers(table: pd.DataFrame | None) -> dict[str, float]:
    normalizers: dict[str, float] = {}
    if table is None or table.empty:
        return normalizers
    for col in SUBTYPE_LOSS_COLUMNS:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            normalizers[col] = float(np.median(values))
    return normalizers


def add_staticnorm_macro(table: pd.DataFrame, normalizers: dict[str, float]) -> pd.DataFrame:
    if table.empty or not normalizers:
        return table
    result = table.copy()
    norm_cols: list[str] = []
    for col in SUBTYPE_LOSS_COLUMNS:
        denom = finite_float(normalizers.get(col))
        if col not in result.columns or not np.isfinite(denom) or denom <= 0.0:
            continue
        norm_col = f"{col}_staticnorm"
        result[norm_col] = pd.to_numeric(result[col], errors="coerce") / denom
        norm_cols.append(norm_col)
    if norm_cols:
        result[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN] = result[norm_cols].apply(
            lambda row: finite_mean([finite_float(value) for value in row.to_list()]),
            axis=1,
        )
    return result


def capture_env_state(env: Any) -> dict[str, Any]:
    return {
        "rng_state": copy.deepcopy(env.rng.bit_generator.state),
        "runtimes": {
            sid: (
                runtime.mode,
                int(runtime.warm_remaining),
                None if runtime.last_observed_step is None else int(runtime.last_observed_step),
                int(runtime.warmup_abort_count),
            )
            for sid, runtime in env.runtimes.items()
        },
        "current_idx": int(env.current_idx),
        "episode_start_idx": int(env.episode_start_idx),
        "episode_end_idx": int(env.episode_end_idx),
        "last_observation": np.array(env.last_observation, dtype=float, copy=True),
        "observed_mask": np.array(env.observed_mask, dtype=float, copy=True),
        "history": np.array(env.history, dtype=float, copy=True),
        "mask_history": np.array(env.mask_history, dtype=float, copy=True),
        "posterior_variance": np.array(env.posterior_variance, dtype=float, copy=True),
        "previous_action_mask": np.array(env.previous_action_mask, dtype=float, copy=True),
        "sensor_on_counts": np.array(env.sensor_on_counts, dtype=float, copy=True),
        "elapsed_steps": int(env.elapsed_steps),
        "dwell_hold_remaining": int(env.dwell_hold_remaining),
        "current_energy": float(env.current_energy),
        "energy_deficit_steps": int(env.energy_deficit_steps),
        "energy_deficit_total": float(env.energy_deficit_total),
        "last_info": copy.deepcopy(env.last_info),
    }


def restore_env_state(env: Any, snapshot: dict[str, Any]) -> None:
    env.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
    for sid, values in snapshot["runtimes"].items():
        mode, warm_remaining, last_observed_step, warmup_abort_count = values
        runtime = env.runtimes[sid]
        runtime.mode = mode
        runtime.warm_remaining = int(warm_remaining)
        runtime.last_observed_step = last_observed_step
        runtime.warmup_abort_count = int(warmup_abort_count)
    env.current_idx = int(snapshot["current_idx"])
    env.episode_start_idx = int(snapshot["episode_start_idx"])
    env.episode_end_idx = int(snapshot["episode_end_idx"])
    env.last_observation = np.array(snapshot["last_observation"], dtype=float, copy=True)
    env.observed_mask = np.array(snapshot["observed_mask"], dtype=float, copy=True)
    env.history = np.array(snapshot["history"], dtype=float, copy=True)
    env.mask_history = np.array(snapshot["mask_history"], dtype=float, copy=True)
    env.posterior_variance = np.array(snapshot["posterior_variance"], dtype=float, copy=True)
    env.previous_action_mask = np.array(snapshot["previous_action_mask"], dtype=float, copy=True)
    env.sensor_on_counts = np.array(snapshot["sensor_on_counts"], dtype=float, copy=True)
    env.elapsed_steps = int(snapshot["elapsed_steps"])
    env.dwell_hold_remaining = int(snapshot["dwell_hold_remaining"])
    env.current_energy = float(snapshot["current_energy"])
    env.energy_deficit_steps = int(snapshot["energy_deficit_steps"])
    env.energy_deficit_total = float(snapshot["energy_deficit_total"])
    env.last_info = copy.deepcopy(snapshot["last_info"])


class ForecastGreedyOneStepPolicy:
    """Privileged myopic diagnostic using final-step forecast loss for each mask."""

    def __init__(self, candidate_masks: np.ndarray, *, name: str = "forecast_greedy_one_step") -> None:
        self.candidate_masks = np.asarray(candidate_masks, dtype=bool)
        self.name = str(name)

    def reset(self) -> None:
        return None

    def act_mask(self, env: Any) -> np.ndarray:
        snapshot = capture_env_state(env)
        previous = np.asarray(env.previous_action_mask, dtype=bool).reshape(-1)
        best_key: tuple[float, float, int] | None = None
        best_mask = np.asarray(self.candidate_masks[0], dtype=bool).copy()
        for action_idx, mask in enumerate(self.candidate_masks):
            restore_env_state(env, snapshot)
            try:
                _, _, _, info = env.step_mask(np.asarray(mask, dtype=bool))
            except Exception:
                continue
            loss = finite_float(info.get("oracle_loss"))
            if not np.isfinite(loss):
                loss = float("inf")
            selected = np.asarray(info.get("selected_mask"), dtype=bool).reshape(-1)
            switch = float(np.mean(np.abs(selected.astype(float) - previous.astype(float))))
            key = (loss, switch, int(action_idx))
            if best_key is None or key < best_key:
                best_key = key
                best_mask = np.asarray(mask, dtype=bool).copy()
        restore_env_state(env, snapshot)
        return best_mask

    def act_scores(self, env: Any) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


class TraceDistilledForecastValuePolicy:
    """Online policy distilled from policy-training receding forecast values."""

    def __init__(
        self,
        *,
        candidate_masks: np.ndarray,
        trace: pd.DataFrame,
        seed: int,
        name: str = "trace_distilled_forecast_value",
    ) -> None:
        from sklearn.ensemble import ExtraTreesRegressor

        self.candidate_masks = np.asarray(candidate_masks, dtype=bool)
        self.name = str(name)
        self.feature_columns = sorted(
            (column for column in trace.columns if column.startswith("online_state_")),
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        self.cost_columns = sorted(
            (column for column in trace.columns if column.startswith("candidate_cost_")),
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if len(self.cost_columns) != len(self.candidate_masks):
            raise ValueError("Trace candidate costs do not match the executable action geometry")
        features = np.nan_to_num(trace[self.feature_columns].to_numpy(dtype=float))
        costs = trace[self.cost_columns].to_numpy(dtype=float)
        if not np.all(np.isfinite(costs)):
            raise ValueError("Policy-training candidate costs must be finite")
        self.model = ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=4,
            max_features=0.7,
            n_jobs=-1,
            random_state=int(seed),
        ).fit(features, costs)

    def reset(self) -> None:
        return None

    def act_mask(self, env: Any) -> np.ndarray:
        features = np.asarray(env._state(), dtype=float).reshape(1, -1)
        predicted_costs = np.asarray(self.model.predict(features)[0], dtype=float)
        for action_idx, mask in enumerate(self.candidate_masks):
            projection = env.projector.project_mask(mask, env.runtimes)
            if not bool(projection.feasible) or not np.array_equal(
                projection.selected_mask.astype(bool), mask
            ):
                predicted_costs[action_idx] = float("inf")
        return np.asarray(self.candidate_masks[int(np.argmin(predicted_costs))], dtype=bool).copy()

    def act_scores(self, env: Any) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


class ContextAlertBanditPolicy:
    """Context-alert bandit driven by supplied synthetic warning scores."""

    def __init__(
        self,
        *,
        sensors: list[Any],
        candidate_masks: np.ndarray,
        action_indices: dict[str, int],
        threshold: float,
        name: str,
    ) -> None:
        self.sensors = list(sensors)
        self.candidate_masks = np.asarray(candidate_masks, dtype=bool)
        self.action_indices = dict(action_indices)
        self.threshold = float(threshold)
        self.name = str(name)
        self.alert_columns = {
            "particle": "agent_context_particle_alert",
            "flux": "agent_context_flux_alert",
            "thermal": "agent_context_thermal_alert",
        }

    def reset(self) -> None:
        return None

    def act_mask(self, env: Any) -> np.ndarray:
        row = env.truth_df.iloc[int(env.current_idx)]
        best_label = "calm"
        best_value = -float("inf")
        for label, column in self.alert_columns.items():
            value = finite_float(row.get(column, 0.0))
            if value > best_value:
                best_value = value
                best_label = label
        if not np.isfinite(best_value) or best_value < self.threshold:
            best_label = "calm"
        action_idx = int(self.action_indices.get(best_label, self.action_indices.get("calm", 0)))
        return np.asarray(self.candidate_masks[action_idx], dtype=bool).copy()

    def act_scores(self, env: Any) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


class QualityAwareContextPolicy(ContextAlertBanditPolicy):
    """Select a feasible subset from online warnings and reported channel quality."""

    def __init__(
        self,
        *,
        action_score_table: pd.DataFrame,
        quality_penalty: float,
        use_alert_context: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.quality_penalty = float(quality_penalty)
        self.use_alert_context = bool(use_alert_context)
        self.quality_columns = [
            f"agent_context_quality_{sensor.sensor_id}" for sensor in self.sensors
        ]
        indexed = action_score_table.set_index("action_idx")
        self.regime_scores: dict[str, np.ndarray] = {}
        score_columns = {
            "calm": "oracle_loss_non_event",
            "particle": "oracle_loss_subtype_particle",
            "flux": "oracle_loss_subtype_flux",
            "thermal": "oracle_loss_subtype_thermal",
        }
        for label, column in score_columns.items():
            scores = np.full(len(self.candidate_masks), np.inf, dtype=float)
            for action_idx in range(len(self.candidate_masks)):
                if action_idx not in indexed.index:
                    continue
                value = finite_float(indexed.loc[action_idx].get(column, float("nan")))
                if not np.isfinite(value):
                    value = finite_float(
                        indexed.loc[action_idx].get("oracle_loss_mean", float("nan"))
                    )
                scores[action_idx] = value
            finite = scores[np.isfinite(scores) & (scores > 0.0)]
            if finite.size == 0:
                raise ValueError(f"No finite validation scores for {label}")
            self.regime_scores[label] = scores / float(np.min(finite))

    def act_mask(self, env: Any) -> np.ndarray:
        row = env.truth_df.iloc[int(env.current_idx)]
        label = "calm"
        if self.use_alert_context:
            best_value = -float("inf")
            for candidate_label, column in self.alert_columns.items():
                value = finite_float(row.get(column, 0.0))
                if value > best_value:
                    best_value = value
                    label = candidate_label
            if not np.isfinite(best_value) or best_value < self.threshold:
                label = "calm"
        quality = np.asarray(
            [
                np.clip(finite_float(row.get(column, 1.0)), 0.0, 1.0)
                for column in self.quality_columns
            ],
            dtype=float,
        )
        scores = self.regime_scores[label].copy()
        for action_idx, mask in enumerate(self.candidate_masks):
            projection = env.projector.project_mask(mask, env.runtimes)
            if not bool(projection.feasible) or not np.array_equal(
                projection.selected_mask.astype(bool), mask
            ):
                scores[action_idx] = float("inf")
                continue
            selected = np.flatnonzero(mask)
            degradation = float(np.mean(1.0 - quality[selected])) if selected.size else 1.0
            scores[action_idx] += self.quality_penalty * degradation
        return np.asarray(self.candidate_masks[int(np.argmin(scores))], dtype=bool).copy()


class IntensityBinnedContextPolicy(ContextAlertBanditPolicy):
    """Online context policy with calibration-selected low/high intensity actions."""

    def __init__(self, *, high_threshold: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.high_threshold = float(high_threshold)
        if self.high_threshold <= self.threshold:
            raise ValueError("high_threshold must exceed the calm threshold")

    def act_mask(self, env: Any) -> np.ndarray:
        row = env.truth_df.iloc[int(env.current_idx)]
        best_label = "calm"
        best_value = -float("inf")
        for label, column in self.alert_columns.items():
            value = finite_float(row.get(column, 0.0))
            if value > best_value:
                best_value = value
                best_label = label
        if not np.isfinite(best_value) or best_value < self.threshold:
            action_label = "calm"
        else:
            level = "high" if best_value >= self.high_threshold else "low"
            action_label = f"{best_label}_{level}"
        action_idx = int(self.action_indices.get(action_label, self.action_indices.get("calm", 0)))
        return np.asarray(self.candidate_masks[action_idx], dtype=bool).copy()


def best_action_by_column(table: pd.DataFrame, column: str, *, fallback: str = "oracle_loss_mean") -> int:
    score_col = column if column in table.columns else fallback
    values = pd.to_numeric(table[score_col], errors="coerce")
    candidates = table[np.isfinite(values)].copy()
    if candidates.empty:
        raise ValueError(f"No finite candidates for {score_col}")
    candidates["_score"] = values[np.isfinite(values)]
    return int(candidates.sort_values(["_score", "oracle_loss_mean"]).iloc[0]["action_idx"])


def build_context_action_indices(validation_table: pd.DataFrame) -> dict[str, int]:
    return {
        "calm": best_action_by_column(validation_table, "oracle_loss_non_event"),
        "particle": best_action_by_column(validation_table, "oracle_loss_subtype_particle"),
        "flux": best_action_by_column(validation_table, "oracle_loss_subtype_flux"),
        "thermal": best_action_by_column(validation_table, "oracle_loss_subtype_thermal"),
    }


def coordinate_select_context_actions(
    validation_table: pd.DataFrame,
    evaluate_mapping: Callable[[dict[str, int]], tuple[float, float]],
    *,
    pool_size: int = 4,
    passes: int = 2,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Select a complete context policy using constrained calibration replay."""
    labels = ("calm", "particle", "flux", "thermal")
    columns = {
        "calm": "oracle_loss_non_event",
        "particle": "oracle_loss_subtype_particle",
        "flux": "oracle_loss_subtype_flux",
        "thermal": "oracle_loss_subtype_thermal",
    }
    selected = build_context_action_indices(validation_table)
    pools: dict[str, list[int]] = {}
    for label in labels:
        column = columns[label]
        ranked = validation_table.copy()
        ranked["_score"] = pd.to_numeric(ranked[column], errors="coerce")
        ranked = ranked[np.isfinite(ranked["_score"])].sort_values(
            ["_score", "oracle_loss_mean", "action_idx"]
        )
        candidates = [int(value) for value in ranked["action_idx"].head(max(1, int(pool_size)))]
        pools[label] = list(dict.fromkeys([int(selected[label]), *candidates]))

    ledger: list[dict[str, Any]] = []

    def score(mapping: dict[str, int], *, pass_idx: int, changed_label: str) -> tuple[float, float]:
        primary, secondary = evaluate_mapping(mapping)
        ledger.append(
            {
                "pass": int(pass_idx),
                "changed_label": str(changed_label),
                **{label: int(mapping[label]) for label in labels},
                "selection_primary": float(primary),
                "selection_secondary": float(secondary),
            }
        )
        return float(primary), float(secondary)

    best = score(selected, pass_idx=0, changed_label="initial")
    for pass_idx in range(1, max(1, int(passes)) + 1):
        changed = False
        for label in labels:
            label_best = best
            label_action = int(selected[label])
            for action_idx in pools[label]:
                trial = dict(selected)
                trial[label] = int(action_idx)
                trial_score = score(trial, pass_idx=pass_idx, changed_label=label)
                if trial_score < label_best:
                    label_best = trial_score
                    label_action = int(action_idx)
            if label_action != int(selected[label]):
                selected[label] = label_action
                best = label_best
                changed = True
        if not changed:
            break
    return selected, pd.DataFrame(ledger)


def coordinate_select_intensity_actions(
    validation_table: pd.DataFrame,
    evaluate_mapping: Callable[[dict[str, int]], tuple[float, float]],
    *,
    pool_size: int = 4,
    passes: int = 2,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Calibrate calm and subtype-specific low/high actions on held-out replay."""
    labels = ("calm", "particle_low", "particle_high", "flux_low", "flux_high", "thermal_low", "thermal_high")
    columns = {
        "calm": "oracle_loss_non_event",
        "particle_low": "oracle_loss_subtype_particle",
        "particle_high": "oracle_loss_subtype_particle",
        "flux_low": "oracle_loss_subtype_flux",
        "flux_high": "oracle_loss_subtype_flux",
        "thermal_low": "oracle_loss_subtype_thermal",
        "thermal_high": "oracle_loss_subtype_thermal",
    }
    base = build_context_action_indices(validation_table)
    selected = {label: int(base[label.split("_")[0]]) for label in labels}
    pools: dict[str, list[int]] = {}
    for label in labels:
        ranked = validation_table.copy()
        ranked["_score"] = pd.to_numeric(ranked[columns[label]], errors="coerce")
        ranked = ranked[np.isfinite(ranked["_score"])].sort_values(["_score", "oracle_loss_mean", "action_idx"])
        candidates = [int(value) for value in ranked["action_idx"].head(max(1, int(pool_size)))]
        pools[label] = list(dict.fromkeys([int(selected[label]), *candidates]))

    ledger: list[dict[str, Any]] = []

    def score(mapping: dict[str, int], *, pass_idx: int, changed_label: str) -> tuple[float, float]:
        primary, secondary = evaluate_mapping(mapping)
        ledger.append({
            "pass": int(pass_idx),
            "changed_label": str(changed_label),
            **{label: int(mapping[label]) for label in labels},
            "selection_primary": float(primary),
            "selection_secondary": float(secondary),
        })
        return float(primary), float(secondary)

    best = score(selected, pass_idx=0, changed_label="initial")
    for pass_idx in range(1, max(1, int(passes)) + 1):
        changed = False
        for label in labels:
            label_best = best
            label_action = int(selected[label])
            for action_idx in pools[label]:
                trial = dict(selected)
                trial[label] = int(action_idx)
                trial_score = score(trial, pass_idx=pass_idx, changed_label=label)
                if trial_score < label_best:
                    label_best = trial_score
                    label_action = int(action_idx)
            if label_action != int(selected[label]):
                selected[label] = label_action
                best = label_best
                changed = True
        if not changed:
            break
    return selected, pd.DataFrame(ledger)


def build_physical_context_action_indices(
    metadata: dict[str, Any],
    sensors: list[Any],
    candidate_masks: np.ndarray,
) -> dict[str, int]:
    sensor_ids = [str(spec.sensor_id) for spec in sensors]
    definitions = dict(metadata.get("oracle_subtype_teacher_sensors", {}))
    out: dict[str, int] = {}
    for label in ("calm", "particle", "flux", "thermal"):
        wanted = {str(sensor_id) for sensor_id in definitions.get(label, ())}
        target = np.asarray([sensor_id in wanted for sensor_id in sensor_ids], dtype=bool)
        matches = np.flatnonzero(np.all(np.asarray(candidate_masks, dtype=bool) == target, axis=1))
        if len(matches) != 1:
            raise ValueError(f"Physical {label} mask is not a unique candidate: {sorted(wanted)}")
        out[label] = int(matches[0])
    return out


def build_continuity_guarded_context_action_indices(
    metadata: dict[str, Any],
    sensors: list[Any],
    candidate_masks: np.ndarray,
    validation_table: pd.DataFrame,
) -> dict[str, int]:
    """Select specialist masks while retaining maximal calm-mask coverage."""
    sensor_ids = [str(spec.sensor_id) for spec in sensors]
    masks = np.asarray(candidate_masks, dtype=bool)
    definitions = dict(metadata.get("oracle_subtype_teacher_sensors", {}))
    calm_action = best_action_by_column(validation_table, "oracle_loss_non_event")
    calm_mask = masks[int(calm_action)]
    physical_calm = {str(sensor_id) for sensor_id in definitions.get("calm", ())}
    indexed = validation_table.set_index("action_idx")
    out = {"calm": int(calm_action)}

    for label in ("particle", "flux", "thermal"):
        physical_event = {str(sensor_id) for sensor_id in definitions.get(label, ())}
        required_ids = physical_event - physical_calm
        required_indices = [idx for idx, sensor_id in enumerate(sensor_ids) if sensor_id in required_ids]
        eligible = [
            idx
            for idx, mask in enumerate(masks)
            if required_indices and bool(np.all(mask[required_indices])) and idx in indexed.index
        ]
        if not eligible:
            out[label] = build_physical_context_action_indices(metadata, sensors, masks)[label]
            continue
        overlap = {idx: int(np.sum(masks[idx] & calm_mask)) for idx in eligible}
        max_overlap = max(overlap.values())
        continuity_candidates = [idx for idx in eligible if overlap[idx] == max_overlap]
        score_column = f"oracle_loss_subtype_{label}"
        ranked = indexed.loc[continuity_candidates].copy()
        ranked["_score"] = pd.to_numeric(ranked[score_column], errors="coerce")
        ranked = ranked[np.isfinite(ranked["_score"])]
        if ranked.empty:
            out[label] = min(continuity_candidates)
        else:
            out[label] = int(
                ranked.reset_index()
                .sort_values(["_score", "oracle_loss_mean", "action_idx"])
                .iloc[0]["action_idx"]
            )
    return out


def rollout_macro_subtype_loss(rollout_path: Path, truth: pd.DataFrame) -> dict[str, float]:
    if not rollout_path.exists() or "event_subtype_id" not in truth.columns:
        return {}
    data = np.load(rollout_path, allow_pickle=False)
    losses = np.asarray(data["oracle_losses"], dtype=float).reshape(-1)
    step_indices = np.asarray(data["step_indices"], dtype=int).reshape(-1)
    if losses.size != step_indices.size:
        return {}
    valid = (step_indices >= 0) & (step_indices < len(truth))
    subtype_values = np.zeros_like(step_indices, dtype=int)
    subtype_values[valid] = truth["event_subtype_id"].to_numpy(dtype=int)[step_indices[valid]]
    finite = np.isfinite(losses)
    out: dict[str, float] = {}
    subtype_losses: list[float] = []
    for subtype_id, label in SUBTYPE_LABELS.items():
        mask = (subtype_values == int(subtype_id)) & finite
        subtype_loss = float(np.mean(losses[mask])) if np.any(mask) else float("nan")
        out[f"oracle_loss_subtype_{label}"] = subtype_loss
        if np.isfinite(subtype_loss):
            subtype_losses.append(subtype_loss)
    out[MACRO_SUBTYPE_LOSS_COLUMN] = finite_mean(subtype_losses)
    return out


def reference_metrics(
    run_dir: Path,
    router_eval_dir: str,
    *,
    truth: pd.DataFrame,
    normalizers: dict[str, float],
) -> dict[str, Any]:
    eval_dir = run_dir / router_eval_dir
    if not (eval_dir / "v2_custom_ppo_metrics.csv").exists():
        eval_dir = run_dir
    path = eval_dir / "v2_custom_ppo_metrics.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    out: dict[str, Any] = {}
    for policy in ("custom_ppo", "validation_selected_static", "feasible_static_projected", "round_robin", "aoi", "random"):
        rows = table[table["policy"].astype(str) == policy]
        if rows.empty:
            continue
        row = rows.iloc[0]
        out[f"{policy}_oracle_loss_mean"] = finite_float(row.get("oracle_loss_mean"))
        if MACRO_SUBTYPE_LOSS_COLUMN in row.index:
            out[f"{policy}_{MACRO_SUBTYPE_LOSS_COLUMN}"] = finite_float(row.get(MACRO_SUBTYPE_LOSS_COLUMN))
    for policy in ("custom_ppo", "validation_selected_static", "feasible_static_projected", "round_robin", "aoi", "random"):
        macro_values = rollout_macro_subtype_loss(eval_dir / f"rollout_{policy}.npz", truth)
        for key, value in macro_values.items():
            out[f"{policy}_{key}"] = value
        normalized: list[float] = []
        for col in SUBTYPE_LOSS_COLUMNS:
            denom = finite_float(normalizers.get(col))
            value = finite_float(macro_values.get(col))
            if np.isfinite(value) and np.isfinite(denom) and denom > 0.0:
                normalized.append(value / denom)
        if normalized:
            out[f"{policy}_{STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN}"] = finite_mean(normalized)
    return out


def evaluate_run(
    run_dir: Path,
    *,
    out_dir: Path,
    router_eval_dir: str,
    replay_dir: str,
    oracle_device: str,
    policies: tuple[str, ...],
    context_thresholds: tuple[float, ...],
    context_high_threshold: float,
    quality_penalties: tuple[float, ...],
    greedy_max_steps: int,
    context_action_source: str,
    trace_training_subdir: str,
) -> pd.DataFrame:
    helpers = load_module(ROOT / "scripts" / "23_v2_train_ppo.py", "_framework_baseline_helpers")
    ops = load_module(ROOT / "scripts" / "64_v31_eval_saved_run_operational_baselines.py", "_framework_baseline_ops")
    gate = load_module(ROOT / "scripts" / "70_v31_split_replay_gate.py", "_framework_baseline_gate")

    run_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    helpers.STATE_COLUMNS = tuple(metadata.get("state_columns", helpers.STATE_COLUMNS))
    policy_cfg = dict(metadata.get("custom_ppo", {}))
    alert_cfg = dict(metadata.get("agent_alert_context", {}))
    if bool(policy_cfg.get("subtype_router_enabled", False)):
        raise ValueError(f"Framework supplements reject hard-router policy: {run_dir}")
    if bool(alert_cfg.get("include_event_flag_in_state", True)):
        raise ValueError(f"Framework supplements reject exact online event flag: {run_dir}")
    if bool(alert_cfg.get("truth_event_labels_used_online", True)):
        raise ValueError(f"Framework supplements reject online truth-label use: {run_dir}")
    truth = helpers.ensure_state_columns(pd.read_csv(resolve_path(metadata["truth_csv"], run_dir=run_dir)))
    sensors = gate.load_sensor_specs(resolve_path(metadata["sensor_cfg"], run_dir=run_dir))
    constraints = ops.constraints_from_metadata(metadata)
    oracle, eval_cfg, _static_cfg, eval_steps, eval_starts = gate.build_eval_cfg(
        helpers=helpers,
        ops=ops,
        metadata=metadata,
        truth=truth,
        run_dir=run_dir,
        env_min_dwell_steps=None,
        oracle_device=str(oracle_device),
    )
    eval_metadata_path = run_dir / router_eval_dir / "v2_ppo_metadata.json"
    if eval_metadata_path.exists():
        eval_metadata = json.loads(eval_metadata_path.read_text(encoding="utf-8"))
        if eval_metadata.get("eval_steps") is not None:
            eval_steps = int(eval_metadata["eval_steps"])
            eval_cfg = replace(eval_cfg, episode_len=int(eval_steps))
        if eval_metadata.get("eval_start_indices") is not None:
            eval_starts = tuple(int(x) for x in eval_metadata["eval_start_indices"])
    candidate_masks = helpers.build_projected_candidate_masks(sensors, constraints)

    validation_path = run_dir / "validation_static_candidates.csv"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation static candidates: {validation_path}")
    validation_table = pd.read_csv(validation_path)
    context_action_table = validation_table
    context_score_columns = (
        "oracle_loss_non_event",
        "oracle_loss_subtype_particle",
        "oracle_loss_subtype_flux",
        "oracle_loss_subtype_thermal",
    )
    if any(
        column not in context_action_table.columns
        or not np.isfinite(pd.to_numeric(context_action_table[column], errors="coerce")).any()
        for column in context_score_columns
    ):
        fallback_path = run_dir / "reward_staticnorm_fallback_candidates.csv"
        if fallback_path.exists():
            context_action_table = pd.read_csv(fallback_path)
    static_table_path = run_dir / replay_dir / "split_static_candidate_event_table.csv"
    static_table = pd.read_csv(static_table_path) if static_table_path.exists() else validation_table.copy()
    normalizers = subtype_static_normalizers(static_table)
    intensity_binned = str(context_action_source) == "intensity_replay_calibrated"
    if str(context_action_source) == "physical":
        action_indices = build_physical_context_action_indices(metadata, sensors, candidate_masks)
    elif str(context_action_source) == "continuity_guarded":
        action_indices = build_continuity_guarded_context_action_indices(
            metadata,
            sensors,
            candidate_masks,
            context_action_table,
        )
    elif str(context_action_source) == "hybrid":
        action_indices = build_physical_context_action_indices(metadata, sensors, candidate_masks)
        action_indices["calm"] = best_action_by_column(context_action_table, "oracle_loss_non_event")
    elif str(context_action_source) == "guarded_hybrid":
        action_indices = build_physical_context_action_indices(metadata, sensors, candidate_masks)
        calm_action = best_action_by_column(context_action_table, "oracle_loss_non_event")
        action_indices["calm"] = calm_action
        indexed = context_action_table.set_index("action_idx")
        for label in ("particle", "flux", "thermal"):
            score_column = f"oracle_loss_subtype_{label}"
            specialist_score = finite_float(indexed.loc[action_indices[label], score_column])
            calm_score = finite_float(indexed.loc[calm_action, score_column])
            if not np.isfinite(specialist_score) or (
                np.isfinite(calm_score) and specialist_score >= calm_score
            ):
                action_indices[label] = calm_action
    elif str(context_action_source) in {"replay_calibrated", "intensity_replay_calibrated"}:
        checkpoint_cfg = dict(metadata.get("checkpoint_selection", {}))
        calibration_starts = tuple(int(value) for value in checkpoint_cfg.get("start_indices", ()))
        calibration_steps = int(checkpoint_cfg.get("steps", 0))
        if not calibration_starts or calibration_steps <= 0:
            raise ValueError("Replay-calibrated context actions require checkpoint-selection starts")
        calibration_cfg = replace(
            eval_cfg,
            episode_len=calibration_steps,
            seed=int(metadata["seed"]) + 12_000,
        )
        best_static_ordinary = float(pd.to_numeric(validation_table["oracle_loss_mean"], errors="coerce").min())
        static_macro_values = pd.to_numeric(
            add_staticnorm_macro(validation_table, normalizers).get(
                STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN, pd.Series(dtype=float)
            ),
            errors="coerce",
        )
        best_static_macro = float(static_macro_values.min())

        def evaluate_context_mapping(mapping: dict[str, int]) -> tuple[float, float]:
            policy_class = IntensityBinnedContextPolicy if intensity_binned else ContextAlertBanditPolicy
            policy_kwargs: dict[str, Any] = {
                "sensors": sensors,
                "candidate_masks": candidate_masks,
                "action_indices": mapping,
                "threshold": 0.5,
                "name": "context_calibration_candidate",
            }
            if intensity_binned:
                policy_kwargs["high_threshold"] = float(context_high_threshold)
            policy = policy_class(**policy_kwargs)
            result, metrics = helpers.evaluate_score_policy_over_starts(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=calibration_cfg,
                oracle=oracle,
                policy=policy,
                steps=calibration_steps,
                start_indices=calibration_starts,
            )
            row = append_subtype_loss_split(dict(metrics), result, truth)
            scored = add_staticnorm_macro(pd.DataFrame([row]), normalizers).iloc[0]
            ordinary = finite_float(scored.get("oracle_loss_mean"))
            macro = finite_float(scored.get(STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN))
            ordinary_ratio = ordinary / best_static_ordinary
            macro_ratio = macro / best_static_macro
            return max(ordinary_ratio, macro_ratio), ordinary_ratio + macro_ratio

        selector = coordinate_select_intensity_actions if intensity_binned else coordinate_select_context_actions
        action_indices, calibration_ledger = selector(context_action_table, evaluate_context_mapping)
        calibration_ledger.to_csv(out_dir / "context_replay_calibration_ledger.csv", index=False)
    else:
        action_indices = build_context_action_indices(context_action_table)

    policy_objects: list[Any] = []
    selected_action_rows: list[dict[str, Any]] = []
    if "context_bandit" in policies:
        for threshold in context_thresholds:
            if intensity_binned:
                name = (
                    f"context_alert_intensity_t{str(threshold).replace('.', 'p')}_"
                    f"h{str(context_high_threshold).replace('.', 'p')}"
                )
                policy_objects.append(IntensityBinnedContextPolicy(
                    sensors=sensors,
                    candidate_masks=candidate_masks,
                    action_indices=action_indices,
                    threshold=float(threshold),
                    high_threshold=float(context_high_threshold),
                    name=name,
                ))
            else:
                name = f"context_alert_bandit_t{str(threshold).replace('.', 'p')}"
                policy_objects.append(ContextAlertBanditPolicy(
                    sensors=sensors,
                    candidate_masks=candidate_masks,
                    action_indices=action_indices,
                    threshold=float(threshold),
                    name=name,
                ))
            selected_action_rows.append(
                {
                    "policy": name,
                    **action_indices,
                    "threshold": float(threshold),
                    "high_threshold": float(context_high_threshold) if intensity_binned else float("nan"),
                    "action_source": str(context_action_source),
                }
            )
    if "quality_context_bandit" in policies:
        for penalty in quality_penalties:
            name = f"quality_context_bandit_t0p5_p{str(penalty).replace('.', 'p')}"
            policy_objects.append(
                QualityAwareContextPolicy(
                    sensors=sensors,
                    candidate_masks=candidate_masks,
                    action_indices=action_indices,
                    threshold=0.5,
                    name=name,
                    action_score_table=context_action_table,
                    quality_penalty=float(penalty),
                )
            )
    if "forecast_greedy" in policies:
        policy_objects.append(ForecastGreedyOneStepPolicy(candidate_masks, name="forecast_greedy_one_step"))
    if "trace_distilled" in policies:
        trace_path = run_dir / str(trace_training_subdir) / "receding_oracle_trace.csv"
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing policy-training receding trace: {trace_path}")
        policy_objects.append(TraceDistilledForecastValuePolicy(
            candidate_masks=candidate_masks,
            trace=pd.read_csv(trace_path),
            seed=int(metadata["seed"]),
        ))
    if "quality_only" in policies:
        for penalty in quality_penalties:
            name = f"quality_only_calibrated_p{str(penalty).replace('.', 'p')}"
            policy_objects.append(
                QualityAwareContextPolicy(
                    sensors=sensors,
                    candidate_masks=candidate_masks,
                    action_indices=action_indices,
                    threshold=0.5,
                    name=name,
                    action_score_table=context_action_table,
                    quality_penalty=float(penalty),
                    use_alert_context=False,
                )
            )
    if "event_label" in policies:
        if "event_subtype_id" not in truth.columns:
            raise ValueError("event-label reference requires event_subtype_id for offline replay")
        lookahead = int(metadata.get("oracle_subtype_teacher_lookahead_steps", 0))
        policy_objects.append(
            gate.SubtypeMaskPolicy(
                name=f"event_label_reference_l{lookahead}",
                subtype_ids=truth["event_subtype_id"].to_numpy(dtype=int),
                calm_mask=np.asarray(candidate_masks[action_indices["calm"]], dtype=bool),
                subtype_masks={
                    1: np.asarray(candidate_masks[action_indices["particle"]], dtype=bool),
                    2: np.asarray(candidate_masks[action_indices["flux"]], dtype=bool),
                    3: np.asarray(candidate_masks[action_indices["thermal"]], dtype=bool),
                },
                lookahead_steps=lookahead,
            )
        )
        selected_action_rows.append(
            {
                "policy": f"event_label_reference_l{lookahead}",
                **action_indices,
                "lookahead_steps": lookahead,
                "action_source": str(context_action_source),
            }
        )

    rows: list[dict[str, Any]] = []
    effective_steps = int(eval_steps)
    if int(greedy_max_steps) > 0 and "forecast_greedy" in policies:
        effective_steps = min(effective_steps, int(greedy_max_steps))
    for policy in policy_objects:
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=effective_steps,
            start_indices=eval_starts,
        )
        row = append_subtype_loss_split(dict(metrics), result, truth)
        row["seed"] = parse_seed(run_dir)
        row["run_dir"] = str(run_dir)
        row["eval_steps_per_start"] = int(effective_steps)
        row["eval_start_count"] = int(len(eval_starts))
        row["diagnostic_privilege"] = (
            "final_future_loss"
            if policy.name == "forecast_greedy_one_step"
            else (
                "exact_final_event_subtype"
                if str(policy.name).startswith("event_label_reference")
                else (
                    "policy_training_future_loss_distillation"
                    if policy.name == "trace_distilled_forecast_value"
                    else "supplied_synthetic_warning_scores"
                )
            )
        )
        rows.append(row)
        gate.save_rollout_npz(
            out_dir / f"rollout_{policy.name}.npz",
            result,
            sensor_ids=[str(spec.sensor_id) for spec in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )

    metrics_table = add_staticnorm_macro(pd.DataFrame(rows), normalizers)
    ref = reference_metrics(run_dir, router_eval_dir, truth=truth, normalizers=normalizers)
    for key, value in ref.items():
        metrics_table[key] = value
    metrics_table["margin_loss_vs_custom_ppo"] = (
        metrics_table["oracle_loss_mean"].astype(float) - metrics_table.get("custom_ppo_oracle_loss_mean", float("nan"))
    )
    if (
        STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN in metrics_table.columns
        and f"custom_ppo_{STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN}" in metrics_table.columns
    ):
        metrics_table[f"margin_{STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN}_vs_custom_ppo"] = (
            metrics_table[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN].astype(float)
            - metrics_table[f"custom_ppo_{STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN}"].astype(float)
        )
    metrics_table.to_csv(out_dir / "framework_baseline_metrics.csv", index=False)
    if selected_action_rows:
        pd.DataFrame(selected_action_rows).to_csv(out_dir / "context_bandit_action_map.csv", index=False)
    return metrics_table


def aggregate(results: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "framework_baseline_seed_metrics.csv", index=False)
    rows: list[dict[str, Any]] = []
    for policy, group in results.groupby("policy"):
        loss_margin = pd.to_numeric(group["margin_loss_vs_custom_ppo"], errors="coerce")
        macro_col = f"margin_{STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN}_vs_custom_ppo"
        macro_margin = pd.to_numeric(group[macro_col], errors="coerce") if macro_col in group.columns else pd.Series(dtype=float)
        rows.append(
            {
                "policy": str(policy),
                "complete_seeds": int(group["seed"].nunique()),
                "mean_oracle_loss": float(pd.to_numeric(group["oracle_loss_mean"], errors="coerce").mean()),
                "mean_loss_margin_vs_custom_ppo": float(loss_margin.mean()),
                "median_loss_margin_vs_custom_ppo": float(loss_margin.median()),
                "pdppo_step_win_count": int((loss_margin > 0).sum()),
                "pdppo_step_loss_count": int((loss_margin < 0).sum()),
                "mean_staticnorm_macro": (
                    float(pd.to_numeric(group[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN], errors="coerce").mean())
                    if STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN in group.columns
                    else float("nan")
                ),
                "mean_staticnorm_macro_margin_vs_custom_ppo": float(macro_margin.mean()) if not macro_margin.empty else float("nan"),
                "pdppo_staticnorm_macro_win_count": int((macro_margin > 0).sum()) if not macro_margin.empty else 0,
                "diagnostic_privilege": "|".join(sorted(set(group["diagnostic_privilege"].astype(str)))),
                "mean_switches_per_step": float(pd.to_numeric(group["switches_per_step"], errors="coerce").mean()),
                "mean_mid_duty_sensor_count": float(pd.to_numeric(group["mid_duty_sensor_count"], errors="coerce").mean()),
                "mean_warmup_abort_count": float(pd.to_numeric(group["warmup_abort_count"], errors="coerce").mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("policy")
    summary.to_csv(out_dir / "framework_baseline_summary.csv", index=False)
    payload = {
        "complete_rows": int(len(results)),
        "policies": rows,
        "interpretation": (
            "Positive margins mean the supplementary baseline has higher loss than PD-PPO. "
            "forecast_greedy_one_step is a privileged myopic diagnostic because it chooses masks "
            "using final-test future loss; context_alert_bandit uses supplied synthetic warning-score "
            "columns; event_label_reference uses exact final-test subtype labels and is privileged."
        ),
    }
    (out_dir / "framework_baseline_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Framework Baseline Supplement", "", payload["interpretation"], "", "## Summary", ""]
    if summary.empty:
        lines.append("No completed rows.")
    else:
        lines.append(dataframe_to_markdown(summary, float_digits=6))
    (out_dir / "framework_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def dataframe_to_markdown(frame: pd.DataFrame, *, float_digits: int = 6) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    if frame.empty:
        return ""

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.{float_digits}f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    rows = [[format_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def expand_run_dirs(pattern: str, seeds: tuple[int, ...]) -> list[Path]:
    paths = [Path(p) for p in sorted(glob.glob(str(pattern)))]
    if seeds:
        wanted = set(int(seed) for seed in seeds)
        paths = [path for path in paths if parse_seed(path) in wanted]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run framework-supporting replay baselines on final benchmark runs.")
    parser.add_argument(
        "--run-glob",
        default=(
            "reports/v31_metpair_backbone_context_ortholinear_balancedobjective_scenebal2_"
            "seed*_h075ctxolscbal2_20260621"
        ),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--out-root", default="reports/aggregate/framework_baseline_supplements_20260702")
    parser.add_argument("--router-eval-dir", default="eval_router_conf05_scenebal2_20260621")
    parser.add_argument("--replay-dir", default="replay_gate_explicit_static_noguard")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument(
        "--trace-training-subdir",
        default="receding_oracle_l8_rl_train_trace",
        help="Policy-training trace directory used by the trace-distilled diagnostic.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=[
            "context_bandit",
            "quality_context_bandit",
            "quality_only",
            "forecast_greedy",
            "event_label",
            "trace_distilled",
        ],
        default=["context_bandit", "forecast_greedy", "event_label"],
    )
    parser.add_argument("--context-thresholds", nargs="+", type=float, default=[0.5])
    parser.add_argument("--context-high-threshold", type=float, default=0.75)
    parser.add_argument("--quality-penalties", nargs="+", type=float, default=[0.25, 1.0, 4.0])
    parser.add_argument(
        "--context-action-source",
        choices=[
            "validation",
            "physical",
            "hybrid",
            "guarded_hybrid",
            "continuity_guarded",
            "replay_calibrated",
            "intensity_replay_calibrated",
        ],
        default="validation",
    )
    parser.add_argument(
        "--greedy-max-steps",
        type=int,
        default=-1,
        help="If positive, cap per-start rollout length for the expensive privileged forecast-greedy diagnostic.",
    )
    parser.add_argument(
        "--reuse-existing-seed-metrics",
        action="store_true",
        help="Load a completed per-seed metrics CSV instead of replaying that seed.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Write per-seed artifacts only; a later resume pass can aggregate them.",
    )
    args = parser.parse_args()

    run_dirs = expand_run_dirs(str(args.run_glob), tuple(int(x) for x in args.seeds))
    if not run_dirs:
        raise SystemExit(f"No run directories matched {args.run_glob!r}")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        seed = parse_seed(run_dir)
        seed_out = out_root / f"seed{seed}"
        seed_metrics_path = seed_out / "framework_baseline_metrics.csv"
        if bool(args.reuse_existing_seed_metrics) and seed_metrics_path.is_file():
            print(f"framework_baseline_reuse seed={seed} path={seed_metrics_path}", flush=True)
            table = pd.read_csv(seed_metrics_path)
        else:
            print(f"framework_baseline_start seed={seed} run_dir={run_dir}", flush=True)
            table = evaluate_run(
                run_dir,
                out_dir=seed_out,
                router_eval_dir=str(args.router_eval_dir),
                replay_dir=str(args.replay_dir),
                oracle_device=str(args.oracle_device),
                policies=tuple(str(x) for x in args.policies),
                context_thresholds=tuple(float(x) for x in args.context_thresholds),
                context_high_threshold=float(args.context_high_threshold),
                quality_penalties=tuple(float(x) for x in args.quality_penalties),
                greedy_max_steps=int(args.greedy_max_steps),
                context_action_source=str(args.context_action_source),
                trace_training_subdir=str(args.trace_training_subdir),
            )
        all_rows.append(table)
        print(f"framework_baseline_done seed={seed} rows={len(table)}", flush=True)
    if bool(args.no_aggregate):
        print(f"framework_baseline_seed_artifacts_done seeds={len(all_rows)}", flush=True)
        return
    result = pd.concat(all_rows, ignore_index=True)
    payload = aggregate(result, out_root)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
