#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.custom_ppo import CustomPPO, CustomPPOConfig, evaluate_custom_ppo  # noqa: E402
from v2.env import WarmupEnvConfig  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.policies import MinDwellPolicyWrapper, StaticMaskPolicy  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


SUBTYPE_LABELS = {
    1: "particle",
    2: "flux",
    3: "thermal",
}
SUBTYPE_LOSS_COLUMNS = tuple(f"oracle_loss_subtype_{label}" for label in SUBTYPE_LABELS.values())
MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event"
STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event_staticnorm"
MACRO_SUBTYPE_COUNT_COLUMN = "macro_subtype_event_count"


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


def energy_kwargs(args: argparse.Namespace) -> dict[str, float | bool]:
    return {
        "energy_account_enabled": bool(args.energy_account),
        "energy_capacity": float(args.energy_capacity),
        "initial_energy": float(args.initial_energy),
        "harvest_per_step": float(args.harvest_per_step),
        "reserve_energy": float(args.reserve_energy),
        "lambda_energy_deficit": float(args.lambda_energy_deficit),
        "soc_soft_penalty_buffer": float(args.soc_soft_penalty_buffer),
        "lambda_soc_soft_penalty": float(args.lambda_soc_soft_penalty),
    }


def resolve_candidate_action_index(
    sensors: list[object],
    candidate_masks: np.ndarray,
    sensor_ids: list[str] | None,
    *,
    label: str,
) -> int:
    if not sensor_ids:
        return -1
    wanted = {str(sensor_id) for sensor_id in sensor_ids}
    mask = np.asarray([str(spec.sensor_id) in wanted for spec in sensors], dtype=bool)
    for idx, candidate in enumerate(np.asarray(candidate_masks, dtype=bool)):
        if np.array_equal(candidate.reshape(-1), mask):
            return int(idx)
    known = [str(spec.sensor_id) for spec in sensors]
    raise ValueError(
        f"{label} teacher mask is not an exact candidate: {sorted(wanted)}; "
        f"known sensors={known}"
    )


def parse_mask_pool_spec(value: str | None) -> list[list[str]]:
    if value is None or not str(value).strip():
        return []
    masks: list[list[str]] = []
    for mask_text in str(value).split(";"):
        sensor_ids = [part for part in mask_text.split(",") if part]
        if sensor_ids:
            masks.append(sensor_ids)
    return masks


def finite_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def control_source_required_files(reward_loss_normalization: str) -> tuple[str, ...]:
    required = [
        "truth_v31_split.csv",
        "v2_ppo_metadata.json",
        "split_protocol_manifest.json",
        "custom_ppo.pt",
        "validation_static_candidates.csv",
    ]
    if str(reward_loss_normalization) == "staticnorm_subtype":
        required.extend(
            [
                "reward_staticnorm_candidates.csv",
                "reward_staticnorm_normalizers.json",
            ]
        )
    return tuple(required)


def validate_control_source(
    *,
    source_dir: Path,
    truth_path: Path,
    sensor_cfg_path: Path,
    candidate_masks: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    required = control_source_required_files(str(args.reward_loss_normalization))
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"control source {source_dir} is missing: {missing}")
    metadata = load_json(source_dir / "v2_ppo_metadata.json")
    manifest = load_json(source_dir / "split_protocol_manifest.json")
    oracle_filename = "v2_tcn_oracle.pt" if str(metadata.get("oracle_type", "tcn")) == "tcn" else "v2_linear_oracle.npz"
    if not (source_dir / oracle_filename).is_file():
        raise FileNotFoundError(f"control source is missing frozen evaluator: {source_dir / oracle_filename}")
    if int(metadata.get("seed", -1)) != int(args.seed):
        raise ValueError(f"control source seed {metadata.get('seed')} != requested seed {args.seed}")
    source_truth = source_dir / "truth_v31_split.csv"
    if sha256_file(source_truth) != sha256_file(truth_path):
        raise ValueError("control source truth and requested truth CSV are not byte-identical")
    if Path(str(metadata.get("sensor_cfg", ""))).name != sensor_cfg_path.name:
        raise ValueError("control source sensor configuration does not match requested sensor configuration")
    constraints = dict(metadata.get("constraints", {}))
    expected_constraints = {
        "max_active": None if args.max_active is None else int(args.max_active),
        "per_step_budget": float(args.per_step_budget),
        "startup_peak_budget": float(args.startup_peak_budget),
        "required_sensor_ids": [str(value) for value in args.required_sensors],
    }
    for key, expected in expected_constraints.items():
        actual = constraints.get(key)
        if isinstance(expected, float):
            if actual is None or not np.isclose(float(actual), expected):
                raise ValueError(f"control source constraint {key}={actual} != {expected}")
        elif actual != expected:
            raise ValueError(f"control source constraint {key}={actual} != {expected}")
    if int(metadata.get("lookback", -1)) != int(args.lookback) or int(metadata.get("horizon", -1)) != int(args.horizon):
        raise ValueError("control source lookback/horizon does not match requested protocol")
    source_eval = tuple(int(value) for value in metadata.get("eval_start_indices", ()))
    requested_eval = tuple(int(value) for value in (args.eval_start_indices or ()))
    if requested_eval and source_eval != requested_eval:
        raise ValueError("control source final-test start indices do not match requested indices")
    partition = dict(metadata.get("partition_protocol", {}))
    source_static = tuple(int(value) for value in partition.get("static_selection_start_indices", ()))
    requested_static = tuple(int(value) for value in (args.static_selection_start_indices or ()))
    if requested_static and source_static != requested_static:
        raise ValueError("control source validation start indices do not match requested indices")

    import torch

    checkpoint = torch.load(source_dir / "custom_ppo.pt", map_location="cpu", weights_only=False)
    source_masks = np.asarray(checkpoint.get("candidate_masks"), dtype=bool)
    if source_masks.shape != candidate_masks.shape or not np.array_equal(source_masks, candidate_masks):
        raise ValueError("control source candidate masks do not match the requested action surface")
    return metadata, manifest


def load_control_oracle(
    *,
    source_dir: Path,
    metadata: dict[str, object],
    out_dir: Path,
    inference_device: str,
) -> tuple[object, Path, Path]:
    oracle_type = str(metadata.get("oracle_type", "tcn"))
    filename = "v2_tcn_oracle.pt" if oracle_type == "tcn" else "v2_linear_oracle.npz"
    source_path = source_dir / filename
    if not source_path.is_file():
        raise FileNotFoundError(f"control source oracle is missing: {source_path}")
    output_path = out_dir / filename
    shutil.copy2(source_path, output_path)
    if oracle_type == "tcn":
        oracle = TCNFrozenForecastOracle.load(output_path, device=str(inference_device))
    elif oracle_type == "linear":
        oracle = LinearFrozenForecastOracle.load(str(output_path))
    else:
        raise ValueError(f"Unsupported control source oracle_type={oracle_type!r}")
    return oracle, output_path, source_path


def sort_table_by_score(table: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if table.empty:
        return table
    requested = str(score_col)
    actual = requested if requested in table.columns else "oracle_loss_mean"
    values = pd.to_numeric(table.get(actual), errors="coerce")
    if not np.any(np.isfinite(values)):
        actual = "oracle_loss_mean"
        values = pd.to_numeric(table.get(actual), errors="coerce")
    ranked = table.copy()
    ranked["_selection_score"] = values
    sort_columns = ["_selection_score", "oracle_loss_mean"]
    if "action_idx" in ranked.columns:
        sort_columns.append("action_idx")
    ranked = ranked.sort_values(sort_columns, na_position="last").drop(
        columns=["_selection_score"]
    )
    return ranked.reset_index(drop=True)


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


def subtype_normalizer_tuple(normalizers: dict[str, float]) -> tuple[float, float, float]:
    return tuple(float(normalizers.get(col, float("nan"))) for col in SUBTYPE_LOSS_COLUMNS)


def missing_subtype_normalizer_columns(normalizers: dict[str, float]) -> list[str]:
    missing: list[str] = []
    for col in SUBTYPE_LOSS_COLUMNS:
        value = float(normalizers.get(col, float("nan")))
        if not np.isfinite(value) or value <= 0.0:
            missing.append(col)
    return missing


def subtype_fallback_start_indices(
    truth: pd.DataFrame,
    *,
    missing_columns: list[str],
    start_idx: int,
    end_idx: int,
    steps: int,
    per_subtype: int = 4,
) -> tuple[int, ...]:
    if not missing_columns or "event_subtype_id" not in truth.columns:
        return ()
    col_to_subtype = {f"oracle_loss_subtype_{label}": int(subtype_id) for subtype_id, label in SUBTYPE_LABELS.items()}
    subtype_ids = [col_to_subtype[col] for col in missing_columns if col in col_to_subtype]
    if not subtype_ids:
        return ()
    values = truth["event_subtype_id"].to_numpy(dtype=int)
    lower = max(0, int(start_idx))
    upper = min(len(truth), int(end_idx))
    rollout_steps = max(1, int(steps))
    if upper <= lower:
        lower, upper = 0, len(truth)
    max_start_global = max(0, len(truth) - rollout_steps - 1)
    if upper - lower >= rollout_steps:
        min_start = lower
        max_start = min(max_start_global, upper - rollout_steps)
    else:
        min_start = 0
        max_start = max_start_global
    starts: list[int] = []
    seen: set[int] = set()
    for subtype_id in subtype_ids:
        subtype_positions = np.flatnonzero(values[lower:upper] == int(subtype_id)) + lower
        if subtype_positions.size == 0:
            subtype_positions = np.flatnonzero(values == int(subtype_id))
        if subtype_positions.size == 0:
            continue
        pick_count = min(max(1, int(per_subtype)), int(subtype_positions.size))
        pick_offsets = np.linspace(0, int(subtype_positions.size) - 1, num=pick_count, dtype=int)
        for pos in subtype_positions[pick_offsets]:
            start = int(np.clip(int(pos) - rollout_steps // 2, min_start, max_start))
            if start not in seen:
                starts.append(start)
                seen.add(start)
    return tuple(starts)


def finite_median(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    return float(np.median(arr)) if arr.size else float("nan")


def add_staticnorm_macro(table: pd.DataFrame, normalizers: dict[str, float]) -> pd.DataFrame:
    if table.empty or not normalizers:
        return table
    result = table.copy()
    normalized_cols: list[str] = []
    for col in SUBTYPE_LOSS_COLUMNS:
        denom = float(normalizers.get(col, float("nan")))
        if col not in result.columns or not np.isfinite(denom) or denom <= 0.0:
            continue
        norm_col = f"{col}_staticnorm"
        result[norm_col] = pd.to_numeric(result[col], errors="coerce") / denom
        normalized_cols.append(norm_col)
    if normalized_cols:
        result[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN] = result[normalized_cols].apply(
            lambda row: finite_mean([float(value) for value in row.to_list()]),
            axis=1,
        )
    return result


def add_subtype_macro_metrics(metrics: dict[str, object], result: object, truth: pd.DataFrame) -> dict[str, object]:
    row = dict(metrics)
    losses = np.asarray(getattr(result, "oracle_losses", np.asarray([], dtype=float)), dtype=float).reshape(-1)
    step_indices = np.asarray(getattr(result, "step_indices", np.asarray([], dtype=int)), dtype=int).reshape(-1)
    event_flags = np.asarray(getattr(result, "event_flags", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)

    finite = np.isfinite(losses)
    if losses.size == event_flags.size:
        event_losses = losses[event_flags & finite]
        non_event_losses = losses[(~event_flags) & finite]
        row["oracle_loss_event"] = float(np.mean(event_losses)) if event_losses.size else float("nan")
        row["oracle_loss_non_event"] = float(np.mean(non_event_losses)) if non_event_losses.size else float("nan")

    subtype_losses: list[float] = []
    if "event_subtype_id" in truth.columns and losses.size == step_indices.size:
        valid = (step_indices >= 0) & (step_indices < len(truth))
        subtype_values = np.zeros_like(step_indices, dtype=int)
        subtype_values[valid] = truth["event_subtype_id"].to_numpy(dtype=int)[step_indices[valid]]
        for subtype_id, label in SUBTYPE_LABELS.items():
            subtype_mask = (subtype_values == int(subtype_id)) & finite
            subtype_loss = float(np.mean(losses[subtype_mask])) if np.any(subtype_mask) else float("nan")
            row[f"oracle_loss_subtype_{label}"] = subtype_loss
            row[f"steps_subtype_{label}"] = int(np.sum(subtype_values == int(subtype_id)))
            if np.isfinite(subtype_loss):
                subtype_losses.append(subtype_loss)
    else:
        for label in SUBTYPE_LABELS.values():
            row[f"oracle_loss_subtype_{label}"] = float("nan")
            row[f"steps_subtype_{label}"] = 0

    row[MACRO_SUBTYPE_LOSS_COLUMN] = finite_mean(subtype_losses)
    row[MACRO_SUBTYPE_COUNT_COLUMN] = int(len(subtype_losses))
    return row


def append_eval_row(rows: list[dict[str, object]], metrics: dict[str, object], result: object, truth: pd.DataFrame) -> None:
    rows.append(add_subtype_macro_metrics(metrics, result, truth))


def resolve_candidate_action_indices(
    sensors: list[object],
    candidate_masks: np.ndarray,
    mask_specs: list[list[str]],
    *,
    label: str,
) -> tuple[int, ...]:
    return tuple(
        resolve_candidate_action_index(
            sensors,
            candidate_masks,
            sensor_ids,
            label=f"{label}_{idx}",
        )
        for idx, sensor_ids in enumerate(mask_specs)
    )


def main() -> None:
    helpers = load_train_helpers()
    parser = argparse.ArgumentParser(description="Train the v2 custom masked action-embedding PPO.")
    parser.add_argument("--truth-csv", default="data/generated/v2_public_weather_truth_custom_ppo.csv")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Taishan"])
    parser.add_argument("--truth-steps", type=int, default=8192)
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
    parser.add_argument("--event-subtype-flux-latent-linear-scale", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-latent-linear-offset", type=float, default=1.5)
    parser.add_argument("--event-subtype-flux-latent-linear-clip", type=float, default=4.0)
    parser.add_argument("--event-subtype-thermal-latent-surface-scale-c", type=float, default=0.0)
    parser.add_argument("--event-subtype-latent-target-lag-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-lead-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-noise-std", type=float, default=0.08)
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", default="reports/v2_custom_ppo_probe/budget1p70_seed41")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--policy-checkpoint-source", default=None)
    parser.add_argument(
        "--evaluation-policy-mode",
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument("--evaluation-sampling-seed", type=int, default=None)
    parser.add_argument("--evaluation-sampling-temperature", type=float, default=1.0)
    parser.add_argument("--evaluation-temperature-candidates", nargs="*", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--oracle-type", choices=["linear", "tcn"], default="tcn")
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-event-fraction", type=float, default=0.50)
    parser.add_argument("--oracle-full-open-repeat", type=int, default=3)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-learning-rate", type=float, default=1e-3)
    parser.add_argument("--oracle-channels", type=int, default=64)
    parser.add_argument("--oracle-levels", type=int, default=3)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument(
        "--control-source-run-dir",
        default=None,
        help="Reuse truth-linked frozen evaluator and validation assets from an existing run.",
    )
    parser.add_argument(
        "--validate-control-source-only",
        action="store_true",
        help="Validate matched-control assets and exit before loading models or training.",
    )
    parser.add_argument("--oracle-loss-clip", type=float, default=10.0)
    parser.add_argument("--oracle-candidate-mask-repeat", type=int, default=0)
    parser.add_argument("--oracle-candidate-mask-limit", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-repeat", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-lookahead-steps", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-calm-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-particle-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-flux-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-thermal-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-disable-mask-channels", action="store_true")
    parser.add_argument("--target-weights", nargs="*", type=float, default=list(helpers.DEFAULT_TARGET_WEIGHTS))
    parser.add_argument("--target-scales", nargs="*", type=float, default=list(helpers.DEFAULT_TARGET_SCALES))
    parser.add_argument("--subtype-loss-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-particle-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-flux-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-thermal-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--train-episode-len", type=int, default=512)
    parser.add_argument("--train-start-indices", nargs="*", type=int, default=None)
    parser.add_argument("--train-start-min", type=int, default=None)
    parser.add_argument("--train-start-max", type=int, default=None)
    parser.add_argument("--normalization-start-idx", type=int, default=None)
    parser.add_argument("--normalization-end-idx", type=int, default=None)
    parser.add_argument("--oracle-start-idx", type=int, default=None)
    parser.add_argument("--oracle-end-idx", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--eval-start-indices", nargs="*", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--channel-marginal-entropy-coef", type=float, default=0.0)
    parser.add_argument(
        "--separate-actor-critic-grad-clip",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--awbc-coef", type=float, default=0.1)
    parser.add_argument("--awbc-decay-timesteps", type=int, default=0)
    parser.add_argument("--awbc-label-stride", type=int, default=4)
    parser.add_argument("--awbc-event-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-selection-interval-updates", type=int, default=0)
    parser.add_argument(
        "--checkpoint-selection-score",
        choices=[
            "oracle_loss_mean",
            "oracle_loss_macro_subtype_event",
            "oracle_loss_macro_subtype_event_staticnorm",
        ],
        default="oracle_loss_mean",
    )
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--bc-pretrain-epochs", type=int, default=4)
    parser.add_argument("--bc-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--bc-pretrain-loss-coef", type=float, default=1.0)
    parser.add_argument(
        "--bc-pretrain-target-mode",
        choices=["hard", "soft_forecast_value"],
        default="hard",
    )
    parser.add_argument("--bc-soft-temperature", type=float, default=1.0)
    parser.add_argument("--subtype-aux-coef", type=float, default=0.0)
    parser.add_argument("--subtype-aux-classes", type=int, default=4)
    parser.add_argument("--subtype-aux-lookahead-steps", type=int, default=0)
    parser.add_argument("--subtype-action-ce-coef", type=float, default=0.0)
    parser.add_argument(
        "--subtype-action-supervision-mode",
        choices=["exact_action", "positive_sensor_inclusion"],
        default="exact_action",
    )
    parser.add_argument("--subtype-action-event-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-action-margin-coef", type=float, default=0.0)
    parser.add_argument("--subtype-action-margin", type=float, default=0.5)
    parser.add_argument("--subtype-router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-router-min-confidence", type=float, default=0.0)
    parser.add_argument("--subtype-router-low-confidence-action", type=int, default=-1)
    parser.add_argument(
        "--awbc-teacher-mode",
        choices=[
            "oracle_greedy",
            "event_pair",
            "event_cyclic",
            "subtype_auto",
            "subtype_static_auto",
            "context_alert",
            "energy_mpc",
        ],
        default="oracle_greedy",
    )
    parser.add_argument("--awbc-teacher-calm-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-event-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-event-lookahead-steps", type=int, default=0)
    parser.add_argument("--awbc-teacher-alert-threshold", type=float, default=0.5)
    parser.add_argument("--awbc-teacher-energy-mpc-horizon", type=int, default=4)
    parser.add_argument("--awbc-teacher-energy-mpc-soc-bins", type=int, default=16)
    parser.add_argument("--awbc-teacher-energy-mpc-low-soc-ratio", type=float, default=0.25)
    parser.add_argument("--awbc-teacher-energy-mpc-high-soc-ratio", type=float, default=0.75)
    parser.add_argument("--awbc-teacher-energy-mpc-terminal-soc-weight", type=float, default=0.0)
    parser.add_argument("--awbc-teacher-energy-mpc-max-actions", type=int, default=0)
    parser.add_argument("--awbc-teacher-energy-mpc-low-power-action", type=int, default=-1)
    parser.add_argument("--awbc-teacher-calm-pool-spec", default=None)
    parser.add_argument("--awbc-teacher-event-pool-spec", default=None)
    parser.add_argument("--awbc-teacher-subtype-calm-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-particle-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-flux-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-thermal-sensors", nargs="*", default=None)
    parser.add_argument(
        "--awbc-teacher-auto-score-mode",
        choices=["raw", "staticnorm"],
        default="raw",
        help=(
            "Score family for subtype_static_auto teacher selection. Calm always uses "
            "oracle_loss_non_event; event subtypes use raw or static-normalized subtype losses."
        ),
    )
    parser.add_argument("--awbc-teacher-dwell-steps", type=int, default=1)
    parser.add_argument("--prior-kl-coef", type=float, default=1.0)
    parser.add_argument("--use-oracle-candidate-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.0)
    parser.add_argument("--candidate-prior-start-indices", nargs="*", type=int, default=None)
    parser.add_argument("--static-selection-start-indices", nargs="*", type=int, default=None)
    parser.add_argument("--static-selection-steps", type=int, default=512)
    parser.add_argument(
        "--static-selection-score",
        choices=[
            "oracle_loss_mean",
            "oracle_loss_macro_subtype_event",
            "oracle_loss_macro_subtype_event_staticnorm",
        ],
        default="oracle_loss_mean",
        help="Score used to choose validation_selected_static from feasible static candidates.",
    )
    parser.add_argument(
        "--metrics-sort-score",
        choices=[
            "oracle_loss_mean",
            "oracle_loss_macro_subtype_event",
            "oracle_loss_macro_subtype_event_staticnorm",
        ],
        default="oracle_loss_mean",
        help="Score used only for sorting v2_custom_ppo_metrics.csv.",
    )
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--greedy-lookahead-steps", type=int, default=4)
    parser.add_argument("--use-action-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-action-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nonlinear-action-embedding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trainable-action-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-aware-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-gated-actor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--context-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--context-feature-dim", type=int, default=0)
    parser.add_argument("--context-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--context-fusion-mode",
        choices=["concat", "gated_add", "subtype_moe"],
        default="concat",
    )
    parser.add_argument("--context-layer-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--soc-aux-horizon", type=int, default=0)
    parser.add_argument("--soc-aux-coef", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="Optional policy/training RNG seed; data generation and evaluation retain --seed.",
    )
    parser.add_argument("--per-step-budget", type=float, default=1.7)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument(
        "--max-active",
        type=int,
        default=None,
        help="Optional cardinality cap. Omit to let power constraints define the subset size.",
    )
    parser.add_argument("--required-sensors", nargs="*", default=list(helpers.DEFAULT_REQUIRED_SENSOR_IDS))
    parser.add_argument("--disable-coverage-groups", action="store_true")
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--event-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--event-subtype-particle-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--event-subtype-flux-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--event-subtype-thermal-reward-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--reward-loss-normalization",
        choices=["none", "staticnorm_subtype"],
        default="none",
        help="Optional normalization applied to oracle loss before PPO reward shaping.",
    )
    parser.add_argument(
        "--reward-proxy-mode",
        choices=["forecast", "forecast_gain", "aoi", "coverage", "uncertainty", "instant_error"],
        default="forecast",
        help=(
            "Training loss proxy for PPO reward. Final evaluation still reports the fixed forecast evaluator loss."
        ),
    )
    parser.add_argument("--energy-account", action="store_true")
    parser.add_argument("--energy-capacity", type=float, default=0.0)
    parser.add_argument("--initial-energy", type=float, default=0.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.0)
    parser.add_argument("--reserve-energy", type=float, default=0.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--soc-soft-penalty-buffer", type=float, default=0.0)
    parser.add_argument("--lambda-soc-soft-penalty", type=float, default=0.0)
    parser.add_argument("--lambda-duty-balance", type=float, default=0.0)
    parser.add_argument("--duty-balance-low", type=float, default=0.05)
    parser.add_argument("--duty-balance-high", type=float, default=0.95)
    parser.add_argument("--duty-balance-grace-steps", type=int, default=64)
    parser.add_argument("--duty-score-feedback", type=float, default=0.0)
    parser.add_argument("--duty-score-target", type=float, default=0.40)
    parser.add_argument("--duty-hard-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--duty-hard-low", type=float, default=0.08)
    parser.add_argument("--duty-hard-high", type=float, default=0.92)
    parser.add_argument("--duty-hard-score", type=float, default=8.0)
    parser.add_argument("--min-dwell-steps", type=int, default=1)
    parser.add_argument("--include-agent-cycle-phase", action="store_true")
    parser.add_argument("--agent-cycle-period-steps", type=int, default=0)
    parser.add_argument("--agent-cycle-dwell-steps", type=int, default=1)
    parser.add_argument("--include-observable-regime-belief", action="store_true")
    parser.add_argument("--regime-belief-lookback", type=int, default=6)
    parser.add_argument("--agent-context-columns", nargs="*", default=None)
    parser.add_argument("--include-event-flag-in-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-alert-context-features", action="store_true")
    parser.add_argument("--alert-context-columns", nargs="*", default=None)
    parser.add_argument("--alert-context-threshold", type=float, default=0.5)
    parser.add_argument("--alert-context-trend-lookback", type=int, default=6)
    parser.add_argument(
        "--measurement-update-mode",
        choices=["direct", "variance_weighted"],
        default="direct",
        help="How a new noisy measurement updates the carried estimator state.",
    )
    parser.add_argument("--eval-duty-constrained-baselines", action="store_true")
    parser.add_argument("--baseline-duty-hard-low", type=float, default=None)
    parser.add_argument("--baseline-duty-hard-high", type=float, default=None)
    parser.add_argument("--baseline-duty-hard-score", type=float, default=None)
    parser.add_argument("--baseline-duty-score-feedback", type=float, default=None)
    parser.add_argument("--primary-eval-duty-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-switch-limited-baselines", action="store_true")
    parser.add_argument("--baseline-min-dwell-steps", nargs="+", type=int, default=[6, 12])
    parser.add_argument("--ppo-max-candidate-warmup", type=int, default=-1)
    parser.add_argument("--event-start-prob", type=float, default=0.67)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    policy_seed = int(args.seed if args.policy_seed is None else args.policy_seed)

    target_weights = tuple(float(x) for x in args.target_weights)
    target_scales = tuple(float(x) for x in args.target_scales)
    if len(target_weights) != len(helpers.REWARD_TARGET_COLUMNS):
        raise ValueError(f"--target-weights must contain {len(helpers.REWARD_TARGET_COLUMNS)} values")
    if len(target_scales) != len(helpers.REWARD_TARGET_COLUMNS):
        raise ValueError(f"--target-scales must contain {len(helpers.REWARD_TARGET_COLUMNS)} values")
    subtype_particle_target_weights = helpers.optional_target_weights(
        args.subtype_particle_target_weights,
        name="--subtype-particle-target-weights",
    )
    subtype_flux_target_weights = helpers.optional_target_weights(
        args.subtype_flux_target_weights,
        name="--subtype-flux-target-weights",
    )
    subtype_thermal_target_weights = helpers.optional_target_weights(
        args.subtype_thermal_target_weights,
        name="--subtype-thermal-target-weights",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = helpers.ensure_truth(args)
    truth = helpers.ensure_state_columns(pd.read_csv(truth_path))
    sensor_cfg_path = resolve_sensor_cfg(str(args.sensor_cfg))
    sensors = load_sensor_specs(sensor_cfg_path)
    coverage_groups = () if bool(args.disable_coverage_groups) else helpers.DEFAULT_COVERAGE_GROUPS
    constraints = PowerConstraintsV2(
        max_active=None if args.max_active is None else int(args.max_active),
        per_step_budget=float(args.per_step_budget),
        startup_peak_budget=float(args.startup_peak_budget),
        required_sensor_ids=tuple(str(sensor_id) for sensor_id in args.required_sensors),
        coverage_groups=coverage_groups,
    )
    candidate_masks = helpers.build_projected_candidate_masks(
        sensors,
        constraints,
        max_candidate_warmup=None if int(args.ppo_max_candidate_warmup) < 0 else int(args.ppo_max_candidate_warmup),
    )
    control_source_dir = Path(args.control_source_run_dir).resolve() if args.control_source_run_dir else None
    control_source_metadata: dict[str, object] | None = None
    control_source_manifest: dict[str, object] | None = None
    if control_source_dir is not None:
        control_source_metadata, control_source_manifest = validate_control_source(
            source_dir=control_source_dir,
            truth_path=Path(truth_path),
            sensor_cfg_path=sensor_cfg_path,
            candidate_masks=candidate_masks,
            args=args,
        )
    if bool(args.validate_control_source_only):
        if control_source_dir is None or control_source_metadata is None:
            raise ValueError("--validate-control-source-only requires --control-source-run-dir")
        oracle_filename = (
            "v2_tcn_oracle.pt"
            if str(control_source_metadata.get("oracle_type", "tcn")) == "tcn"
            else "v2_linear_oracle.npz"
        )
        payload = {
            "status": "passed",
            "seed": int(args.seed),
            "control_source_run_dir": str(control_source_dir),
            "truth_sha256": sha256_file(truth_path),
            "oracle_sha256": sha256_file(control_source_dir / oracle_filename),
            "candidate_mask_count": int(candidate_masks.shape[0]),
            "eval_start_indices": [int(value) for value in (args.eval_start_indices or ())],
            "static_selection_start_indices": [int(value) for value in (args.static_selection_start_indices or ())],
        }
        preflight_path = out_dir / "control_source_preflight.json"
        preflight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(preflight_path)
        return
    awbc_teacher_calm_action = -1
    awbc_teacher_event_action = -1
    awbc_teacher_calm_actions: tuple[int, ...] = ()
    awbc_teacher_event_actions: tuple[int, ...] = ()
    awbc_teacher_subtype_calm_action = -1
    awbc_teacher_subtype_particle_action = -1
    awbc_teacher_subtype_flux_action = -1
    awbc_teacher_subtype_thermal_action = -1
    if str(args.awbc_teacher_mode) == "event_pair":
        awbc_teacher_calm_action = resolve_candidate_action_index(
            sensors,
            candidate_masks,
            args.awbc_teacher_calm_sensors,
            label="calm",
        )
        awbc_teacher_event_action = resolve_candidate_action_index(
            sensors,
            candidate_masks,
            args.awbc_teacher_event_sensors,
            label="event",
        )
    if str(args.awbc_teacher_mode) == "event_cyclic":
        calm_specs = parse_mask_pool_spec(args.awbc_teacher_calm_pool_spec)
        event_specs = parse_mask_pool_spec(args.awbc_teacher_event_pool_spec)
        if not calm_specs or not event_specs:
            raise ValueError("event_cyclic teacher requires calm and event pool specs")
        awbc_teacher_calm_actions = resolve_candidate_action_indices(
            sensors,
            candidate_masks,
            calm_specs,
            label="calm_pool",
        )
        awbc_teacher_event_actions = resolve_candidate_action_indices(
            sensors,
            candidate_masks,
            event_specs,
            label="event_pool",
        )
    if str(args.awbc_teacher_mode) in {"subtype_auto", "context_alert"}:
        if "event_subtype_id" not in truth.columns:
            raise ValueError("subtype_auto AWBC teacher requires truth column event_subtype_id")
        awbc_teacher_subtype_calm_action = resolve_candidate_action_index(
            sensors,
            candidate_masks,
            args.awbc_teacher_subtype_calm_sensors,
            label="subtype_calm",
        )
        awbc_teacher_subtype_particle_action = resolve_candidate_action_index(
            sensors,
            candidate_masks,
            args.awbc_teacher_subtype_particle_sensors,
            label="subtype_particle",
        )
        awbc_teacher_subtype_flux_action = resolve_candidate_action_index(
            sensors,
            candidate_masks,
            args.awbc_teacher_subtype_flux_sensors,
            label="subtype_flux",
        )
        awbc_teacher_subtype_thermal_action = resolve_candidate_action_index(
            sensors,
            candidate_masks,
            args.awbc_teacher_subtype_thermal_sensors,
            label="subtype_thermal",
        )
    if str(args.awbc_teacher_mode) == "subtype_static_auto" and "event_subtype_id" not in truth.columns:
        raise ValueError("subtype_static_auto AWBC teacher requires truth column event_subtype_id")
    if float(args.subtype_aux_coef) > 0.0 and "event_subtype_id" not in truth.columns:
        raise ValueError("subtype auxiliary loss requires truth column event_subtype_id")
    oracle_subtype_teacher_calm_mask = None
    oracle_subtype_teacher_particle_mask = None
    oracle_subtype_teacher_flux_mask = None
    oracle_subtype_teacher_thermal_mask = None
    if int(args.oracle_subtype_teacher_repeat) > 0:
        oracle_subtype_teacher_calm_mask = candidate_masks[
            resolve_candidate_action_index(
                sensors,
                candidate_masks,
                args.oracle_subtype_teacher_calm_sensors,
                label="oracle_subtype_calm",
            )
        ]
        oracle_subtype_teacher_particle_mask = candidate_masks[
            resolve_candidate_action_index(
                sensors,
                candidate_masks,
                args.oracle_subtype_teacher_particle_sensors,
                label="oracle_subtype_particle",
            )
        ]
        oracle_subtype_teacher_flux_mask = candidate_masks[
            resolve_candidate_action_index(
                sensors,
                candidate_masks,
                args.oracle_subtype_teacher_flux_sensors,
                label="oracle_subtype_flux",
            )
        ]
        if args.oracle_subtype_teacher_thermal_sensors:
            oracle_subtype_teacher_thermal_mask = candidate_masks[
                resolve_candidate_action_index(
                    sensors,
                    candidate_masks,
                    args.oracle_subtype_teacher_thermal_sensors,
                    label="oracle_subtype_thermal",
                )
            ]
    control_source_oracle_path: Path | None = None
    if control_source_dir is not None and control_source_metadata is not None:
        oracle, oracle_path, control_source_oracle_path = load_control_oracle(
            source_dir=control_source_dir,
            metadata=control_source_metadata,
            out_dir=out_dir,
            inference_device=str(args.oracle_inference_device),
        )
        args.oracle_type = str(control_source_metadata.get("oracle_type", args.oracle_type))
    else:
        oracle_truth = truth
        if args.oracle_start_idx is not None or args.oracle_end_idx is not None:
            oracle_start = int(args.oracle_start_idx or 0)
            oracle_end = int(args.oracle_end_idx or len(truth))
            if oracle_start < 0 or oracle_end <= oracle_start or oracle_end > len(truth):
                raise ValueError(f"Invalid oracle partition [{oracle_start}, {oracle_end}) for truth length {len(truth)}")
            oracle_truth = truth.iloc[oracle_start:oracle_end].reset_index(drop=True)
        oracle = helpers.train_oracle(
            oracle_truth,
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
            subtype_teacher_repeat=int(args.oracle_subtype_teacher_repeat),
            subtype_teacher_lookahead_steps=int(args.oracle_subtype_teacher_lookahead_steps),
            subtype_teacher_calm_mask=oracle_subtype_teacher_calm_mask,
            subtype_teacher_particle_mask=oracle_subtype_teacher_particle_mask,
            subtype_teacher_flux_mask=oracle_subtype_teacher_flux_mask,
            subtype_teacher_thermal_mask=oracle_subtype_teacher_thermal_mask,
            base_freq_s=int(args.freq_s),
            seed=int(args.seed),
        )
        oracle_path = out_dir / ("v2_tcn_oracle.pt" if args.oracle_type == "tcn" else "v2_linear_oracle.npz")
        oracle.save(str(oracle_path))
        if args.oracle_type == "tcn":
            oracle.to_device(str(args.oracle_inference_device))

    normalization_mean = None
    normalization_std = None
    normalization_start = 0
    normalization_end = len(truth)
    if args.normalization_start_idx is not None or args.normalization_end_idx is not None:
        normalization_start = int(args.normalization_start_idx or 0)
        normalization_end = int(args.normalization_end_idx or len(truth))
        if normalization_start < 0 or normalization_end <= normalization_start or normalization_end > len(truth):
            raise ValueError(
                f"Invalid normalization partition [{normalization_start}, {normalization_end}) for truth length {len(truth)}"
            )
        normalization_values = truth.iloc[normalization_start:normalization_end][list(helpers.STATE_COLUMNS)].to_numpy(dtype=float)
        normalization_mean = tuple(float(x) for x in np.mean(normalization_values, axis=0))
        normalization_std = tuple(float(x) for x in np.maximum(np.std(normalization_values, axis=0), 1e-6))
    process_values = truth.iloc[normalization_start:normalization_end][list(helpers.STATE_COLUMNS)].to_numpy(dtype=float)
    process_mean = np.asarray(
        normalization_mean if normalization_mean is not None else np.mean(process_values, axis=0),
        dtype=float,
    )
    process_std = np.asarray(
        normalization_std if normalization_std is not None else np.maximum(np.std(process_values, axis=0), 1e-6),
        dtype=float,
    )
    normalized_process_values = (process_values - process_mean.reshape(1, -1)) / process_std.reshape(1, -1)
    process_differences = np.diff(normalized_process_values, axis=0)
    uncertainty_process_variance = tuple(
        float(value)
        for value in np.clip(np.nanvar(process_differences, axis=0), 1e-6, 1.0)
    )
    train_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        reward_proxy_mode=str(args.reward_proxy_mode),
        lookback=int(args.lookback),
        episode_len=int(args.train_episode_len),
        seed=policy_seed,
        base_freq_s=int(args.freq_s),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
        event_reward_multiplier=float(args.event_reward_multiplier),
        event_subtype_particle_reward_multiplier=float(args.event_subtype_particle_reward_multiplier),
        event_subtype_flux_reward_multiplier=float(args.event_subtype_flux_reward_multiplier),
        event_subtype_thermal_reward_multiplier=float(args.event_subtype_thermal_reward_multiplier),
        lambda_duty_balance=float(args.lambda_duty_balance),
        duty_balance_low=float(args.duty_balance_low),
        duty_balance_high=float(args.duty_balance_high),
        duty_balance_grace_steps=int(args.duty_balance_grace_steps),
        duty_score_feedback=float(args.duty_score_feedback),
        duty_score_target=float(args.duty_score_target),
        duty_hard_guard=bool(args.duty_hard_guard),
        duty_hard_low=float(args.duty_hard_low),
        duty_hard_high=float(args.duty_hard_high),
        duty_hard_score=float(args.duty_hard_score),
        min_dwell_steps=max(1, int(args.min_dwell_steps)),
        include_agent_cycle_phase=bool(args.include_agent_cycle_phase),
        agent_cycle_period_steps=int(args.agent_cycle_period_steps),
        agent_cycle_dwell_steps=max(1, int(args.agent_cycle_dwell_steps)),
        include_observable_regime_belief=bool(args.include_observable_regime_belief),
        regime_belief_lookback=max(1, int(args.regime_belief_lookback)),
        agent_context_columns=tuple(str(x) for x in (args.agent_context_columns or ())),
        include_event_flag_in_state=bool(args.include_event_flag_in_state),
        include_alert_context_features=bool(args.include_alert_context_features),
        alert_context_columns=tuple(str(x) for x in (args.alert_context_columns or WarmupEnvConfig.alert_context_columns)),
        alert_context_threshold=float(args.alert_context_threshold),
        alert_context_trend_lookback=max(1, int(args.alert_context_trend_lookback)),
        uncertainty_process_variance=uncertainty_process_variance,
        measurement_update_mode=str(args.measurement_update_mode),
        **energy_kwargs(args),
    )
    reward_loss_normalizers: tuple[float, float, float] | None = None
    reward_loss_default_normalizer = 1.0
    reward_loss_normalizer_map: dict[str, float] = {}
    reward_staticnorm_table = None
    if str(args.reward_loss_normalization) == "staticnorm_subtype" and control_source_dir is not None:
        source_table_path = control_source_dir / "reward_staticnorm_candidates.csv"
        source_normalizer_path = control_source_dir / "reward_staticnorm_normalizers.json"
        reward_staticnorm_table = pd.read_csv(source_table_path)
        normalizer_payload = load_json(source_normalizer_path)
        stored_normalizers = dict(normalizer_payload.get("normalizers", {}))
        reward_loss_normalizer_map = {
            column: float(stored_normalizers.get(column, float("nan")))
            for column in SUBTYPE_LOSS_COLUMNS
        }
        reward_loss_normalizers = subtype_normalizer_tuple(reward_loss_normalizer_map)
        source_reward = dict((control_source_metadata or {}).get("reward_shaping", {}))
        reward_loss_default_normalizer = float(
            source_reward.get(
                "reward_loss_default_normalizer",
                finite_median(reward_staticnorm_table["oracle_loss_mean"]),
            )
        )
        if (
            any((not np.isfinite(value) or value <= 0.0) for value in reward_loss_normalizers)
            or not np.isfinite(reward_loss_default_normalizer)
            or reward_loss_default_normalizer <= 0.0
        ):
            raise ValueError("control source contains invalid staticnorm reward normalizers")
        shutil.copy2(source_table_path, out_dir / "reward_staticnorm_candidates.csv")
        shutil.copy2(source_normalizer_path, out_dir / "reward_staticnorm_normalizers.json")
        source_fallback = control_source_dir / "reward_staticnorm_fallback_candidates.csv"
        if source_fallback.is_file():
            shutil.copy2(source_fallback, out_dir / source_fallback.name)
        train_cfg = replace(
            train_cfg,
            oracle_loss_reward_normalizers=reward_loss_normalizers,
            oracle_loss_reward_default_normalizer=float(reward_loss_default_normalizer),
        )
    elif str(args.reward_loss_normalization) == "staticnorm_subtype":
        if not args.static_selection_start_indices:
            raise ValueError("staticnorm_subtype reward normalization requires --static-selection-start-indices")
        reward_norm_cfg = (
            train_cfg
            if bool(args.primary_eval_duty_guard)
            else replace(train_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
        )
        _, reward_staticnorm_table = build_oracle_candidate_prior(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=reward_norm_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            start_indices=tuple(int(x) for x in args.static_selection_start_indices),
            steps=int(args.static_selection_steps),
            scale=0.0,
        )
        reward_loss_normalizer_map = subtype_static_normalizers(reward_staticnorm_table)
        missing_normalizers = missing_subtype_normalizer_columns(reward_loss_normalizer_map)
        fallback_indices: tuple[int, ...] = ()
        fallback_normalizer_map: dict[str, float] = {}
        if missing_normalizers:
            fallback_indices = subtype_fallback_start_indices(
                truth,
                missing_columns=missing_normalizers,
                start_idx=normalization_start,
                end_idx=normalization_end,
                steps=int(args.static_selection_steps),
                per_subtype=max(1, min(4, len(tuple(args.static_selection_start_indices)))),
            )
            if fallback_indices:
                _, fallback_table = build_oracle_candidate_prior(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    cfg=reward_norm_cfg,
                    oracle=oracle,
                    candidate_masks=candidate_masks,
                    start_indices=fallback_indices,
                    steps=int(args.static_selection_steps),
                    scale=0.0,
                )
                fallback_table.to_csv(out_dir / "reward_staticnorm_fallback_candidates.csv", index=False)
                fallback_normalizer_map = subtype_static_normalizers(fallback_table)
                for col in missing_normalizers:
                    value = float(fallback_normalizer_map.get(col, float("nan")))
                    if np.isfinite(value) and value > 0.0:
                        reward_loss_normalizer_map[col] = value
        reward_staticnorm_table = add_staticnorm_macro(reward_staticnorm_table, reward_loss_normalizer_map)
        reward_staticnorm_table.to_csv(out_dir / "reward_staticnorm_candidates.csv", index=False)
        with open(out_dir / "reward_staticnorm_normalizers.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "normalizers": {col: float(reward_loss_normalizer_map.get(col, float("nan"))) for col in SUBTYPE_LOSS_COLUMNS},
                    "missing_from_static_selection": missing_normalizers,
                    "fallback_start_indices": [int(x) for x in fallback_indices],
                    "fallback_normalizers": {
                        col: float(fallback_normalizer_map.get(col, float("nan"))) for col in SUBTYPE_LOSS_COLUMNS
                    },
                    "normalization_start_idx": int(normalization_start),
                    "normalization_end_idx": int(normalization_end),
                    "static_selection_start_indices": [int(x) for x in tuple(args.static_selection_start_indices)],
                },
                f,
                indent=2,
            )
        reward_loss_normalizers = subtype_normalizer_tuple(reward_loss_normalizer_map)
        reward_loss_default_normalizer = finite_median(reward_staticnorm_table["oracle_loss_mean"])
        if (
            any((not np.isfinite(value) or value <= 0.0) for value in reward_loss_normalizers)
            or not np.isfinite(reward_loss_default_normalizer)
            or reward_loss_default_normalizer <= 0.0
        ):
            raise ValueError(
                "Could not derive finite positive staticnorm reward normalizers from validation static candidates"
            )
        train_cfg = replace(
            train_cfg,
            oracle_loss_reward_normalizers=reward_loss_normalizers,
            oracle_loss_reward_default_normalizer=float(reward_loss_default_normalizer),
        )
    awbc_teacher_auto_selection: dict[str, object] = {}
    if str(args.awbc_teacher_mode) == "subtype_static_auto":
        if not args.static_selection_start_indices:
            raise ValueError("subtype_static_auto AWBC teacher requires --static-selection-start-indices")
        auto_table = reward_staticnorm_table
        if auto_table is None:
            auto_cfg = (
                train_cfg
                if bool(args.primary_eval_duty_guard)
                else replace(train_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
            )
            _, auto_table = build_oracle_candidate_prior(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=auto_cfg,
                oracle=oracle,
                candidate_masks=candidate_masks,
                start_indices=tuple(int(x) for x in args.static_selection_start_indices),
                steps=int(args.static_selection_steps),
                scale=0.0,
            )
        auto_table = auto_table.copy()
        auto_table.to_csv(out_dir / "awbc_subtype_static_auto_candidates.csv", index=False)

        def select_auto_action(label: str, score_col: str) -> tuple[int, dict[str, object]]:
            effective_score = score_col if score_col in auto_table.columns else "oracle_loss_mean"
            values = pd.to_numeric(auto_table.get(effective_score), errors="coerce")
            if not np.any(np.isfinite(values)):
                effective_score = "oracle_loss_mean"
            ranked = sort_table_by_score(auto_table, effective_score)
            if ranked.empty:
                raise ValueError(f"Cannot select subtype_static_auto {label}: no static candidates")
            row = ranked.iloc[0]
            action_idx = int(row["action_idx"])
            return action_idx, {
                "action_idx": action_idx,
                "sensor_ids": str(row.get("sensor_ids", "")),
                "score_column": effective_score,
                "score": float(row.get(effective_score, float("nan"))),
                "oracle_loss_mean": float(row.get("oracle_loss_mean", float("nan"))),
                "oracle_loss_event": float(row.get("oracle_loss_event", float("nan"))),
                "oracle_loss_non_event": float(row.get("oracle_loss_non_event", float("nan"))),
            }

        subtype_suffix = "_staticnorm" if str(args.awbc_teacher_auto_score_mode) == "staticnorm" else ""
        awbc_teacher_subtype_calm_action, calm_auto = select_auto_action("calm", "oracle_loss_non_event")
        awbc_teacher_subtype_particle_action, particle_auto = select_auto_action(
            "particle",
            f"oracle_loss_subtype_particle{subtype_suffix}",
        )
        awbc_teacher_subtype_flux_action, flux_auto = select_auto_action(
            "flux",
            f"oracle_loss_subtype_flux{subtype_suffix}",
        )
        awbc_teacher_subtype_thermal_action, thermal_auto = select_auto_action(
            "thermal",
            f"oracle_loss_subtype_thermal{subtype_suffix}",
        )
        awbc_teacher_auto_selection = {
            "mode": str(args.awbc_teacher_mode),
            "auto_score_mode": str(args.awbc_teacher_auto_score_mode),
            "candidate_table": str(out_dir / "awbc_subtype_static_auto_candidates.csv"),
            "calm": calm_auto,
            "particle": particle_auto,
            "flux": flux_auto,
            "thermal": thermal_auto,
        }
        (out_dir / "awbc_subtype_static_auto_selection.json").write_text(
            json.dumps(awbc_teacher_auto_selection, indent=2),
            encoding="utf-8",
        )
    candidate_prior_logits = None
    candidate_prior_table = None
    if bool(args.use_oracle_candidate_prior):
        candidate_prior_cfg = replace(train_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
        prior_start_indices = (
            tuple(int(x) for x in args.candidate_prior_start_indices)
            if args.candidate_prior_start_indices
            else helpers.select_eval_start_indices(
                truth,
                steps=int(args.candidate_prior_steps),
                horizon=int(args.horizon),
                n_rollouts=int(args.candidate_prior_rollouts),
                event_fraction=float(args.eval_event_fraction),
                seed=int(args.seed) + 811,
            )
        )
        candidate_prior_logits, candidate_prior_table = build_oracle_candidate_prior(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=candidate_prior_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            start_indices=prior_start_indices,
            steps=int(args.candidate_prior_steps),
            scale=float(args.candidate_prior_scale),
        )
        candidate_prior_table.to_csv(out_dir / "custom_ppo_candidate_prior.csv", index=False)
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=train_cfg,
        oracle=oracle,
        candidate_masks=candidate_masks,
        cfg=CustomPPOConfig(
            total_timesteps=int(args.total_timesteps),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            channel_marginal_entropy_coef=float(args.channel_marginal_entropy_coef),
            separate_actor_critic_grad_clip=bool(args.separate_actor_critic_grad_clip),
            awbc_coef=float(args.awbc_coef),
            awbc_decay_timesteps=max(0, int(args.awbc_decay_timesteps)),
            awbc_label_stride=int(args.awbc_label_stride),
            awbc_event_only=bool(args.awbc_event_only),
            bc_pretrain_steps=int(args.bc_pretrain_steps),
            bc_pretrain_epochs=int(args.bc_pretrain_epochs),
            bc_pretrain_batch_size=int(args.bc_pretrain_batch_size),
            bc_pretrain_loss_coef=float(args.bc_pretrain_loss_coef),
            bc_pretrain_target_mode=str(args.bc_pretrain_target_mode),
            bc_soft_temperature=float(args.bc_soft_temperature),
            subtype_aux_coef=float(args.subtype_aux_coef),
            subtype_aux_classes=max(2, int(args.subtype_aux_classes)),
            subtype_aux_lookahead_steps=max(0, int(args.subtype_aux_lookahead_steps)),
            subtype_action_ce_coef=float(args.subtype_action_ce_coef),
            subtype_action_supervision_mode=str(args.subtype_action_supervision_mode),
            subtype_action_event_only=bool(args.subtype_action_event_only),
            subtype_action_margin_coef=float(args.subtype_action_margin_coef),
            subtype_action_margin=float(args.subtype_action_margin),
            subtype_router_enabled=bool(args.subtype_router),
            subtype_router_min_confidence=float(args.subtype_router_min_confidence),
            subtype_router_low_confidence_action=int(args.subtype_router_low_confidence_action),
            prior_kl_coef=float(args.prior_kl_coef),
            embed_dim=int(args.embed_dim),
            hidden_dim=int(args.hidden_dim),
            greedy_lookahead_steps=int(args.greedy_lookahead_steps),
            awbc_teacher_mode=str(args.awbc_teacher_mode),
            awbc_teacher_event_lookahead_steps=int(args.awbc_teacher_event_lookahead_steps),
            awbc_teacher_alert_threshold=float(args.awbc_teacher_alert_threshold),
            awbc_teacher_energy_mpc_horizon=int(args.awbc_teacher_energy_mpc_horizon),
            awbc_teacher_energy_mpc_soc_bins=int(args.awbc_teacher_energy_mpc_soc_bins),
            awbc_teacher_energy_mpc_low_soc_ratio=float(args.awbc_teacher_energy_mpc_low_soc_ratio),
            awbc_teacher_energy_mpc_high_soc_ratio=float(args.awbc_teacher_energy_mpc_high_soc_ratio),
            awbc_teacher_energy_mpc_terminal_soc_weight=float(args.awbc_teacher_energy_mpc_terminal_soc_weight),
            awbc_teacher_energy_mpc_max_actions=int(args.awbc_teacher_energy_mpc_max_actions),
            awbc_teacher_energy_mpc_low_power_action=int(args.awbc_teacher_energy_mpc_low_power_action),
            awbc_teacher_calm_action=int(awbc_teacher_calm_action),
            awbc_teacher_event_action=int(awbc_teacher_event_action),
            awbc_teacher_calm_actions=tuple(int(x) for x in awbc_teacher_calm_actions),
            awbc_teacher_event_actions=tuple(int(x) for x in awbc_teacher_event_actions),
            awbc_teacher_subtype_calm_action=int(awbc_teacher_subtype_calm_action),
            awbc_teacher_subtype_particle_action=int(awbc_teacher_subtype_particle_action),
            awbc_teacher_subtype_flux_action=int(awbc_teacher_subtype_flux_action),
            awbc_teacher_subtype_thermal_action=int(awbc_teacher_subtype_thermal_action),
            awbc_teacher_dwell_steps=int(args.awbc_teacher_dwell_steps),
            event_start_prob=float(args.event_start_prob),
            use_action_mask=bool(args.use_action_mask),
            use_action_embedding=bool(args.use_action_embedding),
            nonlinear_action_embedding=bool(args.nonlinear_action_embedding),
            trainable_action_prior=bool(args.trainable_action_prior),
            event_aware_critic=bool(args.event_aware_critic),
            event_gated_actor=bool(args.event_gated_actor),
            context_encoder_enabled=bool(args.context_encoder),
            context_feature_dim=max(0, int(args.context_feature_dim)),
            context_hidden_dim=max(1, int(args.context_hidden_dim)),
            context_fusion_mode=str(args.context_fusion_mode),
            context_layer_norm=bool(args.context_layer_norm),
            temporal_encoder_enabled=bool(args.temporal_encoder),
            temporal_history_steps=int(args.lookback),
            temporal_state_dim=len(helpers.STATE_COLUMNS),
            temporal_hidden_dim=max(1, int(args.temporal_hidden_dim)),
            soc_aux_horizon=int(args.soc_aux_horizon),
            soc_aux_coef=float(args.soc_aux_coef),
            device=str(args.device),
            seed=policy_seed,
            history_path=str(out_dir / "custom_ppo_training_history_live.json"),
            train_start_indices=tuple(int(x) for x in (args.train_start_indices or ())),
            train_start_min=args.train_start_min,
            train_start_max=args.train_start_max,
        ),
        candidate_prior_logits=candidate_prior_logits,
    )
    checkpoint_selection_rows: list[dict[str, float | int]] = []
    best_checkpoint_state = None
    best_checkpoint_score = float("inf")
    best_checkpoint_update = 0
    checkpoint_interval = max(0, int(args.checkpoint_selection_interval_updates))
    checkpoint_starts = tuple(int(x) for x in (args.static_selection_start_indices or ()))
    checkpoint_score_name = str(args.checkpoint_selection_score)
    checkpoint_normalizers: dict[str, float] = {}
    if (checkpoint_interval > 0 or args.evaluation_temperature_candidates) and control_source_dir is not None:
        checkpoint_static_path = control_source_dir / "validation_static_candidates.csv"
        if checkpoint_static_path.is_file():
            checkpoint_normalizers = subtype_static_normalizers(pd.read_csv(checkpoint_static_path))
    if checkpoint_score_name == STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN:
        if not checkpoint_normalizers:
            raise ValueError(
                "static-normalized macro checkpoint selection requires a control source "
                "with validation_static_candidates.csv"
            )
    checkpoint_cfg = replace(
        train_cfg,
        episode_len=int(args.static_selection_steps),
        seed=int(args.seed) + 8500,
        duty_score_feedback=0.0,
        duty_hard_guard=False,
    )

    def select_checkpoint(
        current_trainer: CustomPPO,
        update_idx: int,
        timesteps: int,
        _metrics: dict[str, float | int],
    ) -> None:
        nonlocal best_checkpoint_state, best_checkpoint_score, best_checkpoint_update
        if checkpoint_interval <= 0 or not checkpoint_starts:
            return
        final_update = int(timesteps) >= int(args.total_timesteps)
        if int(update_idx) % checkpoint_interval != 0 and not final_update:
            return
        selection_result, selection_metrics = evaluate_custom_ppo(
            trainer=current_trainer,
            truth_df=truth,
            sensor_specs=sensors,
            constraints=constraints,
            cfg=checkpoint_cfg,
            oracle=oracle,
            steps=int(args.static_selection_steps),
            start_indices=checkpoint_starts,
            policy_name="custom_ppo_checkpoint_selection",
        )
        selection_metrics = add_subtype_macro_metrics(selection_metrics, selection_result, truth)
        if checkpoint_normalizers:
            selection_metrics = add_staticnorm_macro(
                pd.DataFrame([selection_metrics]), checkpoint_normalizers
            ).iloc[0].to_dict()
        score = float(selection_metrics[checkpoint_score_name])
        checkpoint_selection_rows.append(
            {
                "update": int(update_idx),
                "timesteps": int(timesteps),
                "oracle_loss_mean": float(selection_metrics["oracle_loss_mean"]),
                "oracle_loss_macro_subtype_event": float(
                    selection_metrics["oracle_loss_macro_subtype_event"]
                ),
                "oracle_loss_macro_subtype_event_staticnorm": float(
                    selection_metrics.get(
                        "oracle_loss_macro_subtype_event_staticnorm",
                        float("nan"),
                    )
                ),
                "selection_score_name": checkpoint_score_name,
                "selection_score": score,
            }
        )
        if np.isfinite(score) and score < best_checkpoint_score:
            best_checkpoint_score = score
            best_checkpoint_update = int(update_idx)
            best_checkpoint_state = copy.deepcopy(current_trainer.model.state_dict())

    if args.policy_checkpoint_source:
        trainer.load_policy_checkpoint(Path(args.policy_checkpoint_source))
    else:
        trainer.train(on_update=select_checkpoint)
    if best_checkpoint_state is not None:
        trainer.model.load_state_dict(best_checkpoint_state)
    if checkpoint_selection_rows:
        pd.DataFrame(checkpoint_selection_rows).to_csv(
            out_dir / "custom_ppo_checkpoint_selection.csv", index=False
        )
    selected_evaluation_temperature = float(args.evaluation_sampling_temperature)
    temperature_selection_rows: list[dict[str, float]] = []
    temperature_candidates = tuple(float(x) for x in (args.evaluation_temperature_candidates or ()))
    if temperature_candidates:
        if not checkpoint_starts or not checkpoint_normalizers:
            raise ValueError(
                "evaluation temperature selection requires validation starts and static normalizers"
            )
        if control_source_dir is None:
            raise ValueError("evaluation temperature selection requires a control source")
        static_table = pd.read_csv(control_source_dir / "validation_static_candidates.csv")
        static_ordinary = float(static_table["oracle_loss_mean"].min())
        static_macro = float(static_table[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN].min())
        best_temperature_score = float("inf")
        for candidate_idx, temperature in enumerate(temperature_candidates):
            result, metrics = evaluate_custom_ppo(
                trainer=trainer,
                truth_df=truth,
                sensor_specs=sensors,
                constraints=constraints,
                cfg=checkpoint_cfg,
                oracle=oracle,
                steps=int(args.static_selection_steps),
                start_indices=checkpoint_starts,
                policy_name="custom_ppo_temperature_selection",
                deterministic=float(temperature) <= 0.0,
                sampling_seed=int(policy_seed) + 930_000 + int(candidate_idx),
                sampling_temperature=max(float(temperature), 1.0e-6),
            )
            metrics = add_subtype_macro_metrics(metrics, result, truth)
            metrics = add_staticnorm_macro(
                pd.DataFrame([metrics]), checkpoint_normalizers
            ).iloc[0].to_dict()
            score = 0.5 * (
                float(metrics["oracle_loss_mean"]) / max(static_ordinary, 1.0e-8)
                + float(metrics[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN]) / max(static_macro, 1.0e-8)
            )
            temperature_selection_rows.append(
                {
                    "temperature": float(temperature),
                    "oracle_loss_mean": float(metrics["oracle_loss_mean"]),
                    STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN: float(
                        metrics[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN]
                    ),
                    "relative_joint_score": float(score),
                }
            )
            if score < best_temperature_score:
                best_temperature_score = float(score)
                selected_evaluation_temperature = float(temperature)
        pd.DataFrame(temperature_selection_rows).to_csv(
            out_dir / "custom_ppo_temperature_selection.csv", index=False
        )
    model_path = out_dir / "custom_ppo.pt"
    trainer.save(model_path)
    if args.checkpoint_path:
        trainer.save(Path(args.checkpoint_path))
    trainer.save_history(out_dir / "custom_ppo_training_history.json")
    write_training_log(trainer.history, out_dir / "custom_ppo_training_log.csv")

    if args.eval_start_indices:
        eval_start_indices = tuple(int(x) for x in args.eval_start_indices)
    else:
        eval_start_indices = helpers.select_eval_start_indices(
            truth,
            steps=int(args.eval_steps),
            horizon=int(args.horizon),
            n_rollouts=int(args.eval_rollouts),
            event_fraction=float(args.eval_event_fraction),
            seed=int(args.seed) + 1777,
        )
    eval_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        reward_proxy_mode=str(args.reward_proxy_mode),
        lookback=int(args.lookback),
        episode_len=int(args.eval_steps),
        seed=int(args.seed) + 9000,
        base_freq_s=int(args.freq_s),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
        event_reward_multiplier=float(args.event_reward_multiplier),
        event_subtype_particle_reward_multiplier=float(args.event_subtype_particle_reward_multiplier),
        event_subtype_flux_reward_multiplier=float(args.event_subtype_flux_reward_multiplier),
        event_subtype_thermal_reward_multiplier=float(args.event_subtype_thermal_reward_multiplier),
        lambda_duty_balance=float(args.lambda_duty_balance),
        duty_balance_low=float(args.duty_balance_low),
        duty_balance_high=float(args.duty_balance_high),
        duty_balance_grace_steps=int(args.duty_balance_grace_steps),
        duty_score_feedback=float(args.duty_score_feedback),
        duty_score_target=float(args.duty_score_target),
        duty_hard_guard=bool(args.duty_hard_guard),
        duty_hard_low=float(args.duty_hard_low),
        duty_hard_high=float(args.duty_hard_high),
        duty_hard_score=float(args.duty_hard_score),
        min_dwell_steps=max(1, int(args.min_dwell_steps)),
        include_agent_cycle_phase=bool(args.include_agent_cycle_phase),
        agent_cycle_period_steps=int(args.agent_cycle_period_steps),
        agent_cycle_dwell_steps=max(1, int(args.agent_cycle_dwell_steps)),
        include_observable_regime_belief=bool(args.include_observable_regime_belief),
        regime_belief_lookback=max(1, int(args.regime_belief_lookback)),
        agent_context_columns=tuple(str(x) for x in (args.agent_context_columns or ())),
        include_event_flag_in_state=bool(args.include_event_flag_in_state),
        include_alert_context_features=bool(args.include_alert_context_features),
        alert_context_columns=tuple(str(x) for x in (args.alert_context_columns or WarmupEnvConfig.alert_context_columns)),
        alert_context_threshold=float(args.alert_context_threshold),
        alert_context_trend_lookback=max(1, int(args.alert_context_trend_lookback)),
        uncertainty_process_variance=uncertainty_process_variance,
        measurement_update_mode=str(args.measurement_update_mode),
        oracle_loss_reward_normalizers=reward_loss_normalizers,
        oracle_loss_reward_default_normalizer=float(reward_loss_default_normalizer),
        **energy_kwargs(args),
    )
    baseline_eval_cfg = (
        eval_cfg
        if bool(args.primary_eval_duty_guard)
        else replace(eval_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
    )
    rows = []
    custom_result, custom_metrics = evaluate_custom_ppo(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=int(args.eval_steps),
        start_indices=eval_start_indices,
        deterministic=(
            selected_evaluation_temperature <= 0.0
            if temperature_candidates
            else str(args.evaluation_policy_mode) == "deterministic"
        ),
        sampling_seed=args.evaluation_sampling_seed,
        sampling_temperature=max(selected_evaluation_temperature, 1.0e-6),
    )
    append_eval_row(rows, custom_metrics, custom_result, truth)
    save_rollout_npz(
        out_dir / "rollout_custom_ppo.npz",
        custom_result,
        sensor_ids=[s.sensor_id for s in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )

    eval_policy_names = ["custom_ppo"]
    dwell_steps = tuple(sorted({max(1, int(value)) for value in args.baseline_min_dwell_steps}))
    if bool(args.eval_switch_limited_baselines):
        for dwell in dwell_steps:
            dwell_name = f"custom_ppo_dwell{int(dwell)}"
            result, metrics = evaluate_custom_ppo(
                trainer=trainer,
                truth_df=truth,
                sensor_specs=sensors,
                constraints=constraints,
                cfg=eval_cfg,
                oracle=oracle,
                steps=int(args.eval_steps),
                start_indices=eval_start_indices,
                policy_name=dwell_name,
                min_dwell_steps=int(dwell),
                deterministic=(
                    selected_evaluation_temperature <= 0.0
                    if temperature_candidates
                    else str(args.evaluation_policy_mode) == "deterministic"
                ),
                sampling_seed=args.evaluation_sampling_seed,
                sampling_temperature=max(selected_evaluation_temperature, 1.0e-6),
            )
            append_eval_row(rows, metrics, result, truth)
            eval_policy_names.append(dwell_name)
            save_rollout_npz(
                out_dir / f"rollout_{dwell_name}.npz",
                result,
                sensor_ids=[s.sensor_id for s in sensors],
                state_columns=helpers.STATE_COLUMNS,
            )

    oracle_static_summary = None
    oracle_static_mask: tuple[bool, ...] | None = None
    selected_static_table = candidate_prior_table
    selected_static_source = "candidate_prior_min_oracle_loss"
    selected_static_name = "oracle_static_projected"
    if args.static_selection_start_indices and control_source_dir is not None:
        source_static_path = control_source_dir / "validation_static_candidates.csv"
        selected_static_table = pd.read_csv(source_static_path)
        shutil.copy2(source_static_path, out_dir / "validation_static_candidates.csv")
        selected_static_source = f"control_source_validation_min_{args.static_selection_score}"
        selected_static_name = "validation_selected_static"
    elif args.static_selection_start_indices:
        static_selection_cfg = (
            train_cfg
            if bool(args.primary_eval_duty_guard)
            else replace(train_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
        )
        _, selected_static_table = build_oracle_candidate_prior(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=static_selection_cfg,
            oracle=oracle,
            candidate_masks=candidate_masks,
            start_indices=tuple(int(x) for x in args.static_selection_start_indices),
            steps=int(args.static_selection_steps),
            scale=0.0,
        )
        if reward_loss_normalizer_map:
            selected_static_table = add_staticnorm_macro(selected_static_table, reward_loss_normalizer_map)
        selected_static_table = sort_table_by_score(selected_static_table, str(args.static_selection_score))
        selected_static_table.to_csv(out_dir / "validation_static_candidates.csv", index=False)
        selected_static_source = f"validation_min_{args.static_selection_score}"
        selected_static_name = "validation_selected_static"
    if selected_static_table is not None and not selected_static_table.empty:
        selected_static_table = sort_table_by_score(selected_static_table, str(args.static_selection_score))
        best_prior = selected_static_table.iloc[0]
        best_action_idx = int(best_prior["action_idx"])
        oracle_static_mask = tuple(bool(x) for x in np.asarray(candidate_masks[best_action_idx], dtype=bool))
        oracle_static_policy = StaticMaskPolicy(mask=oracle_static_mask, name=selected_static_name)
        oracle_static_result, oracle_static_metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=baseline_eval_cfg,
            oracle=oracle,
            policy=oracle_static_policy,
            steps=int(args.eval_steps),
            start_indices=eval_start_indices,
        )
        append_eval_row(rows, oracle_static_metrics, oracle_static_result, truth)
        eval_policy_names.append(str(oracle_static_policy.name))
        save_rollout_npz(
            out_dir / f"rollout_{selected_static_name}.npz",
            oracle_static_result,
            sensor_ids=[s.sensor_id for s in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )
        oracle_static_summary = {
            "source": selected_static_source,
            "action_idx": best_action_idx,
            "sensor_ids": str(best_prior.get("sensor_ids", "")),
            "static_selection_score": str(args.static_selection_score),
            "candidate_prior_oracle_loss_mean": float(best_prior.get("oracle_loss_mean", float("nan"))),
            "candidate_prior_oracle_loss_macro_subtype_event": float(
                best_prior.get(MACRO_SUBTYPE_LOSS_COLUMN, float("nan"))
            ),
            "candidate_prior_oracle_loss_macro_subtype_event_staticnorm": float(
                best_prior.get(STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN, float("nan"))
            ),
            "candidate_prior_power_mean": float(best_prior.get("power_mean", float("nan"))),
        }

    full_open_result, full_open_metrics = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=PowerConstraintsV2(),
        cfg=baseline_eval_cfg,
        oracle=oracle,
        policy=helpers.FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        steps=int(args.eval_steps),
        start_indices=eval_start_indices,
    )
    append_eval_row(rows, full_open_metrics, full_open_result, truth)
    eval_policy_names.append("full_open_unconstrained")
    save_rollout_npz(
        out_dir / "rollout_full_open_unconstrained.npz",
        full_open_result,
        sensor_ids=[s.sensor_id for s in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )

    for policy in helpers.default_policies(len(sensors), seed=int(args.seed) + 100):
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=baseline_eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(args.eval_steps),
            start_indices=eval_start_indices,
        )
        append_eval_row(rows, metrics, result, truth)
        eval_policy_names.append(str(result.policy_name))
        save_rollout_npz(
            out_dir / f"rollout_{result.policy_name}.npz",
            result,
            sensor_ids=[s.sensor_id for s in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )

    constrained_baseline_settings = {
        "enabled": bool(args.eval_duty_constrained_baselines),
        "duty_hard_low": float(args.baseline_duty_hard_low)
        if args.baseline_duty_hard_low is not None
        else float(args.duty_hard_low),
        "duty_hard_high": float(args.baseline_duty_hard_high)
        if args.baseline_duty_hard_high is not None
        else float(args.duty_hard_high),
        "duty_hard_score": float(args.baseline_duty_hard_score)
        if args.baseline_duty_hard_score is not None
        else float(args.duty_hard_score),
        "duty_score_feedback": float(args.baseline_duty_score_feedback)
        if args.baseline_duty_score_feedback is not None
        else float(args.duty_score_feedback),
    }
    if bool(args.eval_duty_constrained_baselines):
        constrained_baseline_cfg = replace(
            baseline_eval_cfg,
            duty_score_feedback=float(constrained_baseline_settings["duty_score_feedback"]),
            duty_hard_guard=True,
            duty_hard_low=float(constrained_baseline_settings["duty_hard_low"]),
            duty_hard_high=float(constrained_baseline_settings["duty_hard_high"]),
            duty_hard_score=float(constrained_baseline_settings["duty_hard_score"]),
        )
        if oracle_static_mask is not None:
            constrained_static_name = f"duty_constrained_{selected_static_name}"
            constrained_static_policy = StaticMaskPolicy(mask=oracle_static_mask, name=constrained_static_name)
            result, metrics = helpers.evaluate_score_policy_over_starts(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=constrained_baseline_cfg,
                oracle=oracle,
                policy=constrained_static_policy,
                steps=int(args.eval_steps),
                start_indices=eval_start_indices,
            )
            append_eval_row(rows, metrics, result, truth)
            eval_policy_names.append(str(result.policy_name))
            save_rollout_npz(
                out_dir / f"rollout_{result.policy_name}.npz",
                result,
                sensor_ids=[s.sensor_id for s in sensors],
                state_columns=helpers.STATE_COLUMNS,
            )
        for policy in helpers.default_policies(len(sensors), seed=int(args.seed) + 10100):
            original_name = str(policy.name)
            policy.name = f"duty_constrained_{original_name}"
            result, metrics = helpers.evaluate_score_policy_over_starts(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=constrained_baseline_cfg,
                oracle=oracle,
                policy=policy,
                steps=int(args.eval_steps),
                start_indices=eval_start_indices,
            )
            append_eval_row(rows, metrics, result, truth)
            eval_policy_names.append(str(result.policy_name))
            save_rollout_npz(
                out_dir / f"rollout_{result.policy_name}.npz",
                result,
                sensor_ids=[s.sensor_id for s in sensors],
                state_columns=helpers.STATE_COLUMNS,
            )

    dynamic_policy_names = {"round_robin", "aoi", "random"}
    if bool(args.eval_switch_limited_baselines):
        for dwell in dwell_steps:
            for policy in helpers.default_policies(len(sensors), seed=int(args.seed) + 20100 + dwell):
                original_name = str(policy.name)
                if original_name not in dynamic_policy_names:
                    continue
                wrapped = MinDwellPolicyWrapper(
                    base_policy=policy,
                    min_dwell_steps=int(dwell),
                    name=f"dwell{int(dwell)}_{original_name}",
                )
                result, metrics = helpers.evaluate_score_policy_over_starts(
                    truth=truth,
                    sensors=sensors,
                    constraints=constraints,
                    cfg=baseline_eval_cfg,
                    oracle=oracle,
                    policy=wrapped,
                    steps=int(args.eval_steps),
                    start_indices=eval_start_indices,
                )
                append_eval_row(rows, metrics, result, truth)
                eval_policy_names.append(str(result.policy_name))
                save_rollout_npz(
                    out_dir / f"rollout_{result.policy_name}.npz",
                    result,
                    sensor_ids=[s.sensor_id for s in sensors],
                    state_columns=helpers.STATE_COLUMNS,
                )

        if bool(args.eval_duty_constrained_baselines):
            for dwell in dwell_steps:
                for policy in helpers.default_policies(len(sensors), seed=int(args.seed) + 30100 + dwell):
                    original_name = str(policy.name)
                    if original_name not in dynamic_policy_names:
                        continue
                    wrapped = MinDwellPolicyWrapper(
                        base_policy=policy,
                        min_dwell_steps=int(dwell),
                        name=f"duty_dwell{int(dwell)}_{original_name}",
                    )
                    result, metrics = helpers.evaluate_score_policy_over_starts(
                        truth=truth,
                        sensors=sensors,
                        constraints=constraints,
                        cfg=constrained_baseline_cfg,
                        oracle=oracle,
                        policy=wrapped,
                        steps=int(args.eval_steps),
                        start_indices=eval_start_indices,
                    )
                    append_eval_row(rows, metrics, result, truth)
                    eval_policy_names.append(str(result.policy_name))
                    save_rollout_npz(
                        out_dir / f"rollout_{result.policy_name}.npz",
                        result,
                        sensor_ids=[s.sensor_id for s in sensors],
                        state_columns=helpers.STATE_COLUMNS,
                    )

    metrics_frame = pd.DataFrame(rows)
    metrics_normalizers = reward_loss_normalizer_map or subtype_static_normalizers(selected_static_table)
    metrics_frame = add_staticnorm_macro(metrics_frame, metrics_normalizers)
    metrics = sort_table_by_score(metrics_frame, str(args.metrics_sort_score))
    metrics.to_csv(out_dir / "v2_custom_ppo_metrics.csv", index=False)
    if control_source_metadata is not None:
        oracle_rollout_summary = dict(control_source_metadata.get("oracle_pretrain_rollout_summary", {}))
    else:
        oracle_policy_specs = helpers.build_oracle_policy_specs(
            len(sensors),
            constraints,
            seed=int(args.seed),
            full_open_repeat=int(args.oracle_full_open_repeat),
            candidate_masks=candidate_masks,
            candidate_mask_repeat=int(args.oracle_candidate_mask_repeat),
            candidate_mask_limit=int(args.oracle_candidate_mask_limit),
        )
        oracle_rollout_summary = helpers.summarize_oracle_policy_specs(
            oracle_policy_specs,
            rollouts_per_policy=int(args.oracle_rollouts_per_policy),
            per_rollout_steps=helpers.oracle_per_rollout_steps(
                lookback=int(args.lookback),
                horizon=int(args.horizon),
                rollout_steps=int(args.oracle_rollout_steps),
                rollouts_per_policy=int(args.oracle_rollouts_per_policy),
            ),
        )
    metadata = {
        "truth_csv": str(truth_path),
        "sensor_cfg": str(sensor_cfg_path),
        "oracle_path": str(oracle_path),
        "oracle_type": str(args.oracle_type),
        "oracle_rollout_steps": int(args.oracle_rollout_steps),
        "oracle_rollouts_per_policy": int(args.oracle_rollouts_per_policy),
        "oracle_event_fraction": float(args.oracle_event_fraction),
        "oracle_full_open_repeat": int(args.oracle_full_open_repeat),
        "oracle_candidate_mask_repeat": int(args.oracle_candidate_mask_repeat),
        "oracle_candidate_mask_limit": int(args.oracle_candidate_mask_limit),
        "oracle_subtype_teacher_repeat": int(args.oracle_subtype_teacher_repeat),
        "oracle_subtype_teacher_lookahead_steps": int(args.oracle_subtype_teacher_lookahead_steps),
        "oracle_subtype_teacher_sensors": {
            "calm": [str(x) for x in (args.oracle_subtype_teacher_calm_sensors or ())],
            "particle": [str(x) for x in (args.oracle_subtype_teacher_particle_sensors or ())],
            "flux": [str(x) for x in (args.oracle_subtype_teacher_flux_sensors or ())],
            "thermal": [str(x) for x in (args.oracle_subtype_teacher_thermal_sensors or ())],
        },
        "oracle_pretrain_rollout_summary": oracle_rollout_summary,
        "control_source": {
            "enabled": control_source_dir is not None,
            "run_dir": "" if control_source_dir is None else str(control_source_dir),
            "truth_sha256": sha256_file(truth_path),
            "metadata_path": "" if control_source_dir is None else str(control_source_dir / "v2_ppo_metadata.json"),
            "metadata_sha256": ""
            if control_source_dir is None
            else sha256_file(control_source_dir / "v2_ppo_metadata.json"),
            "manifest_path": "" if control_source_dir is None else str(control_source_dir / "split_protocol_manifest.json"),
            "manifest_sha256": ""
            if control_source_dir is None
            else sha256_file(control_source_dir / "split_protocol_manifest.json"),
            "source_oracle_path": "" if control_source_oracle_path is None else str(control_source_oracle_path),
            "source_oracle_sha256": ""
            if control_source_oracle_path is None
            else sha256_file(control_source_oracle_path),
            "copied_oracle_sha256": sha256_file(oracle_path),
            "source_manifest": control_source_manifest,
        },
        "oracle_inference_device": str(args.oracle_inference_device),
        "oracle_use_mask_channels": not bool(args.oracle_disable_mask_channels),
        "reward_target_columns": list(helpers.REWARD_TARGET_COLUMNS),
        "target_weights": list(target_weights),
        "target_scales": list(target_scales),
        "subtype_loss_weighting": bool(args.subtype_loss_weighting),
        "subtype_particle_target_weights": None if subtype_particle_target_weights is None else list(subtype_particle_target_weights),
        "subtype_flux_target_weights": None if subtype_flux_target_weights is None else list(subtype_flux_target_weights),
        "subtype_thermal_target_weights": None if subtype_thermal_target_weights is None else list(subtype_thermal_target_weights),
        "model_path": str(model_path),
        "policy_checkpoint_source": str(args.policy_checkpoint_source or ""),
        "evaluation_policy_mode": str(args.evaluation_policy_mode),
        "evaluation_sampling_seed": args.evaluation_sampling_seed,
        "evaluation_sampling_temperature": float(selected_evaluation_temperature),
        "evaluation_temperature_candidates": [float(x) for x in temperature_candidates],
        "eval_policies": eval_policy_names,
        "seed": int(args.seed),
        "policy_seed": int(policy_seed),
        "freq_s": int(args.freq_s),
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "eval_rollouts": int(args.eval_rollouts),
        "eval_event_fraction": float(args.eval_event_fraction),
        "eval_start_indices": [int(x) for x in eval_start_indices],
        "static_selection_score": str(args.static_selection_score),
        "metrics_sort_score": str(args.metrics_sort_score),
        "train_start_indices": [int(x) for x in (args.train_start_indices or ())],
        "train_start_min": args.train_start_min,
        "train_start_max": args.train_start_max,
        "event_reward_multiplier": float(args.event_reward_multiplier),
        "event_subtype_reward_multipliers": {
            "particle": float(args.event_subtype_particle_reward_multiplier),
            "flux": float(args.event_subtype_flux_reward_multiplier),
            "thermal": float(args.event_subtype_thermal_reward_multiplier),
        },
        "reward_shaping": {
            "lambda_warmup_abort": float(args.lambda_warmup_abort),
            "lambda_switch": float(args.lambda_switch),
            "reward_proxy_mode": str(args.reward_proxy_mode),
            "reward_loss_normalization": str(args.reward_loss_normalization),
            "reward_loss_normalizers": None
            if reward_loss_normalizers is None
            else [float(x) for x in reward_loss_normalizers],
            "reward_loss_default_normalizer": float(reward_loss_default_normalizer),
            "reward_staticnorm_candidates_path": str(out_dir / "reward_staticnorm_candidates.csv")
            if reward_staticnorm_table is not None
            else "",
            "event_subtype_particle_reward_multiplier": float(args.event_subtype_particle_reward_multiplier),
            "event_subtype_flux_reward_multiplier": float(args.event_subtype_flux_reward_multiplier),
            "event_subtype_thermal_reward_multiplier": float(args.event_subtype_thermal_reward_multiplier),
            "lambda_duty_balance": float(args.lambda_duty_balance),
            "duty_balance_low": float(args.duty_balance_low),
            "duty_balance_high": float(args.duty_balance_high),
            "duty_balance_grace_steps": int(args.duty_balance_grace_steps),
            "duty_score_feedback": float(args.duty_score_feedback),
            "duty_score_target": float(args.duty_score_target),
            "duty_hard_guard": bool(args.duty_hard_guard),
            "duty_hard_low": float(args.duty_hard_low),
            "duty_hard_high": float(args.duty_hard_high),
            "duty_hard_score": float(args.duty_hard_score),
            "min_dwell_steps": int(max(1, int(args.min_dwell_steps))),
            "baseline_duty_score_feedback": 0.0,
            "baseline_duty_hard_guard": False,
            "primary_eval_duty_guard": bool(args.primary_eval_duty_guard),
        },
        "energy_account": {
            "enabled": bool(args.energy_account),
            "energy_capacity": float(args.energy_capacity),
            "initial_energy": float(args.initial_energy),
            "harvest_per_step": float(args.harvest_per_step),
            "reserve_energy": float(args.reserve_energy),
            "lambda_energy_deficit": float(args.lambda_energy_deficit),
            "soc_soft_penalty_buffer": float(args.soc_soft_penalty_buffer),
            "lambda_soc_soft_penalty": float(args.lambda_soc_soft_penalty),
        },
        "agent_cycle_phase": {
            "enabled": bool(args.include_agent_cycle_phase),
            "period_steps": int(args.agent_cycle_period_steps),
            "dwell_steps": max(1, int(args.agent_cycle_dwell_steps)),
        },
        "observable_regime_belief": {
            "enabled": bool(args.include_observable_regime_belief),
            "lookback": max(1, int(args.regime_belief_lookback)),
        },
        "agent_context_columns": [str(x) for x in (args.agent_context_columns or ())],
        "agent_alert_context": {
            "include_event_flag_in_state": bool(args.include_event_flag_in_state),
            "include_alert_context_features": bool(args.include_alert_context_features),
            "columns": [str(x) for x in (args.alert_context_columns or WarmupEnvConfig.alert_context_columns)],
            "threshold": float(args.alert_context_threshold),
            "trend_lookback": max(1, int(args.alert_context_trend_lookback)),
            "actor_critic_event_context_source": "online_alert_proxy",
            "truth_event_labels_used_online": False,
        },
        "uncertainty_proxy": {
            "definition": "normalised diagonal predict-update variance over the sample-and-hold state",
            "process_variance": [float(value) for value in uncertainty_process_variance],
            "initial_variance": float(train_cfg.uncertainty_initial_variance),
            "max_variance": float(train_cfg.uncertainty_max_variance),
            "measurement_variance_source": "sensor noise_std propagated to observed state columns",
            "measurement_update_mode": str(train_cfg.measurement_update_mode),
        },
        "custom_ppo": as_serializable_config(
            trainer.cfg,
            candidate_count=int(candidate_masks.shape[0]),
            data_seed=int(args.seed),
            policy_seed=policy_seed,
        ),
        "checkpoint_selection": {
            "enabled": checkpoint_interval > 0,
            "interval_updates": checkpoint_interval,
            "partition": "calibration_validation",
            "start_indices": [int(x) for x in checkpoint_starts],
            "steps": int(args.static_selection_steps),
            "score": str(args.checkpoint_selection_score),
            "selected_update": int(best_checkpoint_update),
            "selected_score": None if not np.isfinite(best_checkpoint_score) else float(best_checkpoint_score),
        },
        "awbc_teacher": {
            "mode": str(args.awbc_teacher_mode),
            "calm_sensors": [str(x) for x in (args.awbc_teacher_calm_sensors or ())],
            "event_sensors": [str(x) for x in (args.awbc_teacher_event_sensors or ())],
            "subtype_calm_sensors": [str(x) for x in (args.awbc_teacher_subtype_calm_sensors or ())],
            "subtype_particle_sensors": [str(x) for x in (args.awbc_teacher_subtype_particle_sensors or ())],
            "subtype_flux_sensors": [str(x) for x in (args.awbc_teacher_subtype_flux_sensors or ())],
            "subtype_thermal_sensors": [str(x) for x in (args.awbc_teacher_subtype_thermal_sensors or ())],
            "auto_score_mode": str(args.awbc_teacher_auto_score_mode),
            "auto_selection": awbc_teacher_auto_selection,
            "calm_pool_spec": str(args.awbc_teacher_calm_pool_spec or ""),
            "event_pool_spec": str(args.awbc_teacher_event_pool_spec or ""),
            "event_lookahead_steps": int(args.awbc_teacher_event_lookahead_steps),
            "alert_threshold": float(args.awbc_teacher_alert_threshold),
            "energy_mpc_horizon": int(args.awbc_teacher_energy_mpc_horizon),
            "energy_mpc_soc_bins": int(args.awbc_teacher_energy_mpc_soc_bins),
            "energy_mpc_low_soc_ratio": float(args.awbc_teacher_energy_mpc_low_soc_ratio),
            "energy_mpc_high_soc_ratio": float(args.awbc_teacher_energy_mpc_high_soc_ratio),
            "energy_mpc_terminal_soc_weight": float(args.awbc_teacher_energy_mpc_terminal_soc_weight),
            "energy_mpc_max_actions": int(args.awbc_teacher_energy_mpc_max_actions),
            "energy_mpc_low_power_action": int(args.awbc_teacher_energy_mpc_low_power_action),
            "dwell_steps": int(args.awbc_teacher_dwell_steps),
            "calm_action": int(awbc_teacher_calm_action),
            "event_action": int(awbc_teacher_event_action),
            "calm_actions": [int(x) for x in awbc_teacher_calm_actions],
            "event_actions": [int(x) for x in awbc_teacher_event_actions],
            "subtype_calm_action": int(awbc_teacher_subtype_calm_action),
            "subtype_particle_action": int(awbc_teacher_subtype_particle_action),
            "subtype_flux_action": int(awbc_teacher_subtype_flux_action),
            "subtype_thermal_action": int(awbc_teacher_subtype_thermal_action),
        },
        "candidate_prior": {
            "enabled": bool(args.use_oracle_candidate_prior),
            "steps": int(args.candidate_prior_steps),
            "rollouts": int(args.candidate_prior_rollouts),
            "scale": float(args.candidate_prior_scale),
            "path": str(out_dir / "custom_ppo_candidate_prior.csv") if candidate_prior_table is not None else "",
            "start_indices": [int(x) for x in (args.candidate_prior_start_indices or ())],
        },
        "oracle_static_projected": oracle_static_summary,
        "selected_static_reference": oracle_static_summary,
        "partition_protocol": {
            "oracle_start_idx": args.oracle_start_idx,
            "oracle_end_idx": args.oracle_end_idx,
            "normalization_start_idx": args.normalization_start_idx,
            "normalization_end_idx": args.normalization_end_idx,
            "static_selection_start_indices": [int(x) for x in (args.static_selection_start_indices or ())],
            "static_selection_steps": int(args.static_selection_steps),
            "static_selection_score": str(args.static_selection_score),
            "metrics_sort_score": str(args.metrics_sort_score),
        },
        "ablation_switches": {
            "use_action_mask": bool(args.use_action_mask),
            "use_action_embedding": bool(args.use_action_embedding),
            "nonlinear_action_embedding": bool(args.nonlinear_action_embedding),
            "trainable_action_prior": bool(args.trainable_action_prior),
            "event_aware_critic": bool(args.event_aware_critic),
            "event_gated_actor": bool(args.event_gated_actor),
            "context_encoder": bool(args.context_encoder),
            "context_fusion_mode": str(args.context_fusion_mode),
            "context_layer_norm": bool(args.context_layer_norm),
            "temporal_encoder": bool(args.temporal_encoder),
            "alert_context_features": bool(args.include_alert_context_features),
            "event_flag_in_state": bool(args.include_event_flag_in_state),
            "soc_aux_enabled": int(args.soc_aux_horizon) > 0 and float(args.soc_aux_coef) > 0.0,
            "awbc_enabled": float(args.awbc_coef) > 0.0,
            "prior_kl_enabled": bool(args.use_oracle_candidate_prior) and float(args.prior_kl_coef) > 0.0,
            "eval_duty_constrained_baselines": bool(args.eval_duty_constrained_baselines),
            "eval_switch_limited_baselines": bool(args.eval_switch_limited_baselines),
        },
        "duty_constrained_baselines": constrained_baseline_settings,
        "switch_limited_baselines": {
            "enabled": bool(args.eval_switch_limited_baselines),
            "min_dwell_steps": [int(value) for value in dwell_steps],
        },
        "constraints": {
            "max_active": None if args.max_active is None else int(args.max_active),
            "per_step_budget": float(args.per_step_budget),
            "startup_peak_budget": float(args.startup_peak_budget),
            "required_sensor_ids": [str(sensor_id) for sensor_id in args.required_sensors],
            "coverage_groups": [
                {"name": str(name), "sensor_ids": [str(sensor_id) for sensor_id in sensor_ids]}
                for name, sensor_ids in coverage_groups
            ],
        },
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
            "event_subtypes_enabled": bool(args.event_subtypes_enabled),
            "event_subtype_particle_prob": float(args.event_subtype_particle_prob),
            "event_subtype_assignment": str(args.event_subtype_assignment),
            "event_subtype_particle_min_parsivel_availability": float(
                args.event_subtype_particle_min_parsivel_availability
            ),
            "event_subtype_flux_prob": float(args.event_subtype_flux_prob),
            "event_subtype_thermal_prob": float(args.event_subtype_thermal_prob),
            "event_subtype_particle_humidity_boost_pct": float(args.event_subtype_particle_humidity_boost_pct),
            "event_subtype_flux_wind_boost_ms": float(args.event_subtype_flux_wind_boost_ms),
            "event_subtype_thermal_air_temp_drop_c": float(args.event_subtype_thermal_air_temp_drop_c),
            "event_subtype_latent_alpha": float(args.event_subtype_latent_alpha),
            "event_subtype_particle_latent_diameter_scale_mm": float(args.event_subtype_particle_latent_diameter_scale_mm),
            "event_subtype_particle_latent_velocity_scale_ms": float(args.event_subtype_particle_latent_velocity_scale_ms),
            "event_subtype_flux_latent_sigma": float(args.event_subtype_flux_latent_sigma),
            "event_subtype_flux_latent_linear_scale": float(args.event_subtype_flux_latent_linear_scale),
            "event_subtype_flux_latent_linear_offset": float(args.event_subtype_flux_latent_linear_offset),
            "event_subtype_flux_latent_linear_clip": float(args.event_subtype_flux_latent_linear_clip),
            "event_subtype_thermal_latent_surface_scale_c": float(args.event_subtype_thermal_latent_surface_scale_c),
            "event_subtype_latent_target_lag_steps": int(args.event_subtype_latent_target_lag_steps),
            "event_subtype_context_lead_steps": int(args.event_subtype_context_lead_steps),
            "event_subtype_context_noise_std": float(args.event_subtype_context_noise_std),
        },
    }
    (out_dir / "v2_ppo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir / "v2_custom_ppo_metrics.csv")
    print(metrics.to_string(index=False))

    if not bool(args.skip_evaluation):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "24_v2_evaluate_rollouts.py"),
                "--run-dir",
                str(out_dir),
                "--per-step-budget",
                str(float(args.per_step_budget)),
                "--startup-peak-budget",
                str(float(args.startup_peak_budget)),
            ],
            check=True,
        )


def as_serializable_config(
    cfg: CustomPPOConfig,
    *,
    candidate_count: int,
    data_seed: int,
    policy_seed: int,
) -> dict[str, float | int | str]:
    payload = {
        "total_timesteps": int(cfg.total_timesteps),
        "n_steps": int(cfg.n_steps),
        "batch_size": int(cfg.batch_size),
        "n_epochs": int(cfg.n_epochs),
        "learning_rate": float(cfg.learning_rate),
        "gamma": float(cfg.gamma),
        "gae_lambda": float(cfg.gae_lambda),
        "clip_range": float(cfg.clip_range),
        "ent_coef": float(cfg.ent_coef),
        "channel_marginal_entropy_coef": float(cfg.channel_marginal_entropy_coef),
        "vf_coef": float(cfg.vf_coef),
        "awbc_coef": float(cfg.awbc_coef),
        "awbc_decay_timesteps": int(cfg.awbc_decay_timesteps),
        "awbc_label_stride": int(cfg.awbc_label_stride),
        "awbc_event_only": bool(cfg.awbc_event_only),
        "bc_pretrain_steps": int(cfg.bc_pretrain_steps),
        "bc_pretrain_epochs": int(cfg.bc_pretrain_epochs),
        "bc_pretrain_batch_size": int(cfg.bc_pretrain_batch_size),
        "bc_pretrain_loss_coef": float(cfg.bc_pretrain_loss_coef),
        "subtype_aux_coef": float(cfg.subtype_aux_coef),
        "subtype_aux_classes": int(cfg.subtype_aux_classes),
        "subtype_aux_lookahead_steps": int(cfg.subtype_aux_lookahead_steps),
        "subtype_router_enabled": bool(cfg.subtype_router_enabled),
        "subtype_router_min_confidence": float(cfg.subtype_router_min_confidence),
        "subtype_router_low_confidence_action": int(cfg.subtype_router_low_confidence_action),
        "awbc_teacher_mode": str(cfg.awbc_teacher_mode),
        "awbc_teacher_event_lookahead_steps": int(cfg.awbc_teacher_event_lookahead_steps),
        "awbc_teacher_alert_threshold": float(cfg.awbc_teacher_alert_threshold),
        "awbc_teacher_energy_mpc_horizon": int(cfg.awbc_teacher_energy_mpc_horizon),
        "awbc_teacher_energy_mpc_soc_bins": int(cfg.awbc_teacher_energy_mpc_soc_bins),
        "awbc_teacher_energy_mpc_low_soc_ratio": float(cfg.awbc_teacher_energy_mpc_low_soc_ratio),
        "awbc_teacher_energy_mpc_high_soc_ratio": float(cfg.awbc_teacher_energy_mpc_high_soc_ratio),
        "awbc_teacher_energy_mpc_terminal_soc_weight": float(cfg.awbc_teacher_energy_mpc_terminal_soc_weight),
        "awbc_teacher_energy_mpc_max_actions": int(cfg.awbc_teacher_energy_mpc_max_actions),
        "awbc_teacher_energy_mpc_low_power_action": int(cfg.awbc_teacher_energy_mpc_low_power_action),
        "awbc_teacher_calm_action": int(cfg.awbc_teacher_calm_action),
        "awbc_teacher_event_action": int(cfg.awbc_teacher_event_action),
        "awbc_teacher_calm_actions": [int(x) for x in cfg.awbc_teacher_calm_actions],
        "awbc_teacher_event_actions": [int(x) for x in cfg.awbc_teacher_event_actions],
        "awbc_teacher_subtype_calm_action": int(cfg.awbc_teacher_subtype_calm_action),
        "awbc_teacher_subtype_particle_action": int(cfg.awbc_teacher_subtype_particle_action),
        "awbc_teacher_subtype_flux_action": int(cfg.awbc_teacher_subtype_flux_action),
        "awbc_teacher_subtype_thermal_action": int(cfg.awbc_teacher_subtype_thermal_action),
        "awbc_teacher_dwell_steps": int(cfg.awbc_teacher_dwell_steps),
        "prior_kl_coef": float(cfg.prior_kl_coef),
        "embed_dim": int(cfg.embed_dim),
        "hidden_dim": int(cfg.hidden_dim),
        "greedy_lookahead_steps": int(cfg.greedy_lookahead_steps),
        "event_start_prob": float(cfg.event_start_prob),
        "use_action_mask": int(bool(cfg.use_action_mask)),
        "use_action_embedding": int(bool(cfg.use_action_embedding)),
        "nonlinear_action_embedding": int(bool(cfg.nonlinear_action_embedding)),
        "trainable_action_prior": int(bool(cfg.trainable_action_prior)),
        "event_aware_critic": int(bool(cfg.event_aware_critic)),
        "event_gated_actor": int(bool(cfg.event_gated_actor)),
        "context_encoder_enabled": int(bool(cfg.context_encoder_enabled)),
        "context_feature_dim": int(cfg.context_feature_dim),
        "context_hidden_dim": int(cfg.context_hidden_dim),
        "context_fusion_mode": str(cfg.context_fusion_mode),
        "context_layer_norm": int(bool(cfg.context_layer_norm)),
        "temporal_encoder_enabled": int(bool(cfg.temporal_encoder_enabled)),
        "temporal_history_steps": int(cfg.temporal_history_steps),
        "temporal_state_dim": int(cfg.temporal_state_dim),
        "temporal_hidden_dim": int(cfg.temporal_hidden_dim),
        "soc_aux_horizon": int(cfg.soc_aux_horizon),
        "soc_aux_coef": float(cfg.soc_aux_coef),
        "device": str(cfg.device),
        "seed": int(cfg.seed),
        "data_seed": int(data_seed),
        "policy_seed": int(policy_seed),
        "candidate_count": int(candidate_count),
        "history_path": str(cfg.history_path or ""),
        "train_start_indices": [int(x) for x in getattr(cfg, "train_start_indices", ())],
        "train_start_min": cfg.train_start_min,
        "train_start_max": cfg.train_start_max,
    }
    return payload


def write_training_log(history: list[dict[str, float | int]], path: Path) -> None:
    """Persist a compact learning-curve CSV for paper plotting.

    The v2 custom PPO loop does not run expensive full forecast evaluation
    during training. The final forecast metric is produced by
    scripts/24_v2_evaluate_rollouts.py after rollout export; this log records
    the available online learning diagnostics with stable column names.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in history:
        rows.append(
            {
                "step": int(item.get("timesteps", 0)),
                "loss": float(item.get("loss", float("nan"))),
                "policy_loss": float(item.get("policy_loss", float("nan"))),
                "value_loss": float(item.get("value_loss", float("nan"))),
                "entropy_mean": float(item.get("entropy", float("nan"))),
                "awbc_loss": float(item.get("awbc_loss", float("nan"))),
                "prior_kl_loss": float(item.get("prior_kl_loss", float("nan"))),
                "soc_aux_loss": float(item.get("soc_aux_loss", float("nan"))),
                "subtype_aux_loss": float(item.get("subtype_aux_loss", float("nan"))),
                "subtype_aux_accuracy": float(item.get("subtype_aux_accuracy", float("nan"))),
                "advantage_mean": float(item.get("advantage_mean", float("nan"))),
                "advantage_std": float(item.get("advantage_std", float("nan"))),
                "event_rate": float(item.get("event_rate", float("nan"))),
                "awbc_label_rate": float(item.get("awbc_label_rate", float("nan"))),
                "greedy_unique_actions": int(item.get("greedy_unique_actions", 0)),
                "bc_pretrain": int(item.get("bc_pretrain", 0)),
                "bc_steps": int(item.get("bc_steps", 0)),
                "bc_accuracy": float(item.get("bc_accuracy", float("nan"))),
                "forecast_weighted_mae_overall": float("nan"),
                "reward_mean": float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def build_oracle_candidate_prior(
    *,
    truth: pd.DataFrame,
    sensors: list[object],
    constraints: PowerConstraintsV2,
    cfg: WarmupEnvConfig,
    oracle: object,
    candidate_masks: np.ndarray,
    start_indices: tuple[int, ...],
    steps: int,
    scale: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    from v2.env import WarmupSchedulingEnv

    rows = []
    masks = np.asarray(candidate_masks, dtype=bool)
    for action_idx, mask in enumerate(masks):
        losses: list[float] = []
        event_losses: list[float] = []
        non_event_losses: list[float] = []
        powers: list[float] = []
        subtype_losses: dict[int, list[float]] = {int(subtype_id): [] for subtype_id in SUBTYPE_LABELS}
        subtype_counts: dict[int, int] = {int(subtype_id): 0 for subtype_id in SUBTYPE_LABELS}
        event_count = 0
        non_event_count = 0
        aborts = 0
        for offset, start_idx in enumerate(start_indices):
            env = WarmupSchedulingEnv(
                truth,
                sensors,
                constraints,
                WarmupEnvConfig(
                    state_columns=cfg.state_columns,
                    reward_target_columns=cfg.reward_target_columns,
                    reward_proxy_mode=cfg.reward_proxy_mode,
                    lookback=cfg.lookback,
                    episode_len=int(steps),
                    seed=int(cfg.seed) + int(offset) + 17_003,
                    base_freq_s=cfg.base_freq_s,
                    event_column=cfg.event_column,
                    normalize_agent_state=cfg.normalize_agent_state,
                    normalization_mean=cfg.normalization_mean,
                    normalization_std=cfg.normalization_std,
                    lambda_warmup_abort=cfg.lambda_warmup_abort,
                    lambda_switch=cfg.lambda_switch,
                    event_reward_multiplier=cfg.event_reward_multiplier,
                    event_subtype_particle_reward_multiplier=cfg.event_subtype_particle_reward_multiplier,
                    event_subtype_flux_reward_multiplier=cfg.event_subtype_flux_reward_multiplier,
                    event_subtype_thermal_reward_multiplier=cfg.event_subtype_thermal_reward_multiplier,
                    oracle_loss_reward_normalizers=cfg.oracle_loss_reward_normalizers,
                    oracle_loss_reward_default_normalizer=cfg.oracle_loss_reward_default_normalizer,
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
                    include_event_flag_in_state=cfg.include_event_flag_in_state,
                    include_alert_context_features=cfg.include_alert_context_features,
                    alert_context_columns=cfg.alert_context_columns,
                    alert_context_threshold=cfg.alert_context_threshold,
                    alert_context_trend_lookback=cfg.alert_context_trend_lookback,
                    uncertainty_process_variance=cfg.uncertainty_process_variance,
                    uncertainty_initial_variance=cfg.uncertainty_initial_variance,
                    uncertainty_max_variance=cfg.uncertainty_max_variance,
                    measurement_update_mode=cfg.measurement_update_mode,
                ),
                oracle=oracle,
            )
            env.reset(start_idx=int(start_idx))
            for _ in range(int(steps)):
                _, _, done, info = env.step_mask(mask)
                loss = float(info.get("oracle_loss", float("nan")))
                is_event = bool(info.get("event", False))
                subtype_id = int(info.get("event_subtype_id", 0) or 0)
                event_count += int(is_event)
                non_event_count += int(not is_event)
                if subtype_id in subtype_counts:
                    subtype_counts[subtype_id] += 1
                if np.isfinite(loss):
                    losses.append(loss)
                    if is_event:
                        event_losses.append(loss)
                    else:
                        non_event_losses.append(loss)
                    if subtype_id in subtype_losses:
                        subtype_losses[subtype_id].append(loss)
                powers.append(float(info.get("power", 0.0)))
                aborts += int(info.get("warmup_abort_delta", 0))
                if done:
                    break
        subtype_means: dict[str, float] = {}
        subtype_macro_values: list[float] = []
        for subtype_id, label in SUBTYPE_LABELS.items():
            subtype_loss = finite_mean(subtype_losses[int(subtype_id)])
            subtype_means[f"oracle_loss_subtype_{label}"] = subtype_loss
            subtype_means[f"steps_subtype_{label}"] = int(subtype_counts[int(subtype_id)])
            if np.isfinite(subtype_loss):
                subtype_macro_values.append(subtype_loss)
        rows.append(
            {
                "action_idx": int(action_idx),
                "oracle_loss_mean": float(np.mean(losses)) if losses else float("inf"),
                "oracle_loss_event": finite_mean(event_losses),
                "oracle_loss_non_event": finite_mean(non_event_losses),
                "event_rate": (
                    float(event_count) / float(max(1, event_count + non_event_count))
                ),
                **subtype_means,
                MACRO_SUBTYPE_LOSS_COLUMN: finite_mean(subtype_macro_values),
                MACRO_SUBTYPE_COUNT_COLUMN: int(len(subtype_macro_values)),
                "power_mean": float(np.mean(powers)) if powers else 0.0,
                "warmup_abort_count": int(aborts),
                "sensor_ids": "|".join(str(sensors[i].sensor_id) for i in np.flatnonzero(mask)),
                **{f"sensor_{i}": int(v) for i, v in enumerate(mask.astype(int))},
            }
        )
    table = pd.DataFrame(rows)
    table = add_staticnorm_macro(table, subtype_static_normalizers(table))
    table = table.sort_values("oracle_loss_mean").reset_index(drop=True)
    costs = table.sort_values("action_idx")["oracle_loss_mean"].to_numpy(dtype=float)
    finite = np.isfinite(costs)
    if not np.any(finite):
        logits = np.zeros_like(costs, dtype=np.float32)
    else:
        filled = costs.copy()
        filled[~finite] = float(np.max(filled[finite]))
        centered = filled - float(np.mean(filled))
        denom = float(np.std(centered))
        if denom < 1e-8:
            logits = np.zeros_like(filled, dtype=np.float32)
        else:
            logits = (-centered / denom * float(scale)).astype(np.float32)
    table["prior_logit"] = logits[table["action_idx"].to_numpy(dtype=int)]
    return logits, table


if __name__ == "__main__":
    main()
