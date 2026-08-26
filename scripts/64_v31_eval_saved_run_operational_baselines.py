#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.custom_ppo import CustomPPO, CustomPPOConfig, evaluate_custom_ppo  # noqa: E402
from v2.env import WarmupEnvConfig  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle  # noqa: E402
from v2.policies import FullOpenUnconstrainedScorePolicy, MinDwellPolicyWrapper, StaticMaskPolicy  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import rollout_metrics, save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


def configure_torch_threads() -> None:
    thread_value = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or "8"
    try:
        thread_count = max(1, int(thread_value))
    except ValueError:
        thread_count = 8
    try:
        import torch

        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(max(1, min(4, thread_count)))
    except Exception:
        return


def load_helpers() -> Any:
    path = ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("_v2_train_ppo_helpers_operational_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(value: str | Path, *, run_dir: Path) -> Path:
    path = Path(value)
    candidates = [path, run_dir / path.name, ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve {value!r} from {run_dir}")


def constraints_from_metadata(metadata: dict[str, Any]) -> PowerConstraintsV2:
    source = dict(metadata["constraints"])
    coverage_groups = tuple(
        (str(item["name"]), tuple(str(sensor_id) for sensor_id in item["sensor_ids"]))
        for item in source.get("coverage_groups", [])
    )
    return PowerConstraintsV2(
        max_active=None if source.get("max_active") is None else int(source["max_active"]),
        per_step_budget=float(source["per_step_budget"]),
        startup_peak_budget=float(source["startup_peak_budget"]),
        required_sensor_ids=tuple(str(x) for x in source.get("required_sensor_ids", [])),
        coverage_groups=coverage_groups,
    )


def env_kwargs_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    energy = dict(metadata.get("energy_account", {}))
    shaping = dict(metadata.get("reward_shaping", {}))
    regime_belief = dict(metadata.get("observable_regime_belief", {}))
    alert_context = dict(metadata.get("agent_alert_context", {}))
    sensor_quality = dict(metadata.get("sensor_quality", {}))
    return {
        "lambda_warmup_abort": float(shaping.get("lambda_warmup_abort", 0.08)),
        "lambda_switch": float(shaping.get("lambda_switch", 0.002)),
        "event_reward_multiplier": float(metadata.get("event_reward_multiplier", 1.0)),
        "event_subtype_particle_reward_multiplier": float(
            shaping.get(
                "event_subtype_particle_reward_multiplier",
                dict(metadata.get("event_subtype_reward_multipliers", {})).get("particle", 1.0),
            )
        ),
        "event_subtype_flux_reward_multiplier": float(
            shaping.get(
                "event_subtype_flux_reward_multiplier",
                dict(metadata.get("event_subtype_reward_multipliers", {})).get("flux", 1.0),
            )
        ),
        "event_subtype_thermal_reward_multiplier": float(
            shaping.get(
                "event_subtype_thermal_reward_multiplier",
                dict(metadata.get("event_subtype_reward_multipliers", {})).get("thermal", 1.0),
            )
        ),
        "lambda_duty_balance": float(shaping.get("lambda_duty_balance", 0.0)),
        "duty_balance_low": float(shaping.get("duty_balance_low", 0.05)),
        "duty_balance_high": float(shaping.get("duty_balance_high", 0.95)),
        "duty_balance_grace_steps": int(shaping.get("duty_balance_grace_steps", 64)),
        "duty_score_feedback": float(shaping.get("duty_score_feedback", 0.0)),
        "duty_score_target": float(shaping.get("duty_score_target", 0.40)),
        "duty_hard_guard": bool(shaping.get("duty_hard_guard", False)),
        "duty_hard_low": float(shaping.get("duty_hard_low", 0.08)),
        "duty_hard_high": float(shaping.get("duty_hard_high", 0.92)),
        "duty_hard_score": float(shaping.get("duty_hard_score", 8.0)),
        "min_dwell_steps": int(shaping.get("min_dwell_steps", 1)),
        "energy_account_enabled": bool(energy.get("enabled", False)),
        "energy_capacity": float(energy.get("energy_capacity", 0.0)),
        "initial_energy": float(energy.get("initial_energy", 0.0)),
        "harvest_per_step": float(energy.get("harvest_per_step", 0.0)),
        "reserve_energy": float(energy.get("reserve_energy", 0.0)),
        "lambda_energy_deficit": float(energy.get("lambda_energy_deficit", 1.0)),
        "soc_soft_penalty_buffer": float(energy.get("soc_soft_penalty_buffer", 0.0)),
        "lambda_soc_soft_penalty": float(energy.get("lambda_soc_soft_penalty", 0.0)),
        "include_observable_regime_belief": bool(regime_belief.get("enabled", False)),
        "regime_belief_lookback": int(regime_belief.get("lookback", 6)),
        "agent_context_columns": tuple(str(x) for x in metadata.get("agent_context_columns", ())),
        "sensor_quality_columns": tuple(str(x) for x in sensor_quality.get("columns", ())),
        "sensor_quality_max_noise_multiplier": float(sensor_quality.get("max_noise_multiplier", 1.0)),
        "sensor_quality_availability_floor": float(sensor_quality.get("availability_floor", 1.0)),
        "include_event_flag_in_state": bool(alert_context.get("include_event_flag_in_state", True)),
        "include_alert_context_features": bool(alert_context.get("include_alert_context_features", False)),
        "alert_context_columns": tuple(
            str(x)
            for x in alert_context.get("columns", WarmupEnvConfig.alert_context_columns)
        ),
        "alert_context_threshold": float(alert_context.get("threshold", 0.5)),
        "alert_context_trend_lookback": int(alert_context.get("trend_lookback", 6)),
    }


def normalization_stats(
    truth: pd.DataFrame,
    *,
    state_columns: tuple[str, ...],
    metadata: dict[str, Any],
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    protocol = dict(metadata.get("partition_protocol", {}))
    start = protocol.get("normalization_start_idx")
    end = protocol.get("normalization_end_idx")
    if start is None and end is None:
        return None, None
    start_idx = int(start or 0)
    end_idx = int(end or len(truth))
    if start_idx < 0 or end_idx <= start_idx or end_idx > len(truth):
        raise ValueError(f"Invalid normalization range [{start_idx}, {end_idx}) for truth length {len(truth)}")
    values = truth.iloc[start_idx:end_idx][list(state_columns)].to_numpy(dtype=float)
    return (
        tuple(float(x) for x in np.mean(values, axis=0)),
        tuple(float(x) for x in np.maximum(np.std(values, axis=0), 1e-6)),
    )


def infer_eval_steps(metadata: dict[str, Any], *, run_dir: Path) -> int:
    if metadata.get("eval_steps") is not None:
        return int(metadata["eval_steps"])
    rollout_path = run_dir / "rollout_custom_ppo.npz"
    if rollout_path.exists():
        data = np.load(rollout_path, allow_pickle=False)
        step_count = int(data["rewards"].shape[0])
        rollout_count = int(metadata.get("eval_rollouts", 1))
        if rollout_count > 0 and step_count % rollout_count == 0:
            return int(step_count // rollout_count)
    return 1024


def full_final_partition_scope(
    metadata: dict[str, Any],
    *,
    run_dir: Path,
    truth_length: int,
    horizon: int,
) -> tuple[int, int, int]:
    """Return the full held-out interval for which all future labels exist.

    The chronological final partition can extend to the end of the generated
    truth sequence.  A forecast loss at time ``t`` requires labels through
    ``t + horizon``; the last ``horizon`` rows therefore cannot be scored.
    """

    manifest_path = run_dir / "split_protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    partitions = dict(manifest.get("partitions", {}))
    final_range = partitions.get("final_test")
    if final_range is None:
        source_manifest = dict(dict(metadata.get("control_source", {})).get("source_manifest", {}))
        final_range = dict(source_manifest.get("partitions", {})).get("final_test")
    if not isinstance(final_range, (list, tuple)) or len(final_range) != 2:
        raise ValueError(f"Cannot resolve final_test partition from {manifest_path}")

    partition_start, partition_end = (int(final_range[0]), int(final_range[1]))
    scoreable_end = min(partition_end, int(truth_length) - int(horizon))
    if partition_start < 0 or partition_end > int(truth_length) or scoreable_end <= partition_start:
        raise ValueError(
            "Invalid full-final scope: "
            f"partition=[{partition_start}, {partition_end}), "
            f"scoreable_end={scoreable_end}, truth_length={truth_length}, horizon={horizon}"
        )
    return partition_start, partition_end, scoreable_end


def load_oracle(metadata: dict[str, Any], *, run_dir: Path, device: str) -> Any:
    oracle_path = resolve_path(str(metadata["oracle_path"]), run_dir=run_dir)
    if str(metadata.get("oracle_type", "linear")) == "tcn":
        return TCNFrozenForecastOracle.load(oracle_path, device=str(device))
    return LinearFrozenForecastOracle.load(str(oracle_path))


def load_trainer(
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    truth: pd.DataFrame,
    sensors: list[Any],
    constraints: PowerConstraintsV2,
    env_cfg: WarmupEnvConfig,
    oracle: Any,
    device: str,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[CustomPPO, np.ndarray]:
    import torch

    model_path = resolve_path(str(metadata.get("model_path", run_dir / "custom_ppo.pt")), run_dir=run_dir)
    payload = torch.load(str(model_path), map_location=str(device), weights_only=False)
    allowed = {field.name for field in dataclasses.fields(CustomPPOConfig)}
    cfg_raw = {key: value for key, value in dict(payload.get("cfg", {})).items() if key in allowed}
    cfg_raw["device"] = str(device)
    cfg_raw["history_path"] = None
    if cfg_overrides:
        cfg_raw.update({key: value for key, value in cfg_overrides.items() if key in allowed})
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=env_cfg,
        oracle=oracle,
        candidate_masks=np.asarray(payload["candidate_masks"], dtype=bool),
        cfg=CustomPPOConfig(**cfg_raw),
        candidate_prior_logits=payload.get("candidate_prior_logits"),
    )
    trainer.model.load_state_dict(payload["state_dict"])
    trainer.model.eval()
    return trainer, np.asarray(payload["candidate_masks"], dtype=bool)


def selected_static_policy(run_dir: Path, candidate_masks: np.ndarray) -> StaticMaskPolicy | None:
    score_col = "oracle_loss_mean"
    metadata_path = run_dir / "v2_ppo_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            score_col = str(metadata.get("static_selection_score") or score_col)
        except Exception:
            score_col = "oracle_loss_mean"
    candidates = [
        (run_dir / "validation_static_candidates.csv", "validation_selected_static"),
        (run_dir / "custom_ppo_candidate_prior.csv", "oracle_static_projected"),
    ]
    for table_path, name in candidates:
        if table_path.exists():
            table = pd.read_csv(table_path)
            if not table.empty:
                actual_score = score_col if score_col in table.columns else "oracle_loss_mean"
                values = pd.to_numeric(table.get(actual_score), errors="coerce")
                if np.any(np.isfinite(values)):
                    table = table.assign(_selection_score=values).sort_values(
                        ["_selection_score", "oracle_loss_mean"],
                        na_position="last",
                    )
                row = table.iloc[0]
                mask = tuple(bool(x) for x in np.asarray(candidate_masks[int(row["action_idx"])], dtype=bool))
                return StaticMaskPolicy(mask=mask, name=name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate a saved V3.1 custom-PPO run with operational baselines.")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument(
        "--eval-full-final-partition",
        action="store_true",
        help="Replay the complete held-out final interval for which all forecast targets exist.",
    )
    parser.add_argument("--eval-steps-override", type=int, default=None)
    parser.add_argument("--eval-start-indices-override", nargs="+", type=int, default=None)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Evaluate only the saved custom PPO and validation-selected static reference.",
    )
    parser.add_argument("--eval-duty-constrained-baselines", action="store_true")
    parser.add_argument("--baseline-duty-hard-low", type=float, default=0.12)
    parser.add_argument("--baseline-duty-hard-high", type=float, default=0.85)
    parser.add_argument("--baseline-duty-hard-score", type=float, default=12.0)
    parser.add_argument("--baseline-duty-score-feedback", type=float, default=2.5)
    parser.add_argument("--eval-switch-limited-baselines", action="store_true")
    parser.add_argument("--baseline-min-dwell-steps", nargs="+", type=int, default=[6, 12])
    parser.add_argument("--env-min-dwell-steps", type=int, default=None)
    parser.add_argument("--env-harvest-per-step", type=float, default=None)
    parser.add_argument("--env-duty-hard-guard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--env-duty-score-feedback", type=float, default=None)
    parser.add_argument("--env-lambda-duty-balance", type=float, default=None)
    parser.add_argument("--subtype-router", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subtype-router-min-confidence", type=float, default=None)
    parser.add_argument("--subtype-router-low-confidence-action", type=int, default=None)
    parser.add_argument("--subtype-router-calm-action", type=int, default=None)
    parser.add_argument("--subtype-router-particle-action", type=int, default=None)
    parser.add_argument("--subtype-router-flux-action", type=int, default=None)
    parser.add_argument("--subtype-router-thermal-action", type=int, default=None)
    parser.add_argument("--skip-rollout-evaluation", action="store_true")
    args = parser.parse_args()

    configure_torch_threads()
    helpers = load_helpers()
    source_run_dir = Path(args.source_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source_run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth_path = resolve_path(str(metadata["truth_csv"]), run_dir=source_run_dir)
    truth = pd.read_csv(truth_path)
    sensor_cfg = resolve_path(str(metadata["sensor_cfg"]), run_dir=source_run_dir)
    sensors = load_sensor_specs(sensor_cfg)
    constraints = constraints_from_metadata(metadata)
    oracle = load_oracle(metadata, run_dir=source_run_dir, device=str(args.oracle_device))
    norm_mean, norm_std = normalization_stats(
        truth,
        state_columns=helpers.STATE_COLUMNS,
        metadata=metadata,
    )
    if args.eval_full_final_partition and (
        args.eval_steps_override is not None or args.eval_start_indices_override is not None
    ):
        raise ValueError("--eval-full-final-partition cannot be combined with explicit evaluation overrides")

    evaluation_scope: dict[str, Any]
    if args.eval_full_final_partition:
        partition_start, partition_end, scoreable_end = full_final_partition_scope(
            metadata,
            run_dir=source_run_dir,
            truth_length=len(truth),
            horizon=int(oracle.cfg.horizon),
        )
        eval_starts = (int(partition_start),)
        eval_steps = int(scoreable_end - partition_start)
        evaluation_scope = {
            "mode": "full_scoreable_final_partition",
            "final_partition": [int(partition_start), int(partition_end)],
            "scoreable_interval": [int(partition_start), int(scoreable_end)],
            "excluded_tail_steps_without_complete_future": int(partition_end - scoreable_end),
        }
    else:
        eval_steps = int(args.eval_steps_override or infer_eval_steps(metadata, run_dir=source_run_dir))
        eval_starts = tuple(
            int(x)
            for x in (
                args.eval_start_indices_override
                if args.eval_start_indices_override is not None
                else metadata["eval_start_indices"]
            )
        )
        evaluation_scope = {
            "mode": "explicit_override" if (
                args.eval_steps_override is not None or args.eval_start_indices_override is not None
            ) else "saved_evaluation_windows",
            "scoreable_interval": None,
        }
    if eval_steps <= 0 or not eval_starts:
        raise ValueError(f"Invalid evaluation request: starts={eval_starts}, steps={eval_steps}")
    env_kwargs = env_kwargs_from_metadata(metadata)
    if args.env_min_dwell_steps is not None:
        env_kwargs["min_dwell_steps"] = int(max(1, int(args.env_min_dwell_steps)))
    if args.env_harvest_per_step is not None:
        env_kwargs["harvest_per_step"] = float(args.env_harvest_per_step)
    if args.env_duty_hard_guard is not None:
        env_kwargs["duty_hard_guard"] = bool(args.env_duty_hard_guard)
    if args.env_duty_score_feedback is not None:
        env_kwargs["duty_score_feedback"] = float(args.env_duty_score_feedback)
    if args.env_lambda_duty_balance is not None:
        env_kwargs["lambda_duty_balance"] = float(args.env_lambda_duty_balance)
    cfg_overrides: dict[str, Any] = {}
    if args.subtype_router is not None:
        cfg_overrides["subtype_router_enabled"] = bool(args.subtype_router)
    if args.subtype_router_min_confidence is not None:
        cfg_overrides["subtype_router_min_confidence"] = float(args.subtype_router_min_confidence)
    if args.subtype_router_low_confidence_action is not None:
        cfg_overrides["subtype_router_low_confidence_action"] = int(args.subtype_router_low_confidence_action)
    if args.subtype_router_calm_action is not None:
        cfg_overrides["awbc_teacher_subtype_calm_action"] = int(args.subtype_router_calm_action)
    if args.subtype_router_particle_action is not None:
        cfg_overrides["awbc_teacher_subtype_particle_action"] = int(args.subtype_router_particle_action)
    if args.subtype_router_flux_action is not None:
        cfg_overrides["awbc_teacher_subtype_flux_action"] = int(args.subtype_router_flux_action)
    if args.subtype_router_thermal_action is not None:
        cfg_overrides["awbc_teacher_subtype_thermal_action"] = int(args.subtype_router_thermal_action)
    eval_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=int(metadata["lookback"]),
        episode_len=int(eval_steps),
        seed=int(metadata["seed"]) + 9000,
        base_freq_s=int(metadata["freq_s"]),
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        **env_kwargs,
    )
    baseline_eval_cfg = replace(eval_cfg, duty_score_feedback=0.0, duty_hard_guard=False)
    trainer, candidate_masks = load_trainer(
        run_dir=source_run_dir,
        metadata=metadata,
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        env_cfg=eval_cfg,
        oracle=oracle,
        device=str(args.device),
        cfg_overrides=cfg_overrides,
    )

    rows: list[dict[str, Any]] = []
    eval_policies: list[str] = []
    custom_result, custom_metrics = evaluate_custom_ppo(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=int(eval_steps),
        start_indices=eval_starts,
    )
    rows.append(custom_metrics)
    eval_policies.append("custom_ppo")
    save_rollout_npz(out_dir / "rollout_custom_ppo.npz", custom_result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

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
                steps=int(eval_steps),
                start_indices=eval_starts,
                policy_name=dwell_name,
                min_dwell_steps=int(dwell),
            )
            rows.append(metrics)
            eval_policies.append(dwell_name)
            save_rollout_npz(out_dir / f"rollout_{dwell_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    static_policy = selected_static_policy(source_run_dir, candidate_masks)
    if static_policy is not None:
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=baseline_eval_cfg,
            oracle=oracle,
            policy=static_policy,
            steps=int(eval_steps),
            start_indices=eval_starts,
        )
        rows.append(metrics)
        eval_policies.append(str(result.policy_name))
        save_rollout_npz(out_dir / f"rollout_{result.policy_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    if not bool(args.primary_only):
        full_result, full_metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=PowerConstraintsV2(),
            cfg=baseline_eval_cfg,
            oracle=oracle,
            policy=FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
            steps=int(eval_steps),
            start_indices=eval_starts,
        )
        rows.append(full_metrics)
        eval_policies.append("full_open_unconstrained")
        save_rollout_npz(out_dir / "rollout_full_open_unconstrained.npz", full_result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

        for policy in helpers.default_policies(len(sensors), seed=int(metadata["seed"]) + 100):
            result, metrics = helpers.evaluate_score_policy_over_starts(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=baseline_eval_cfg,
                oracle=oracle,
                policy=policy,
                steps=int(eval_steps),
                start_indices=eval_starts,
            )
            rows.append(metrics)
            eval_policies.append(str(result.policy_name))
            save_rollout_npz(out_dir / f"rollout_{result.policy_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    constrained_settings = {
        "enabled": bool(args.eval_duty_constrained_baselines),
        "duty_hard_low": float(args.baseline_duty_hard_low),
        "duty_hard_high": float(args.baseline_duty_hard_high),
        "duty_hard_score": float(args.baseline_duty_hard_score),
        "duty_score_feedback": float(args.baseline_duty_score_feedback),
    }
    if bool(args.eval_duty_constrained_baselines):
        constrained_cfg = replace(
            baseline_eval_cfg,
            duty_score_feedback=float(args.baseline_duty_score_feedback),
            duty_hard_guard=True,
            duty_hard_low=float(args.baseline_duty_hard_low),
            duty_hard_high=float(args.baseline_duty_hard_high),
            duty_hard_score=float(args.baseline_duty_hard_score),
        )
        if static_policy is not None:
            constrained_static = StaticMaskPolicy(
                mask=tuple(bool(x) for x in static_policy.mask),
                name=f"duty_constrained_{static_policy.name}",
            )
            result, metrics = helpers.evaluate_score_policy_over_starts(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=constrained_cfg,
                oracle=oracle,
                policy=constrained_static,
                steps=int(eval_steps),
                start_indices=eval_starts,
            )
            rows.append(metrics)
            eval_policies.append(str(result.policy_name))
            save_rollout_npz(out_dir / f"rollout_{result.policy_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)
        for policy in helpers.default_policies(len(sensors), seed=int(metadata["seed"]) + 10100):
            policy.name = f"duty_constrained_{policy.name}"
            result, metrics = helpers.evaluate_score_policy_over_starts(
                truth=truth,
                sensors=sensors,
                constraints=constraints,
                cfg=constrained_cfg,
                oracle=oracle,
                policy=policy,
                steps=int(eval_steps),
                start_indices=eval_starts,
            )
            rows.append(metrics)
            eval_policies.append(str(result.policy_name))
            save_rollout_npz(out_dir / f"rollout_{result.policy_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    dynamic_policy_names = {"round_robin", "aoi", "random"}
    if bool(args.eval_switch_limited_baselines):
        for dwell in dwell_steps:
            for policy in helpers.default_policies(len(sensors), seed=int(metadata["seed"]) + 20100 + dwell):
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
                    steps=int(eval_steps),
                    start_indices=eval_starts,
                )
                rows.append(metrics)
                eval_policies.append(str(result.policy_name))
                save_rollout_npz(out_dir / f"rollout_{result.policy_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

        if bool(args.eval_duty_constrained_baselines):
            for dwell in dwell_steps:
                for policy in helpers.default_policies(len(sensors), seed=int(metadata["seed"]) + 30100 + dwell):
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
                        cfg=constrained_cfg,
                        oracle=oracle,
                        policy=wrapped,
                        steps=int(eval_steps),
                        start_indices=eval_starts,
                    )
                    rows.append(metrics)
                    eval_policies.append(str(result.policy_name))
                    save_rollout_npz(out_dir / f"rollout_{result.policy_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    metrics = pd.DataFrame(rows).sort_values("oracle_loss_mean")
    metrics.to_csv(out_dir / "v2_custom_ppo_metrics.csv", index=False)
    replay_metadata = dict(metadata)
    replay_metadata.update(
        {
            "source_run_dir": str(source_run_dir),
            "truth_csv": str(truth_path),
            "policy_inference_device": str(args.device),
            "oracle_inference_device": str(args.oracle_device),
            "eval_policies": eval_policies,
            "eval_steps": int(eval_steps),
            "eval_start_indices": [int(value) for value in eval_starts],
            "evaluation_scope": evaluation_scope,
            "operational_baseline_replay": {
                "enabled": True,
                "uses_saved_custom_ppo": True,
                "uses_original_truth": True,
                "uses_original_eval_starts": evaluation_scope["mode"] == "saved_evaluation_windows",
                "primary_only": bool(args.primary_only),
                "skip_rollout_evaluation": bool(args.skip_rollout_evaluation),
                "env_min_dwell_steps": int(env_kwargs.get("min_dwell_steps", 1)),
                "env_harvest_per_step": float(env_kwargs.get("harvest_per_step", 0.0)),
                "custom_ppo_cfg_overrides": cfg_overrides,
                "duty_constrained_baselines": constrained_settings,
                "switch_limited_baselines": {
                    "enabled": bool(args.eval_switch_limited_baselines),
                    "min_dwell_steps": [int(value) for value in dwell_steps],
                },
            },
        }
    )
    (out_dir / "v2_ppo_metadata.json").write_text(json.dumps(replay_metadata, indent=2), encoding="utf-8")
    if not bool(args.skip_rollout_evaluation):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "24_v2_evaluate_rollouts.py"),
                "--run-dir",
                str(out_dir),
                "--per-step-budget",
                str(float(constraints.per_step_budget)),
                "--startup-peak-budget",
                str(float(constraints.startup_peak_budget)),
                "--forecast-oracle-device",
                str(args.oracle_device),
            ],
            check=True,
        )
    print(out_dir / "v2_custom_ppo_metrics.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
