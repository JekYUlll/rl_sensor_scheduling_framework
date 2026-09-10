#!/usr/bin/env python3
"""Prepare causal-specialist assets and refit the frozen forecast oracle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


def _helpers():
    path = ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("v2_train_helpers_v608", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_weights_for_objective(
    objective: str, ordinary_weights: tuple[float, ...] | None
) -> tuple[float, ...] | None:
    if objective == "ordinary":
        if ordinary_weights is None:
            raise ValueError("ordinary objective requires source oracle target weights")
        return tuple(float(x) for x in ordinary_weights)
    if objective == "group_balanced":
        # Keep the six backbone targets at weight 1.0.  Give particle, flux,
        # and thermal specialist groups equal total weight while preserving
        # the ordinary objective's total weight (9.0).
        return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0)
    raise ValueError(f"unsupported objective: {objective}")


def prepare(
    seed: int,
    source_root: Path,
    truth_root: Path,
    output_root: Path,
    budget: float,
    objective: str,
    oracle_loss_clip: float | None,
    context_columns: tuple[str, ...],
) -> dict:
    source = source_root / f"seed{seed}"
    output = output_root / f"seed{seed}"
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    metadata_path = output / "v2_ppo_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_truth = truth_root / f"truth_seed{seed}.csv"
    truth = pd.read_csv(target_truth)
    # The mandatory logger/backbone is not a specialist measurement channel.
    # Older truth generators omit its explicit quality column, but the runtime
    # environment requires the schema and the backbone has unit quality.
    truth["agent_context_quality_cr1000xe_backbone"] = 1.0
    truth_path = output / "truth_v608_causal.csv"
    truth.to_csv(truth_path, index=False)

    resource_path = Path("reports/v525_stage_b_truth_resource_trace_20260909") / f"resource_trace_seed{seed}.csv"
    if hasattr(prepare, "resource_root") and prepare.resource_root is not None:
        resource_path = prepare.resource_root / f"resource_seed{seed}.csv"
    resource_path = resource_path.resolve()
    helpers = _helpers()
    metadata["truth_csv"] = str(truth_path.relative_to(ROOT))
    metadata["agent_context_columns"] = list(context_columns)
    metadata["agent_alert_context"] = {
        **dict(metadata.get("agent_alert_context", {})),
        "include_event_flag_in_state": False,
        "include_alert_context_features": False,
    }
    metadata["dynamic_resource"] = {
        **dict(metadata.get("dynamic_resource", {})),
        "trace_csv": str(resource_path.relative_to(ROOT)),
        "controller": "entity_heater_trace_v1_with_causal_specialist_targets",
    }
    metadata["constraints"] = {
        **dict(metadata.get("constraints", {})),
        "per_step_budget": float(budget),
        "startup_peak_budget": float(budget),
    }
    metadata["state_columns"] = list(helpers.STATE_COLUMNS)
    metadata["reward_target_columns"] = list(helpers.REWARD_TARGET_COLUMNS)
    uncertainty = dict(metadata.get("uncertainty_proxy", {}))
    process_variance = list(uncertainty.get("process_variance", []))
    if len(process_variance) == 12:
        process_variance.extend([0.01, 0.01, 0.01])
    uncertainty["process_variance"] = process_variance
    metadata["uncertainty_proxy"] = uncertainty

    sensors = load_sensor_specs(str(ROOT / metadata["sensor_cfg"]))
    constraints_meta = dict(metadata.get("constraints", {}))
    constraints = PowerConstraintsV2(
        max_active=constraints_meta.get("max_active"),
        per_step_budget=float(constraints_meta.get("per_step_budget", budget)),
        startup_peak_budget=float(constraints_meta.get("startup_peak_budget", budget)),
        required_sensor_ids=tuple(constraints_meta.get("required_sensor_ids", ["cr1000xe_backbone"])),
    )
    oracle_old = TCNFrozenForecastOracle.load(str(output / "v2_tcn_oracle.pt"), device="cpu")
    partition = dict(metadata["partition_protocol"])
    oracle_truth = truth.iloc[int(partition.get("oracle_start_idx", 0)):int(partition.get("oracle_end_idx", 31500))].reset_index(drop=True)
    diag_spec = importlib.util.spec_from_file_location("diag_v608", ROOT / "scripts" / "27_v2_diagnose_action_landscape.py")
    if diag_spec is None or diag_spec.loader is None:
        raise ImportError("27_v2_diagnose_action_landscape.py")
    diag = importlib.util.module_from_spec(diag_spec)
    diag_spec.loader.exec_module(diag)
    candidate_masks = diag.build_candidate_masks(sensors, constraints, max_candidate_warmup=None)
    target_weights = target_weights_for_objective(objective, oracle_old.cfg.target_weights)
    oracle = helpers.train_oracle(
        oracle_truth,
        sensors,
        constraints,
        oracle_type="tcn",
        lookback=oracle_old.cfg.lookback,
        horizon=oracle_old.cfg.horizon,
        rollout_steps=256,
        tcn_epochs=oracle_old.cfg.epochs,
        tcn_batch_size=oracle_old.cfg.batch_size,
        tcn_lr=oracle_old.cfg.learning_rate,
        tcn_channels=oracle_old.cfg.channels,
        tcn_levels=oracle_old.cfg.levels,
        tcn_device="cpu",
        tcn_loss_clip=(oracle_old.cfg.loss_clip if oracle_loss_clip is None else float(oracle_loss_clip)),
        tcn_use_mask_channels=oracle_old.cfg.use_mask_channels,
        target_weights=target_weights,
        target_scales=oracle_old.cfg.target_scales,
        rollouts_per_policy=1,
        event_fraction=0.67,
        full_open_repeat=1,
        candidate_masks=candidate_masks,
        candidate_mask_repeat=1,
        candidate_mask_limit=0,
        base_freq_s=int(metadata.get("freq_s", 3600)),
        seed=seed,
    )
    oracle_path = output / "v2_tcn_oracle.pt"
    oracle.save(str(oracle_path))
    metadata["oracle_path"] = oracle_path.name
    metadata["candidate_mask_count"] = int(len(candidate_masks))
    metadata["oracle_refit"] = {
        "relation": "causal specialist target increment plus entity heater resource trace",
        "objective": objective,
        "target_weights": None if target_weights is None else list(target_weights),
        "oracle_partition": [int(partition.get("oracle_start_idx", 0)), int(partition.get("oracle_end_idx", 31500))],
        "truth_targets_changed": True,
        "event_labels_changed": False,
        "policy_exact_mode_input": False,
        "loss_clip": float(oracle.cfg.loss_clip),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "asset": str(output), "candidate_mask_count": int(len(candidate_masks))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7177, 7178, 7179, 7180])
    parser.add_argument("--resource-root", type=Path, default=None)
    parser.add_argument("--budget", type=float, default=2.15)
    parser.add_argument("--objective", choices=("ordinary", "group_balanced"), default="ordinary")
    parser.add_argument("--oracle-loss-clip", type=float, default=None)
    parser.add_argument("--context-columns", nargs="*", default=[
        "agent_context_forecast_mode_transport",
        "agent_context_forecast_mode_particle",
        "agent_context_forecast_mode_thermal",
    ])
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.truth_root = args.truth_root.resolve()
    args.output_root = args.output_root.resolve()
    prepare.resource_root = args.resource_root.resolve() if args.resource_root is not None else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = [
        prepare(
            seed, args.source_root, args.truth_root, args.output_root, float(args.budget),
            args.objective, args.oracle_loss_clip, tuple(args.context_columns)
        )
        for seed in args.seeds
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
