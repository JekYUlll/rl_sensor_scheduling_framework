#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_row(source_metrics: pd.DataFrame, pattern: str) -> dict[str, Any]:
    policy_col = "policy" if "policy" in source_metrics.columns else "policy_name"
    regex = re.compile(str(pattern))
    mask = source_metrics[policy_col].astype(str).map(lambda value: bool(regex.search(value)))
    candidates = source_metrics[mask].copy()
    if candidates.empty:
        raise ValueError(f"No reference policies matched pattern: {pattern}")
    row = candidates.sort_values("oracle_loss_mean").iloc[0].to_dict()
    row["policy"] = str(row.get(policy_col, ""))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast explicit subtype replay without static-candidate sweep.")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--env-min-dwell-steps", type=int, default=None)
    parser.add_argument("--lead-steps", nargs="+", type=int, default=[0, 2, 4, 8, 10])
    parser.add_argument("--explicit-calm-sensors", nargs="*", required=True)
    parser.add_argument("--explicit-particle-sensors", nargs="*", required=True)
    parser.add_argument("--explicit-flux-sensors", nargs="*", required=True)
    parser.add_argument("--explicit-thermal-sensors", nargs="*", default=None)
    parser.add_argument("--explicit-policy-name", default="split_subtype_explicit_teacher_fast")
    parser.add_argument(
        "--reference-policy-regex",
        default=(
            "^(duty_constrained_.*|feasible_static_projected|"
            "validation_selected_static|round_robin|aoi)$"
        ),
    )
    parser.add_argument("--min-margin-abs", type=float, default=0.001)
    parser.add_argument("--min-margin-rel", type=float, default=0.001)
    args = parser.parse_args()

    helpers = load_module(ROOT / "scripts" / "23_v2_train_ppo.py", "_v31_explicit_fast_helpers")
    ops = load_module(
        ROOT / "scripts" / "64_v31_eval_saved_run_operational_baselines.py",
        "_v31_explicit_fast_ops",
    )
    gate = load_module(ROOT / "scripts" / "70_v31_split_replay_gate.py", "_v31_explicit_fast_gate")

    source_run_dir = Path(args.source_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source_run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth = helpers.ensure_state_columns(pd.read_csv(gate.resolve(str(metadata["truth_csv"]), run_dir=source_run_dir)))
    sensors = gate.load_sensor_specs(gate.resolve(str(metadata["sensor_cfg"]), run_dir=source_run_dir))
    constraints = ops.constraints_from_metadata(metadata)
    oracle, eval_cfg, _static_cfg, eval_steps, eval_starts = gate.build_eval_cfg(
        helpers=helpers,
        ops=ops,
        metadata=metadata,
        truth=truth,
        run_dir=source_run_dir,
        env_min_dwell_steps=args.env_min_dwell_steps,
        oracle_device=str(args.oracle_device),
    )
    candidate_masks = helpers.build_projected_candidate_masks(sensors, constraints)

    if "event_subtype_id" not in truth.columns:
        raise ValueError("Fast explicit replay requires truth column event_subtype_id")
    subtype_ids = truth["event_subtype_id"].to_numpy(dtype=int)
    calm_mask = gate.mask_from_sensor_ids(sensors, args.explicit_calm_sensors, label="--explicit-calm-sensors")
    particle_mask = gate.mask_from_sensor_ids(
        sensors,
        args.explicit_particle_sensors,
        label="--explicit-particle-sensors",
    )
    flux_mask = gate.mask_from_sensor_ids(sensors, args.explicit_flux_sensors, label="--explicit-flux-sensors")
    thermal_mask = (
        gate.mask_from_sensor_ids(sensors, args.explicit_thermal_sensors, label="--explicit-thermal-sensors")
        if gate.parse_sensor_id_tokens(args.explicit_thermal_sensors)
        else calm_mask.copy()
    )
    mask_specs = {
        "calm": calm_mask,
        "particle": particle_mask,
        "flux": flux_mask,
        "thermal": thermal_mask,
    }
    action_indices = {
        label: gate.candidate_index_for_mask(candidate_masks, mask)
        for label, mask in mask_specs.items()
    }

    rows: list[dict[str, Any]] = []
    for lead in sorted({max(0, int(x)) for x in args.lead_steps}):
        name = f"{args.explicit_policy_name}_l{int(lead)}"
        policy = gate.SubtypeMaskPolicy(
            name=name,
            subtype_ids=subtype_ids,
            calm_mask=calm_mask,
            subtype_masks={1: particle_mask, 2: flux_mask, 3: thermal_mask},
            lookahead_steps=int(lead),
        )
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(eval_steps),
            start_indices=eval_starts,
        )
        row = gate.append_loss_split(dict(metrics), result)
        row = gate.append_subtype_loss_split(row, result, truth)
        row["policy"] = name
        row["replay_family"] = "subtype_explicit_fast"
        row["lead_steps"] = int(lead)
        row["dwell_steps"] = 0
        for label, mask in mask_specs.items():
            row[f"{label}_action_idx"] = int(action_indices[label])
            row[f"{label}_sensor_ids"] = gate.sensor_ids_for_mask(sensors, mask)
        rows.append(row)
        gate.save_rollout_npz(
            out_dir / f"rollout_{name}.npz",
            result,
            sensor_ids=[str(spec.sensor_id) for spec in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )

    replay_table = pd.DataFrame(rows).sort_values("oracle_loss_mean").reset_index(drop=True)
    replay_table.to_csv(out_dir / "explicit_replay_fast_metrics.csv", index=False)
    source_metrics = pd.read_csv(source_run_dir / "v2_custom_ppo_metrics.csv")
    source_reference = reference_row(source_metrics, str(args.reference_policy_regex))
    source_loss = float(source_reference["oracle_loss_mean"])
    best = replay_table.iloc[0].to_dict()
    best_loss = float(best["oracle_loss_mean"])
    required_loss = min(
        source_loss - float(args.min_margin_abs),
        source_loss * (1.0 - float(args.min_margin_rel)),
    )
    summary = {
        "source_run_dir": str(source_run_dir),
        "best_policy": str(best["policy"]),
        "best_oracle_loss_mean": best_loss,
        "source_reference_policy": str(source_reference["policy"]),
        "source_reference_oracle_loss_mean": source_loss,
        "required_source_loss": float(required_loss),
        "source_gate_pass": bool(best_loss <= required_loss),
        "margin_vs_source_reference": float(source_loss - best_loss),
        "lead_steps": [int(x) for x in sorted({max(0, int(x)) for x in args.lead_steps})],
        "explicit_action_indices": action_indices,
        "explicit_sensor_ids": {
            label: gate.sensor_ids_for_mask(sensors, mask)
            for label, mask in mask_specs.items()
        },
    }
    (out_dir / "explicit_replay_fast_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(replay_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
