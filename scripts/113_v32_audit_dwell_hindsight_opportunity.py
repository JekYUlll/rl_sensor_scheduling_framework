#!/usr/bin/env python3
"""Measure development-only adaptive opportunity under executable dwell.

This is a privileged ceiling diagnostic.  At each unlocked decision epoch it
evaluates every currently executable candidate over a common frozen-forecaster
lookahead and executes the lowest-cost candidate for the environment's normal
minimum dwell.  It must never be used as a policy target, online feature, or
fair deployment comparator.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.custom_ppo import feasible_candidate_mask, oracle_greedy_candidate_costs  # noqa: E402
from v2.env import WarmupSchedulingEnv  # noqa: E402
from v2.rollout import concat_rollout_results, rollout_metrics, run_policy_rollout  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402


def _load_geometry_module():
    path = Path(__file__).with_name("109_v32_audit_subset_forecast_geometry.py")
    spec = importlib.util.spec_from_file_location("subset_geometry_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DwellHindsightPolicy:
    """Privileged receding-horizon policy used only to estimate opportunity."""

    name = "privileged_common_dwell_hindsight"

    def __init__(self, candidate_masks: np.ndarray, *, lookahead_steps: int) -> None:
        self.candidate_masks = np.asarray(candidate_masks, dtype=bool)
        self.lookahead_steps = max(1, int(lookahead_steps))
        self.decision_rows: list[dict[str, int | float]] = []

    def reset(self) -> None:
        self.decision_rows = []

    def act_mask(self, env: WarmupSchedulingEnv) -> np.ndarray:
        feasible = feasible_candidate_mask(env, self.candidate_masks)
        indices = np.flatnonzero(feasible)
        if indices.size != 1 or int(getattr(env, "dwell_hold_remaining", 0)) <= 0:
            costs = oracle_greedy_candidate_costs(
                env,
                self.candidate_masks,
                lookahead_steps=self.lookahead_steps,
            )
            index = int(indices[np.argmin(costs[indices])])
            self.decision_rows.append(
                {
                    "step_index": int(env.current_idx),
                    "candidate_index": index,
                    "block_cost": float(costs[index]),
                    "dwell_hold_remaining": int(getattr(env, "dwell_hold_remaining", 0)),
                }
            )
        else:
            index = int(indices[0])
        return self.candidate_masks[index]


def _make_env(geometry, meta: dict, truth: pd.DataFrame, sensors: list, constraints, oracle, *, start_seed: int, steps: int):
    cfg = geometry.env_config_from_metadata(
        meta,
        truth,
        seed=int(start_seed),
        episode_len=int(steps),
    )
    return WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        cfg,
        oracle=oracle,
    )


def _run_fixed(geometry, mask: np.ndarray, meta: dict, truth: pd.DataFrame, sensors: list, constraints, oracle, starts: list[int], steps: int):
    policy = geometry.load_diagnostic_module().FixedMaskPolicy(mask, name="fixed_candidate")
    results = []
    for offset, start in enumerate(starts):
        env = _make_env(geometry, meta, truth, sensors, constraints, oracle, start_seed=int(meta.get("seed", 0)) + 2000 + offset, steps=steps)
        results.append(run_policy_rollout(env, policy, steps=steps, start_idx=start))
    return concat_rollout_results(results, policy_name=policy.name)


def audit_run(
    run_dir: Path,
    out_dir: Path,
    *,
    steps: int,
    max_rollouts: int,
    lookahead_steps: int,
    steady_budget: float | None = None,
    startup_budget: float | None = None,
) -> dict:
    geometry = _load_geometry_module()
    meta = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(meta["truth_csv"])
    sensors = load_sensor_specs(meta["sensor_cfg"])
    constraints = geometry.constraints_from_metadata(
        meta,
        sensors,
        steady_budget=steady_budget,
        startup_budget=startup_budget,
    )
    candidates = geometry.load_diagnostic_module().build_candidate_masks(sensors, constraints, max_candidate_warmup=None)
    oracle = geometry.load_oracle(Path(meta["oracle_path"]), str(meta.get("oracle_type", "tcn")))
    starts = [int(x) for x in meta.get("eval_start_indices", [0])][: max(1, int(max_rollouts))]

    fixed_results = [
        _run_fixed(geometry, mask, meta, truth, sensors, constraints, oracle, starts, steps)
        for mask in candidates
    ]
    fixed_losses = [float(rollout_metrics(result)["oracle_loss_mean"]) for result in fixed_results]
    static_index = int(np.nanargmin(np.asarray(fixed_losses, dtype=float)))
    static_result = fixed_results[static_index]

    hindsight_results = []
    decision_rows: list[dict[str, int | float]] = []
    for offset, start in enumerate(starts):
        env = _make_env(geometry, meta, truth, sensors, constraints, oracle, start_seed=int(meta.get("seed", 0)) + 2000 + offset, steps=steps)
        policy = DwellHindsightPolicy(candidates, lookahead_steps=lookahead_steps)
        result = run_policy_rollout(env, policy, steps=steps, start_idx=start)
        hindsight_results.append(result)
        for row in policy.decision_rows:
            decision_rows.append({"rollout": offset, **row})
    hindsight_result = concat_rollout_results(hindsight_results, policy_name="privileged_common_dwell_hindsight")

    static_metrics = rollout_metrics(static_result)
    hindsight_metrics = rollout_metrics(hindsight_result)
    aligned = min(len(static_result.oracle_losses), len(hindsight_result.oracle_losses))
    step_frame = pd.DataFrame(
        {
            "step_index": hindsight_result.step_indices[:aligned],
            "static_oracle_loss": static_result.oracle_losses[:aligned],
            "hindsight_oracle_loss": hindsight_result.oracle_losses[:aligned],
            "static_minus_hindsight": static_result.oracle_losses[:aligned] - hindsight_result.oracle_losses[:aligned],
            "hindsight_switch": np.r_[False, np.any(np.diff(hindsight_result.selected_masks[:aligned], axis=0) != 0, axis=1)] if aligned else np.asarray([], dtype=bool),
        }
    )
    step_frame.to_csv(out_dir / f"dwell_hindsight_steps_seed{int(meta.get('seed', -1))}.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(out_dir / f"dwell_hindsight_decisions_seed{int(meta.get('seed', -1))}.csv", index=False)
    static_sensor_ids = ";".join(spec.sensor_id for spec, active in zip(sensors, candidates[static_index], strict=True) if active)
    summary = {
        "seed": int(meta.get("seed", -1)),
        "steady_budget": float(constraints.per_step_budget),
        "startup_budget": float(constraints.startup_peak_budget),
        "lookahead_steps": int(lookahead_steps),
        "candidate_count": int(len(candidates)),
        "static_candidate_index": static_index,
        "static_selected_sensor_ids": static_sensor_ids,
        "static_oracle_loss_mean": float(static_metrics["oracle_loss_mean"]),
        "hindsight_oracle_loss_mean": float(hindsight_metrics["oracle_loss_mean"]),
        "static_minus_hindsight_loss": float(static_metrics["oracle_loss_mean"] - hindsight_metrics["oracle_loss_mean"]),
        "hindsight_switches_per_step": float(hindsight_metrics["switches_per_step"]),
        "hindsight_warmup_abort_count": int(hindsight_metrics["warmup_abort_count"]),
        "decision_count": int(len(decision_rows)),
        "switch_step_gain_mean": float(step_frame.loc[step_frame.hindsight_switch, "static_minus_hindsight"].mean()) if bool(step_frame.hindsight_switch.any()) else float("nan"),
        "non_switch_step_gain_mean": float(step_frame.loc[~step_frame.hindsight_switch, "static_minus_hindsight"].mean()) if bool((~step_frame.hindsight_switch).any()) else float("nan"),
    }
    (out_dir / f"dwell_hindsight_summary_seed{summary['seed']}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-rollouts", type=int, default=2)
    parser.add_argument("--lookahead-steps", type=int, default=6)
    parser.add_argument("--steady-budget", type=float)
    parser.add_argument("--startup-budget", type=float)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(args.torch_threads)))
    except RuntimeError:
        pass
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        audit_run(
            run_dir,
            args.out_dir,
            steps=int(args.steps),
            max_rollouts=int(args.max_rollouts),
            lookahead_steps=int(args.lookahead_steps),
            steady_budget=args.steady_budget,
            startup_budget=args.startup_budget,
        )
        for run_dir in args.run_dir
    ]
    (args.out_dir / "dwell_hindsight_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
