"""Dump deployable agent observations from a fixed mandatory-backbone replay."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v2.rollout import run_policy_rollout
from v2.policies import RoundRobinScorePolicy


ROOT = Path(__file__).resolve().parents[1]


def load_geometry_module():
    path = Path(__file__).with_name("109_v32_audit_subset_forecast_geometry.py")
    spec = importlib.util.spec_from_file_location("geometry_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump_run(run_dir: Path, out_dir: Path, steps: int, starts: list[int], fixed_power: float, budget: float, probe_policy: str) -> dict:
    geom = load_geometry_module()
    meta = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth = geom.merge_dynamic_resource_trace(pd.read_csv(meta["truth_csv"]), meta)
    sensors = geom.load_sensor_specs(meta["sensor_cfg"])
    constraints = geom.constraints_from_metadata(meta, sensors, steady_budget=1000.0, startup_budget=1000.0)
    oracle_path = Path(meta["oracle_path"])
    if not oracle_path.is_absolute():
        # Asset manifests may store either a repository-relative path or a
        # path relative to the run directory.  Prefer the repository-relative
        # path when it already exists; otherwise use the legacy run-dir form.
        repository_path = ROOT / oracle_path
        oracle_path = repository_path if repository_path.exists() else run_dir / oracle_path
    oracle = geom.load_oracle(oracle_path, str(meta.get("oracle_type", "tcn")))
    masks = geom.load_diagnostic_module().build_candidate_masks(sensors, constraints, max_candidate_warmup=None)
    if probe_policy == "round_robin":
        policy = RoundRobinScorePolicy(n_sensors=len(sensors), group_size=max(1, len(sensors) // 3))
    else:
        policy = geom.FeasibleFixedMaskPolicy(masks[0], masks, name="candidate_000")
    rollouts = []
    for offset, start in enumerate(starts):
        cfg = geom.env_config_from_metadata(meta, truth, seed=int(meta.get("seed", 42)) + 1000 + offset, episode_len=steps)
        cfg = replace(
            cfg,
            dynamic_resource_budget_w=float(budget),
            dynamic_resource_fixed_power_w=float(fixed_power),
            # The exact simulator event flag is privileged at deployment and
            # must not be present in this online-transfer diagnostic.
            include_event_flag_in_state=False,
        )
        env = geom.load_diagnostic_module().WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
        rollouts.append(run_policy_rollout(env, policy, steps=steps, start_idx=int(start)))
    time_idx = np.concatenate([r.step_indices for r in rollouts])
    observations = np.concatenate([r.agent_observations for r in rollouts], axis=0)
    selected = np.concatenate([r.selected_masks for r in rollouts], axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "online_observation_probe.npz", time_idx=time_idx, agent_observations=observations, selected_masks=selected)
    summary = {
        "seed": int(meta.get("seed", -1)),
        "candidate": "candidate_000",
        "probe_policy": probe_policy,
        "starts": starts,
        "steps": int(steps),
        "rows": int(len(time_idx)),
        "observation_dim": int(observations.shape[1]),
        "dynamic_budget": float(budget),
        "fixed_power": float(fixed_power),
        "online_only": True,
        "event_labels_used": False,
        "event_flag_excluded_from_observation": True,
        "target_values_used": False,
    }
    (out_dir / "online_observation_probe.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--start-index", type=int, action="append", required=True)
    parser.add_argument("--dynamic-resource-budget", type=float, default=3.0)
    parser.add_argument("--dynamic-resource-fixed-power", type=float, default=0.4104)
    parser.add_argument("--probe-policy", choices=("fixed", "round_robin"), default="fixed")
    args = parser.parse_args()
    dump_run(args.run_dir, args.out_dir, args.steps, args.start_index, args.dynamic_resource_fixed_power, args.dynamic_resource_budget, args.probe_policy)


if __name__ == "__main__":
    main()
