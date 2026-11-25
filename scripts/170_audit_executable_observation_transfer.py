"""Audit online subset-value transfer with real dwell-aware execution.

This is a policy-free diagnostic. Candidate-loss regressors are fitted only
on the training partition, then executed through WarmupSchedulingEnv so that
dynamic feasibility, startup, and minimum dwell are enforced by the same
environment used by PD-PPO. It is not PPO evidence and does not expose event
labels or future targets.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2.rollout import run_policy_rollout


def load_geometry():
    path = Path(__file__).with_name("109_v32_audit_subset_forecast_geometry.py")
    spec = importlib.util.spec_from_file_location("geometry", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(root: Path, run_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = root / path
    return candidate if candidate.exists() else run_dir / path


def load_observations(path: Path) -> pd.DataFrame:
    with np.load(path) as data:
        times = np.asarray(data["time_idx"], dtype=int)
        observations = np.asarray(data["agent_observations"], dtype=float)
    if len(times) != len(observations) or len(np.unique(times)) != len(times):
        raise ValueError(f"observation rows are not unique and aligned: {path}")
    frame = pd.DataFrame(observations, index=times)
    frame.index.name = "time_idx"
    frame.columns = [f"obs_{i:04d}" for i in range(frame.shape[1])]
    return frame.reset_index()


def enrich(losses, observations, trace, *, budget: float, fixed_power: float):
    frame = losses.merge(observations, on="time_idx", how="inner", validate="many_to_one")
    trace = trace.set_index("time_idx")
    power_columns = {
        column.removeprefix("resource_effective_power_"): column
        for column in trace.columns
        if column.startswith("resource_effective_power_")
    }
    rows = []
    for row in frame.itertuples(index=False):
        selected = [x for x in str(row.selected_sensor_ids).split(";") if x and x != "cr1000xe_backbone"]
        idx = int(row.time_idx)
        if idx not in trace.index:
            continue
        cost = float(fixed_power)
        for sensor_id in selected:
            if sensor_id not in power_columns:
                raise KeyError(f"missing dynamic power column for {sensor_id}")
            cost += float(trace.loc[idx, power_columns[sensor_id]])
        item = row._asdict()
        item["dynamic_feasible"] = bool(cost <= budget + 1.0e-9)
        rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no loss/observation rows could be aligned")
    return result


class ExecutableValuePolicy:
    def __init__(self, models, candidates):
        self.name = "online_value_policy"
        self.models = models
        self.candidates = np.asarray(candidates, dtype=bool)

    def reset(self):
        pass

    def act_mask(self, env):
        previous = np.asarray(env.previous_action_mask, dtype=bool)
        if int(getattr(env, "dwell_hold_remaining", 0)) > 0:
            return previous.copy()
        state = np.nan_to_num(np.asarray(env._state(), dtype=float).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        available = np.asarray([env.is_mask_executable(mask) for mask in self.candidates], dtype=bool)
        predicted = np.full(len(self.candidates), np.inf, dtype=float)
        for idx, model in self.models.items():
            if available[idx]:
                predicted[idx] = float(model.predict(state)[0])
        if np.isfinite(predicted).any():
            return self.candidates[int(np.argmin(predicted))].copy()
        if env.is_mask_executable(previous):
            return previous.copy()
        raise RuntimeError("no executable candidate available")


class FixedMaskPolicy:
    def __init__(self, mask):
        self.name = "train_selected_static"
        self.mask = np.asarray(mask, dtype=bool)

    def reset(self):
        pass

    def act_mask(self, env):
        if env.is_mask_executable(self.mask):
            return self.mask.copy()
        previous = np.asarray(env.previous_action_mask, dtype=bool)
        if env.is_mask_executable(previous):
            return previous.copy()
        return self.mask.copy()


def make_env(geom, meta, run_dir, *, budget, fixed_power, oracle, seed_offset):
    truth_path = resolve(ROOT, run_dir, meta["truth_csv"])
    truth = geom.merge_dynamic_resource_trace(pd.read_csv(truth_path), meta)
    sensors = geom.load_sensor_specs(resolve(ROOT, run_dir, meta["sensor_cfg"]))
    constraints = geom.constraints_from_metadata(meta, sensors, steady_budget=1000.0, startup_budget=1000.0)
    cfg = geom.env_config_from_metadata(meta, truth, seed=int(meta["seed"]) + int(seed_offset), episode_len=256)
    cfg = replace(cfg, dynamic_resource_budget_w=float(budget), dynamic_resource_fixed_power_w=float(fixed_power), include_event_flag_in_state=False)
    env = geom.load_diagnostic_module().WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
    masks = geom.load_diagnostic_module().build_candidate_masks(sensors, constraints, max_candidate_warmup=None)
    return env, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--train-observations", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--fixed-power", type=float, required=True)
    parser.add_argument("--test-start", type=int, action="append", required=True)
    parser.add_argument("--split-index", type=int, default=70000)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    geom = load_geometry()
    meta = json.loads((args.run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    observations = load_observations(args.train_observations)
    losses = pd.read_csv(args.losses)
    trace = pd.read_csv(resolve(ROOT, args.run_dir, meta["dynamic_resource"]["trace_csv"]))
    frame = enrich(losses, observations, trace, budget=args.budget, fixed_power=args.fixed_power)
    train = frame[(frame.time_idx < int(args.split_index)) & frame.dynamic_feasible].copy()
    if train.empty:
        raise ValueError("no feasible training rows")
    feature_columns = [c for c in train.columns if c.startswith("obs_")]
    models = {}
    for candidate, group in train.groupby("candidate"):
        if len(group) < 40:
            continue
        idx = int(str(candidate).removeprefix("candidate_"))
        model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=20, max_features=0.8, random_state=int(args.seed), n_jobs=1)
        model.fit(group[feature_columns], group["oracle_loss"])
        models[idx] = model
    static_row = train.groupby(["candidate", "selected_sensor_ids"], as_index=False)["oracle_loss"].mean().sort_values("oracle_loss").iloc[0]
    static_idx = int(str(static_row.candidate).removeprefix("candidate_"))
    oracle = geom.load_oracle(args.oracle, "tcn")
    rows = []
    for offset, start in enumerate(args.test_start):
        env, masks = make_env(geom, meta, args.run_dir, budget=args.budget, fixed_power=args.fixed_power, oracle=oracle, seed_offset=30000 + offset)
        result = run_policy_rollout(env, ExecutableValuePolicy(models, masks), steps=int(args.steps), start_idx=int(start))
        env_static, static_masks = make_env(geom, meta, args.run_dir, budget=args.budget, fixed_power=args.fixed_power, oracle=oracle, seed_offset=31000 + offset)
        static = run_policy_rollout(env_static, FixedMaskPolicy(static_masks[static_idx]), steps=int(args.steps), start_idx=int(start))
        transfer_loss = float(np.nanmean(result.oracle_losses))
        static_loss = float(np.nanmean(static.oracle_losses))
        rows.append({"seed": int(args.seed), "start": int(start), "transfer_loss": transfer_loss, "static_loss": static_loss, "transfer_minus_static": transfer_loss - static_loss, "switches": int(np.sum(np.any(result.selected_masks[1:] != result.selected_masks[:-1], axis=1))), "warmup_abort_count": int(result.warmup_abort_count)})
    summary = {"seed": int(args.seed), "rows": rows, "mean_transfer_minus_static": float(np.mean([r["transfer_minus_static"] for r in rows])), "diagnostic_status": "train_only_observable_regressors; real_env_dwell_execution; event_labels_excluded; no_PPO"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / f"seed{args.seed}_window_results.csv", index=False)
    (args.out_dir / f"seed{args.seed}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
