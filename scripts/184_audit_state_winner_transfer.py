#!/usr/bin/env python3
"""Audit causal heater-state winner transfer on a held-out start."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from v2.rollout import run_policy_rollout
from v2.sensor_spec import load_sensor_specs


def geometry_module():
    path = Path(__file__).with_name("109_v32_audit_subset_forecast_geometry.py")
    spec = importlib.util.spec_from_file_location("geometry", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HeaterWinnerPolicy:
    def __init__(self, winners, static_idx, masks, trace, sensors):
        self.name = "heater_state_winner_lookup"
        self.winners = winners
        self.static_idx = int(static_idx)
        self.masks = np.asarray(masks, dtype=bool)
        self.trace = trace.set_index("time_idx")
        self.sensors = sensors
        self.state_columns = tuple(
            column for column in (
                "resource_heater_on_met_station_core",
                "resource_heater_on_laser_disdrometer",
                "resource_heater_on_radiometer_basic",
                "resource_heater_on_surface_temp_ir",
                "resource_heater_on_fc4_flux",
            ) if column in self.trace.columns
        )

    def reset(self):
        pass

    def _state(self, idx):
        values = [int(self.trace.loc[int(idx), column]) for column in self.state_columns]
        return "heater_" + "".join(str(value) for value in values)

    def act_mask(self, env):
        name = self._state(env.current_idx)
        candidate = self.winners.get(name, self.static_idx)
        desired = self.masks[int(candidate)]
        if env.is_mask_executable(desired):
            return desired.copy()
        previous = np.asarray(env.previous_action_mask, dtype=bool)
        if env.is_mask_executable(previous):
            return previous.copy()
        for mask in self.masks:
            if env.is_mask_executable(mask):
                return mask.copy()
        raise RuntimeError("no executable fallback mask")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geom = geometry_module()
    rows = []
    for train_path in sorted(args.run_root.glob("seed*/train/subset_forecast_geometry_seed*.json")):
        seed = int(train_path.stem.rsplit("seed", 1)[-1])
        test_path = args.run_root / f"seed{seed}" / "test" / train_path.name
        train = json.loads(train_path.read_text())
        test = json.loads(test_path.read_text())
        meta = json.loads((args.assets_root / f"seed{seed}" / "v2_ppo_metadata.json").read_text())
        truth = geom.merge_dynamic_resource_trace(pd.read_csv(meta["truth_csv"]), meta)
        sensors = load_sensor_specs(meta["sensor_cfg"])
        constraints = geom.constraints_from_metadata(meta, sensors, steady_budget=10.0, startup_budget=10.0)
        cfg = geom.env_config_from_metadata(meta, truth, seed=seed, episode_len=256)
        oracle = geom.load_oracle(Path(args.assets_root / f"seed{seed}" / "v2_tcn_oracle.pt"), "tcn")
        oracle.cfg = geom.replace(oracle.cfg, loss_clip=1.0e9)
        masks = geom.load_diagnostic_module().build_candidate_masks(sensors, constraints, max_candidate_warmup=None)
        winners = {str(k): int(str(v).removeprefix("candidate_")) for k, v in train["operating_condition_best_candidates"].items()}
        static_idx = int(str(train["operating_domain_best_static_candidate"]).removeprefix("candidate_"))
        policy = HeaterWinnerPolicy(winners, static_idx, masks, pd.read_csv(meta["dynamic_resource"]["trace_csv"]), sensors)
        env = geom.load_diagnostic_module().WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
        test_start = 82600
        transfer = run_policy_rollout(env, policy, steps=256, start_idx=test_start)
        static_policy = HeaterWinnerPolicy({}, static_idx, masks, pd.read_csv(meta["dynamic_resource"]["trace_csv"]), sensors)
        env_static = geom.load_diagnostic_module().WarmupSchedulingEnv(truth, sensors, constraints, cfg, oracle=oracle)
        static = run_policy_rollout(env_static, static_policy, steps=256, start_idx=test_start)
        transfer_loss = float(np.nanmean(transfer.oracle_losses))
        static_loss = float(np.nanmean(static.oracle_losses))
        rows.append({"seed": seed, "transfer_loss": transfer_loss, "static_loss": static_loss, "transfer_minus_static": transfer_loss-static_loss, "transfer_switches": int(np.sum(np.any(transfer.selected_masks[1:] != transfer.selected_masks[:-1], axis=1))), "warmup_abort_count": int(transfer.warmup_abort_count)})
    summary = {"seeds": rows, "all_positive_margin": all(row["transfer_minus_static"] < 0 for row in rows), "diagnostic_status": "chronological_observable_heater_winner_lookup; held_out_start; no_PPO"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
