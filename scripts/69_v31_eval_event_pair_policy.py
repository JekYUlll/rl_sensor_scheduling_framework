#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.env import WarmupEnvConfig  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(path_value: str | Path, *, run_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path, run_dir / path.name, ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve {path_value!r} from {run_dir}")


def parse_policy_spec(value: str) -> tuple[str, tuple[str, ...], tuple[str, ...], int]:
    try:
        name, payload = str(value).split("=", 1)
        calm_text, event_text, lookahead_text = payload.split("|", 2)
    except ValueError as exc:
        raise ValueError(
            "Policy spec must be name=calm_id,calm_id|event_id,event_id|lookahead"
        ) from exc
    calm = tuple(part for part in calm_text.split(",") if part)
    event = tuple(part for part in event_text.split(",") if part)
    if not calm or not event:
        raise ValueError(f"Policy spec requires non-empty calm and event masks: {value}")
    return str(name), calm, event, int(lookahead_text)


def parse_mask_pool(text: str) -> tuple[tuple[str, ...], ...]:
    masks: list[tuple[str, ...]] = []
    for mask_text in str(text).split(";"):
        mask = tuple(part for part in mask_text.split(",") if part)
        if mask:
            masks.append(mask)
    return tuple(masks)


def parse_cyclic_policy_spec(value: str) -> tuple[str, tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...], int, int]:
    try:
        name, payload = str(value).split("=", 1)
        calm_text, event_text, lookahead_text, dwell_text = payload.split("|", 3)
    except ValueError as exc:
        raise ValueError(
            "Cyclic spec must be name=calm1;calm2|event1;event2|lookahead|dwell"
        ) from exc
    calm = parse_mask_pool(calm_text)
    event = parse_mask_pool(event_text)
    if not calm or not event:
        raise ValueError(f"Cyclic policy spec requires non-empty calm/event mask pools: {value}")
    return str(name), calm, event, int(lookahead_text), int(dwell_text)


def mask_for_sensor_ids(sensors: list[Any], sensor_ids: tuple[str, ...]) -> np.ndarray:
    wanted = {str(sensor_id) for sensor_id in sensor_ids}
    mask = np.asarray([str(spec.sensor_id) in wanted for spec in sensors], dtype=bool)
    missing = sorted(wanted - {str(spec.sensor_id) for spec in sensors})
    if missing:
        raise ValueError(f"Unknown sensors in policy mask: {missing}")
    return mask


def mask_pool_for_sensor_ids(sensors: list[Any], mask_specs: tuple[tuple[str, ...], ...]) -> tuple[np.ndarray, ...]:
    return tuple(mask_for_sensor_ids(sensors, item) for item in mask_specs)


class EventPairMaskPolicy:
    def __init__(
        self,
        *,
        name: str,
        calm_mask: np.ndarray,
        event_mask: np.ndarray,
        lookahead_steps: int,
    ) -> None:
        self.name = str(name)
        self.calm_mask = np.asarray(calm_mask, dtype=bool).reshape(-1)
        self.event_mask = np.asarray(event_mask, dtype=bool).reshape(-1)
        self.lookahead_steps = int(max(0, int(lookahead_steps)))

    def reset(self) -> None:
        return None

    def act_mask(self, env: object) -> np.ndarray:
        event_flags = np.asarray(getattr(env, "event_flags"), dtype=bool)
        current_idx = int(getattr(env, "current_idx"))
        end_idx = min(len(event_flags), current_idx + self.lookahead_steps + 1)
        trigger = bool(np.any(event_flags[current_idx:end_idx]))
        return self.event_mask.copy() if trigger else self.calm_mask.copy()

    def act_scores(self, env: object) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


class CyclicEventMaskPolicy:
    def __init__(
        self,
        *,
        name: str,
        calm_masks: tuple[np.ndarray, ...],
        event_masks: tuple[np.ndarray, ...],
        lookahead_steps: int,
        dwell_steps: int,
    ) -> None:
        self.name = str(name)
        self.calm_masks = tuple(np.asarray(mask, dtype=bool).reshape(-1) for mask in calm_masks)
        self.event_masks = tuple(np.asarray(mask, dtype=bool).reshape(-1) for mask in event_masks)
        self.lookahead_steps = int(max(0, int(lookahead_steps)))
        self.dwell_steps = int(max(1, int(dwell_steps)))

    def reset(self) -> None:
        return None

    def act_mask(self, env: object) -> np.ndarray:
        event_flags = np.asarray(getattr(env, "event_flags"), dtype=bool)
        current_idx = int(getattr(env, "current_idx"))
        start_idx = int(getattr(env, "episode_start_idx", current_idx))
        end_idx = min(len(event_flags), current_idx + self.lookahead_steps + 1)
        trigger = bool(np.any(event_flags[current_idx:end_idx]))
        masks = self.event_masks if trigger else self.calm_masks
        phase = max(0, current_idx - start_idx) // self.dwell_steps
        return masks[int(phase) % len(masks)].copy()

    def act_scores(self, env: object) -> np.ndarray:
        mask = self.act_mask(env)
        return np.where(mask, 1.0, -1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate event-pair mask policies on a saved V3.1 split run.")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--env-min-dwell-steps", type=int, default=None)
    parser.add_argument("--env-harvest-per-step", type=float, default=None)
    parser.add_argument("--env-energy-capacity", type=float, default=None)
    parser.add_argument("--env-initial-energy", type=float, default=None)
    parser.add_argument("--env-reserve-energy", type=float, default=None)
    parser.add_argument(
        "--policy-spec",
        action="append",
        default=[],
        help="Format: name=calm_id,calm_id|event_id,event_id|lookahead",
    )
    parser.add_argument(
        "--cyclic-policy-spec",
        action="append",
        default=[],
        help="Format: name=calm1_ids;calm2_ids|event1_ids;event2_ids|lookahead|dwell",
    )
    args = parser.parse_args()
    if not args.policy_spec and not args.cyclic_policy_spec:
        parser.error("At least one --policy-spec or --cyclic-policy-spec is required")

    helpers = load_module(ROOT / "scripts" / "23_v2_train_ppo.py", "_v31_eval_event_pair_helpers")
    ops = load_module(
        ROOT / "scripts" / "64_v31_eval_saved_run_operational_baselines.py",
        "_v31_saved_run_ops",
    )

    source_run_dir = Path(args.source_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source_run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(resolve(str(metadata["truth_csv"]), run_dir=source_run_dir))
    sensors = load_sensor_specs(resolve(str(metadata["sensor_cfg"]), run_dir=source_run_dir))
    constraints = ops.constraints_from_metadata(metadata)
    oracle = ops.load_oracle(metadata, run_dir=source_run_dir, device=str(args.oracle_device))
    norm_mean, norm_std = ops.normalization_stats(
        truth,
        state_columns=helpers.STATE_COLUMNS,
        metadata=metadata,
    )
    eval_steps = ops.infer_eval_steps(metadata, run_dir=source_run_dir)
    eval_starts = tuple(int(x) for x in metadata["eval_start_indices"])
    env_kwargs = ops.env_kwargs_from_metadata(metadata)
    if args.env_min_dwell_steps is not None:
        env_kwargs["min_dwell_steps"] = int(max(1, int(args.env_min_dwell_steps)))
    if args.env_harvest_per_step is not None:
        env_kwargs["harvest_per_step"] = float(args.env_harvest_per_step)
    if args.env_energy_capacity is not None:
        env_kwargs["energy_capacity"] = float(args.env_energy_capacity)
    if args.env_initial_energy is not None:
        env_kwargs["initial_energy"] = float(args.env_initial_energy)
    if args.env_reserve_energy is not None:
        env_kwargs["reserve_energy"] = float(args.env_reserve_energy)
    eval_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=int(metadata["lookback"]),
        episode_len=int(eval_steps),
        seed=int(metadata["seed"]) + 12_000,
        base_freq_s=int(metadata["freq_s"]),
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        **env_kwargs,
    )

    metrics_rows: list[dict[str, object]] = []
    for spec_text in args.policy_spec:
        name, calm_ids, event_ids, lookahead = parse_policy_spec(str(spec_text))
        policy = EventPairMaskPolicy(
            name=name,
            calm_mask=mask_for_sensor_ids(sensors, calm_ids),
            event_mask=mask_for_sensor_ids(sensors, event_ids),
            lookahead_steps=int(lookahead),
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
        metrics_rows.append(dict(metrics))
        save_rollout_npz(
            out_dir / f"rollout_{name}.npz",
            result,
            sensor_ids=[str(spec.sensor_id) for spec in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )
    for spec_text in args.cyclic_policy_spec:
        name, calm_ids, event_ids, lookahead, dwell = parse_cyclic_policy_spec(str(spec_text))
        policy = CyclicEventMaskPolicy(
            name=name,
            calm_masks=mask_pool_for_sensor_ids(sensors, calm_ids),
            event_masks=mask_pool_for_sensor_ids(sensors, event_ids),
            lookahead_steps=int(lookahead),
            dwell_steps=int(dwell),
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
        metrics_rows.append(dict(metrics))
        save_rollout_npz(
            out_dir / f"rollout_{name}.npz",
            result,
            sensor_ids=[str(spec.sensor_id) for spec in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("oracle_loss_mean")
    metrics_path = out_dir / "event_pair_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(metrics_path)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
