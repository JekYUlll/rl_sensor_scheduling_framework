#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.custom_ppo import CustomPPO, CustomPPOConfig, evaluate_custom_ppo  # noqa: E402
from v2.env import WarmupEnvConfig  # noqa: E402
from v2.forecast_eval import load_oracle_from_metadata  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402


EPISODE_TYPES = {
    "all": -1.0,
    "calm": 0.0,
    "mixed": 0.5,
    "event": 1.0,
}


def condition_label(episode_type: str) -> str:
    if str(episode_type) == "event-heavy":
        return "event"
    return str(episode_type)


def load_train_helpers() -> Any:
    path = ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("_v2_train_ppo_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(path: str | Path, *, run_dir: Path) -> Path:
    source = Path(path)
    if source.exists():
        return source
    for base in (Path.cwd(), ROOT, run_dir, *run_dir.parents):
        candidate = base / source
        if candidate.exists():
            return candidate
    return source


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def seed_from_run_dir(run_dir: Path) -> int:
    for part in run_dir.name.split("_"):
        if part.startswith("seed"):
            try:
                return int(part.replace("seed", ""))
            except ValueError:
                pass
    return 0


def budget_from_run_dir(run_dir: Path) -> float:
    for part in run_dir.name.split("_"):
        if part.startswith("budget"):
            try:
                return float(part.replace("budget", "").replace("p", "."))
            except ValueError:
                pass
    return float("nan")


def constraints_from_metadata(metadata: dict[str, Any], *, budget_override: float | None, peak_override: float | None) -> PowerConstraintsV2:
    payload = dict(metadata.get("constraints", {}))
    coverage_groups = []
    for item in payload.get("coverage_groups", []):
        coverage_groups.append((str(item.get("name", "")), tuple(str(x) for x in item.get("sensor_ids", []))))
    return PowerConstraintsV2(
        max_active=payload.get("max_active"),
        per_step_budget=float(budget_override) if budget_override is not None else payload.get("per_step_budget"),
        startup_peak_budget=float(peak_override) if peak_override is not None else payload.get("startup_peak_budget"),
        required_sensor_ids=tuple(str(x) for x in payload.get("required_sensor_ids", [])),
        coverage_groups=tuple(coverage_groups),
    )


def env_kwargs_from_metadata(metadata: dict[str, Any]) -> dict[str, float | bool]:
    energy = dict(metadata.get("energy_account", {}))
    event_multiplier = metadata.get("event_reward_multiplier", 1.0)
    if event_multiplier is None:
        event_multiplier = 1.0
    return {
        "event_reward_multiplier": float(event_multiplier),
        "energy_account_enabled": bool(energy.get("enabled", False)),
        "energy_capacity": float(energy.get("energy_capacity", 0.0)),
        "initial_energy": float(energy.get("initial_energy", 0.0)),
        "harvest_per_step": float(energy.get("harvest_per_step", 0.0)),
        "reserve_energy": float(energy.get("reserve_energy", 0.0)),
        "lambda_energy_deficit": float(energy.get("lambda_energy_deficit", 1.0)),
        "soc_soft_penalty_buffer": float(energy.get("soc_soft_penalty_buffer", 0.0)),
        "lambda_soc_soft_penalty": float(energy.get("lambda_soc_soft_penalty", 0.0)),
    }


def load_custom_trainer(
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    truth: pd.DataFrame,
    sensors: list[Any],
    constraints: PowerConstraintsV2,
    oracle: Any,
    device: str,
) -> CustomPPO:
    import torch

    model_path = resolve_path(metadata.get("model_path", run_dir / "custom_ppo.pt"), run_dir=run_dir)
    if not model_path.exists():
        candidate = run_dir / "custom_ppo.pt"
        if candidate.exists():
            model_path = candidate
    checkpoint = torch.load(str(model_path), map_location=str(device), weights_only=False)
    cfg_raw = dict(checkpoint.get("cfg", {}))
    allowed = {field.name for field in dataclasses.fields(CustomPPOConfig)}
    cfg_raw = {key: value for key, value in cfg_raw.items() if key in allowed}
    cfg_raw["device"] = str(device)
    cfg = CustomPPOConfig(**cfg_raw)
    env_cfg = WarmupEnvConfig(
        state_columns=tuple(metadata.get("reward_state_columns", metadata.get("state_columns", load_train_helpers().STATE_COLUMNS))),
        reward_target_columns=tuple(metadata.get("reward_target_columns", load_train_helpers().REWARD_TARGET_COLUMNS)),
        lookback=int(metadata.get("lookback", getattr(oracle.cfg, "lookback", 20))),
        episode_len=int(metadata.get("eval_steps", 1024)),
        seed=int(metadata.get("seed", seed_from_run_dir(run_dir))),
        base_freq_s=int(metadata.get("freq_s", 10800)),
        lambda_warmup_abort=float(metadata.get("reward_shaping", {}).get("lambda_warmup_abort", 0.08)),
        lambda_switch=float(metadata.get("reward_shaping", {}).get("lambda_switch", 0.002)),
        **env_kwargs_from_metadata(metadata),
    )
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=env_cfg,
        oracle=oracle,
        candidate_masks=np.asarray(checkpoint["candidate_masks"], dtype=bool),
        cfg=cfg,
        candidate_prior_logits=checkpoint.get("candidate_prior_logits"),
    )
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.history = list(checkpoint.get("history", []))
    return trainer


def select_condition_start_indices(
    truth: pd.DataFrame,
    *,
    steps: int,
    horizon: int,
    n_rollouts: int,
    episode_type: str,
    seed: int,
    event_column: str = "event_flag",
    stride_divisor: int = 8,
    calm_max_event_rate: float = 0.20,
    mixed_min_event_rate: float = 0.35,
    mixed_max_event_rate: float = 0.65,
    event_min_event_rate: float = 0.75,
    strict: bool = False,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    max_start = max(0, len(truth) - int(steps) - int(horizon) - 1)
    if max_start <= 0:
        return (0,), {
            "candidate_count": 1,
            "selected_event_rates": [],
            "warning": "truth sequence is shorter than one requested evaluation window",
        }
    flags = (
        truth[event_column].astype(bool).to_numpy()
        if event_column in truth.columns
        else np.zeros(len(truth), dtype=bool)
    )
    stride = max(1, int(steps) // max(1, int(stride_divisor)))
    starts = np.arange(0, max_start + 1, stride, dtype=int)
    if starts.size == 0:
        return (0,), {
            "candidate_count": 1,
            "selected_event_rates": [],
            "warning": "no valid start index found",
        }
    rates = np.asarray([float(np.mean(flags[start : start + int(steps)])) for start in starts], dtype=float)
    episode = condition_label(episode_type)
    warning = ""
    if episode == "all":
        candidate_idx = np.arange(starts.size, dtype=int)
    elif episode == "calm":
        mask = rates < float(calm_max_event_rate)
        target = 0.0
        fallback_order = np.argsort(rates)
    elif episode == "event":
        mask = rates > float(event_min_event_rate)
        target = 1.0
        fallback_order = np.argsort(-rates)
    else:
        target = 0.5 * (float(mixed_min_event_rate) + float(mixed_max_event_rate))
        mask = (rates >= float(mixed_min_event_rate)) & (rates <= float(mixed_max_event_rate))
        fallback_order = np.argsort(np.abs(rates - target))
    if episode != "all":
        candidate_idx = np.flatnonzero(mask)
        if candidate_idx.size < int(n_rollouts):
            warning = (
                f"{episode} candidate pool has {candidate_idx.size} windows for {n_rollouts} requested "
                f"rollouts; event-rate range in scan is [{float(np.min(rates)):.3f}, {float(np.max(rates)):.3f}]"
            )
            if strict:
                raise RuntimeError(warning)
            needed = max(int(n_rollouts) * 4, int(n_rollouts))
            candidate_idx = fallback_order[:needed]
        else:
            order_within = candidate_idx[np.argsort(np.abs(rates[candidate_idx] - target))]
            candidate_idx = order_within[: max(int(n_rollouts) * 4, int(n_rollouts))]
    top = starts[candidate_idx]
    rng = np.random.default_rng(int(seed))
    if top.size >= int(n_rollouts):
        chosen = rng.choice(top, size=int(n_rollouts), replace=False)
    else:
        chosen = rng.choice(starts, size=int(n_rollouts), replace=True)
    chosen = np.sort(chosen)
    chosen_rates = [float(np.mean(flags[int(idx) : int(idx) + int(steps)])) for idx in chosen]
    diagnostics = {
        "candidate_count": int(candidate_idx.size),
        "scan_count": int(starts.size),
        "stride": int(stride),
        "min_event_rate_scanned": float(np.min(rates)),
        "max_event_rate_scanned": float(np.max(rates)),
        "thresholds": {
            "calm_max_event_rate": float(calm_max_event_rate),
            "mixed_min_event_rate": float(mixed_min_event_rate),
            "mixed_max_event_rate": float(mixed_max_event_rate),
            "event_min_event_rate": float(event_min_event_rate),
        },
        "selected_event_rates": chosen_rates,
    }
    if warning:
        diagnostics["warning"] = warning
        print(f"[warn] {warning}", flush=True)
    return tuple(int(x) for x in chosen), diagnostics


def evaluate_episode_type(
    *,
    run_dir: Path,
    episode_type: str,
    out_dir: Path,
    metadata: dict[str, Any],
    truth: pd.DataFrame,
    sensors: list[Any],
    constraints: PowerConstraintsV2,
    oracle: Any,
    trainer: CustomPPO,
    steps: int,
    n_rollouts: int,
    seed: int,
    per_step_budget: float | None,
    startup_peak_budget: float | None,
    stride_divisor: int,
    calm_max_event_rate: float,
    mixed_min_event_rate: float,
    mixed_max_event_rate: float,
    event_min_event_rate: float,
    strict_condition_bands: bool,
    skip_summary_eval: bool,
) -> None:
    helpers = load_train_helpers()
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(metadata.get("horizon", getattr(oracle.cfg, "horizon", 8)))
    start_indices, condition_diagnostics = select_condition_start_indices(
        truth,
        steps=int(steps),
        horizon=int(horizon),
        n_rollouts=int(n_rollouts),
        episode_type=str(episode_type),
        seed=int(seed),
        stride_divisor=int(stride_divisor),
        calm_max_event_rate=float(calm_max_event_rate),
        mixed_min_event_rate=float(mixed_min_event_rate),
        mixed_max_event_rate=float(mixed_max_event_rate),
        event_min_event_rate=float(event_min_event_rate),
        strict=bool(strict_condition_bands),
    )
    eval_cfg = WarmupEnvConfig(
        state_columns=tuple(helpers.STATE_COLUMNS),
        reward_target_columns=tuple(metadata.get("reward_target_columns", helpers.REWARD_TARGET_COLUMNS)),
        lookback=int(metadata.get("lookback", getattr(oracle.cfg, "lookback", 20))),
        episode_len=int(steps),
        seed=int(seed) + 9000,
        base_freq_s=int(metadata.get("freq_s", 10800)),
        lambda_warmup_abort=float(metadata.get("reward_shaping", {}).get("lambda_warmup_abort", 0.08)),
        lambda_switch=float(metadata.get("reward_shaping", {}).get("lambda_switch", 0.002)),
        **env_kwargs_from_metadata(metadata),
    )
    custom_result, _ = evaluate_custom_ppo(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=int(steps),
        start_indices=start_indices,
    )
    save_rollout_npz(out_dir / "rollout_custom_ppo.npz", custom_result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    full_result, _ = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=PowerConstraintsV2(),
        cfg=eval_cfg,
        oracle=oracle,
        policy=helpers.FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        steps=int(steps),
        start_indices=start_indices,
    )
    save_rollout_npz(out_dir / "rollout_full_open_unconstrained.npz", full_result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    for policy in helpers.default_policies(len(sensors), seed=int(seed) + 100):
        result, _ = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(steps),
            start_indices=start_indices,
        )
        save_rollout_npz(out_dir / f"rollout_{result.policy_name}.npz", result, sensor_ids=[s.sensor_id for s in sensors], state_columns=helpers.STATE_COLUMNS)

    meta = dict(metadata)
    meta["eval_start_indices"] = [int(x) for x in start_indices]
    meta["eval_steps"] = int(steps)
    meta["eval_rollouts"] = int(n_rollouts)
    meta["condition_episode_type"] = str(episode_type)
    meta["condition_event_rates"] = [
        float(np.mean(truth["event_flag"].astype(bool).to_numpy()[idx : idx + int(steps)]))
        for idx in start_indices
    ] if "event_flag" in truth.columns else []
    meta["condition_sampling"] = condition_diagnostics
    meta["truth_csv"] = str(resolve_path(metadata["truth_csv"], run_dir=run_dir))
    meta["oracle_path"] = str(resolve_path(metadata["oracle_path"], run_dir=run_dir))
    (out_dir / "v2_ppo_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    eval_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "24_v2_evaluate_rollouts.py"),
        "--run-dir",
        str(out_dir),
    ]
    if per_step_budget is not None:
        eval_cmd.extend(["--per-step-budget", str(per_step_budget)])
    if startup_peak_budget is not None:
        eval_cmd.extend(["--startup-peak-budget", str(startup_peak_budget)])
    if not bool(skip_summary_eval):
        subprocess.run(eval_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate a trained v2 PD-PPO run on calm/mixed/event episodes.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-root", default="reports/v2_supplement_experiments/E1_condition_eval")
    parser.add_argument("--episode-types", nargs="+", default=list(EPISODE_TYPES))
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--rollouts", type=int, default=6)
    parser.add_argument("--per-step-budget", type=float, default=None)
    parser.add_argument("--startup-peak-budget", type=float, default=None)
    parser.add_argument("--forecast-oracle-device", default="cpu")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--condition-stride-divisor", type=int, default=8)
    parser.add_argument("--calm-max-event-rate", type=float, default=0.20)
    parser.add_argument("--mixed-min-event-rate", type=float, default=0.35)
    parser.add_argument("--mixed-max-event-rate", type=float, default=0.65)
    parser.add_argument("--event-min-event-rate", type=float, default=0.75)
    parser.add_argument("--strict-condition-bands", action="store_true")
    parser.add_argument("--skip-summary-eval", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    truth_path = resolve_path(metadata["truth_csv"], run_dir=run_dir)
    sensor_cfg = resolve_path(metadata.get("sensor_cfg", "configs/sensors/windblown_sensors_balanced.yaml"), run_dir=run_dir)
    truth = pd.read_csv(truth_path)
    sensors = load_sensor_specs(sensor_cfg)
    constraints = constraints_from_metadata(
        metadata,
        budget_override=args.per_step_budget,
        peak_override=args.startup_peak_budget,
    )
    oracle = load_oracle_from_metadata(metadata, run_dir=run_dir, device=str(args.forecast_oracle_device))
    trainer = load_custom_trainer(
        run_dir=run_dir,
        metadata=metadata,
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        oracle=oracle,
        device=str(args.policy_device),
    )
    budget = float(args.per_step_budget) if args.per_step_budget is not None else budget_from_run_dir(run_dir)
    seed = seed_from_run_dir(run_dir)
    for episode_type in args.episode_types:
        out_dir = Path(args.out_root) / str(episode_type) / f"budget{budget_tag(budget)}_seed{seed}"
        evaluate_episode_type(
            run_dir=run_dir,
            episode_type=str(episode_type),
            out_dir=out_dir,
            metadata=metadata,
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            oracle=oracle,
            trainer=trainer,
            steps=int(args.steps),
            n_rollouts=int(args.rollouts),
            seed=int(seed) + 55_000,
            per_step_budget=args.per_step_budget,
            startup_peak_budget=args.startup_peak_budget,
            stride_divisor=int(args.condition_stride_divisor),
            calm_max_event_rate=float(args.calm_max_event_rate),
            mixed_min_event_rate=float(args.mixed_min_event_rate),
            mixed_max_event_rate=float(args.mixed_max_event_rate),
            event_min_event_rate=float(args.event_min_event_rate),
            strict_condition_bands=bool(args.strict_condition_bands),
            skip_summary_eval=bool(args.skip_summary_eval),
        )


if __name__ == "__main__":
    main()
