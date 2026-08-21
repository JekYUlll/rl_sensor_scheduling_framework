#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.dqn import DQNConfig, DQNTrainer, evaluate_dqn  # noqa: E402
from v2.env import WarmupEnvConfig  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


def load_train_helpers():
    path = ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("_v2_train_ppo_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_or_train_oracle(
    *,
    helpers,
    args: argparse.Namespace,
    truth: pd.DataFrame,
    sensors: list,
    constraints: PowerConstraintsV2,
    target_weights: tuple[float, ...],
    target_scales: tuple[float, ...],
    out_dir: Path,
) -> tuple[object, Path]:
    checkpoint = Path(str(args.oracle_checkpoint)) if args.oracle_checkpoint else None
    if checkpoint is not None and checkpoint.exists():
        if str(args.oracle_type) == "tcn":
            oracle = TCNFrozenForecastOracle.load(checkpoint, device=str(args.oracle_inference_device))
        else:
            oracle = LinearFrozenForecastOracle.load(str(checkpoint))
        return oracle, checkpoint

    oracle = helpers.train_oracle(
        truth,
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
        rollouts_per_policy=int(args.oracle_rollouts_per_policy),
        event_fraction=float(args.oracle_event_fraction),
        full_open_repeat=int(args.oracle_full_open_repeat),
        base_freq_s=int(args.freq_s),
        seed=int(args.seed),
    )
    oracle_path = out_dir / ("v2_tcn_oracle.pt" if args.oracle_type == "tcn" else "v2_linear_oracle.npz")
    oracle.save(str(oracle_path))
    if args.oracle_type == "tcn":
        oracle.to_device(str(args.oracle_inference_device))
    return oracle, oracle_path


def write_training_log(history: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in history:
        rows.append(
            {
                "step": int(item.get("timesteps", 0)),
                "loss": float(item.get("loss", float("nan"))),
                "epsilon": float(item.get("epsilon", float("nan"))),
                "reward_mean": float(item.get("reward_mean", float("nan"))),
                "episode_return_mean": float(item.get("episode_return_mean", float("nan"))),
                "unique_actions": int(item.get("unique_actions", 0)),
                "replay_size": int(item.get("replay_size", 0)),
                "forecast_weighted_mae_overall": float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    helpers = load_train_helpers()
    parser = argparse.ArgumentParser(description="Train the v2 DQN candidate-subset scheduler.")
    parser.add_argument("--truth-csv", default="data/generated/v2_public_weather_truth_dqn.csv")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Taishan"])
    parser.add_argument("--truth-steps", type=int, default=8192)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--blowing-snow-event-coverage", type=float, default=0.30)
    parser.add_argument("--blowing-snow-min-duration-steps", type=int, default=10)
    parser.add_argument("--blowing-snow-max-duration-steps", type=int, default=30)
    parser.add_argument("--blowing-snow-lead-steps", type=int, default=5)
    parser.add_argument("--blowing-snow-wind-margin-ms", type=float, default=1.5)
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", default="reports/v2_dqn_probe/budget1p70_seed41")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--oracle-type", choices=["linear", "tcn"], default="tcn")
    parser.add_argument("--oracle-checkpoint", default=None)
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
    parser.add_argument("--oracle-loss-clip", type=float, default=10.0)
    parser.add_argument("--oracle-disable-mask-channels", action="store_true")
    parser.add_argument("--target-weights", nargs="*", type=float, default=list(helpers.DEFAULT_TARGET_WEIGHTS))
    parser.add_argument("--target-scales", nargs="*", type=float, default=list(helpers.DEFAULT_TARGET_SCALES))
    parser.add_argument("--train-episode-len", type=int, default=512)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step-return", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--exploration-fraction", type=float, default=0.20)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--oracle-prefill-steps", type=int, default=0)
    parser.add_argument("--oracle-prefill-lookahead-steps", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--per-step-budget", type=float, default=1.7)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--required-sensors", nargs="*", default=list(helpers.DEFAULT_REQUIRED_SENSOR_IDS))
    parser.add_argument("--disable-coverage-groups", action="store_true")
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--dqn-max-candidate-warmup", type=int, default=-1)
    parser.add_argument("--event-start-prob", type=float, default=0.67)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    target_weights = tuple(float(x) for x in args.target_weights)
    target_scales = tuple(float(x) for x in args.target_scales)
    if len(target_weights) != len(helpers.REWARD_TARGET_COLUMNS):
        raise ValueError(f"--target-weights must contain {len(helpers.REWARD_TARGET_COLUMNS)} values")
    if len(target_scales) != len(helpers.REWARD_TARGET_COLUMNS):
        raise ValueError(f"--target-scales must contain {len(helpers.REWARD_TARGET_COLUMNS)} values")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = helpers.ensure_truth(args)
    truth = pd.read_csv(truth_path)
    sensors = load_sensor_specs(args.sensor_cfg)
    coverage_groups = () if bool(args.disable_coverage_groups) else helpers.DEFAULT_COVERAGE_GROUPS
    constraints = PowerConstraintsV2(
        max_active=int(args.max_active),
        per_step_budget=float(args.per_step_budget),
        startup_peak_budget=float(args.startup_peak_budget),
        required_sensor_ids=tuple(str(sensor_id) for sensor_id in args.required_sensors),
        coverage_groups=coverage_groups,
    )
    candidate_masks = helpers.build_projected_candidate_masks(
        sensors,
        constraints,
        max_candidate_warmup=None if int(args.dqn_max_candidate_warmup) < 0 else int(args.dqn_max_candidate_warmup),
    )
    oracle, oracle_path = load_or_train_oracle(
        helpers=helpers,
        args=args,
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        target_weights=target_weights,
        target_scales=target_scales,
        out_dir=out_dir,
    )

    train_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=int(args.lookback),
        episode_len=int(args.train_episode_len),
        seed=int(args.seed),
        base_freq_s=int(args.freq_s),
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
    )
    trainer = DQNTrainer(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=train_cfg,
        oracle=oracle,
        candidate_masks=candidate_masks,
        cfg=DQNConfig(
            total_timesteps=int(args.total_timesteps),
            replay_size=int(args.replay_size),
            learning_starts=int(args.learning_starts),
            batch_size=int(args.batch_size),
            train_freq=int(args.train_freq),
            gradient_steps=int(args.gradient_steps),
            target_update_interval=int(args.target_update_interval),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            n_step_return=int(args.n_step_return),
            hidden_dim=int(args.hidden_dim),
            exploration_fraction=float(args.exploration_fraction),
            exploration_final_eps=float(args.exploration_final_eps),
            oracle_prefill_steps=int(args.oracle_prefill_steps),
            oracle_prefill_lookahead_steps=int(args.oracle_prefill_lookahead_steps),
            event_start_prob=float(args.event_start_prob),
            device=str(args.device),
            seed=int(args.seed),
            log_interval=int(args.log_interval),
            history_path=str(out_dir / "dqn_training_history_live.json"),
        ),
    )
    trainer.train()
    model_path = out_dir / "dqn.pt"
    trainer.save(model_path)
    if args.checkpoint_path:
        trainer.save(Path(args.checkpoint_path))
    trainer.save_history(out_dir / "dqn_training_history.json")
    write_training_log(trainer.history, out_dir / "dqn_training_log.csv")

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
        lookback=int(args.lookback),
        episode_len=int(args.eval_steps),
        seed=int(args.seed) + 9000,
        base_freq_s=int(args.freq_s),
        lambda_warmup_abort=float(args.lambda_warmup_abort),
        lambda_switch=float(args.lambda_switch),
    )
    rows = []
    dqn_result, dqn_metrics = evaluate_dqn(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=int(args.eval_steps),
        start_indices=eval_start_indices,
    )
    rows.append(dqn_metrics)
    save_rollout_npz(
        out_dir / "rollout_dqn.npz",
        dqn_result,
        sensor_ids=[s.sensor_id for s in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )

    full_open_result, full_open_metrics = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=PowerConstraintsV2(),
        cfg=eval_cfg,
        oracle=oracle,
        policy=helpers.FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        steps=int(args.eval_steps),
        start_indices=eval_start_indices,
    )
    rows.append(full_open_metrics)
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
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(args.eval_steps),
            start_indices=eval_start_indices,
        )
        rows.append(metrics)
        save_rollout_npz(
            out_dir / f"rollout_{result.policy_name}.npz",
            result,
            sensor_ids=[s.sensor_id for s in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )

    metrics = pd.DataFrame(rows).sort_values("oracle_loss_mean")
    metrics.to_csv(out_dir / "v2_dqn_metrics.csv", index=False)
    oracle_policy_specs = helpers.build_oracle_policy_specs(
        len(sensors),
        constraints,
        seed=int(args.seed),
        full_open_repeat=int(args.oracle_full_open_repeat),
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
        "sensor_cfg": str(args.sensor_cfg),
        "oracle_path": str(oracle_path),
        "oracle_type": str(args.oracle_type),
        "oracle_rollout_steps": int(args.oracle_rollout_steps),
        "oracle_rollouts_per_policy": int(args.oracle_rollouts_per_policy),
        "oracle_event_fraction": float(args.oracle_event_fraction),
        "oracle_full_open_repeat": int(args.oracle_full_open_repeat),
        "oracle_pretrain_rollout_summary": oracle_rollout_summary,
        "oracle_inference_device": str(args.oracle_inference_device),
        "oracle_use_mask_channels": not bool(args.oracle_disable_mask_channels),
        "reward_target_columns": list(helpers.REWARD_TARGET_COLUMNS),
        "target_weights": list(target_weights),
        "target_scales": list(target_scales),
        "model_path": str(model_path),
        "eval_policies": ["dqn", "full_open_unconstrained", *[str(policy.name) for policy in helpers.default_policies(len(sensors), seed=int(args.seed) + 100)]],
        "seed": int(args.seed),
        "freq_s": int(args.freq_s),
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "eval_rollouts": int(args.eval_rollouts),
        "eval_event_fraction": float(args.eval_event_fraction),
        "eval_start_indices": [int(x) for x in eval_start_indices],
        "dqn": {**asdict(trainer.cfg), "candidate_count": int(candidate_masks.shape[0])},
        "constraints": {
            "max_active": int(args.max_active),
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
            "blowing_snow_min_duration_steps": int(args.blowing_snow_min_duration_steps),
            "blowing_snow_max_duration_steps": int(args.blowing_snow_max_duration_steps),
            "blowing_snow_lead_steps": int(args.blowing_snow_lead_steps),
            "blowing_snow_wind_margin_ms": float(args.blowing_snow_wind_margin_ms),
        },
    }
    # Keep the historical filename because scripts/24_v2_evaluate_rollouts.py
    # already uses it as the run metadata contract.
    (out_dir / "v2_ppo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "v2_dqn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir / "v2_dqn_metrics.csv")
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


if __name__ == "__main__":
    main()
