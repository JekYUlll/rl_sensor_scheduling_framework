#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2.dqn import DQNConfig, DQNTrainer, evaluate_dqn  # noqa: E402
from v2.env import WarmupEnvConfig  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


def load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_oracle_artifact(path: Path, *, oracle_type: str, device: str):
    if oracle_type == "tcn":
        return TCNFrozenForecastOracle.load(path, device=device)
    if oracle_type == "linear":
        return LinearFrozenForecastOracle.load(str(path))
    raise ValueError(f"Unsupported oracle type: {oracle_type}")


def load_frozen_oracle(source_dir: Path, metadata: dict[str, object], out_dir: Path, device: str):
    oracle_type = str(metadata.get("oracle_type", "tcn"))
    filename = "v2_tcn_oracle.pt" if oracle_type == "tcn" else "v2_linear_oracle.npz"
    source = source_dir / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = out_dir / filename
    shutil.copy2(source, destination)
    oracle = load_oracle_artifact(destination, oracle_type=oracle_type, device=device)
    return oracle, destination, source


def main() -> None:
    helpers = load_script("_v31_dqn_helpers", "23_v2_train_ppo.py")
    paper_helpers = load_script("_v31_dqn_metric_helpers", "25_v2_train_custom_ppo.py")

    parser = argparse.ArgumentParser(description="Train a mask-matched DQN on frozen SCENEBAL-2 assets.")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step-return", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--exploration-fraction", type=float, default=0.20)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Positive values bound PyTorch CPU threads, primarily for evaluation-only workers.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load an existing complete DQN checkpoint and run only final evaluation.",
    )
    parser.add_argument(
        "--eval-oracle-device",
        default="",
        help="Optional final-rollout evaluator device; empty reuses --oracle-device.",
    )
    parser.add_argument(
        "--eval-step-limit",
        type=int,
        default=0,
        help="Positive values shorten evaluation for implementation smoke tests only.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    required = (
        "truth_v31_split.csv",
        "v2_ppo_metadata.json",
        "split_protocol_manifest.json",
        "custom_ppo.pt",
        "reward_staticnorm_candidates.csv",
        "reward_staticnorm_normalizers.json",
        "validation_static_candidates.csv",
    )
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source run is missing required assets: {missing}")

    metadata = load_json(source_dir / "v2_ppo_metadata.json")
    manifest = load_json(source_dir / "split_protocol_manifest.json")
    seed = int(metadata["seed"])
    truth_path = source_dir / "truth_v31_split.csv"
    truth = helpers.ensure_state_columns(pd.read_csv(truth_path))
    sensor_cfg = resolve_repo_path(str(metadata["sensor_cfg"]))
    sensors = load_sensor_specs(sensor_cfg)

    constraints_payload = dict(metadata["constraints"])
    constraints = PowerConstraintsV2(
        max_active=int(constraints_payload["max_active"]),
        per_step_budget=float(constraints_payload["per_step_budget"]),
        startup_peak_budget=float(constraints_payload["startup_peak_budget"]),
        required_sensor_ids=tuple(str(value) for value in constraints_payload["required_sensor_ids"]),
        coverage_groups=(),
    )

    import torch

    if int(args.cpu_threads) > 0:
        torch.set_num_threads(int(args.cpu_threads))
        torch.set_num_interop_threads(1)

    source_checkpoint = torch.load(source_dir / "custom_ppo.pt", map_location="cpu", weights_only=False)
    candidate_masks = np.asarray(source_checkpoint["candidate_masks"], dtype=bool)
    if candidate_masks.ndim != 2 or candidate_masks.shape[1] != len(sensors):
        raise ValueError("Source candidate masks do not match the sensor configuration")

    oracle, oracle_path, source_oracle_path = load_frozen_oracle(
        source_dir,
        metadata,
        out_dir,
        str(args.oracle_device),
    )
    if sha256_file(oracle_path) != sha256_file(source_oracle_path):
        raise RuntimeError("Copied frozen evaluator differs from its source")

    partition = dict(metadata["partition_protocol"])
    normalization_start = int(partition["normalization_start_idx"])
    normalization_end = int(partition["normalization_end_idx"])
    normalization_values = truth.iloc[normalization_start:normalization_end][list(helpers.STATE_COLUMNS)].to_numpy(dtype=float)
    normalization_mean = tuple(float(value) for value in np.mean(normalization_values, axis=0))
    normalization_std = tuple(float(value) for value in np.maximum(np.std(normalization_values, axis=0), 1e-6))
    normalized_values = (
        normalization_values - np.asarray(normalization_mean).reshape(1, -1)
    ) / np.asarray(normalization_std).reshape(1, -1)
    process_variance = tuple(
        float(value) for value in np.clip(np.nanvar(np.diff(normalized_values, axis=0), axis=0), 1e-6, 1.0)
    )

    normalizer_payload = load_json(source_dir / "reward_staticnorm_normalizers.json")
    normalizer_map = {
        str(key): float(value)
        for key, value in dict(normalizer_payload["normalizers"]).items()
    }
    normalizer_tuple = paper_helpers.subtype_normalizer_tuple(normalizer_map)
    if any(not np.isfinite(value) or value <= 0.0 for value in normalizer_tuple):
        raise ValueError("Source static-normalisation constants are invalid")

    reward_cfg = dict(metadata["reward_shaping"])
    regime_cfg = dict(metadata.get("observable_regime_belief", {}))
    train_cfg = WarmupEnvConfig(
        state_columns=tuple(helpers.STATE_COLUMNS),
        reward_target_columns=tuple(helpers.REWARD_TARGET_COLUMNS),
        reward_proxy_mode="forecast",
        lookback=int(metadata["lookback"]),
        episode_len=512,
        seed=seed,
        base_freq_s=int(metadata["freq_s"]),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        lambda_warmup_abort=float(reward_cfg["lambda_warmup_abort"]),
        lambda_switch=float(reward_cfg["lambda_switch"]),
        event_reward_multiplier=float(metadata.get("event_reward_multiplier", 1.0)),
        event_subtype_particle_reward_multiplier=float(reward_cfg["event_subtype_particle_reward_multiplier"]),
        event_subtype_flux_reward_multiplier=float(reward_cfg["event_subtype_flux_reward_multiplier"]),
        event_subtype_thermal_reward_multiplier=float(reward_cfg["event_subtype_thermal_reward_multiplier"]),
        min_dwell_steps=int(reward_cfg["min_dwell_steps"]),
        include_observable_regime_belief=bool(regime_cfg.get("enabled", False)),
        regime_belief_lookback=int(regime_cfg.get("lookback", 6)),
        agent_context_columns=tuple(str(value) for value in metadata.get("agent_context_columns", ())),
        include_event_flag_in_state=False,
        include_alert_context_features=False,
        uncertainty_process_variance=process_variance,
        oracle_loss_reward_normalizers=normalizer_tuple,
        oracle_loss_reward_default_normalizer=float(reward_cfg["reward_loss_default_normalizer"]),
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
            event_start_prob=float(dict(metadata["custom_ppo"]).get("event_start_prob", 0.85)),
            device=str(args.device),
            seed=seed,
            log_interval=1_000,
            history_path=str(out_dir / "dqn_training_history_live.json"),
            train_start_indices=tuple(int(value) for value in metadata.get("train_start_indices", ())),
            train_start_min=int(metadata["train_start_min"]),
            train_start_max=int(metadata["train_start_max"]),
        ),
    )
    model_path = out_dir / "dqn.pt"
    history_path = out_dir / "dqn_training_history.json"
    if bool(args.skip_training):
        if not model_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(
                f"--skip-training requires {model_path} and {history_path}"
            )
        checkpoint = torch.load(model_path, map_location=trainer.device, weights_only=False)
        checkpoint_masks = np.asarray(checkpoint["candidate_masks"], dtype=bool)
        if not np.array_equal(checkpoint_masks, candidate_masks):
            raise ValueError("Saved DQN checkpoint candidate masks differ from the source masks")
        if int(checkpoint.get("obs_dim", -1)) != int(trainer.obs_dim):
            raise ValueError("Saved DQN checkpoint observation dimension differs from the environment")
        trainer.q_net.load_state_dict(checkpoint["state_dict"])
        trainer.target_q_net.load_state_dict(checkpoint["target_state_dict"])
        trainer.history = json.loads(history_path.read_text(encoding="utf-8"))
        if not trainer.history or int(trainer.history[-1].get("timesteps", -1)) != int(args.total_timesteps):
            raise ValueError("Saved DQN history does not certify the requested training length")
    else:
        trainer.train()
        trainer.save(model_path)
        trainer.save_history(history_path)

    oracle_type = str(metadata.get("oracle_type", "tcn"))
    eval_oracle_device = str(args.eval_oracle_device or args.oracle_device)
    if eval_oracle_device != str(args.oracle_device):
        oracle = load_oracle_artifact(
            oracle_path,
            oracle_type=oracle_type,
            device=eval_oracle_device,
        )
        trainer.oracle = oracle

    eval_starts = tuple(int(value) for value in metadata["eval_start_indices"])
    eval_steps = int(dict(manifest["final_test"])["eval_steps"])
    if int(args.eval_step_limit) > 0:
        eval_steps = min(eval_steps, int(args.eval_step_limit))
        eval_starts = eval_starts[:1]
    eval_cfg = replace(train_cfg, episode_len=eval_steps, seed=seed + 9_000)
    rows: list[dict[str, object]] = []

    dqn_result, dqn_metrics = evaluate_dqn(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=eval_steps,
        start_indices=eval_starts,
    )
    paper_helpers.append_eval_row(rows, dqn_metrics, dqn_result, truth)
    save_rollout_npz(
        out_dir / "rollout_dqn.npz",
        dqn_result,
        sensor_ids=[sensor.sensor_id for sensor in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )

    selected = dict(metadata["selected_static_reference"])
    selected_idx = int(selected["action_idx"])
    static_policy = StaticMaskPolicy(
        mask=tuple(bool(value) for value in candidate_masks[selected_idx]),
        name="validation_selected_static",
    )
    policies = [
        static_policy,
        helpers.FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        *helpers.default_policies(len(sensors), seed=seed + 100),
    ]
    seen = {"dqn"}
    for policy in policies:
        if str(policy.name) in seen:
            continue
        seen.add(str(policy.name))
        policy_constraints = PowerConstraintsV2() if str(policy.name) == "full_open_unconstrained" else constraints
        result, metrics = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=policy_constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=eval_steps,
            start_indices=eval_starts,
        )
        paper_helpers.append_eval_row(rows, metrics, result, truth)
        save_rollout_npz(
            out_dir / f"rollout_{result.policy_name}.npz",
            result,
            sensor_ids=[sensor.sensor_id for sensor in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )

    metrics = paper_helpers.add_staticnorm_macro(pd.DataFrame(rows), normalizer_map)
    metrics = metrics.sort_values("oracle_loss_macro_subtype_event_staticnorm").reset_index(drop=True)
    metrics.to_csv(out_dir / "v31_matched_dqn_metrics.csv", index=False)
    shutil.copy2(source_dir / "reward_staticnorm_candidates.csv", out_dir / "reward_staticnorm_candidates.csv")
    shutil.copy2(source_dir / "reward_staticnorm_normalizers.json", out_dir / "reward_staticnorm_normalizers.json")
    shutil.copy2(source_dir / "validation_static_candidates.csv", out_dir / "validation_static_candidates.csv")

    output_metadata = {
        "method": "masked_double_dqn",
        "seed": seed,
        "source_run_dir": str(source_dir),
        "truth_path": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
        "source_metadata_sha256": sha256_file(source_dir / "v2_ppo_metadata.json"),
        "source_manifest_sha256": sha256_file(source_dir / "split_protocol_manifest.json"),
        "source_oracle_sha256": sha256_file(source_oracle_path),
        "copied_oracle_sha256": sha256_file(oracle_path),
        "candidate_mask_count": int(candidate_masks.shape[0]),
        "candidate_masks": candidate_masks.astype(int).tolist(),
        "online_observation_contract": {
            "include_exact_event_flag": False,
            "agent_context_columns": list(train_cfg.agent_context_columns),
            "training_partition_event_sampling": True,
            "final_test_event_labels_used_by_policy": False,
        },
        "reward": {
            "mode": "forecast",
            "frozen_evaluator": str(oracle_path),
            "training_evaluator_device": str(args.oracle_device),
            "evaluation_evaluator_device": eval_oracle_device,
            "evaluation_cpu_threads": int(args.cpu_threads),
            "static_normalisers": list(normalizer_tuple),
        },
        "dqn": asdict(trainer.cfg),
        "partitions": {
            "train_start_min": int(metadata["train_start_min"]),
            "train_start_max": int(metadata["train_start_max"]),
            "eval_start_indices": list(eval_starts),
            "eval_steps": eval_steps,
            "smoke_evaluation": int(args.eval_step_limit) > 0,
        },
        "constraints": constraints_payload,
        "selected_static_reference": selected,
    }
    (out_dir / "v31_matched_dqn_metadata.json").write_text(
        json.dumps(output_metadata, indent=2),
        encoding="utf-8",
    )
    print(out_dir / "v31_matched_dqn_metrics.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
