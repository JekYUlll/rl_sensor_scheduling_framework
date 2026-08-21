#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_sources.public_weather_synthesis import PublicWeatherSynthesisConfig, generate_public_weather_truth  # noqa: E402
from v2.custom_ppo import CustomPPO, CustomPPOConfig, evaluate_custom_ppo  # noqa: E402
from v2.env import WarmupEnvConfig  # noqa: E402
from v2.policies import FullOpenUnconstrainedScorePolicy, StaticMaskPolicy  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle  # noqa: E402


def load_helpers():
    path = ROOT / "scripts" / "23_v2_train_ppo.py"
    spec = importlib.util.spec_from_file_location("_v2_train_ppo_helpers_independent_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def non_overlapping_random_starts(
    *,
    sequence_steps: int,
    eval_steps: int,
    horizon: int,
    n_rollouts: int,
    seed: int,
) -> tuple[int, ...]:
    required_span = int(n_rollouts) * int(eval_steps) + int(horizon) + 1
    if int(sequence_steps) < required_span:
        raise ValueError("Independent test truth is too short for the requested evaluation windows")
    rng = np.random.default_rng(int(seed))
    slack = int(sequence_steps) - required_span
    gaps = rng.multinomial(slack, np.full(int(n_rollouts) + 1, 1.0 / float(int(n_rollouts) + 1)))
    starts: list[int] = []
    cursor = int(gaps[0])
    for idx in range(int(n_rollouts)):
        starts.append(int(cursor))
        cursor += int(eval_steps) + int(gaps[idx + 1])
    return tuple(starts)


def _resolve_path(value: str, *, run_dir: Path) -> Path:
    path = Path(value)
    candidates = [path, ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve source path {value!r} for {run_dir}")


def _truth_config(
    *,
    metadata: dict[str, object],
    dataset_metadata: dict[str, object],
    antaws_root: Path,
    test_seed: int,
    truth_steps: int | None,
) -> PublicWeatherSynthesisConfig:
    design = dict(metadata.get("truth_event_design", {}))
    return PublicWeatherSynthesisConfig(
        antaws_root=antaws_root,
        stations=tuple(str(x) for x in dataset_metadata["stations"]),
        steps=int(truth_steps or dataset_metadata["steps"]),
        freq_s=int(metadata["freq_s"]),
        seed=int(test_seed),
        phase_keep_fraction=float(dataset_metadata.get("phase_keep_fraction", 0.15)),
        match_marginal_distribution=bool(dataset_metadata.get("match_marginal_distribution", True)),
        blowing_snow_event_coverage=float(design.get("blowing_snow_event_coverage", 0.28)),
        blowing_snow_event_model=str(design.get("blowing_snow_event_model", "semi_markov")),
        blowing_snow_min_duration_steps=int(design.get("blowing_snow_min_duration_steps", 12)),
        blowing_snow_max_duration_steps=int(design.get("blowing_snow_max_duration_steps", 24)),
        blowing_snow_min_gap_steps=int(design.get("blowing_snow_min_gap_steps", 4)),
        blowing_snow_lead_steps=int(design.get("blowing_snow_lead_steps", 6)),
        blowing_snow_wind_margin_ms=float(design.get("blowing_snow_wind_margin_ms", 1.2)),
        cred_hysteresis_on=float(design.get("cred_hysteresis_on", 0.6)),
        cred_hysteresis_off=float(design.get("cred_hysteresis_off", 0.3)),
        flux_wind_exponent=float(design.get("flux_wind_exponent", 3.0)),
        event_microstructure_sigma=float(design.get("event_microstructure_sigma", 0.0)),
        event_microstructure_alpha=float(design.get("event_microstructure_alpha", 0.18)),
        event_microstructure_diameter_scale=float(design.get("event_microstructure_diameter_scale", 0.0)),
        event_microstructure_velocity_scale=float(design.get("event_microstructure_velocity_scale", 0.0)),
    )


def ensure_independent_truth(
    *,
    run_dir: Path,
    metadata: dict[str, object],
    out_dir: Path,
    antaws_root: Path,
    test_seed: int,
    truth_steps: int | None,
) -> Path:
    truth_dir = out_dir / "test_truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    truth_path = truth_dir / f"truth_test_seed{int(test_seed)}.csv"
    metadata_path = truth_dir / f"truth_test_seed{int(test_seed)}_metadata.json"
    if truth_path.exists() and metadata_path.exists():
        return truth_path
    source_dataset_metadata = json.loads(
        (run_dir / "dataset_validation" / "synthetic_metadata.json").read_text(encoding="utf-8")
    )
    cfg = _truth_config(
        metadata=metadata,
        dataset_metadata=source_dataset_metadata,
        antaws_root=antaws_root,
        test_seed=int(test_seed),
        truth_steps=truth_steps,
    )
    truth, generated_metadata = generate_public_weather_truth(cfg)
    truth.to_csv(truth_path, index=False)
    provenance = {
        "role": "independent_posthoc_test_truth",
        "source_training_run": str(run_dir),
        "training_seed": int(metadata["seed"]),
        "test_seed": int(test_seed),
        "generator_metadata": generated_metadata,
    }
    metadata_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return truth_path


def constraints_from_metadata(metadata: dict[str, object]) -> PowerConstraintsV2:
    source = dict(metadata["constraints"])
    coverage_groups = tuple(
        (str(item["name"]), tuple(str(sensor_id) for sensor_id in item["sensor_ids"]))
        for item in source.get("coverage_groups", [])
    )
    return PowerConstraintsV2(
        max_active=int(source["max_active"]),
        per_step_budget=float(source["per_step_budget"]),
        startup_peak_budget=float(source["startup_peak_budget"]),
        required_sensor_ids=tuple(str(x) for x in source.get("required_sensor_ids", [])),
        coverage_groups=coverage_groups,
    )


def load_trainer_for_replay(
    *,
    run_dir: Path,
    truth: pd.DataFrame,
    sensors: list,
    constraints: PowerConstraintsV2,
    env_cfg: WarmupEnvConfig,
    oracle: TCNFrozenForecastOracle,
    device: str,
) -> tuple[CustomPPO, np.ndarray]:
    import torch

    payload = torch.load(run_dir / "custom_ppo.pt", map_location="cpu", weights_only=False)
    allowed_fields = {item.name for item in fields(CustomPPOConfig)}
    config_data = {key: value for key, value in dict(payload["cfg"]).items() if key in allowed_fields}
    config_data["device"] = str(device)
    config_data["history_path"] = None
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=env_cfg,
        oracle=oracle,
        candidate_masks=np.asarray(payload["candidate_masks"], dtype=bool),
        cfg=CustomPPOConfig(**config_data),
        candidate_prior_logits=payload.get("candidate_prior_logits"),
    )
    trainer.model.load_state_dict(payload["state_dict"])
    trainer.model.eval()
    return trainer, np.asarray(payload["candidate_masks"], dtype=bool)


def evaluate_run(
    *,
    source_run_dir: Path,
    result_run_dir: Path,
    output_root: Path,
    antaws_root: Path,
    test_seed_offset: int,
    eval_steps: int,
    eval_rollouts: int,
    truth_steps: int | None,
    device: str,
) -> Path:
    helpers = load_helpers()
    metadata = json.loads((source_run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    training_seed = int(metadata["seed"])
    test_seed = int(training_seed + test_seed_offset)
    truth_path = ensure_independent_truth(
        run_dir=source_run_dir,
        metadata=metadata,
        out_dir=output_root,
        antaws_root=antaws_root,
        test_seed=test_seed,
        truth_steps=truth_steps,
    )
    truth = pd.read_csv(truth_path)
    sensor_cfg = _resolve_path(str(metadata["sensor_cfg"]), run_dir=source_run_dir)
    sensors = load_sensor_specs(sensor_cfg)
    constraints = constraints_from_metadata(metadata)
    oracle_path = _resolve_path(str(metadata["oracle_path"]), run_dir=source_run_dir)
    oracle = TCNFrozenForecastOracle.load(oracle_path, device=str(device))
    source_truth_path = _resolve_path(str(metadata["truth_csv"]), run_dir=source_run_dir)
    source_truth = pd.read_csv(source_truth_path)
    source_state_values = source_truth[list(helpers.STATE_COLUMNS)].to_numpy(dtype=float)
    normalization_mean = tuple(float(x) for x in np.mean(source_state_values, axis=0))
    normalization_std = tuple(float(x) for x in np.maximum(np.std(source_state_values, axis=0), 1e-6))
    starts = non_overlapping_random_starts(
        sequence_steps=len(truth),
        eval_steps=int(eval_steps),
        horizon=int(metadata["horizon"]),
        n_rollouts=int(eval_rollouts),
        seed=int(test_seed) + 1777,
    )
    eval_cfg = WarmupEnvConfig(
        state_columns=helpers.STATE_COLUMNS,
        reward_target_columns=helpers.REWARD_TARGET_COLUMNS,
        lookback=int(metadata["lookback"]),
        episode_len=int(eval_steps),
        seed=int(test_seed) + 9000,
        base_freq_s=int(metadata["freq_s"]),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )
    trainer, candidate_masks = load_trainer_for_replay(
        run_dir=source_run_dir,
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        env_cfg=eval_cfg,
        oracle=oracle,
        device=str(device),
    )
    result_run_dir.mkdir(parents=True, exist_ok=True)
    policies: list[str] = []
    custom_result, _ = evaluate_custom_ppo(
        trainer=trainer,
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        steps=int(eval_steps),
        start_indices=starts,
    )
    policies.append("custom_ppo")
    save_rollout_npz(
        result_run_dir / "rollout_custom_ppo.npz",
        custom_result,
        sensor_ids=[sensor.sensor_id for sensor in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )

    prior = pd.read_csv(source_run_dir / "custom_ppo_candidate_prior.csv").iloc[0]
    prior_mask = tuple(bool(x) for x in candidate_masks[int(prior["action_idx"])])
    prior_policy = StaticMaskPolicy(mask=prior_mask, name="prior_selected_static")
    prior_result, _ = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=eval_cfg,
        oracle=oracle,
        policy=prior_policy,
        steps=int(eval_steps),
        start_indices=starts,
    )
    policies.append(prior_policy.name)
    save_rollout_npz(
        result_run_dir / "rollout_prior_selected_static.npz",
        prior_result,
        sensor_ids=[sensor.sensor_id for sensor in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )

    full_result, _ = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=PowerConstraintsV2(),
        cfg=eval_cfg,
        oracle=oracle,
        policy=FullOpenUnconstrainedScorePolicy(n_sensors=len(sensors)),
        steps=int(eval_steps),
        start_indices=starts,
    )
    policies.append("full_open_unconstrained")
    save_rollout_npz(
        result_run_dir / "rollout_full_open_unconstrained.npz",
        full_result,
        sensor_ids=[sensor.sensor_id for sensor in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )
    for policy in helpers.default_policies(len(sensors), seed=int(test_seed) + 100):
        result, _ = helpers.evaluate_score_policy_over_starts(
            truth=truth,
            sensors=sensors,
            constraints=constraints,
            cfg=eval_cfg,
            oracle=oracle,
            policy=policy,
            steps=int(eval_steps),
            start_indices=starts,
        )
        policies.append(str(result.policy_name))
        save_rollout_npz(
            result_run_dir / f"rollout_{result.policy_name}.npz",
            result,
            sensor_ids=[sensor.sensor_id for sensor in sensors],
            state_columns=helpers.STATE_COLUMNS,
        )

    replay_metadata = dict(metadata)
    replay_metadata.update(
        {
            "truth_csv": str(truth_path.resolve()),
            "oracle_path": str(oracle_path.resolve()),
            "oracle_inference_device": str(device),
            "eval_policies": policies,
            "eval_steps": int(eval_steps),
            "eval_rollouts": int(eval_rollouts),
            "eval_start_indices": [int(x) for x in starts],
            "history_initial_state_mean": list(normalization_mean),
            "independent_replay": {
                "source_training_run": str(source_run_dir),
                "training_seed": int(training_seed),
                "test_seed": int(test_seed),
                "test_seed_offset": int(test_seed_offset),
                "window_sampling": "uniform_random_non_overlapping",
                "prior_selected_static_source": "minimum candidate-prior oracle loss during original training run",
                "prior_selected_static_sensor_ids": str(prior["sensor_ids"]),
                "normalization_reference": str(source_truth_path),
                "normalization_mean": list(normalization_mean),
                "normalization_std": list(normalization_std),
            },
        }
    )
    (result_run_dir / "v2_ppo_metadata.json").write_text(json.dumps(replay_metadata, indent=2), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "24_v2_evaluate_rollouts.py"),
            "--run-dir",
            str(result_run_dir),
            "--per-step-budget",
            str(float(constraints.per_step_budget)),
            "--startup-peak-budget",
            str(float(constraints.startup_peak_budget)),
            "--forecast-oracle-device",
            str(device),
        ],
        check=True,
    )
    return result_run_dir / "evaluation" / "v2_eval_overall.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved V3.1 S2 policies on independent test truth.")
    parser.add_argument("--source-dir", default="reports/v31_s2_main")
    parser.add_argument("--out-dir", default="reports/v31_s2_independent_replay")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--budgets", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--test-seed-offset", type=int, default=10000)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--truth-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source_raw = Path(args.source_dir) / "raw"
    source_runs = sorted(source_raw.glob("budget*_seed*"))
    if args.labels:
        wanted = set(str(label) for label in args.labels)
        source_runs = [path for path in source_runs if path.name in wanted]
    if args.budgets or args.seeds:
        filtered = []
        for run_dir in source_runs:
            metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
            budget = float(metadata["constraints"]["per_step_budget"])
            seed = int(metadata["seed"])
            if args.budgets and budget not in {float(x) for x in args.budgets}:
                continue
            if args.seeds and seed not in {int(x) for x in args.seeds}:
                continue
            filtered.append(run_dir)
        source_runs = filtered
    if not source_runs:
        raise FileNotFoundError(f"No selected S2 runs found under {source_raw}")

    output_root = Path(args.out_dir)
    long_rows: list[pd.DataFrame] = []
    for source_run in source_runs:
        result_run = output_root / "raw" / source_run.name
        eval_path = evaluate_run(
            source_run_dir=source_run,
            result_run_dir=result_run,
            output_root=output_root,
            antaws_root=Path(args.antaws_root),
            test_seed_offset=int(args.test_seed_offset),
            eval_steps=int(args.eval_steps),
            eval_rollouts=int(args.eval_rollouts),
            truth_steps=args.truth_steps,
            device=str(args.device),
        )
        metadata = json.loads((result_run / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
        values = pd.read_csv(eval_path)
        values.insert(0, "test_seed", int(metadata["independent_replay"]["test_seed"]))
        values.insert(0, "training_seed", int(metadata["seed"]))
        values.insert(0, "budget", float(metadata["constraints"]["per_step_budget"]))
        values.insert(0, "run_tag", source_run.name)
        long_rows.append(values)

    long = pd.concat(long_rows, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_root / "v31_independent_replay_long.csv", index=False)
    metric_columns = [
        column
        for column in [
            "forecast_weighted_mae_overall",
            "forecast_weighted_mae_event",
            "forecast_weighted_mae_non_event",
            "oracle_loss_mean",
            "power_mean",
            "warmup_abort_rate",
        ]
        if column in long.columns
    ]
    stats = long.groupby(["budget", "policy"], as_index=False)[metric_columns].agg(["mean", "std", "count"])
    stats.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in stats.columns
    ]
    stats.reset_index(drop=True).to_csv(output_root / "v31_independent_replay_stats.csv", index=False)
    summary = {
        "source_dir": str(args.source_dir),
        "run_count": int(len(source_runs)),
        "test_seed_offset": int(args.test_seed_offset),
        "eval_steps": int(args.eval_steps),
        "eval_rollouts": int(args.eval_rollouts),
        "window_sampling": "uniform_random_non_overlapping",
        "policies": sorted(str(value) for value in long["policy"].unique()),
        "interpretation": (
            "Posthoc independent-truth replay of previously trained policies and "
            "training-prior-selected static masks; no policy or oracle retraining."
        ),
    }
    (output_root / "v31_independent_replay_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(output_root / "v31_independent_replay_stats.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
