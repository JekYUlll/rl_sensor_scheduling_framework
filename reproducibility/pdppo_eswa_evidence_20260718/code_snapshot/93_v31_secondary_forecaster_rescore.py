#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
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
from v2.evaluation import load_rollout_npz  # noqa: E402
from v2.forecast_eval import forecast_loss_samples  # noqa: E402
from v2.policies import StaticMaskPolicy  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.rollout import save_rollout_npz  # noqa: E402
from v2.sensor_spec import load_sensor_specs  # noqa: E402


SUBTYPE_LABELS = {1: "particle", 2: "flux", 3: "thermal"}
PRIMARY_SCORE = "oracle_loss_macro_subtype_event_staticnorm"


def load_script(name: str, filename: str) -> Any:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def first_existing_path(candidates: list[Path], *, label: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve {label}; checked: {joined}")


def bootstrap_mean_ci(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.mean(rng.choice(data, size=(int(draws), int(data.size)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_samples(samples: pd.DataFrame, *, policy: str) -> dict[str, Any]:
    finite = samples.loc[np.isfinite(pd.to_numeric(samples["forecast_loss"], errors="coerce"))].copy()
    row: dict[str, Any] = {
        "policy": str(policy),
        "forecast_samples": int(len(finite)),
        "oracle_loss_mean": float(finite["forecast_loss"].mean()) if len(finite) else float("nan"),
    }
    subtype_values: list[float] = []
    for subtype_id, label in SUBTYPE_LABELS.items():
        selected = finite.loc[finite["event_subtype_id"].astype(int) == int(subtype_id), "forecast_loss"]
        loss = float(selected.mean()) if len(selected) else float("nan")
        row[f"oracle_loss_subtype_{label}"] = loss
        row[f"steps_subtype_{label}"] = int(len(selected))
        if np.isfinite(loss):
            subtype_values.append(loss)
    row["oracle_loss_macro_subtype_event"] = (
        float(np.mean(subtype_values)) if subtype_values else float("nan")
    )
    row["macro_subtype_event_count"] = int(len(subtype_values))
    return row


def build_constraints(metadata: dict[str, Any]) -> PowerConstraintsV2:
    payload = dict(metadata["constraints"])
    return PowerConstraintsV2(
        max_active=int(payload["max_active"]),
        per_step_budget=float(payload["per_step_budget"]),
        startup_peak_budget=float(payload["startup_peak_budget"]),
        required_sensor_ids=tuple(str(value) for value in payload["required_sensor_ids"]),
        coverage_groups=(),
    )


def build_env_cfg(
    *,
    truth: pd.DataFrame,
    metadata: dict[str, Any],
    helpers: Any,
    episode_len: int,
) -> WarmupEnvConfig:
    partition = dict(metadata["partition_protocol"])
    start = int(partition["normalization_start_idx"])
    end = int(partition["normalization_end_idx"])
    values = truth.iloc[start:end][list(helpers.STATE_COLUMNS)].to_numpy(dtype=float)
    mean = tuple(float(value) for value in np.mean(values, axis=0))
    std = tuple(float(value) for value in np.maximum(np.std(values, axis=0), 1e-6))
    normalized = (values - np.asarray(mean).reshape(1, -1)) / np.asarray(std).reshape(1, -1)
    process_variance = tuple(
        float(value)
        for value in np.clip(np.nanvar(np.diff(normalized, axis=0), axis=0), 1e-6, 1.0)
    )
    reward = dict(metadata["reward_shaping"])
    regime = dict(metadata.get("observable_regime_belief", {}))
    return WarmupEnvConfig(
        state_columns=tuple(helpers.STATE_COLUMNS),
        reward_target_columns=tuple(helpers.REWARD_TARGET_COLUMNS),
        reward_proxy_mode="forecast",
        lookback=int(metadata["lookback"]),
        episode_len=int(episode_len),
        seed=int(metadata["seed"]) + 31_000,
        base_freq_s=int(metadata["freq_s"]),
        normalization_mean=mean,
        normalization_std=std,
        lambda_warmup_abort=float(reward["lambda_warmup_abort"]),
        lambda_switch=float(reward["lambda_switch"]),
        min_dwell_steps=int(reward["min_dwell_steps"]),
        include_observable_regime_belief=bool(regime.get("enabled", False)),
        regime_belief_lookback=int(regime.get("lookback", 6)),
        agent_context_columns=tuple(str(value) for value in metadata.get("agent_context_columns", ())),
        include_event_flag_in_state=False,
        include_alert_context_features=False,
        uncertainty_process_variance=process_variance,
    )


def load_candidate_masks(run_dir: Path, *, expected_sensors: int) -> np.ndarray:
    import torch

    checkpoint = torch.load(run_dir / "custom_ppo.pt", map_location="cpu", weights_only=False)
    masks = np.asarray(checkpoint["candidate_masks"], dtype=bool)
    if masks.ndim != 2 or masks.shape[1] != int(expected_sensors):
        raise ValueError(f"Candidate mask shape mismatch in {run_dir}: {masks.shape}")
    if masks.shape[0] != 6:
        raise ValueError(f"Expected six candidate masks in {run_dir}, found {masks.shape[0]}")
    return masks


def fit_secondary_oracle(
    *,
    run_dir: Path,
    output_dir: Path,
    truth: pd.DataFrame,
    metadata: dict[str, Any],
    sensors: list[Any],
    constraints: PowerConstraintsV2,
    candidate_masks: np.ndarray,
    helpers: Any,
    paper_helpers: Any,
) -> Any:
    partition = dict(metadata["partition_protocol"])
    start = int(partition["oracle_start_idx"])
    end = int(partition["oracle_end_idx"])
    oracle_truth = truth.iloc[start:end].reset_index(drop=True)
    teacher = dict(metadata.get("oracle_subtype_teacher_sensors", {}))

    def teacher_mask(label: str) -> np.ndarray | None:
        sensor_ids = [str(value) for value in teacher.get(label, ())]
        if not sensor_ids:
            return None
        action_idx = paper_helpers.resolve_candidate_action_index(
            sensors,
            candidate_masks,
            sensor_ids,
            label=f"secondary_forecaster_{label}",
        )
        return np.asarray(candidate_masks[action_idx], dtype=bool)

    oracle = helpers.train_oracle(
        oracle_truth,
        sensors,
        constraints,
        oracle_type="linear",
        lookback=int(metadata["lookback"]),
        horizon=int(metadata["horizon"]),
        rollout_steps=int(metadata["oracle_rollout_steps"]),
        tcn_epochs=1,
        tcn_batch_size=128,
        tcn_lr=1e-3,
        tcn_channels=8,
        tcn_levels=1,
        tcn_device="cpu",
        tcn_loss_clip=0.0,
        tcn_use_mask_channels=True,
        target_weights=tuple(float(value) for value in metadata["target_weights"]),
        target_scales=tuple(float(value) for value in metadata["target_scales"]),
        subtype_loss_weighting=bool(metadata.get("subtype_loss_weighting", False)),
        subtype_particle_target_weights=tuple(
            float(value) for value in metadata.get("subtype_particle_target_weights", ())
        ) or None,
        subtype_flux_target_weights=tuple(
            float(value) for value in metadata.get("subtype_flux_target_weights", ())
        ) or None,
        subtype_thermal_target_weights=tuple(
            float(value) for value in metadata.get("subtype_thermal_target_weights", ())
        ) or None,
        rollouts_per_policy=int(metadata["oracle_rollouts_per_policy"]),
        event_fraction=float(metadata["oracle_event_fraction"]),
        full_open_repeat=int(metadata["oracle_full_open_repeat"]),
        candidate_masks=candidate_masks,
        candidate_mask_repeat=int(metadata.get("oracle_candidate_mask_repeat", 0)),
        candidate_mask_limit=int(metadata.get("oracle_candidate_mask_limit", 0)),
        subtype_teacher_repeat=int(metadata.get("oracle_subtype_teacher_repeat", 0)),
        subtype_teacher_lookahead_steps=int(metadata.get("oracle_subtype_teacher_lookahead_steps", 0)),
        subtype_teacher_calm_mask=teacher_mask("calm"),
        subtype_teacher_particle_mask=teacher_mask("particle"),
        subtype_teacher_flux_mask=teacher_mask("flux"),
        subtype_teacher_thermal_mask=teacher_mask("thermal"),
        base_freq_s=int(metadata["freq_s"]),
        seed=int(metadata["seed"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle.save(str(output_dir / "secondary_linear_oracle.npz"))
    return oracle


def rescore_seed(
    *,
    run_dir: Path,
    output_dir: Path,
    additional_rollout_dirs: list[Path],
    helpers: Any,
    paper_helpers: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = (
        "v2_ppo_metadata.json",
        "custom_ppo.pt",
        "rollout_custom_ppo.npz",
        "rollout_validation_selected_static.npz",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing: {missing}")

    metadata = load_json(run_dir / "v2_ppo_metadata.json")
    control_source = dict(metadata.get("control_source", {}))
    source_dir_value = str(
        control_source.get("source_run_dir") or control_source.get("run_dir") or ""
    ).strip()
    source_dir = Path(source_dir_value).resolve() if source_dir_value else run_dir.resolve()
    truth_from_metadata = resolve_repo_path(str(metadata.get("truth_csv", "truth_v31_split.csv")))
    truth_path = first_existing_path(
        [
            run_dir / "truth_v31_split.csv",
            truth_from_metadata,
            source_dir / "truth_v31_split.csv",
        ],
        label="source truth",
    )
    manifest_path = first_existing_path(
        [
            run_dir / "split_protocol_manifest.json",
            resolve_repo_path(str(control_source.get("manifest_path", "split_protocol_manifest.json"))),
            source_dir / "split_protocol_manifest.json",
        ],
        label="split protocol manifest",
    )
    manifest = load_json(manifest_path)
    policy_cfg = dict(metadata.get("custom_ppo", {}))
    alerts = dict(metadata.get("agent_alert_context", {}))
    if bool(policy_cfg.get("subtype_router_enabled", False)):
        raise ValueError(f"Secondary rescoring rejects hard-router policy: {run_dir}")
    if bool(alerts.get("include_event_flag_in_state", True)):
        raise ValueError(f"Secondary rescoring rejects exact online event flag: {run_dir}")
    if bool(alerts.get("truth_event_labels_used_online", True)):
        raise ValueError(f"Secondary rescoring rejects online truth-label use: {run_dir}")

    truth = helpers.ensure_state_columns(pd.read_csv(truth_path))
    sensors = load_sensor_specs(resolve_repo_path(str(metadata["sensor_cfg"])))
    constraints = build_constraints(metadata)
    candidate_masks = load_candidate_masks(run_dir, expected_sensors=len(sensors))
    oracle = fit_secondary_oracle(
        run_dir=run_dir,
        output_dir=output_dir,
        truth=truth,
        metadata=metadata,
        sensors=sensors,
        constraints=constraints,
        candidate_masks=candidate_masks,
        helpers=helpers,
        paper_helpers=paper_helpers,
    )

    partition = dict(metadata["partition_protocol"])
    validation_steps = int(partition["static_selection_steps"])
    validation_cfg = build_env_cfg(
        truth=truth,
        metadata=metadata,
        helpers=helpers,
        episode_len=validation_steps,
    )
    _, validation_table = paper_helpers.build_oracle_candidate_prior(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=validation_cfg,
        oracle=oracle,
        candidate_masks=candidate_masks,
        start_indices=tuple(int(value) for value in partition["static_selection_start_indices"]),
        steps=validation_steps,
        scale=0.0,
    )
    validation_table.to_csv(output_dir / "secondary_validation_static_candidates.csv", index=False)
    normalizers = paper_helpers.subtype_static_normalizers(validation_table)
    selected = validation_table.sort_values(
        [PRIMARY_SCORE, "oracle_loss_mean", "action_idx"],
        ascending=True,
    ).iloc[0]
    selected_idx = int(selected["action_idx"])

    final_cfg = replace(
        validation_cfg,
        episode_len=int(dict(manifest["final_test"])["eval_steps"]),
        seed=int(metadata["seed"]) + 32_000,
    )
    linear_static_policy = StaticMaskPolicy(
        mask=tuple(bool(value) for value in candidate_masks[selected_idx]),
        name="secondary_validation_selected_static",
    )
    linear_static_result, _ = helpers.evaluate_score_policy_over_starts(
        truth=truth,
        sensors=sensors,
        constraints=constraints,
        cfg=final_cfg,
        oracle=oracle,
        policy=linear_static_policy,
        steps=int(final_cfg.episode_len),
        start_indices=tuple(int(value) for value in metadata["eval_start_indices"]),
    )
    linear_static_path = output_dir / "rollout_secondary_validation_selected_static.npz"
    save_rollout_npz(
        linear_static_path,
        linear_static_result,
        sensor_ids=[sensor.sensor_id for sensor in sensors],
        state_columns=helpers.STATE_COLUMNS,
    )

    rollout_paths: dict[str, Path] = {}
    for directory in [run_dir, *additional_rollout_dirs]:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("rollout_*.npz")):
            rollout_paths.setdefault(path.stem.removeprefix("rollout_"), path)
    rollout_paths["secondary_validation_selected_static"] = linear_static_path

    rows: list[dict[str, Any]] = []
    for policy_hint, path in rollout_paths.items():
        rollout = load_rollout_npz(path)
        samples = forecast_loss_samples(
            rollout,
            truth_df=truth,
            oracle=oracle,
            metadata=metadata,
        )
        row = summarize_samples(samples, policy=str(rollout.policy or policy_hint))
        row["rollout_path"] = str(path)
        rows.append(row)
    metrics = paper_helpers.add_staticnorm_macro(pd.DataFrame(rows), normalizers)
    metrics = metrics.sort_values(PRIMARY_SCORE).reset_index(drop=True)
    metrics.to_csv(output_dir / "secondary_forecaster_metrics.csv", index=False)

    provenance = {
        "seed": int(metadata["seed"]),
        "source_run_dir": str(run_dir),
        "control_source_run_dir": str(source_dir),
        "truth_sha256": sha256_file(truth_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "source_policy_metadata_sha256": sha256_file(run_dir / "v2_ppo_metadata.json"),
        "source_policy_checkpoint_sha256": sha256_file(run_dir / "custom_ppo.pt"),
        "secondary_forecaster": "multioutput_ridge",
        "secondary_forecaster_path": str(output_dir / "secondary_linear_oracle.npz"),
        "secondary_forecaster_sha256": sha256_file(output_dir / "secondary_linear_oracle.npz"),
        "forecaster_fit_partition": [
            int(partition["oracle_start_idx"]),
            int(partition["oracle_end_idx"]),
        ],
        "validation_start_indices": [int(value) for value in partition["static_selection_start_indices"]],
        "validation_steps": validation_steps,
        "final_start_indices": [int(value) for value in metadata["eval_start_indices"]],
        "final_steps": int(final_cfg.episode_len),
        "selected_secondary_static_action": selected_idx,
        "selected_secondary_static_sensors": str(selected["sensor_ids"]),
        "offline_event_labels_used_only_for_loss_grouping": True,
        "policy_or_trajectory_retraining": False,
    }
    (output_dir / "secondary_forecaster_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return metrics, provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a disjoint ridge forecaster and rescore frozen clean PD-PPO trajectories."
    )
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(117, 141)))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--additional-rollout-dir-template",
        action="append",
        default=[],
        help="Optional path template containing {seed}; rollout_*.npz files are rescored.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()

    helpers = load_script("_secondary_forecaster_train_helpers", "23_v2_train_ppo.py")
    paper_helpers = load_script("_secondary_forecaster_metric_helpers", "25_v2_train_custom_ppo.py")
    all_rows: list[pd.DataFrame] = []
    provenance_by_seed: dict[str, Any] = {}
    for seed in [int(value) for value in args.seeds]:
        run_dir = args.reports_root / (
            f"v31_scenebal2_matched_reward_forecast_noexactevent_seed{seed}_"
            f"h075forecastctrl_{args.date_tag}"
        )
        seed_out = args.out_dir / f"seed{seed}"
        additional = [Path(template.format(seed=seed)) for template in args.additional_rollout_dir_template]
        metrics, provenance = rescore_seed(
            run_dir=run_dir,
            output_dir=seed_out,
            additional_rollout_dirs=additional,
            helpers=helpers,
            paper_helpers=paper_helpers,
        )
        metrics.insert(0, "seed", seed)
        all_rows.append(metrics)
        provenance_by_seed[str(seed)] = provenance

    combined = pd.concat(all_rows, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_dir / "secondary_forecaster_seed_policy_metrics.csv", index=False)
    paired_rows: list[dict[str, Any]] = []
    for seed, group in combined.groupby("seed"):
        learned = group.loc[group["policy"] == "custom_ppo"]
        original_static = group.loc[group["policy"] == "validation_selected_static"]
        secondary_static = group.loc[group["policy"] == "secondary_validation_selected_static"]
        if len(learned) != 1 or len(original_static) != 1 or len(secondary_static) != 1:
            raise ValueError(f"Missing required policies for seed {seed}")
        learned_row = learned.iloc[0]
        original_row = original_static.iloc[0]
        secondary_row = secondary_static.iloc[0]
        paired_rows.append(
            {
                "seed": int(seed),
                "pdppo_step_loss": float(learned_row["oracle_loss_mean"]),
                "original_static_step_loss": float(original_row["oracle_loss_mean"]),
                "secondary_static_step_loss": float(secondary_row["oracle_loss_mean"]),
                "step_margin_vs_original_static": float(
                    original_row["oracle_loss_mean"] - learned_row["oracle_loss_mean"]
                ),
                "step_margin_vs_secondary_static": float(
                    secondary_row["oracle_loss_mean"] - learned_row["oracle_loss_mean"]
                ),
                "pdppo_macro_score": float(learned_row[PRIMARY_SCORE]),
                "original_static_macro_score": float(original_row[PRIMARY_SCORE]),
                "secondary_static_macro_score": float(secondary_row[PRIMARY_SCORE]),
                "macro_margin_vs_original_static": float(
                    original_row[PRIMARY_SCORE] - learned_row[PRIMARY_SCORE]
                ),
                "macro_margin_vs_secondary_static": float(
                    secondary_row[PRIMARY_SCORE] - learned_row[PRIMARY_SCORE]
                ),
            }
        )
    paired = pd.DataFrame(paired_rows).sort_values("seed").reset_index(drop=True)
    paired.to_csv(args.out_dir / "secondary_forecaster_paired_metrics.csv", index=False)

    summary_row: dict[str, Any] = {"n_seeds": int(len(paired))}
    for margin_name in (
        "step_margin_vs_original_static",
        "step_margin_vs_secondary_static",
        "macro_margin_vs_original_static",
        "macro_margin_vs_secondary_static",
    ):
        values = paired[margin_name].to_numpy(dtype=float)
        ci = bootstrap_mean_ci(
            values,
            draws=int(args.bootstrap_draws),
            seed=93_000 + len(summary_row),
        )
        summary_row[f"{margin_name}_mean"] = float(np.mean(values))
        summary_row[f"{margin_name}_wins"] = int(np.sum(values > 0.0))
        summary_row[f"{margin_name}_ci95_low"] = ci[0]
        summary_row[f"{margin_name}_ci95_high"] = ci[1]
    summary = pd.DataFrame([summary_row])
    summary.to_csv(args.out_dir / "secondary_forecaster_summary.csv", index=False)
    (args.out_dir / "secondary_forecaster_protocol.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "date_tag": str(args.date_tag),
                "seeds": [int(value) for value in args.seeds],
                "positive_margin_definition": "static loss minus PD-PPO loss",
                "provenance_by_seed": provenance_by_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out_dir / "secondary_forecaster_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
