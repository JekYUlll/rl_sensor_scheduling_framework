#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_SCORE = "oracle_loss_macro_subtype_event_staticnorm"
STEP_SCORE = "oracle_loss_mean"
BEHAVIOR_COLUMNS = (
    "switches_per_step",
    "warmup_abort_count",
    "always_on_sensor_count",
    "always_off_sensor_count",
    "mid_duty_sensor_count",
    "duty_entropy",
    "duty_min",
    "duty_max",
)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows)) if rows else len(columns[idx])
        for idx in range(len(columns))
    ]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    header = render(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([header, separator, *(render(row) for row in rows)])


def find_policy(table: pd.DataFrame, policy: str) -> pd.Series:
    rows = table.loc[table["policy"].astype(str) == str(policy)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {policy!r} row, found {len(rows)}")
    return rows.iloc[0]


def bootstrap_mean_ci(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.mean(rng.choice(data, size=(int(draws), int(data.size)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_behavior_auditor() -> Any:
    path = Path(__file__).with_name("71_v31_behavior_complexity_audit.py")
    spec = importlib.util.spec_from_file_location("v31_behavior_complexity_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_rollout


def audit_behavior(path: Path) -> dict[str, Any]:
    audit_rollout = load_behavior_auditor()
    return audit_rollout(
        path,
        max_period=64,
        fixed_top1_threshold=0.95,
        simple_top3_threshold=0.85,
        simple_period_threshold=0.90,
        min_unique_masks=5,
        min_mask_entropy_bits=1.50,
        min_transition_entropy_bits=1.25,
        min_event_sensor_l1=0.50,
        min_event_mi_bits=0.10,
        min_subtype_sensor_l1=1.00,
        min_subtype_mi_bits=0.25,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect mask-matched Double-DQN and forecast-PPO results.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--dqn-date-tag", default="20260718pilot")
    parser.add_argument("--ppo-date-tag", default="20260718pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[117, 118])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    protocol: dict[str, object] = {}
    for seed in args.seeds:
        dqn_dir = args.reports_root / f"v31_scenebal2_matched_dqn_seed{seed}_h075_{args.dqn_date_tag}"
        ppo_dir = args.reports_root / (
            f"v31_scenebal2_matched_reward_forecast_noexactevent_seed{seed}_"
            f"h075forecastctrl_{args.ppo_date_tag}"
        )
        dqn_metrics_path = dqn_dir / "v31_matched_dqn_metrics.csv"
        dqn_metadata_path = dqn_dir / "v31_matched_dqn_metadata.json"
        ppo_metrics_path = ppo_dir / "v2_custom_ppo_metrics.csv"
        ppo_metadata_path = ppo_dir / "v2_ppo_metadata.json"
        ppo_checkpoint_path = ppo_dir / "custom_ppo.pt"
        dqn_rollout_path = dqn_dir / "rollout_dqn.npz"
        ppo_rollout_path = ppo_dir / "rollout_custom_ppo.npz"
        for path in (
            dqn_metrics_path,
            dqn_metadata_path,
            ppo_metrics_path,
            ppo_metadata_path,
            ppo_checkpoint_path,
            dqn_rollout_path,
            ppo_rollout_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        dqn_table = pd.read_csv(dqn_metrics_path)
        ppo_table = pd.read_csv(ppo_metrics_path)
        dqn_metadata = json.loads(dqn_metadata_path.read_text(encoding="utf-8"))
        ppo_metadata = json.loads(ppo_metadata_path.read_text(encoding="utf-8"))
        ppo_source = dict(ppo_metadata.get("control_source", {}))
        ppo_alert = dict(ppo_metadata.get("agent_alert_context", {}))
        ppo_policy = dict(ppo_metadata.get("custom_ppo", {}))
        dqn_online = dict(dqn_metadata.get("online_observation_contract", {}))

        if str(dqn_metadata.get("method")) != "masked_double_dqn":
            raise ValueError(f"Unexpected DQN method in {dqn_metadata_path}")
        if bool(dqn_online.get("include_exact_event_flag", True)):
            raise ValueError(f"DQN exposes the exact event flag for seed {seed}")
        if bool(dqn_online.get("final_test_event_labels_used_by_policy", True)):
            raise ValueError(f"DQN uses final event labels for seed {seed}")
        if bool(ppo_alert.get("include_event_flag_in_state", True)):
            raise ValueError(f"Forecast PPO exposes the exact event flag for seed {seed}")
        if bool(ppo_alert.get("truth_event_labels_used_online", True)):
            raise ValueError(f"Forecast PPO records online truth-label use for seed {seed}")
        if bool(ppo_policy.get("subtype_router_enabled", False)):
            raise ValueError(f"Forecast PPO uses the hard subtype-to-action router for seed {seed}")

        truth_hash = str(dqn_metadata.get("truth_sha256"))
        oracle_hash = str(dqn_metadata.get("source_oracle_sha256"))
        if truth_hash != str(ppo_source.get("truth_sha256")):
            raise ValueError(f"Truth checksum differs between DQN and PPO for seed {seed}")
        if oracle_hash != str(ppo_source.get("source_oracle_sha256")):
            raise ValueError(f"Frozen evaluator checksum differs between DQN and PPO for seed {seed}")
        if str(dqn_metadata.get("copied_oracle_sha256")) != oracle_hash:
            raise ValueError(f"DQN copied evaluator checksum differs for seed {seed}")
        if str(dqn_metadata.get("source_metadata_sha256")) != str(ppo_source.get("metadata_sha256")):
            raise ValueError(f"Source metadata checksum differs between DQN and PPO for seed {seed}")
        if str(dqn_metadata.get("source_manifest_sha256")) != str(ppo_source.get("manifest_sha256")):
            raise ValueError(f"Source split-manifest checksum differs between DQN and PPO for seed {seed}")
        dqn_partition = dict(dqn_metadata.get("partitions", {}))
        dqn_reward = dict(dqn_metadata.get("reward", {}))
        training_evaluator_device = str(dqn_reward.get("training_evaluator_device", ""))
        evaluation_evaluator_device = str(dqn_reward.get("evaluation_evaluator_device", ""))
        if training_evaluator_device != "cuda":
            raise ValueError(
                f"DQN training evaluator device differs from the frozen protocol for seed {seed}; "
                f"found {training_evaluator_device!r}"
            )
        if evaluation_evaluator_device != "cpu":
            raise ValueError(
                f"DQN final scoring must use the common CPU evaluator for seed {seed}; "
                f"found {evaluation_evaluator_device!r}"
            )
        ppo_eval_starts = tuple(int(value) for value in ppo_metadata.get("eval_start_indices", ()))
        dqn_eval_starts = tuple(int(value) for value in dqn_partition.get("eval_start_indices", ()))
        if dqn_eval_starts != ppo_eval_starts:
            raise ValueError(f"Final-test starts differ between DQN and PPO for seed {seed}")
        dqn_candidate_count = int(dqn_metadata.get("candidate_mask_count", -1))
        ppo_candidate_count = int(ppo_policy.get("candidate_count", -1))
        if dqn_candidate_count != ppo_candidate_count or dqn_candidate_count != 6:
            raise ValueError(f"Candidate-mask count differs between DQN and PPO for seed {seed}")
        import torch

        ppo_checkpoint = torch.load(ppo_checkpoint_path, map_location="cpu", weights_only=False)
        ppo_candidate_masks = np.asarray(ppo_checkpoint["candidate_masks"], dtype=bool)
        dqn_candidate_masks = np.asarray(dqn_metadata.get("candidate_masks", ()), dtype=bool)
        if not np.array_equal(dqn_candidate_masks, ppo_candidate_masks):
            raise ValueError(f"Candidate masks differ between DQN and PPO for seed {seed}")
        if dict(dqn_metadata.get("constraints", {})) != dict(ppo_metadata.get("constraints", {})):
            raise ValueError(f"Power and execution constraints differ for seed {seed}")
        if tuple(str(value) for value in dqn_online.get("agent_context_columns", ())) != tuple(
            str(value) for value in ppo_metadata.get("agent_context_columns", ())
        ):
            raise ValueError(f"Online context columns differ for seed {seed}")
        source_manifest = dict(ppo_source.get("source_manifest", {}))
        expected_eval_steps = int(dict(source_manifest.get("final_test", {})).get("eval_steps", -1))
        if int(dqn_partition.get("eval_steps", -1)) != expected_eval_steps:
            raise ValueError(f"Final-test length differs between DQN and PPO for seed {seed}")
        if str(dqn_reward.get("mode")) != "forecast":
            raise ValueError(f"DQN reward is not forecast loss for seed {seed}")

        dqn = find_policy(dqn_table, "dqn")
        ppo = find_policy(ppo_table, "custom_ppo")
        dqn_static = find_policy(dqn_table, "validation_selected_static")
        ppo_static = find_policy(ppo_table, "validation_selected_static")
        for score in (STEP_SCORE, PRIMARY_SCORE):
            if not np.isclose(float(dqn_static[score]), float(ppo_static[score]), rtol=0.0, atol=1e-9):
                raise ValueError(f"Static replay {score} differs between DQN and PPO for seed {seed}")
        ppo_behavior = audit_behavior(ppo_rollout_path)
        dqn_behavior = audit_behavior(dqn_rollout_path)

        row: dict[str, object] = {
            "seed": int(seed),
            "ppo_run_dir": str(ppo_dir),
            "dqn_run_dir": str(dqn_dir),
            "ppo_step_loss": float(ppo[STEP_SCORE]),
            "dqn_step_loss": float(dqn[STEP_SCORE]),
            "step_margin_dqn_minus_ppo": float(dqn[STEP_SCORE] - ppo[STEP_SCORE]),
            "ppo_macro_score": float(ppo[PRIMARY_SCORE]),
            "dqn_macro_score": float(dqn[PRIMARY_SCORE]),
            "macro_margin_dqn_minus_ppo": float(dqn[PRIMARY_SCORE] - ppo[PRIMARY_SCORE]),
            "static_step_loss": float(ppo_static[STEP_SCORE]),
            "static_macro_score": float(ppo_static[PRIMARY_SCORE]),
            "ppo_step_margin_vs_static": float(ppo_static[STEP_SCORE] - ppo[STEP_SCORE]),
            "dqn_step_margin_vs_static": float(dqn_static[STEP_SCORE] - dqn[STEP_SCORE]),
            "ppo_macro_margin_vs_static": float(ppo_static[PRIMARY_SCORE] - ppo[PRIMARY_SCORE]),
            "dqn_macro_margin_vs_static": float(dqn_static[PRIMARY_SCORE] - dqn[PRIMARY_SCORE]),
            "truth_sha256": truth_hash,
            "oracle_sha256": oracle_hash,
            "ppo_behavior_gate_pass": bool(ppo_behavior["behavior_complexity_gate_pass"]),
            "dqn_behavior_gate_pass": bool(dqn_behavior["behavior_complexity_gate_pass"]),
        }
        for prefix, policy_row in (("ppo", ppo), ("dqn", dqn)):
            for column in BEHAVIOR_COLUMNS:
                row[f"{prefix}_{column}"] = (
                    float(policy_row[column]) if column in policy_row.index else float("nan")
                )
        for prefix, behavior in (("ppo", ppo_behavior), ("dqn", dqn_behavior)):
            for key in (
                "unique_mask_count",
                "mask_entropy_bits",
                "transition_entropy_bits",
                "subtype_mask_mi_bits",
                "fixed_like",
                "simple_cycle_like",
            ):
                row[f"{prefix}_{key}"] = behavior[key]
        rows.append(row)
        protocol[str(seed)] = {
            "truth_sha256": truth_hash,
            "oracle_sha256": oracle_hash,
            "candidate_mask_count": dqn_candidate_count,
            "eval_start_indices": list(dqn_eval_starts),
            "eval_steps": int(dqn_partition.get("eval_steps", -1)),
            "training_evaluator_device": training_evaluator_device,
            "evaluation_evaluator_device": evaluation_evaluator_device,
        }

    seed_metrics = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    step_margins = seed_metrics["step_margin_dqn_minus_ppo"].to_numpy(dtype=float)
    macro_margins = seed_metrics["macro_margin_dqn_minus_ppo"].to_numpy(dtype=float)
    step_ci = bootstrap_mean_ci(step_margins, draws=int(args.bootstrap_draws), seed=71_811)
    macro_ci = bootstrap_mean_ci(macro_margins, draws=int(args.bootstrap_draws), seed=71_812)
    summary = pd.DataFrame(
        [
            {
                "n_seeds": int(len(seed_metrics)),
                "ppo_step_wins_vs_dqn": int(np.sum(step_margins > 0.0)),
                "ppo_macro_wins_vs_dqn": int(np.sum(macro_margins > 0.0)),
                "step_margin_dqn_minus_ppo_mean": float(np.mean(step_margins)),
                "step_margin_ci95_low": step_ci[0],
                "step_margin_ci95_high": step_ci[1],
                "macro_margin_dqn_minus_ppo_mean": float(np.mean(macro_margins)),
                "macro_margin_ci95_low": macro_ci[0],
                "macro_margin_ci95_high": macro_ci[1],
                "ppo_macro_wins_vs_static": int(np.sum(seed_metrics["ppo_macro_margin_vs_static"] > 0.0)),
                "dqn_macro_wins_vs_static": int(np.sum(seed_metrics["dqn_macro_margin_vs_static"] > 0.0)),
                "ppo_warmup_abort_total": int(np.nansum(seed_metrics["ppo_warmup_abort_count"])),
                "dqn_warmup_abort_total": int(np.nansum(seed_metrics["dqn_warmup_abort_count"])),
                "ppo_behavior_gate_passes": int(np.sum(seed_metrics["ppo_behavior_gate_pass"])),
                "dqn_behavior_gate_passes": int(np.sum(seed_metrics["dqn_behavior_gate_pass"])),
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(args.out_dir / "matched_dqn_seed_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "matched_dqn_summary.csv", index=False)
    (args.out_dir / "matched_dqn_protocol_audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "positive_margin_definition": "DQN loss minus forecast-PPO loss",
                "dqn_date_tag": str(args.dqn_date_tag),
                "ppo_date_tag": str(args.ppo_date_tag),
                "seeds": [int(seed) for seed in args.seeds],
                "protocol_by_seed": protocol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Matched learned-policy summary",
        "",
        "Positive margins mean the corrected forecast-reward PPO has lower held-out loss than masked Double-DQN.",
        "The collector verifies identical truth, frozen evaluator, candidate-mask count, final starts, exact-event exclusion, and fixed-schedule replay scores.",
        "The same action-trace complexity audit is reported for both learned policies; it is descriptive for the DQN baseline rather than a prerequisite for retaining it.",
        "",
        dataframe_to_markdown(summary),
    ]
    (args.out_dir / "matched_dqn_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "matched_dqn_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
