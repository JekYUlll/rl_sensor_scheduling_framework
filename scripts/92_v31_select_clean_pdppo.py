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


def find_policy(table: pd.DataFrame, policy: str) -> pd.Series:
    rows = table.loc[table["policy"].astype(str) == str(policy)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {policy!r} row, found {len(rows)}")
    return rows.iloc[0]


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows)) if rows else len(columns[idx])
        for idx in range(len(columns))
    ]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    return "\n".join(
        [
            render(columns),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(render(row) for row in rows),
        ]
    )


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


def run_dir(reports_root: Path, *, seed: int, date_tag: str) -> Path:
    return reports_root / (
        f"v31_scenebal2_matched_reward_forecast_noexactevent_seed{seed}_"
        f"h075forecastctrl_{date_tag}"
    )


def protocol_signature(metadata: dict[str, Any]) -> dict[str, Any]:
    control = dict(metadata.get("control_source", {}))
    partition = dict(metadata.get("partition_protocol", {}))
    policy = dict(metadata.get("custom_ppo", {}))
    return {
        "truth_sha256": str(control.get("truth_sha256")),
        "oracle_sha256": str(control.get("source_oracle_sha256")),
        "copied_oracle_sha256": str(control.get("copied_oracle_sha256")),
        "candidate_count": int(policy.get("candidate_count", -1)),
        "static_selection_start_indices": tuple(
            int(value) for value in partition.get("static_selection_start_indices", ())
        ),
        "eval_start_indices": tuple(int(value) for value in metadata.get("eval_start_indices", ())),
    }


def load_candidate(
    *,
    reports_root: Path,
    candidate: str,
    date_tag: str,
    expected_context_encoder: bool,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    protocols: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        directory = run_dir(reports_root, seed=seed, date_tag=date_tag)
        metrics_path = directory / "v2_custom_ppo_metrics.csv"
        metadata_path = directory / "v2_ppo_metadata.json"
        rollout_path = directory / "rollout_custom_ppo.npz"
        for path in (metrics_path, metadata_path, rollout_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        metrics = pd.read_csv(metrics_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        control = dict(metadata.get("control_source", {}))
        alerts = dict(metadata.get("agent_alert_context", {}))
        reward = dict(metadata.get("reward_shaping", {}))
        policy = dict(metadata.get("custom_ppo", {}))
        if not bool(control.get("enabled", False)):
            raise ValueError(f"{directory} did not reuse frozen source assets")
        if str(reward.get("reward_proxy_mode")) != "forecast":
            raise ValueError(f"{directory} is not a forecast-reward run")
        if bool(alerts.get("include_event_flag_in_state", True)):
            raise ValueError(f"{directory} exposes the exact event flag online")
        if bool(alerts.get("truth_event_labels_used_online", True)):
            raise ValueError(f"{directory} records online truth-label use")
        if bool(policy.get("subtype_router_enabled", False)):
            raise ValueError(f"{directory} uses the hard subtype-to-action router")
        if bool(policy.get("context_encoder_enabled", False)) != bool(expected_context_encoder):
            raise ValueError(f"{directory} has an unexpected context-encoder setting")
        signature = protocol_signature(metadata)
        if signature["oracle_sha256"] != signature["copied_oracle_sha256"]:
            raise ValueError(f"{directory} frozen evaluator checksum mismatch")
        if signature["candidate_count"] != 6:
            raise ValueError(f"{directory} has {signature['candidate_count']} actions instead of 6")

        learned = find_policy(metrics, "custom_ppo")
        static = find_policy(metrics, "validation_selected_static")
        behavior = audit_behavior(rollout_path)
        rows.append(
            {
                "candidate": candidate,
                "date_tag": date_tag,
                "seed": int(seed),
                "run_dir": str(directory),
                "context_encoder": bool(expected_context_encoder),
                "step_loss": float(learned[STEP_SCORE]),
                "static_step_loss": float(static[STEP_SCORE]),
                "step_margin_vs_static": float(static[STEP_SCORE] - learned[STEP_SCORE]),
                "macro_score": float(learned[PRIMARY_SCORE]),
                "static_macro_score": float(static[PRIMARY_SCORE]),
                "macro_margin_vs_static": float(static[PRIMARY_SCORE] - learned[PRIMARY_SCORE]),
                "switches_per_step": float(learned.get("switches_per_step", np.nan)),
                "warmup_abort_count": float(learned.get("warmup_abort_count", np.nan)),
                "always_on_sensor_count": float(learned.get("always_on_sensor_count", np.nan)),
                "always_off_sensor_count": float(learned.get("always_off_sensor_count", np.nan)),
                "mid_duty_sensor_count": float(learned.get("mid_duty_sensor_count", np.nan)),
                "unique_mask_count": int(behavior["unique_mask_count"]),
                "mask_entropy_bits": float(behavior["mask_entropy_bits"]),
                "subtype_mask_mi_bits": float(behavior["subtype_mask_mi_bits"]),
                "behavior_gate_pass": bool(behavior["behavior_complexity_gate_pass"]),
                "truth_sha256": signature["truth_sha256"],
                "oracle_sha256": signature["oracle_sha256"],
            }
        )
        protocols[int(seed)] = signature
    return rows, protocols


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in seed_metrics.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": str(candidate),
                "n_seeds": int(len(group)),
                "step_loss_mean": float(group["step_loss"].mean()),
                "macro_score_mean": float(group["macro_score"].mean()),
                "step_margin_vs_static_mean": float(group["step_margin_vs_static"].mean()),
                "macro_margin_vs_static_mean": float(group["macro_margin_vs_static"].mean()),
                "step_wins_vs_static": int(np.sum(group["step_margin_vs_static"] > 0.0)),
                "macro_wins_vs_static": int(np.sum(group["macro_margin_vs_static"] > 0.0)),
                "behavior_gate_passes": int(np.sum(group["behavior_gate_pass"])),
                "warmup_abort_total": int(np.nansum(group["warmup_abort_count"])),
                "switches_per_step_mean": float(group["switches_per_step"].mean()),
                "mask_entropy_bits_mean": float(group["mask_entropy_bits"].mean()),
            }
        )
    return pd.DataFrame(rows)


def candidate_valid(summary_row: pd.Series) -> bool:
    n = int(summary_row["n_seeds"])
    return bool(
        int(summary_row["step_wins_vs_static"]) == n
        and int(summary_row["macro_wins_vs_static"]) == n
        and int(summary_row["behavior_gate_passes"]) == n
        and int(summary_row["warmup_abort_total"]) == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and select clean actor-only PD-PPO pilot candidates.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--plain-tag", default="20260718cleanpilot")
    parser.add_argument("--context-tag", default="20260718capilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[117, 118])
    parser.add_argument("--material-macro-improvement", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    plain_rows, plain_protocol = load_candidate(
        reports_root=args.reports_root,
        candidate="plain_pdppo",
        date_tag=str(args.plain_tag),
        expected_context_encoder=False,
        seeds=[int(seed) for seed in args.seeds],
    )
    context_rows, context_protocol = load_candidate(
        reports_root=args.reports_root,
        candidate="ca_pdppo",
        date_tag=str(args.context_tag),
        expected_context_encoder=True,
        seeds=[int(seed) for seed in args.seeds],
    )
    if plain_protocol != context_protocol:
        raise ValueError("Plain and context candidates do not share the same frozen protocol")

    seed_metrics = pd.DataFrame([*plain_rows, *context_rows]).sort_values(
        ["seed", "candidate"]
    ).reset_index(drop=True)
    paired = seed_metrics.pivot(index="seed", columns="candidate", values=["step_loss", "macro_score"])
    paired.columns = [f"{metric}_{candidate}" for metric, candidate in paired.columns]
    paired = paired.reset_index()
    paired["macro_improvement_ca_vs_plain"] = (
        paired["macro_score_plain_pdppo"] - paired["macro_score_ca_pdppo"]
    )
    paired["step_improvement_ca_vs_plain"] = (
        paired["step_loss_plain_pdppo"] - paired["step_loss_ca_pdppo"]
    )

    summary = summarize(seed_metrics)
    plain_summary = summary.loc[summary["candidate"] == "plain_pdppo"].iloc[0]
    context_summary = summary.loc[summary["candidate"] == "ca_pdppo"].iloc[0]
    plain_valid = candidate_valid(plain_summary)
    context_valid = candidate_valid(context_summary)
    mean_macro_improvement = float(paired["macro_improvement_ca_vs_plain"].mean())
    context_paired_wins = int(np.sum(paired["macro_improvement_ca_vs_plain"] > 0.0))
    required_wins = int(len(paired))

    selected: str | None
    reason: str
    if plain_valid and context_valid:
        if (
            context_paired_wins == required_wins
            and mean_macro_improvement >= float(args.material_macro_improvement)
            and int(context_summary["step_wins_vs_static"]) >= int(plain_summary["step_wins_vs_static"])
        ):
            selected = "ca_pdppo"
            reason = "context encoder clears the prespecified material-improvement and behavior gates"
        else:
            selected = "plain_pdppo"
            reason = "plain actor is valid and the context encoder does not justify its added complexity"
    elif plain_valid:
        selected = "plain_pdppo"
        reason = "only the plain actor clears all pilot gates"
    elif context_valid:
        selected = "ca_pdppo"
        reason = "only the context-aware actor clears all pilot gates"
    else:
        selected = None
        reason = "neither clean actor clears the frozen pilot gates; do not expand"

    decision = {
        "status": "selected" if selected is not None else "blocked",
        "selected_candidate": selected,
        "reason": reason,
        "plain_valid": plain_valid,
        "context_valid": context_valid,
        "context_macro_wins_vs_plain": context_paired_wins,
        "paired_seed_count": required_wins,
        "mean_macro_improvement_ca_vs_plain": mean_macro_improvement,
        "material_macro_improvement_threshold": float(args.material_macro_improvement),
        "seeds": [int(seed) for seed in args.seeds],
        "plain_tag": str(args.plain_tag),
        "context_tag": str(args.context_tag),
        "protocol": {
            str(seed): {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in payload.items()
            }
            for seed, payload in plain_protocol.items()
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(args.out_dir / "clean_candidate_seed_metrics.csv", index=False)
    paired.to_csv(args.out_dir / "clean_candidate_paired_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "clean_candidate_summary.csv", index=False)
    (args.out_dir / "clean_candidate_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "# Clean PD-PPO pilot selection",
        "",
        "Both candidates execute only feasibility-masked actor logits at final evaluation.",
        "Exact simulator event labels are excluded from the online policy interface.",
        "",
        dataframe_to_markdown(summary),
        "",
        dataframe_to_markdown(paired),
        "",
        f"Decision: `{selected or 'no expansion'}`. {reason}.",
    ]
    (args.out_dir / "clean_candidate_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "clean_candidate_decision.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
