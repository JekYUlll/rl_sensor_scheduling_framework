#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DECISION_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "upgrade_allseed_strict": {
        "next_layer": "claim_update",
        "next_action": "Upgrade the final specialist-budget claim only after confirming every pre-registered aggregate and artifact is present.",
    },
    "upgrade_sign_bounded": {
        "next_layer": "claim_update_no_blind_expansion",
        "next_action": "Use sign-bounded strong wording; do not run another same-configuration seed wave unless it answers a concrete unresolved uncertainty.",
    },
    "pivot_true_static_step_sign_failure": {
        "next_layer": "simulator/data generation",
        "next_action": "Start a bounded simulator/data-balance pilot on failing seeds plus a clean control seed while preserving PPO and the met-plus-one-specialist sensing premise.",
    },
    "pivot_behavior_failure": {
        "next_layer": "PPO-REGIME-2 observation/auxiliary architecture",
        "next_action": "Start a bounded PPO observation, regime-belief, memory/lead-context, or auxiliary-head pilot; reject any fixed-like or simple-cycle policy.",
    },
    "pivot_replay_headroom_failure": {
        "next_layer": "simulator/data generation",
        "next_action": "Do not tune PPO first; repair the scenario so explicit dynamic replay again beats true fixed static and operational references.",
    },
    "diagnose_operational_failure": {
        "next_layer": "reward/evaluation or PPO credit assignment",
        "next_action": "Check rule-switching realism and reward/oracle credit assignment before changing the simulator.",
    },
    "hold_true_static_macro_boundary": {
        "next_layer": "EVAL-REWARD-2 narrow protocol audit",
        "next_action": "Audit true-static macro scale/window alignment first; if no protocol bug exists, treat the claim as bounded and do not upgrade.",
    },
    "incomplete_wait": {
        "next_layer": "wait_or_repair_artifacts",
        "next_action": "Wait for missing seeds or repair missing artifacts; do not update claim wording.",
    },
}

PIVOT_PROTOCOL_REPORT = "reports/aggregate/pdppo_next_layer_pivot_designs_20260621.md"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def as_bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"})


def finite_positive_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.map(lambda value: bool(np.isfinite(value) and value > 0.0))


def count_true(frame: pd.DataFrame, column: str) -> int:
    return int(as_bool_series(frame, column).sum())


def seeds_where(frame: pd.DataFrame, mask: pd.Series) -> list[int]:
    if "seed" not in frame.columns:
        return []
    seeds = frame.loc[mask.fillna(False), "seed"].tolist()
    result: list[int] = []
    for seed in seeds:
        try:
            result.append(int(seed))
        except Exception:
            continue
    return result


def fmt_count(count: int, total: int) -> str:
    return f"{count}/{total}"


def decision_from_gates(gates: dict[str, Any], expected_seeds: int | None) -> tuple[str, str]:
    complete = as_int(gates["complete_seeds"])
    if expected_seeds is not None and complete < expected_seeds:
        return "incomplete_wait", "aggregate has fewer complete seeds than expected"
    if complete <= 0:
        return "incomplete_wait", "no complete seeds"

    if gates["true_static_step_positive_count"] < complete:
        return "pivot_true_static_step_sign_failure", "PPO loses in sign to true fixed static on at least one seed"
    if gates["behavior_gate_count"] < complete:
        return "pivot_behavior_failure", "learned behavior is fixed-like, cyclic, or not state-dependent on at least one seed"
    if gates["replay_step_gate_count"] < complete or gates["replay_macro_gate_count"] < complete:
        return "pivot_replay_headroom_failure", "explicit dynamic replay headroom is missing on at least one seed"
    if gates["step_operational_gate_count"] < complete or gates["macro_operational_gate_count"] < complete:
        return "diagnose_operational_failure", "operational static/rule baselines beat PPO on at least one objective"
    if gates["true_static_macro_gate_count"] < complete:
        return "hold_true_static_macro_boundary", "true-static macro gate is not all-seed clean"
    if gates["true_static_step_strict_gate_count"] == complete:
        return "upgrade_allseed_strict", "all pre-registered gates including strict true-static step pass"
    return (
        "upgrade_sign_bounded",
        "all operational/replay/behavior/true-static-sign gates pass; strict true-static step remains a boundary",
    )


def recommendation_for(decision: str) -> dict[str, str]:
    default = {
        "next_layer": "manual_review",
        "next_action": "Review the decision audit and choose a bounded next unit before launching new experiments.",
    }
    rec = dict(DECISION_RECOMMENDATIONS.get(decision, default))
    rec["protocol_report"] = PIVOT_PROTOCOL_REPORT
    return rec


def markdown_list(values: list[int]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide final-benchmark stress-wave claim status from aggregate files.")
    parser.add_argument("--oldclaim-dir", required=True)
    parser.add_argument("--macro-dir", default=None)
    parser.add_argument("--raw-macro-dir", default=None)
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument("--label", default="final benchmark stress aggregate")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    oldclaim_dir = resolve(args.oldclaim_dir)
    macro_dir = resolve(args.macro_dir) if args.macro_dir else None
    raw_macro_dir = resolve(args.raw_macro_dir) if args.raw_macro_dir else None
    out_json = resolve(args.out_json)
    out_md = resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    old_summary = load_json(oldclaim_dir / "oldclaim_summary.json")
    old_table = pd.read_csv(oldclaim_dir / "oldclaim_seed_summary.csv")
    complete_mask = as_bool_series(old_table, "complete")
    complete_table = old_table.loc[complete_mask].copy()
    complete = as_int(old_summary.get("complete_seeds", len(complete_table)))

    true_static_step_positive = finite_positive_series(complete_table, "step_margin_vs_replay_static_reference")
    gates: dict[str, Any] = {
        "complete_seeds": complete,
        "step_operational_gate_count": as_int(old_summary.get("step_operational_gate_count", count_true(complete_table, "step_operational_gate_pass"))),
        "macro_operational_gate_count": as_int(old_summary.get("macro_operational_gate_count", count_true(complete_table, "macro_operational_gate_pass"))),
        "replay_step_gate_count": as_int(old_summary.get("replay_gate_count", count_true(complete_table, "replay_gate_pass"))),
        "replay_macro_gate_count": as_int(old_summary.get("replay_macro_gate_count", count_true(complete_table, "replay_macro_gate_pass"))),
        "behavior_gate_count": as_int(old_summary.get("behavior_gate_count", count_true(complete_table, "behavior_gate_pass"))),
        "true_static_macro_gate_count": as_int(
            old_summary.get("learned_true_static_macro_gate_count", count_true(complete_table, "learned_true_static_macro_gate_pass"))
        ),
        "true_static_step_strict_gate_count": as_int(
            old_summary.get("learned_true_static_step_gate_count", count_true(complete_table, "learned_true_static_step_gate_pass"))
        ),
        "true_static_step_positive_count": int(true_static_step_positive.sum()),
        "old_claim_step_gate_count": as_int(old_summary.get("old_claim_step_gate_count", count_true(complete_table, "old_claim_step_gate_pass"))),
        "old_claim_macro_gate_count": as_int(old_summary.get("old_claim_macro_gate_count", count_true(complete_table, "old_claim_macro_gate_pass"))),
    }

    decision, rationale = decision_from_gates(gates, args.expected_seeds)
    failure_masks = {
        "step_operational_fail_seeds": ~as_bool_series(complete_table, "step_operational_gate_pass"),
        "macro_operational_fail_seeds": ~as_bool_series(complete_table, "macro_operational_gate_pass"),
        "replay_step_fail_seeds": ~as_bool_series(complete_table, "replay_gate_pass"),
        "replay_macro_fail_seeds": ~as_bool_series(complete_table, "replay_macro_gate_pass"),
        "behavior_fail_seeds": ~as_bool_series(complete_table, "behavior_gate_pass"),
        "true_static_step_sign_fail_seeds": ~true_static_step_positive,
        "true_static_step_strict_fail_seeds": ~as_bool_series(complete_table, "learned_true_static_step_gate_pass"),
        "true_static_macro_fail_seeds": ~as_bool_series(complete_table, "learned_true_static_macro_gate_pass"),
    }
    failures = {name: seeds_where(complete_table, mask) for name, mask in failure_masks.items()}

    margins: dict[str, Any] = {}
    if "step_margin_vs_replay_static_reference" in complete_table.columns:
        values = pd.to_numeric(complete_table["step_margin_vs_replay_static_reference"], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            margins["true_static_step_margin_min"] = float(finite.min())
            margins["true_static_step_margin_median"] = float(finite.median())
            margins["true_static_step_margin_mean"] = float(finite.mean())
            margins["true_static_step_margin_max"] = float(finite.max())
    if "step_margin_vs_best_operational_baseline" in complete_table.columns:
        values = pd.to_numeric(complete_table["step_margin_vs_best_operational_baseline"], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            margins["operational_step_margin_min"] = float(finite.min())
            margins["operational_step_margin_median"] = float(finite.median())
            margins["operational_step_margin_mean"] = float(finite.mean())
            margins["operational_step_margin_max"] = float(finite.max())

    macro_summary = load_json(macro_dir / "metpair_claim_summary.json") if macro_dir else {}
    raw_macro_summary = load_json(raw_macro_dir / "metpair_claim_summary.json") if raw_macro_dir else {}

    recommendation = recommendation_for(decision)

    result = {
        "label": args.label,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "oldclaim_dir": rel(oldclaim_dir),
        "macro_dir": rel(macro_dir) if macro_dir else None,
        "raw_macro_dir": rel(raw_macro_dir) if raw_macro_dir else None,
        "expected_seeds": args.expected_seeds,
        "decision": decision,
        "rationale": rationale,
        "recommendation": recommendation,
        "gates": gates,
        "failures": failures,
        "margins": margins,
        "oldclaim_summary": old_summary,
        "macro_summary": macro_summary,
        "raw_macro_summary": raw_macro_summary,
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# {args.label} Decision Audit")
    lines.append("")
    lines.append(f"- Generated: `{result['generated']}`")
    lines.append(f"- Decision: `{decision}`")
    lines.append(f"- Rationale: {rationale}")
    lines.append(f"- Recommended next layer: `{recommendation['next_layer']}`")
    lines.append(f"- Recommended next action: {recommendation['next_action']}")
    lines.append(f"- Pivot protocol report: `{recommendation['protocol_report']}`")
    lines.append(f"- Old-claim aggregate: `{rel(oldclaim_dir)}`")
    if macro_dir:
        lines.append(f"- Macro aggregate: `{rel(macro_dir)}`")
    if raw_macro_dir:
        lines.append(f"- Raw macro aggregate: `{rel(raw_macro_dir)}`")
    if args.expected_seeds is not None:
        lines.append(f"- Expected complete seeds: `{args.expected_seeds}`")
    lines.append("")
    lines.append("## Gate Counts")
    lines.append("")
    lines.append("| Gate | Count |")
    lines.append("|---|---:|")
    gate_labels = {
        "step_operational_gate_count": "Operational step",
        "macro_operational_gate_count": "Operational macro",
        "replay_step_gate_count": "Explicit replay step",
        "replay_macro_gate_count": "Explicit replay macro",
        "behavior_gate_count": "Behavior complexity",
        "true_static_macro_gate_count": "Replay-normalized true-static macro",
        "true_static_step_positive_count": "True-static step positive sign",
        "true_static_step_strict_gate_count": "Strict-margin true-static step",
    }
    for key, label in gate_labels.items():
        lines.append(f"| {label} | `{fmt_count(as_int(gates[key]), complete)}` |")
    lines.append("")
    lines.append("## Failure Seeds")
    lines.append("")
    for key, seeds in failures.items():
        lines.append(f"- `{key}`: `{markdown_list(seeds)}`")
    lines.append("")
    if margins:
        lines.append("## Margins")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key, value in margins.items():
            lines.append(f"| {key} | `{float(value):.6f}` |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if decision == "upgrade_allseed_strict":
        lines.append("The aggregate supports upgrading the claim with all pre-registered gates clean, including strict-margin true-static step.")
    elif decision == "upgrade_sign_bounded":
        lines.append(
            "The aggregate supports all-seed operational, replay, behavior, true-static macro, and true-static step sign wording. "
            "Do not claim universal strict-margin true-static step dominance."
        )
    elif decision == "incomplete_wait":
        lines.append("Do not update claim wording yet. Wait for missing seeds or artifacts.")
    else:
        lines.append(
            "Do not upgrade claim wording from this aggregate. Follow the recommended bounded next layer named above."
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(rel(out_json))
    print(rel(out_md))
    print(
        json.dumps(
            {
                "decision": decision,
                "rationale": rationale,
                "recommendation": recommendation,
                "gates": gates,
                "failures": failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
