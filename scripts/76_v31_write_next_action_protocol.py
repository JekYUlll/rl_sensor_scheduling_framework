#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def as_list(values: Any) -> list[Any]:
    return values if isinstance(values, list) else []


def md_code(value: Any) -> str:
    return f"`{value}`"


def fmt_seed_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def gate_rows(gates: dict[str, Any]) -> list[str]:
    labels = [
        ("step_operational_gate_count", "Operational step"),
        ("macro_operational_gate_count", "Operational macro"),
        ("replay_step_gate_count", "Explicit replay step"),
        ("replay_macro_gate_count", "Explicit replay macro"),
        ("behavior_gate_count", "Behavior complexity"),
        ("true_static_macro_gate_count", "Replay-normalized true-static macro"),
        ("true_static_step_positive_count", "True-static step positive sign"),
        ("true_static_step_strict_gate_count", "Strict-margin true-static step"),
    ]
    total = int(gates.get("complete_seeds", 0) or 0)
    rows = ["| Gate | Count |", "|---|---:|"]
    for key, label in labels:
        count = int(gates.get(key, 0) or 0)
        rows.append(f"| {label} | `{count}/{total}` |")
    return rows


def bounded_unit(decision: str, failures: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    if decision == "upgrade_allseed_strict":
        return (
            "Claim Update",
            [
                "Update the manuscript/report claim only after verifying the 24-seed aggregate files and artifacts are present.",
                "Do not change experiment settings from this decision alone.",
            ],
            [
                "Rendered paper/report contains the upgraded all-gate wording.",
                "Aggregate JSON/CSV and decision audit remain archived.",
            ],
        )
    if decision == "upgrade_sign_bounded":
        return (
            "Sign-Bounded Claim Update",
            [
                "Use sign-bounded strong wording and explicitly preserve the strict-margin true-static step boundary.",
                "Do not launch another same-configuration seed wave unless it answers a new, concrete uncertainty.",
                "If universal strict-margin step dominance is still required, pivot to simulator balance or reward/oracle calibration instead of seed expansion.",
            ],
            [
                "Claim wording names operational, replay, behavior, true-static macro, and true-static sign gates.",
                "Claim wording does not assert universal strict-margin true-static step dominance.",
            ],
        )
    if decision in {"pivot_true_static_step_sign_failure", "pivot_replay_headroom_failure"}:
        failure_seeds = as_list(failures.get("true_static_step_sign_fail_seeds")) + as_list(
            failures.get("replay_step_fail_seeds")
        ) + as_list(failures.get("replay_macro_fail_seeds"))
        return (
            "SCENEBAL-2 Simulator/Data Balance Pilot",
            [
                f"Use failing seeds `{fmt_seed_list(failure_seeds)}` plus at least one clean control seed.",
                "Preserve PPO/PD-PPO as the final learned scheduler.",
                "Preserve the met-backbone plus one-specialist sensing premise.",
                "Modify simulator/data balance before PPO coefficient tuning.",
                "Run structural replay/short PPO only as a bounded pilot before any seed expansion.",
            ],
            [
                "Operational step and macro pass on all pilot seeds.",
                "Explicit replay step and macro pass on all pilot seeds.",
                "Behavior audit passes on all pilot seeds.",
                "True-static step sign is positive on all pilot seeds.",
            ],
        )
    if decision == "pivot_behavior_failure":
        return (
            "PPO-REGIME-2 Observation/Auxiliary Pilot",
            [
                f"Use behavior-failing seeds `{fmt_seed_list(as_list(failures.get('behavior_fail_seeds')))}` plus a clean control seed.",
                "Keep the SCENEBAL-1 simulator and sensor geometry fixed initially.",
                "Improve deployable regime belief through observation features, memory/lead context, or auxiliary representation heads.",
                "Avoid high-weight direct action CE/margin losses unless diagnostics show pure imitation failure.",
            ],
            [
                "Behavior audit passes with nontrivial unique mask count.",
                "Event/mask mutual information remains positive.",
                "No seed is fixed-like or simple-cycle-like.",
                "Operational step and macro improve on the failure seeds.",
            ],
        )
    if decision == "diagnose_operational_failure":
        return (
            "Reward/Evaluation Or PPO Credit Diagnostic",
            [
                f"Inspect operational-failing seeds `{fmt_seed_list(as_list(failures.get('step_operational_fail_seeds')) + as_list(failures.get('macro_operational_fail_seeds')))}`.",
                "Check whether the winning rule baseline depends on unrealistic switching or an evaluation mismatch.",
                "If replay headroom remains clean, diagnose reward/oracle credit assignment before changing the simulator.",
            ],
            [
                "Failure is classified as protocol mismatch, PPO-credit failure, or true scenario weakness.",
                "A written keep/pivot decision exists before any training wave is launched.",
            ],
        )
    if decision == "hold_true_static_macro_boundary":
        return (
            "EVAL-REWARD-2 Narrow Protocol Audit",
            [
                f"Inspect macro-failing seeds `{fmt_seed_list(as_list(failures.get('true_static_macro_fail_seeds')))}`.",
                "Recompute only the failing aggregate with an auditable collector/window/scale change if a mismatch is demonstrated.",
                "Do not drop losing seeds or redefine baselines after seeing the failure.",
            ],
            [
                "Before/after tables identify the exact protocol mismatch if one exists.",
                "If no protocol bug exists, claim wording remains bounded.",
            ],
        )
    if decision == "incomplete_wait":
        return (
            "Wait Or Artifact Repair",
            [
                "Do not update claim wording.",
                "Wait for missing seeds or repair missing artifacts named by the watcher.",
                "Rerun the decision audit after artifacts are complete.",
            ],
            [
                "All expected seeds are complete.",
                "Oracle, PPO, evaluation, replay, and behavior artifacts exist for every expected seed.",
            ],
        )
    return (
        "Manual Bounded Review",
        [
            "Review the decision audit manually.",
            "Choose a bounded next unit before launching any new experiment.",
        ],
        ["A written protocol exists before launch."],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the next action protocol from a SCENEBAL-1 decision audit.")
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--label", default="SCENEBAL-1 next action")
    args = parser.parse_args()

    decision_path = resolve(args.decision_json)
    out_path = resolve(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audit = load_json(decision_path)
    decision = str(audit.get("decision", "unknown"))
    recommendation = audit.get("recommendation", {}) if isinstance(audit.get("recommendation"), dict) else {}
    failures = audit.get("failures", {}) if isinstance(audit.get("failures"), dict) else {}
    gates = audit.get("gates", {}) if isinstance(audit.get("gates"), dict) else {}
    unit_name, steps, criteria = bounded_unit(decision, failures)

    lines: list[str] = []
    lines.append(f"# {args.label} Protocol")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Decision audit: `{rel(decision_path)}`")
    lines.append(f"- Decision: `{decision}`")
    lines.append(f"- Recommended next layer: `{recommendation.get('next_layer', 'unknown')}`")
    lines.append(f"- Recommended next action: {recommendation.get('next_action', 'not provided')}")
    lines.append(f"- Pivot protocol report: `{recommendation.get('protocol_report', 'not provided')}`")
    lines.append(f"- Materialized bounded unit: `{unit_name}`")
    lines.append("")
    lines.append("## Gate Snapshot")
    lines.append("")
    lines.extend(gate_rows(gates))
    lines.append("")
    lines.append("## Failure Seeds")
    lines.append("")
    for key in sorted(failures):
        lines.append(f"- `{key}`: `{fmt_seed_list(as_list(failures.get(key)))}`")
    lines.append("")
    lines.append("## Immediate Steps")
    lines.append("")
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("")
    for idx, criterion in enumerate(criteria, start=1):
        lines.append(f"{idx}. {criterion}")
    lines.append("")
    lines.append("## Hard Constraints")
    lines.append("")
    lines.append("- PPO/PD-PPO remains the final learned scheduler.")
    lines.append("- The accepted behavior must not be fixed-like or simple-cycle-like.")
    lines.append("- The met-backbone plus one-specialist sensing setup remains the baseline.")
    lines.append("- Moderate sensor/noise changes are allowed only as explainable simulated variants.")
    lines.append("- Do not run more than ten bounded no-improvement units in the same modification direction.")
    lines.append("- Every launched pivot unit must produce aggregate evidence and a written keep/pivot decision.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(rel(out_path))
    print(json.dumps({"decision": decision, "bounded_unit": unit_name}, indent=2))


if __name__ == "__main__":
    main()
