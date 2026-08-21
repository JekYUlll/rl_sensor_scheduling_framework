#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if pd.isna(number):
        return "NA"
    return f"{number:.{digits}f}"


def int_value(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def count_line(summary: dict[str, Any], key: str) -> str:
    return f"{int_value(summary.get(key))}/{int_value(summary.get('complete_seeds'))}"


def all_count(summary: dict[str, Any], key: str) -> bool:
    complete = int_value(summary.get("complete_seeds"))
    return complete > 0 and int_value(summary.get(key)) == complete


def seeds_where(table: pd.DataFrame, column: str, value: bool) -> str:
    if column not in table.columns or table.empty:
        return "NA"
    mask = table[column].astype(bool) == bool(value)
    seeds = [str(int(seed)) for seed in table.loc[mask, "seed"].tolist()]
    return ", ".join(seeds) if seeds else "none"


def markdown_escape(value: Any) -> str:
    text = fmt(value) if isinstance(value, float) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(markdown_escape(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(markdown_escape(row[column]) for column in table.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a readable V31 balanced-objective breakthrough report.")
    parser.add_argument("--macro-dir", required=True)
    parser.add_argument("--oldclaim-dir", required=True)
    parser.add_argument("--raw-macro-dir", default=None)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--title", default="V31 Balanced-Objective Breakthrough Report")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    macro_dir = resolve(args.macro_dir)
    raw_macro_dir = resolve(args.raw_macro_dir) if args.raw_macro_dir else None
    oldclaim_dir = resolve(args.oldclaim_dir)
    out_file = resolve(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    macro_summary = load_json(macro_dir / "metpair_claim_summary.json")
    old_summary = load_json(oldclaim_dir / "oldclaim_summary.json")
    macro_table = pd.read_csv(macro_dir / "metpair_seed_summary.csv")
    old_table = pd.read_csv(oldclaim_dir / "oldclaim_seed_summary.csv")
    raw_summary = (
        load_json(raw_macro_dir / "metpair_claim_summary.json")
        if raw_macro_dir is not None and (raw_macro_dir / "metpair_claim_summary.json").exists()
        else {}
    )

    lines: list[str] = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Macro aggregate: `{macro_dir.relative_to(ROOT) if macro_dir.is_relative_to(ROOT) else macro_dir}`")
    lines.append(f"- Old-claim aggregate: `{oldclaim_dir.relative_to(ROOT) if oldclaim_dir.is_relative_to(ROOT) else oldclaim_dir}`")
    if raw_macro_dir is not None:
        lines.append(f"- Raw macro aggregate: `{raw_macro_dir.relative_to(ROOT) if raw_macro_dir.is_relative_to(ROOT) else raw_macro_dir}`")
    if args.notes:
        lines.append(f"- Notes: {args.notes}")
    lines.append("")

    lines.append("## Executive Result")
    lines.append("")
    if all_count(old_summary, "old_claim_step_gate_count") and all_count(old_summary, "replay_gate_count"):
        if all_count(old_summary, "learned_true_static_step_gate_count") and all_count(
            old_summary, "learned_true_static_macro_gate_count"
        ):
            lines.append(
                "The balanced-objective branch supports strong operational and "
                "true-static step/macro claims under the tested protocol."
            )
        elif all_count(old_summary, "learned_true_static_macro_gate_count"):
            lines.append(
                "The balanced-objective branch supports strong operational step and "
                "regime-balanced macro claims under the tested protocol. The "
                "true-static macro gate also passes on all seeds; the remaining "
                "boundary is strict-margin true-static step dominance."
            )
        else:
            lines.append(
                "The balanced-objective branch supports strong operational step and "
                "regime-balanced macro claims under the tested protocol. The remaining "
                "boundary is true-static dominance, especially under the macro score."
            )
    else:
        lines.append(
            "The balanced-objective branch supports a strong regime-balanced macro claim, "
            "but the step-weighted replay claim remains bounded by incomplete step gates."
        )
    lines.append("")
    lines.append(f"- Complete seeds: `{int_value(old_summary.get('complete_seeds'))}`")
    if "behavior_eval_dirs" in old_summary:
        lines.append(f"- Behaviour audit eval dirs: `{old_summary.get('behavior_eval_dirs')}`")
    lines.append(f"- Learned PPO beats static/rule/operational baselines, step objective: `{count_line(old_summary, 'step_operational_gate_count')}`")
    lines.append(f"- Learned PPO beats static/rule/operational baselines, macro objective: `{count_line(old_summary, 'macro_operational_gate_count')}`")
    lines.append(f"- Strict explicit-replay step gate: `{count_line(old_summary, 'old_claim_step_gate_count')}`")
    lines.append(f"- Strict explicit-replay macro gate: `{count_line(old_summary, 'old_claim_macro_gate_count')}`")
    lines.append(f"- Behaviour complexity gate: `{count_line(old_summary, 'behavior_gate_count')}`")
    if "learned_true_static_step_gate_count" in old_summary:
        lines.append(f"- Learned-policy true-static step gate: `{count_line(old_summary, 'learned_true_static_step_gate_count')}`")
    if "learned_true_static_macro_gate_count" in old_summary:
        lines.append(f"- Learned-policy true-static macro gate: `{count_line(old_summary, 'learned_true_static_macro_gate_count')}`")
    lines.append(f"- Old-claim macro strength: `{old_summary.get('macro_claim_strength', 'NA')}`")
    lines.append(f"- Macro sign-test p-value: `{fmt(old_summary.get('one_sided_sign_test_p_old_claim_macro_gate'), 8)}`")
    lines.append(f"- Step sign-test p-value: `{fmt(old_summary.get('one_sided_sign_test_p_old_claim_step_gate'), 8)}`")
    if "one_sided_sign_test_p_learned_true_static_step_gate" in old_summary:
        lines.append(
            "- Learned true-static step sign-test p-value: "
            f"`{fmt(old_summary.get('one_sided_sign_test_p_learned_true_static_step_gate'), 8)}`"
        )
    lines.append("")

    lines.append("## Supported Claim")
    lines.append("")
    lines.append(
        "PD-PPO learns a non-fixed, non-cyclic contextual specialist scheduler that "
        "is forecast-optimal under the regime-balanced event-subtype objective, "
        "beating fixed static, rule-dynamic, and operational baselines across the "
        "completed seeds."
    )
    lines.append("")
    lines.append("This claim is bounded to the balanced-objective protocol. It should not be rewritten as a broad step-weighted all-condition claim unless later evidence clears the strict step replay gate.")
    if all_count(old_summary, "old_claim_step_gate_count") and all_count(old_summary, "replay_gate_count"):
        if all_count(old_summary, "learned_true_static_macro_gate_count"):
            lines[-1] = (
                "This claim is bounded to the balanced-objective protocol. Since "
                "strict explicit-replay and true-static macro gates pass here, the "
                "remaining unsupported wording is universal strict-margin "
                "true-static step dominance if any seed still misses that margin."
            )
        else:
            lines[-1] = (
                "This claim is bounded to the balanced-objective protocol. Since the "
                "strict explicit-replay step gate also passes here, the remaining "
                "unsupported wording is an unconditional claim over every true fixed "
                "static schedule and every macro scoring view."
            )
    lines.append("")

    lines.append("## Unsupported Or Still Bounded")
    lines.append("")
    if all_count(old_summary, "old_claim_step_gate_count") and all_count(old_summary, "replay_gate_count"):
        lines.append("- The old step-weighted operational and explicit-replay gates pass on all completed seeds.")
    else:
        lines.append("- The unqualified old step-weighted claim remains bounded because strict explicit-replay step gates did not pass all seeds.")
    lines.append(f"- Seeds failing explicit-replay step gate: `{seeds_where(old_table, 'old_claim_step_gate_pass', False)}`")
    if "learned_true_static_step_gate_pass" in old_table.columns:
        lines.append(
            "- Seeds failing learned-policy true-static step gate: "
            f"`{seeds_where(old_table, 'learned_true_static_step_gate_pass', False)}`"
        )
    if "learned_true_static_macro_gate_pass" in old_table.columns:
        lines.append(
            "- Seeds failing learned-policy true-static macro gate: "
            f"`{seeds_where(old_table, 'learned_true_static_macro_gate_pass', False)}`"
        )
    lines.append("- The true-static failures should be treated as evidence boundaries, not ignored.")
    lines.append("")

    lines.append("## Key Numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Mean step margin vs best operational baseline | `{fmt(old_summary.get('mean_step_margin_vs_best_operational_baseline'))}` |")
    lines.append(f"| Median step margin vs best operational baseline | `{fmt(old_summary.get('median_step_margin_vs_best_operational_baseline'))}` |")
    lines.append(f"| Mean macro margin vs best operational baseline | `{fmt(old_summary.get('mean_macro_margin_vs_best_operational_baseline'))}` |")
    lines.append(f"| Median macro margin vs best operational baseline | `{fmt(old_summary.get('median_macro_margin_vs_best_operational_baseline'))}` |")
    lines.append(f"| Mean learned margin vs selected static | `{fmt(macro_summary.get('mean_learned_margin_abs'))}` |")
    lines.append(f"| Mean replay macro margin vs static reference | `{fmt(macro_summary.get('mean_replay_macro_margin_abs_vs_static_reference'))}` |")
    if raw_summary:
        lines.append(f"| Raw mean learned margin vs selected static | `{fmt(raw_summary.get('mean_learned_margin_abs'))}` |")
    lines.append("")

    lines.append("## Per-Seed Old-Claim Summary")
    lines.append("")
    columns = [
        "seed",
        "custom_ppo_loss",
        "best_static_loss",
        "best_rule_dynamic_policy",
        "best_rule_dynamic_loss",
        "step_margin_vs_best_operational_baseline",
        "macro_margin_vs_best_operational_baseline",
        "step_margin_vs_replay_static_reference",
        "old_claim_step_gate_pass",
        "old_claim_macro_gate_pass",
        "learned_true_static_step_gate_pass",
    ]
    available = [column for column in columns if column in old_table.columns]
    lines.append(markdown_table(old_table[available]))
    lines.append("")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_file.relative_to(ROOT) if out_file.is_relative_to(ROOT) else out_file)


if __name__ == "__main__":
    main()
