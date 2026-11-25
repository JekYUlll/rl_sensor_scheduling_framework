#!/usr/bin/env python3
"""Write a LaTeX summary table from a SCENEBAL decision audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pvalue(value: object) -> str:
    if value is None:
        return "--"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return "$p=0$"
    if v < 1e-3:
        mantissa, exponent = f"{v:.2e}".split("e")
        return rf"$p={mantissa}\times10^{{{int(exponent)}}}$"
    return rf"$p={v:.4f}$"


def row(label: str, value: str) -> str:
    return rf"    {label} & {value} \\"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-json", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    data = json.loads(args.decision_json.read_text(encoding="utf-8"))
    gates = data.get("gates", {})
    margins = data.get("margins", {})
    old_summary = data.get("oldclaim_summary", {})
    macro_summary = data.get("macro_summary", {})
    n = int(gates.get("complete_seeds") or data.get("expected_seeds") or 0)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{args.caption}}}",
        rf"  \label{{{args.label}}}",
        r"  \small",
        r"  \begin{tabular}{@{}ll@{}}",
        r"    \toprule",
        r"    Quantity & Result \\",
        r"    \midrule",
        row("Complete seeds", str(n)),
        row("Operational step gate pass count", f"{gates.get('step_operational_gate_count', '--')}/{n}"),
        row("Operational macro gate pass count", f"{gates.get('macro_operational_gate_count', '--')}/{n}"),
        row("Explicit replay step gate pass count", f"{gates.get('replay_step_gate_count', '--')}/{n}"),
        row("Explicit replay macro gate pass count", f"{gates.get('replay_macro_gate_count', '--')}/{n}"),
        row("Behaviour-complexity gate pass count", f"{gates.get('behavior_gate_count', '--')}/{n}"),
        row("PD-PPO macro-score wins vs. fixed-mask replay", f"{gates.get('true_static_macro_gate_count', '--')}/{n}"),
        row("PD-PPO step-loss wins vs. fixed-mask replay", f"{gates.get('true_static_step_positive_count', '--')}/{n}"),
        row("Prespecified fixed-mask step-margin criterion", f"{gates.get('true_static_step_strict_gate_count', '--')}/{n}"),
        row(
            "Mean step margin vs. best operational baseline",
            fmt(margins.get("operational_step_margin_mean")),
        ),
        row(
            "Mean macro margin vs. best operational baseline",
            fmt(old_summary.get("mean_macro_margin_vs_best_operational_baseline")),
        ),
        row(
            "Mean learned macro margin vs. true-static reference",
            fmt(macro_summary.get("mean_learned_macro_margin_abs_vs_macro_static_reference")),
        ),
        row(
            "Mean explicit replay macro margin vs. true-static reference",
            fmt(macro_summary.get("mean_replay_macro_margin_abs_vs_static_reference")),
        ),
        row(
            "One-sided sign-test value for operational gates",
            pvalue(old_summary.get("one_sided_sign_test_p_old_claim_step_gate")),
        ),
        row(
            "One-sided sign-test value for true-static step strict margin",
            pvalue(old_summary.get("one_sided_sign_test_p_learned_true_static_step_gate")),
        ),
        row("Minimum strict true-static step margin", fmt(margins.get("true_static_step_margin_min"))),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
