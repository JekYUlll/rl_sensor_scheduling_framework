#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_entry(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Entry must be LABEL=PATH, got: {text}")
    label, path = text.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Entry label is empty: {text}")
    return label, Path(path.strip())


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def row_from_decision(label: str, path: Path, data: dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "decision": "missing",
        }
    gates = data.get("gates") or {}
    old = data.get("oldclaim_summary") or {}
    macro = data.get("macro_summary") or {}
    raw = data.get("raw_macro_summary") or {}
    failures = data.get("failures") or {}
    complete = as_int(gates.get("complete_seeds") or old.get("complete_seeds") or data.get("expected_seeds"))
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "decision": data.get("decision"),
        "complete_seeds": complete,
        "operational_step_gate": as_int(gates.get("step_operational_gate_count")),
        "operational_macro_gate": as_int(gates.get("macro_operational_gate_count")),
        "replay_step_gate": as_int(gates.get("replay_step_gate_count")),
        "replay_macro_gate": as_int(gates.get("replay_macro_gate_count")),
        "behavior_gate": as_int(gates.get("behavior_gate_count")),
        "true_static_macro_gate": as_int(gates.get("true_static_macro_gate_count")),
        "true_static_step_strict_gate": as_int(gates.get("true_static_step_strict_gate_count")),
        "true_static_step_positive_gate": as_int(gates.get("true_static_step_positive_count")),
        "mean_step_margin_vs_operational": as_float(old.get("mean_step_margin_vs_best_operational_baseline")),
        "median_step_margin_vs_operational": as_float(old.get("median_step_margin_vs_best_operational_baseline")),
        "mean_macro_margin_vs_operational": as_float(old.get("mean_macro_margin_vs_best_operational_baseline")),
        "mean_staticnorm_macro_margin_vs_true_static": as_float(
            macro.get("mean_learned_macro_margin_abs_vs_macro_static_reference")
        ),
        "raw_macro_gate": as_int(raw.get("learned_macro_gate_count")),
        "raw_macro_mean_margin_vs_true_static": as_float(
            raw.get("mean_learned_macro_margin_abs_vs_macro_static_reference")
        ),
        "true_static_step_failures": ",".join(str(x) for x in failures.get("true_static_step_strict_fail_seeds", [])),
        "behavior_failures": ",".join(str(x) for x in failures.get("behavior_fail_seeds", [])),
        "all_failures_json": json.dumps(failures, ensure_ascii=False, sort_keys=True),
    }


def gate_text(row: pd.Series, key: str) -> str:
    value = row.get(key)
    n = row.get("complete_seeds")
    if pd.isna(value) or pd.isna(n):
        return "--"
    return f"{int(value)}/{int(n)}"


def write_markdown(df: pd.DataFrame, *, label: str, out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# {label}")
    lines.append("")
    lines.append("## Robustness Gate Summary")
    lines.append("")
    if df.empty:
        lines.append("No robustness entries were available.")
    else:
        lines.append(
            "| Perturbation | Decision | Complete | Operational step | True-static macro | Strict step | Behaviour | Mean step margin | Mean macro margin | Macro margin vs true-static |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in df.iterrows():
            lines.append(
                "| {label} | {decision} | {complete} | {op_step} | {ts_macro} | {strict_step} | {behavior} | {step:.4f} | {macro:.4f} | {tsm:.4f} |".format(
                    label=row.get("label"),
                    decision=row.get("decision"),
                    complete="--" if pd.isna(row.get("complete_seeds")) else int(row.get("complete_seeds")),
                    op_step=gate_text(row, "operational_step_gate"),
                    ts_macro=gate_text(row, "true_static_macro_gate"),
                    strict_step=gate_text(row, "true_static_step_strict_gate"),
                    behavior=gate_text(row, "behavior_gate"),
                    step=as_float(row.get("mean_step_margin_vs_operational")),
                    macro=as_float(row.get("mean_macro_margin_vs_operational")),
                    tsm=as_float(row.get("mean_staticnorm_macro_margin_vs_true_static")),
                )
            )
    lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("")
    lines.append("- These rows are robustness support, not a replacement for the main 24-seed claim.")
    lines.append("- A perturbation is useful support when at least 5/6 seeds pass operational, true-static macro, and behaviour gates.")
    lines.append("- The strict true-static step gate is reported separately because it is the strongest fixed-static shortcut check.")
    lines.append("- The raw unnormalised macro diagnostic remains a metric-boundary check, not the headline criterion.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final-benchmark robustness perturbation decision audits.")
    parser.add_argument("--label", default="Final-Benchmark Robustness Pilot")
    parser.add_argument("--entries", nargs="+", required=True, type=parse_entry)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row_from_decision(label, path, read_json(path)) for label, path in args.entries]
    df = pd.DataFrame(rows)

    csv_path = out_dir / "robustness_summary.csv"
    json_path = out_dir / "robustness_summary.json"
    md_path = out_dir / "robustness_summary.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    write_markdown(df, label=str(args.label), out_path=md_path)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
