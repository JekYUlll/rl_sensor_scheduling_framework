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


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def row_from_decision(label: str, path: Path, data: dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "decision": "missing",
        }
    gates = data.get("gates") or {}
    margins = data.get("margins") or {}
    old = data.get("oldclaim_summary") or {}
    macro = data.get("macro_summary") or {}
    raw = data.get("raw_macro_summary") or {}
    n = as_int(gates.get("complete_seeds") or data.get("expected_seeds") or old.get("complete_seeds"))
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "decision": data.get("decision"),
        "expected_seeds": as_int(data.get("expected_seeds")),
        "complete_seeds": n,
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
        "mean_true_static_step_margin": as_float(margins.get("true_static_step_margin_mean")),
        "min_true_static_step_margin": as_float(margins.get("true_static_step_margin_min")),
        "mean_staticnorm_macro_margin_vs_true_static": as_float(
            macro.get("mean_learned_macro_margin_abs_vs_macro_static_reference")
        ),
        "raw_macro_gate": as_int(raw.get("learned_macro_gate_count")),
        "raw_macro_mean_margin_vs_true_static": as_float(
            raw.get("mean_learned_macro_margin_abs_vs_macro_static_reference")
        ),
        "failures": json.dumps(data.get("failures") or {}, ensure_ascii=False, sort_keys=True),
    }


def add_reference_deltas(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    if df.empty or reference not in set(df["label"].astype(str)):
        return df
    ref = df[df["label"].astype(str) == reference].iloc[0]
    out = df.copy()
    for col in (
        "mean_step_margin_vs_operational",
        "mean_macro_margin_vs_operational",
        "mean_true_static_step_margin",
        "mean_staticnorm_macro_margin_vs_true_static",
    ):
        ref_value = as_float(ref.get(col))
        out[f"delta_{col}_vs_reference"] = out[col].astype(float) - ref_value
        if math.isfinite(ref_value) and abs(ref_value) > 1.0e-12:
            out[f"ratio_{col}_vs_reference"] = out[col].astype(float) / ref_value
        else:
            out[f"ratio_{col}_vs_reference"] = float("nan")
    return out


def gate_text(row: pd.Series, key: str) -> str:
    value = row.get(key)
    n = row.get("complete_seeds")
    if pd.isna(value) or pd.isna(n):
        return "--"
    return f"{int(value)}/{int(n)}"


def write_markdown(df: pd.DataFrame, *, label: str, reference: str, out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# {label}")
    lines.append("")
    lines.append(f"Reference variant: `{reference}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if df.empty:
        lines.append("No entries were available.")
    else:
        lines.append(
            "| Variant | Decision | Complete | Operational step | True-static macro | Strict step | Behaviour | Mean step margin | Mean macro margin | Macro margin vs true-static |"
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
    lines.append("- A mechanism ablation is useful when it clearly degrades gates or margins relative to the full reference.")
    lines.append("- The raw unnormalised macro diagnostic remains a boundary check, not the headline claim.")
    lines.append("- Missing rows mean the remote run or collect phase has not completed yet.")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final-benchmark mechanism ablation decision audits.")
    parser.add_argument("--label", default="Final-Benchmark Mechanism Ablation")
    parser.add_argument("--reference", default="full_reference")
    parser.add_argument("--entries", nargs="+", required=True, type=parse_entry)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row_from_decision(label, path, read_json(path)) for label, path in args.entries]
    df = pd.DataFrame(rows)
    df = add_reference_deltas(df, str(args.reference))

    csv_path = out_dir / "mechanism_ablation_summary.csv"
    json_path = out_dir / "mechanism_ablation_summary.json"
    md_path = out_dir / "mechanism_ablation_summary.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    write_markdown(df, label=str(args.label), reference=str(args.reference), out_path=md_path)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
