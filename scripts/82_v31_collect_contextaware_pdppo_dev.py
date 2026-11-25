#!/usr/bin/env python
"""Collect feature-parity and CA-PD-PPO development-seed comparisons.

The input to this collector is the output of
``81_v31_framework_baseline_supplements.py`` with ``--router-eval-dir .``.
Margins are reported as ``baseline - PD-PPO`` so positive values mean that the
trained PD-PPO variant is better than the compared baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MACRO_COL = "oracle_loss_macro_subtype_event_staticnorm"
MACRO_MARGIN_COL = f"margin_{MACRO_COL}_vs_custom_ppo"
CONTEXT_POLICY = "context_alert_bandit_t0p5"
GREEDY_POLICY = "forecast_greedy_one_step"


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_array(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Variant must use name=out_root")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Variant name is empty")
    return name, Path(path)


def read_custom_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "v2_custom_ppo_metrics.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    rows = table[table["policy"].astype(str) == "custom_ppo"]
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    keep = (
        "oracle_loss_mean",
        MACRO_COL,
        "switches_per_step",
        "always_on_sensor_count",
        "always_off_sensor_count",
        "mid_duty_sensor_count",
        "warmup_abort_count",
        "power_mean",
        "peak_power_max",
        "event_rate",
    )
    return {f"custom_metric_{key}": finite_float(row.get(key)) for key in keep}


def best_family_margin(row: pd.Series, family: list[str], metric: str) -> float:
    custom_key = f"custom_ppo_{metric}"
    custom = finite_float(row.get(custom_key))
    values = [finite_float(row.get(f"{name}_{metric}")) for name in family]
    values = [value for value in values if np.isfinite(value)]
    if not values or not np.isfinite(custom):
        return float("nan")
    return float(min(values) - custom)


def load_variant(name: str, out_root: Path) -> pd.DataFrame:
    metrics_path = out_root / "framework_baseline_seed_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing framework seed metrics for {name}: {metrics_path}")
    data = pd.read_csv(metrics_path)
    rows: list[dict[str, Any]] = []
    for seed, group in data.groupby("seed", dropna=False):
        out: dict[str, Any] = {"variant": name, "seed": int(seed)}
        out["out_root"] = str(out_root)
        run_dir_text = str(group["run_dir"].iloc[0]) if "run_dir" in group.columns and not group.empty else ""
        out["run_dir"] = run_dir_text
        context = group[group["policy"].astype(str) == CONTEXT_POLICY]
        greedy = group[group["policy"].astype(str) == GREEDY_POLICY]
        if not context.empty:
            crow = context.iloc[0]
            out["step_margin_vs_context_bandit"] = finite_float(crow.get("margin_loss_vs_custom_ppo"))
            out["macro_margin_vs_context_bandit"] = finite_float(crow.get(MACRO_MARGIN_COL))
            out["context_bandit_loss"] = finite_float(crow.get("oracle_loss_mean"))
            out["context_bandit_macro"] = finite_float(crow.get(MACRO_COL))
            out["best_static_step_margin"] = best_family_margin(
                crow,
                ["validation_selected_static", "feasible_static_projected"],
                "oracle_loss_mean",
            )
            out["best_static_macro_margin"] = best_family_margin(
                crow,
                ["validation_selected_static", "feasible_static_projected"],
                MACRO_COL,
            )
            out["best_original_dynamic_step_margin"] = best_family_margin(
                crow,
                ["round_robin", "aoi", "random"],
                "oracle_loss_mean",
            )
            out["best_original_dynamic_macro_margin"] = best_family_margin(
                crow,
                ["round_robin", "aoi", "random"],
                MACRO_COL,
            )
            out["custom_ppo_loss"] = finite_float(crow.get("custom_ppo_oracle_loss_mean"))
            out["custom_ppo_macro"] = finite_float(crow.get(f"custom_ppo_{MACRO_COL}"))
        if not greedy.empty:
            grow = greedy.iloc[0]
            out["step_margin_vs_forecast_greedy"] = finite_float(grow.get("margin_loss_vs_custom_ppo"))
            out["macro_margin_vs_forecast_greedy"] = finite_float(grow.get(MACRO_MARGIN_COL))
            out["forecast_greedy_loss"] = finite_float(grow.get("oracle_loss_mean"))
            out["forecast_greedy_macro"] = finite_float(grow.get(MACRO_COL))
        if run_dir_text:
            out.update(read_custom_metrics(Path(run_dir_text)))
        rows.append(out)
    return pd.DataFrame(rows)


def summarise_variant(rows: pd.DataFrame, *, draws: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"variant": str(rows["variant"].iloc[0]), "seed_count": int(len(rows))}
    margin_specs = {
        "context_step": "step_margin_vs_context_bandit",
        "context_macro": "macro_margin_vs_context_bandit",
        "forecast_greedy_step": "step_margin_vs_forecast_greedy",
        "forecast_greedy_macro": "macro_margin_vs_forecast_greedy",
        "best_static_step": "best_static_step_margin",
        "best_static_macro": "best_static_macro_margin",
        "best_original_dynamic_step": "best_original_dynamic_step_margin",
        "best_original_dynamic_macro": "best_original_dynamic_macro_margin",
    }
    for label, column in margin_specs.items():
        values = finite_array(rows[column]) if column in rows.columns else np.asarray([], dtype=float)
        lo, hi = bootstrap_ci(values, draws=draws, seed=seed + len(label))
        out[f"{label}_n"] = int(values.size)
        out[f"{label}_wins"] = int(np.sum(values > 0.0)) if values.size else 0
        out[f"{label}_mean"] = float(np.mean(values)) if values.size else float("nan")
        out[f"{label}_median"] = float(np.median(values)) if values.size else float("nan")
        out[f"{label}_ci_low"] = lo
        out[f"{label}_ci_high"] = hi
    for column in (
        "custom_metric_switches_per_step",
        "custom_metric_always_on_sensor_count",
        "custom_metric_always_off_sensor_count",
        "custom_metric_mid_duty_sensor_count",
        "custom_metric_warmup_abort_count",
    ):
        values = finite_array(rows[column]) if column in rows.columns else np.asarray([], dtype=float)
        key = column.replace("custom_metric_", "behavior_")
        out[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
        out[f"{key}_max"] = float(np.max(values)) if values.size else float("nan")
    return out


def decision_for(summary: pd.DataFrame) -> str:
    ca = summary[summary["variant"].astype(str).str.lower().isin({"ca_pdppo", "ca-pdppo", "ca"})]
    if ca.empty:
        return "no_ca_pdppo_variant_found"
    row = ca.iloc[0]
    mean_macro = finite_float(row.get("context_macro_mean"))
    wins = int(row.get("context_macro_wins", 0))
    n = int(row.get("context_macro_n", 0))
    if np.isfinite(mean_macro) and mean_macro > 0.0 and n >= 24 and wins >= 15:
        return "advance_to_fresh_final_24_seed_evaluation"
    if np.isfinite(mean_macro) and mean_macro > 0.0 and wins >= max(1, n // 2):
        return "competitive_positive_mean_final_gate_not_passed"
    if np.isfinite(mean_macro) and abs(mean_macro) <= 0.002 and wins >= max(1, n // 2):
        return "competitive_with_context_aware_bandit"
    return "context_alert_bandit_remains_stronger"


def dataframe_to_markdown(frame: pd.DataFrame, *, float_digits: int = 6) -> str:
    if frame.empty:
        return ""

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.{float_digits}f}"
        return str(value)

    headers = [str(c) for c in frame.columns]
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def write_report(seed_rows: pd.DataFrame, summary: pd.DataFrame, decision: str, out_dir: Path) -> None:
    display_cols = [
        "variant",
        "seed_count",
        "context_macro_wins",
        "context_macro_mean",
        "context_macro_ci_low",
        "context_macro_ci_high",
        "context_step_wins",
        "context_step_mean",
        "forecast_greedy_macro_wins",
        "forecast_greedy_macro_mean",
        "best_static_macro_wins",
        "best_static_macro_mean",
        "best_original_dynamic_macro_wins",
        "best_original_dynamic_macro_mean",
        "behavior_always_on_sensor_count_max",
        "behavior_always_off_sensor_count_max",
        "behavior_mid_duty_sensor_count_mean",
    ]
    lines = [
        "# Context-Aware PD-PPO Development Summary",
        "",
        "Margins are `baseline - PD-PPO`; positive values mean the PD-PPO variant is better.",
        f"Decision: `{decision}`.",
        "",
        dataframe_to_markdown(summary[[c for c in display_cols if c in summary.columns]], float_digits=6),
        "",
        "Seed-level rows:",
        f"- `{out_dir / 'contextaware_pdppo_dev_seed_metrics.csv'}`",
    ]
    (out_dir / "contextaware_pdppo_dev_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CA-PD-PPO development comparisons.")
    parser.add_argument("--variant", action="append", type=parse_variant, required=True, help="name=framework_baseline_out_root")
    parser.add_argument("--out-dir", default="reports/aggregate/contextaware_pdppo_dev_20260703")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260703)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_variant(name, path) for name, path in args.variant]
    seed_rows = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(
        [summarise_variant(group, draws=int(args.bootstrap_draws), seed=int(args.bootstrap_seed)) for _, group in seed_rows.groupby("variant")]
    ).sort_values("variant")
    decision = decision_for(summary)

    seed_rows.to_csv(out_dir / "contextaware_pdppo_dev_seed_metrics.csv", index=False)
    summary.to_csv(out_dir / "contextaware_pdppo_dev_summary.csv", index=False)
    payload = {
        "decision": decision,
        "variants": summary.to_dict(orient="records"),
    }
    (out_dir / "contextaware_pdppo_dev_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(seed_rows, summary, decision, out_dir)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
