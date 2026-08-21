#!/usr/bin/env python
"""Collect bounded CA-PD-PPO development-wave variants."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTEXT_POLICY = "context_alert_bandit_t0p5"
GREEDY_POLICY = "forecast_greedy_one_step"
MACRO_COL = "oracle_loss_macro_subtype_event_staticnorm"
MACRO_MARGIN_COL = f"margin_{MACRO_COL}_vs_custom_ppo"


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
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Variant must use name=out_root")
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def read_custom_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "v2_custom_ppo_metrics.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    rows = table[table["policy"].astype(str) == "custom_ppo"]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    keys = [
        "switches_per_step",
        "warmup_abort_count",
        "always_on_sensor_count",
        "always_off_sensor_count",
        "mid_duty_sensor_count",
    ]
    return {key: finite_float(row.get(key)) for key in keys}


def best_family_margin(row: pd.Series, family: list[str], metric: str) -> float:
    custom = finite_float(row.get(f"custom_ppo_{metric}"))
    baselines = [finite_float(row.get(f"{name}_{metric}")) for name in family]
    baselines = [v for v in baselines if np.isfinite(v)]
    if not baselines or not np.isfinite(custom):
        return float("nan")
    return float(min(baselines) - custom)


def load_variant(name: str, root: Path) -> pd.DataFrame:
    table = pd.read_csv(root / "framework_baseline_seed_metrics.csv")
    rows: list[dict[str, Any]] = []
    for seed, group in table.groupby("seed"):
        out: dict[str, Any] = {"variant": name, "seed": int(seed), "out_root": str(root)}
        run_dir = Path(str(group["run_dir"].iloc[0]))
        out["run_dir"] = str(run_dir)
        context = group[group["policy"].astype(str) == CONTEXT_POLICY]
        greedy = group[group["policy"].astype(str) == GREEDY_POLICY]
        if not context.empty:
            crow = context.iloc[0]
            out["context_step_margin"] = finite_float(crow.get("margin_loss_vs_custom_ppo"))
            out["context_macro_margin"] = finite_float(crow.get(MACRO_MARGIN_COL))
            out["best_static_macro_margin"] = best_family_margin(
                crow,
                ["validation_selected_static", "feasible_static_projected"],
                MACRO_COL,
            )
            out["best_original_dynamic_macro_margin"] = best_family_margin(
                crow,
                ["round_robin", "aoi", "random"],
                MACRO_COL,
            )
        if not greedy.empty:
            grow = greedy.iloc[0]
            out["forecast_greedy_macro_margin"] = finite_float(grow.get(MACRO_MARGIN_COL))
        metrics = read_custom_metrics(run_dir if run_dir.is_absolute() else Path(".") / run_dir)
        out.update(metrics)
        rows.append(out)
    return pd.DataFrame(rows)


def summarise(rows: pd.DataFrame, *, bootstrap_draws: int, seed: int, switch_limit: float) -> dict[str, Any]:
    out: dict[str, Any] = {"variant": str(rows["variant"].iloc[0]), "seed_count": int(len(rows))}
    for label, col in [
        ("context_macro", "context_macro_margin"),
        ("context_step", "context_step_margin"),
        ("forecast_greedy_macro", "forecast_greedy_macro_margin"),
        ("best_static_macro", "best_static_macro_margin"),
        ("best_original_dynamic_macro", "best_original_dynamic_macro_margin"),
    ]:
        values = finite_array(rows[col]) if col in rows.columns else np.asarray([], dtype=float)
        lo, hi = bootstrap_ci(values, draws=int(bootstrap_draws), seed=int(seed) + len(label))
        out[f"{label}_n"] = int(values.size)
        out[f"{label}_wins"] = int(np.sum(values > 0.0)) if values.size else 0
        out[f"{label}_mean"] = float(values.mean()) if values.size else float("nan")
        out[f"{label}_ci_low"] = lo
        out[f"{label}_ci_high"] = hi
    for col in [
        "switches_per_step",
        "warmup_abort_count",
        "always_on_sensor_count",
        "always_off_sensor_count",
        "mid_duty_sensor_count",
    ]:
        values = finite_array(rows[col]) if col in rows.columns else np.asarray([], dtype=float)
        out[f"{col}_mean"] = float(values.mean()) if values.size else float("nan")
        out[f"{col}_max"] = float(values.max()) if values.size else float("nan")
    out["passes_dev2_gate"] = bool(
        finite_float(out.get("context_macro_mean")) > 0.010
        and int(out.get("context_macro_wins", 0)) >= 15
        and finite_float(out.get("context_macro_ci_low")) >= 0.0
        and int(out.get("forecast_greedy_macro_wins", 0)) >= 23
        and int(out.get("best_static_macro_wins", 0)) == int(out.get("seed_count", 0))
        and finite_float(out.get("warmup_abort_count_max")) == 0.0
        and finite_float(out.get("switches_per_step_mean")) <= float(switch_limit)
    )
    return out


def md_table(frame: pd.DataFrame, *, digits: int = 6) -> str:
    if frame.empty:
        return ""
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)
    headers = [str(c) for c in frame.columns]
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines += ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--switch-limit", type=float, default=0.00662)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_tables = [load_variant(name, path) for name, path in args.variant]
    seed_rows = pd.concat(seed_tables, ignore_index=True)
    summary = pd.DataFrame(
        [
            summarise(group, bootstrap_draws=args.bootstrap_draws, seed=args.bootstrap_seed, switch_limit=args.switch_limit)
            for _, group in seed_rows.groupby("variant", sort=False)
        ]
    )
    summary = summary.sort_values(["passes_dev2_gate", "context_macro_mean"], ascending=[False, False])
    seed_rows.to_csv(args.out_dir / "variant_seed_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "variant_summary.csv", index=False)
    (args.out_dir / "variant_summary.json").write_text(
        json.dumps(
            {
                "switch_limit": float(args.switch_limit),
                "passing_variants": summary[summary["passes_dev2_gate"]]["variant"].astype(str).tolist(),
                "best_variant": str(summary.iloc[0]["variant"]) if not summary.empty else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    display = [
        "variant",
        "seed_count",
        "passes_dev2_gate",
        "context_macro_wins",
        "context_macro_mean",
        "context_macro_ci_low",
        "context_macro_ci_high",
        "context_step_wins",
        "context_step_mean",
        "forecast_greedy_macro_wins",
        "best_static_macro_wins",
        "switches_per_step_mean",
        "warmup_abort_count_max",
    ]
    lines = [
        "# CA-PD-PPO Bounded Dev2 Variant Summary",
        "",
        "Margins are `baseline - PD-PPO`; positive values mean the PD-PPO variant is better.",
        "Gate: context macro mean > 0.010, context macro wins >= 15/24, CI lower >= 0, forecast-greedy macro wins >= 23/24, static macro wins 24/24, no warmup aborts, and switches <= configured limit.",
        "",
        md_table(summary[[c for c in display if c in summary.columns]], digits=6),
        "",
        f"Seed-level rows: `{args.out_dir / 'variant_seed_metrics.csv'}`",
        "",
    ]
    (args.out_dir / "variant_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
