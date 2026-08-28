#!/usr/bin/env python3
"""Aggregate a training-trace-only dynamic reference against fixed static masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=Path("reports"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--budget-label", default="b1p85")
    parser.add_argument("--suffix", default="20260822")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--policy", default="trace_distilled_forecast_value")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        run = args.report_root / f"{args.prefix}_seed{seed}_{args.budget_label}_{args.suffix}"
        static = pd.read_csv(run / "v2_custom_ppo_metrics.csv").set_index("policy").loc[
            "validation_selected_static"
        ]
        policy = pd.read_csv(
            args.baseline_root / f"seed{seed}" / "framework_baseline_metrics.csv"
        ).set_index("policy").loc[args.policy]
        rows.append({
            "seed": int(seed),
            "ordinary_margin_vs_static": float(static["oracle_loss_mean"] - policy["oracle_loss_mean"]),
            "macro_margin_vs_static": float(
                static["oracle_loss_macro_subtype_event_staticnorm"]
                - policy["oracle_loss_macro_subtype_event_staticnorm"]
            ),
            "switches_per_step": float(policy["switches_per_step"]),
            "always_on_sensor_count": int(policy["always_on_sensor_count"]),
            "always_off_sensor_count": int(policy["always_off_sensor_count"]),
            "mid_duty_sensor_count": int(policy["mid_duty_sensor_count"]),
            "warmup_abort_count": int(policy["warmup_abort_count"]),
        })
    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    summary = pd.DataFrame([{
        "seed_count": len(table),
        "ordinary_wins": int((table["ordinary_margin_vs_static"] > 0.0).sum()),
        "macro_wins": int((table["macro_margin_vs_static"] > 0.0).sum()),
        "mean_ordinary_margin": float(table["ordinary_margin_vs_static"].mean()),
        "mean_macro_margin": float(table["macro_margin_vs_static"].mean()),
        "behavior_passes": int(((table["always_on_sensor_count"] == 0) & (table["always_off_sensor_count"] == 0) & (table["mid_duty_sensor_count"] == 5) & (table["warmup_abort_count"] == 0)).sum()),
    }])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
