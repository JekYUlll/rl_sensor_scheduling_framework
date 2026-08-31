#!/usr/bin/env python3
"""Aggregate V339 per-seed metrics without post-hoc seed exclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ORDINARY = "oracle_loss_mean"
MACRO = "oracle_loss_macro_subtype_event_staticnorm"
STATIC = ("validation_selected_static", "feasible_static_projected")
DYNAMIC = ("aoi", "round_robin", "random")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in args.run_dir:
        metrics = pd.read_csv(run_dir / "v2_custom_ppo_metrics.csv")
        custom = metrics.loc[metrics.policy == "custom_ppo"].iloc[0]
        static = metrics.loc[metrics.policy.isin(STATIC)]
        dynamic = metrics.loc[metrics.policy.isin(DYNAMIC)]
        metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text())
        rows.append(
            {
                "seed": int(metadata.get("seed", run_dir.name.split("seed", 1)[1].split("_", 1)[0])),
                "policy_seed": int(metadata.get("policy_seed", -1)),
                "selected_checkpoint_update": metadata.get("selected_checkpoint_update"),
                "pdppo_ordinary": float(custom[ORDINARY]),
                "pdppo_macro": float(custom[MACRO]),
                "static_ordinary_margin": float(static[ORDINARY].min() - custom[ORDINARY]),
                "static_macro_margin": float(static[MACRO].min() - custom[MACRO]),
                "dynamic_ordinary_margin": float(dynamic[ORDINARY].min() - custom[ORDINARY]),
                "dynamic_macro_margin": float(dynamic[MACRO].min() - custom[MACRO]),
                "full_open_ordinary_margin": float(
                    metrics.loc[metrics.policy == "full_open_unconstrained", ORDINARY].iloc[0]
                    - custom[ORDINARY]
                ),
                "full_open_macro_margin": float(
                    metrics.loc[metrics.policy == "full_open_unconstrained", MACRO].iloc[0]
                    - custom[MACRO]
                ),
                "warmup_abort_count": int(custom["warmup_abort_count"]),
                "always_on_sensor_count": int(custom["always_on_sensor_count"]),
                "always_off_sensor_count": int(custom["always_off_sensor_count"]),
                "mid_duty_sensor_count": int(custom["mid_duty_sensor_count"]),
                "switches_per_step": float(custom["switches_per_step"]),
                "power_mean": float(custom["power_mean"]),
                "peak_power_max": float(custom["peak_power_max"]),
            }
        )
    table = pd.DataFrame(rows).sort_values("seed")
    table.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    families = ("static", "dynamic", "full_open")
    summary = []
    for family in families:
        ordinary = table[f"{family}_ordinary_margin"]
        macro = table[f"{family}_macro_margin"]
        summary.append(
            {
                "baseline_family": family,
                "seed_count": len(table),
                "ordinary_wins": int((ordinary > 0).sum()),
                "macro_wins": int((macro > 0).sum()),
                "ordinary_mean_margin": float(ordinary.mean()),
                "macro_mean_margin": float(macro.mean()),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_dir / "family_summary.csv", index=False)
    lines = [
        "# V339 recalibrated-scene PD-PPO development aggregate",
        "",
        "All completed seeds are included. Positive margins mean lower PD-PPO loss.",
        "",
        summary_df.to_string(index=False),
        "",
        f"Behavior rows with zero warm-up aborts and zero constant channels: "
        f"{int(((table.warmup_abort_count == 0) & (table.always_on_sensor_count == 0) & (table.always_off_sensor_count == 0)).sum())}/{len(table)}.",
        "",
        table.to_string(index=False),
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
