#!/usr/bin/env python3
"""Aggregate the V352 cycling-scene structural readiness screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STATIC = ("validation_selected_static", "feasible_static_projected")
ORDINARY = "oracle_loss_mean"
MACRO = "oracle_loss_macro_subtype_event_staticnorm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in args.run_dir:
        metrics = pd.read_csv(run_dir / "v2_custom_ppo_metrics.csv")
        metadata = json.loads((run_dir / "dataset_validation" / "synthetic_metadata.json").read_text())
        static = metrics[metrics.policy.isin(STATIC)]
        static_best = static.iloc[static[MACRO].argmin()]
        receding = pd.read_csv(
            run_dir / "receding_oracle_l6_scene_gate" / "oracle_lift_candidate_table.csv"
        )
        receding = receding[receding["sensor_ids"].astype(str).str.contains("receding_oracle")].iloc[0]
        receding_macro = sum(
            float(receding[f"oracle_loss_subtype_{name}"])
            / (float(static_best[f"oracle_loss_subtype_{name}"])
               / float(static_best[f"oracle_loss_subtype_{name}_staticnorm"]))
            for name in ("particle", "flux", "thermal")
        ) / 3.0
        rows.append({
            "seed": int(metadata["seed"]),
            "event_coverage_actual": float(metadata["blowing_snow_event_coverage_actual"]),
            "event_cluster_count": int(metadata["event_cluster_count"]),
            "particle_rate": float(metadata["event_subtype_particle_rate"]),
            "flux_rate": float(metadata["event_subtype_flux_rate"]),
            "thermal_rate": float(metadata["event_subtype_thermal_rate"]),
            "event_subtype_assignment": str(metadata["event_subtype_assignment"]),
            "event_subtype_cycle_steps": int(metadata["event_subtype_cycle_steps"]),
            "static_best_ordinary": float(static[ORDINARY].min()),
            "static_best_macro": float(static[MACRO].min()),
            "receding_ordinary": float(receding[ORDINARY]),
            "receding_macro": float(receding_macro),
            "receding_minus_static_ordinary": float(static[ORDINARY].min() - receding[ORDINARY]),
            "receding_minus_static_macro": float(static[MACRO].min() - receding_macro),
            "receding_action_coverage": int(receding["receding_action_coverage"]),
            "receding_always_on": int(receding["always_on_sensor_count"]),
            "receding_always_off": int(receding["always_off_sensor_count"]),
            "receding_mid_duty": int(receding["mid_duty_sensor_count"]),
            "receding_switches_per_step": float(receding["switches_per_step"]),
            "receding_warmup_abort": int(receding["warmup_abort_count"]),
        })
    table = pd.DataFrame(rows).sort_values("seed")
    table.to_csv(args.out_dir / "scene_control_seed_metrics.csv", index=False)
    ordinary = table["receding_minus_static_ordinary"]
    macro = table["receding_minus_static_macro"]
    lines = [
        "# V352 cycling-scene control",
        "",
        "This is a pre-training structural screen. The receding reference is privileged and is not a deployable policy.",
        "",
        "- Scene change: event subtype assignment `cycling`, cycle length 12 samples; all other V338 physical and protocol settings are unchanged.",
        f"- Receding versus best static: ordinary {int((ordinary > 0).sum())}/{len(table)} wins, mean margin {ordinary.mean():+.6f}; macro {int((macro > 0).sum())}/{len(table)} wins, mean margin {macro.mean():+.6f}.",
        "- Training decision: authorize PPO only if both seeds show positive oracle margins and all behavior fields are valid.",
        "",
        table.to_string(index=False),
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
