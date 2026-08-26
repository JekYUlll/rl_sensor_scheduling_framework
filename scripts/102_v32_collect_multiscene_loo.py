#!/usr/bin/env python
"""Aggregate frozen leave-one-scene-out PD-PPO evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def behavior_valid(row: pd.Series) -> bool:
    return bool(
        int(row["always_on_sensor_count"]) == 0
        and int(row["always_off_sensor_count"]) <= 1
        and float(row["switches_per_step"]) > 0.0
        and int(row["warmup_abort_count"]) == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default="reports")
    parser.add_argument("--prefix", default="v151_loo")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1501, 1502, 1503, 1504, 1505])
    args = parser.parse_args()

    root = Path(args.report_root)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        run = root / f"{args.prefix}_holdout{seed}_seed{seed}_b1p75_20260822"
        metrics = pd.read_csv(run / "v2_custom_ppo_metrics.csv")
        by_policy = metrics.set_index("policy")
        custom = by_policy.loc["custom_ppo"]
        static = by_policy.loc["validation_selected_static"]
        dynamics = metrics[metrics["policy"].isin(["round_robin", "aoi", "random"])]
        dynamic = dynamics.sort_values("oracle_loss_mean").iloc[0]
        train_seed = next(candidate for candidate in args.seeds if candidate != seed)
        train = root / f"{args.prefix}_train_holdout{seed}_seed{train_seed}_b1p75_20260822"
        metadata = json.loads((train / "v2_ppo_metadata.json").read_text())
        selection = dict(metadata["checkpoint_selection"])
        ordinary_margin = float(static["oracle_loss_mean"] - custom["oracle_loss_mean"])
        macro_margin = float(
            static["oracle_loss_macro_subtype_event_staticnorm"]
            - custom["oracle_loss_macro_subtype_event_staticnorm"]
        )
        dynamic_margin = float(dynamic["oracle_loss_mean"] - custom["oracle_loss_mean"])
        behavior = behavior_valid(custom)
        rows.append({
            "heldout_seed": seed,
            "train_seed": train_seed,
            "selected_update": int(selection["selected_update"]),
            "validation_score": float(selection["selected_score"]),
            "validation_behavior_failures": int(selection["selected_behavior_failure_count"]),
            "pdppo_ordinary": float(custom["oracle_loss_mean"]),
            "static_ordinary": float(static["oracle_loss_mean"]),
            "ordinary_margin_vs_static": ordinary_margin,
            "pdppo_macro": float(custom["oracle_loss_macro_subtype_event_staticnorm"]),
            "static_macro": float(static["oracle_loss_macro_subtype_event_staticnorm"]),
            "macro_margin_vs_static": macro_margin,
            "best_dynamic_policy": str(dynamic["policy"]),
            "ordinary_margin_vs_dynamic": dynamic_margin,
            "always_on": int(custom["always_on_sensor_count"]),
            "always_off": int(custom["always_off_sensor_count"]),
            "mid_duty": int(custom["mid_duty_sensor_count"]),
            "switches_per_step": float(custom["switches_per_step"]),
            "warmup_abort_count": int(custom["warmup_abort_count"]),
            "behavior_gate_pass": behavior,
            "joint_gate_pass": bool(
                ordinary_margin > 0.0 and macro_margin > 0.0 and dynamic_margin > 0.0 and behavior
            ),
        })

    table = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "seed_metrics.csv", index=False)
    summary = pd.DataFrame([{
        "seed_count": len(table),
        "ordinary_static_wins": int((table["ordinary_margin_vs_static"] > 0).sum()),
        "macro_static_wins": int((table["macro_margin_vs_static"] > 0).sum()),
        "dynamic_wins": int((table["ordinary_margin_vs_dynamic"] > 0).sum()),
        "behavior_passes": int(table["behavior_gate_pass"].sum()),
        "joint_gate_passes": int(table["joint_gate_pass"].sum()),
        "mean_ordinary_margin_vs_static": float(table["ordinary_margin_vs_static"].mean()),
        "mean_macro_margin_vs_static": float(table["macro_margin_vs_static"].mean()),
        "mean_ordinary_margin_vs_dynamic": float(table["ordinary_margin_vs_dynamic"].mean()),
        "continue_to_fresh_confirmation": bool(int(table["joint_gate_pass"].sum()) >= 4),
    }])
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
