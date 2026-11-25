#!/usr/bin/env python3
"""Aggregate quality-aware PD-PPO runs and online quality behavior."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def quality_behavior(run: Path, truth_run: Path) -> dict[str, float]:
    rollout = np.load(run / "rollout_custom_ppo.npz", allow_pickle=True)
    truth = pd.read_csv(truth_run / "truth_v31_split.csv")
    step_indices = rollout["step_indices"].astype(int)
    selected = rollout["selected_masks"].astype(bool)
    sensor_ids = rollout["sensor_ids"].astype(str)
    quality = np.column_stack([
        truth.loc[step_indices, f"agent_context_quality_{sensor_id}"].to_numpy(float)
        for sensor_id in sensor_ids
    ])
    high_low_gaps: list[float] = []
    for sensor_idx in range(selected.shape[1]):
        high = quality[:, sensor_idx] >= 0.8
        low = quality[:, sensor_idx] <= 0.4
        if high.any() and low.any():
            high_low_gaps.append(float(selected[high, sensor_idx].mean() - selected[low, sensor_idx].mean()))
    return {
        "selected_quality_mean": float(quality[selected].mean()),
        "unselected_quality_mean": float(quality[~selected].mean()),
        "selected_unselected_quality_gap": float(quality[selected].mean() - quality[~selected].mean()),
        "high_low_quality_duty_gap": float(np.mean(high_low_gaps)) if high_low_gaps else float("nan"),
        "unique_action_count": int(len(np.unique(selected, axis=0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default="reports", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--scene-prefix", default="v152_channel_quality_scene_dev")
    parser.add_argument("--context-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        run = args.report_root / f"{args.prefix}_seed{seed}_b1p75_20260822"
        scene = args.report_root / f"{args.scene_prefix}_seed{seed}_b1p75_20260822"
        metrics = pd.read_csv(run / "v2_custom_ppo_metrics.csv").set_index("policy")
        custom = metrics.loc["custom_ppo"]
        static = metrics.loc["validation_selected_static"]
        dynamic = metrics.loc[["aoi", "round_robin", "random"]].sort_values("oracle_loss_mean").iloc[0]
        context = pd.read_csv(args.context_root / f"seed{seed}" / "framework_baseline_metrics.csv").set_index("policy")
        greedy = context.loc["forecast_greedy_one_step"]
        behavior = quality_behavior(run, scene)
        rows.append({
            "seed": seed,
            "pdppo_ordinary": float(custom["oracle_loss_mean"]),
            "pdppo_macro": float(custom["oracle_loss_macro_subtype_event_staticnorm"]),
            "static_ordinary_margin": float(static["oracle_loss_mean"] - custom["oracle_loss_mean"]),
            "static_macro_margin": float(static["oracle_loss_macro_subtype_event_staticnorm"] - custom["oracle_loss_macro_subtype_event_staticnorm"]),
            "best_dynamic_policy": str(dynamic.name),
            "dynamic_ordinary_margin": float(dynamic["oracle_loss_mean"] - custom["oracle_loss_mean"]),
            "greedy_ordinary_margin": float(greedy["oracle_loss_mean"] - custom["oracle_loss_mean"]),
            "greedy_macro_margin": float(greedy["oracle_loss_macro_subtype_event_staticnorm"] - custom["oracle_loss_macro_subtype_event_staticnorm"]),
            "always_on": int(custom["always_on_sensor_count"]),
            "always_off": int(custom["always_off_sensor_count"]),
            "mid_duty": int(custom["mid_duty_sensor_count"]),
            "switches_per_step": float(custom["switches_per_step"]),
            "warmup_abort_count": int(custom["warmup_abort_count"]),
            **behavior,
        })

    table = pd.DataFrame(rows)
    table["behavior_gate"] = (
        (table["always_on"] == 0)
        & (table["always_off"] <= 1)
        & (table["switches_per_step"] > 0)
        & (table["warmup_abort_count"] == 0)
        & (table["unique_action_count"] > 1)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    summary = pd.DataFrame([{
        "seed_count": len(table),
        "static_ordinary_wins": int((table["static_ordinary_margin"] > 0).sum()),
        "static_macro_wins": int((table["static_macro_margin"] > 0).sum()),
        "dynamic_ordinary_wins": int((table["dynamic_ordinary_margin"] > 0).sum()),
        "greedy_ordinary_wins": int((table["greedy_ordinary_margin"] > 0).sum()),
        "greedy_macro_wins": int((table["greedy_macro_margin"] > 0).sum()),
        "behavior_passes": int(table["behavior_gate"].sum()),
        "mean_static_ordinary_margin": float(table["static_ordinary_margin"].mean()),
        "mean_static_macro_margin": float(table["static_macro_margin"].mean()),
        "mean_dynamic_ordinary_margin": float(table["dynamic_ordinary_margin"].mean()),
        "mean_greedy_ordinary_margin": float(table["greedy_ordinary_margin"].mean()),
        "mean_greedy_macro_margin": float(table["greedy_macro_margin"].mean()),
        "mean_selected_unselected_quality_gap": float(table["selected_unselected_quality_gap"].mean()),
        "mean_high_low_quality_duty_gap": float(table["high_low_quality_duty_gap"].mean()),
    }])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
