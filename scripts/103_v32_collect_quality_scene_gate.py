#!/usr/bin/env python3
"""Aggregate online and horizon-matched diagnostics for a quality scene."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUBTYPES = ("particle", "flux", "thermal")


def staticnorm_macro(row: pd.Series, static: pd.Series) -> float:
    values: list[float] = []
    for subtype in SUBTYPES:
        raw_column = f"oracle_loss_subtype_{subtype}"
        normalized_column = f"{raw_column}_staticnorm"
        normalizer = float(static[raw_column]) / float(static[normalized_column])
        values.append(float(row[raw_column]) / normalizer)
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default="reports", type=Path)
    parser.add_argument("--prefix", default="v152_channel_quality_scene_dev")
    parser.add_argument("--context-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1601, 1602, 1603, 1604, 1605])
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        run = args.report_root / f"{args.prefix}_seed{seed}_b1p75_20260822"
        metrics = pd.read_csv(run / "v2_custom_ppo_metrics.csv").set_index("policy")
        static = metrics.loc["validation_selected_static"]
        context = pd.read_csv(args.context_root / f"seed{seed}" / "framework_baseline_metrics.csv")
        greedy = context.set_index("policy").loc["forecast_greedy_one_step"]
        candidates = pd.read_csv(run / "receding_oracle_l8_scene_gate" / "oracle_lift_candidate_table.csv")
        receding = candidates[candidates["sensor_ids"].astype(str).str.contains("receding_oracle")].iloc[0]
        receding_macro = staticnorm_macro(receding, static)
        rows.append({
            "seed": seed,
            "static_ordinary": float(static["oracle_loss_mean"]),
            "static_macro": float(static["oracle_loss_macro_subtype_event_staticnorm"]),
            "greedy_ordinary": float(greedy["oracle_loss_mean"]),
            "greedy_macro": float(greedy["oracle_loss_macro_subtype_event_staticnorm"]),
            "greedy_ordinary_margin_vs_static": float(static["oracle_loss_mean"] - greedy["oracle_loss_mean"]),
            "greedy_macro_margin_vs_static": float(static["oracle_loss_macro_subtype_event_staticnorm"] - greedy["oracle_loss_macro_subtype_event_staticnorm"]),
            "receding_ordinary": float(receding["oracle_loss_mean"]),
            "receding_macro": receding_macro,
            "receding_ordinary_margin_vs_static": float(static["oracle_loss_mean"] - receding["oracle_loss_mean"]),
            "receding_macro_margin_vs_static": float(static["oracle_loss_macro_subtype_event_staticnorm"] - receding_macro),
            "receding_switches_per_step": float(receding["switches_per_step"]),
            "receding_always_on": int(receding["always_on_sensor_count"]),
            "receding_always_off": int(receding["always_off_sensor_count"]),
            "receding_mid_duty": int(receding["mid_duty_sensor_count"]),
            "receding_warmup_aborts": int(receding["warmup_abort_count"]),
            "receding_action_coverage": int(receding["receding_action_coverage"]),
        })

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    summary = pd.DataFrame([{
        "seed_count": len(table),
        "greedy_ordinary_wins": int((table["greedy_ordinary_margin_vs_static"] > 0).sum()),
        "greedy_macro_wins": int((table["greedy_macro_margin_vs_static"] > 0).sum()),
        "greedy_mean_ordinary_margin": float(table["greedy_ordinary_margin_vs_static"].mean()),
        "greedy_mean_macro_margin": float(table["greedy_macro_margin_vs_static"].mean()),
        "receding_ordinary_wins": int((table["receding_ordinary_margin_vs_static"] > 0).sum()),
        "receding_macro_wins": int((table["receding_macro_margin_vs_static"] > 0).sum()),
        "receding_mean_ordinary_margin": float(table["receding_ordinary_margin_vs_static"].mean()),
        "receding_mean_macro_margin": float(table["receding_macro_margin_vs_static"].mean()),
        "receding_behavior_passes": int(((table["receding_always_on"] == 0) & (table["receding_always_off"] == 0) & (table["receding_mid_duty"] == 6) & (table["receding_switches_per_step"] > 0) & (table["receding_warmup_aborts"] == 0)).sum()),
        "horizon_matched_headroom_gate": bool(
            int((table["receding_ordinary_margin_vs_static"] > 0).sum()) >= 4
            and int((table["receding_macro_margin_vs_static"] > 0).sum()) >= 4
        ),
        "myopic_headroom_gate": bool(
            int((table["greedy_ordinary_margin_vs_static"] > 0).sum()) >= 4
            and int((table["greedy_macro_margin_vs_static"] > 0).sum()) >= 4
        ),
    }])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
