#!/usr/bin/env python3
"""Aggregate the label-free health-diagnostic and receding scene gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def staticnorm_macro(row: pd.Series, static: pd.Series) -> float:
    values: list[float] = []
    for subtype in ("particle", "flux", "thermal"):
        raw = f"oracle_loss_subtype_{subtype}"
        normalizer = float(static[raw]) / float(static[f"{raw}_staticnorm"])
        values.append(float(row[raw]) / normalizer)
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect V232 physical-group reliability scene-gate metrics."
    )
    parser.add_argument("--report-root", type=Path, default=Path("reports"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--budget-label", default="b1p85")
    parser.add_argument("--suffix", default="20260822")
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--quality-policy", required=True)
    parser.add_argument("--receding-subdir", default="receding_oracle_l8_scene_gate")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        run = args.report_root / f"{args.prefix}_seed{seed}_{args.budget_label}_{args.suffix}"
        metrics = pd.read_csv(run / "v2_custom_ppo_metrics.csv").set_index("policy")
        static = metrics.loc["validation_selected_static"]
        quality = pd.read_csv(
            args.context_root / f"seed{seed}" / "framework_baseline_metrics.csv"
        ).set_index("policy").loc[args.quality_policy]
        candidates = pd.read_csv(
            run / args.receding_subdir / "oracle_lift_candidate_table.csv"
        )
        receding = candidates[
            candidates["sensor_ids"].astype(str).str.contains("receding_oracle", regex=False)
        ].iloc[0]
        receding_macro = staticnorm_macro(receding, static)
        rows.append({
            "seed": int(seed),
            "quality_ordinary_margin_vs_static": float(static["oracle_loss_mean"] - quality["oracle_loss_mean"]),
            "quality_macro_margin_vs_static": float(
                static["oracle_loss_macro_subtype_event_staticnorm"]
                - quality["oracle_loss_macro_subtype_event_staticnorm"]
            ),
            "receding_ordinary_margin_vs_static": float(static["oracle_loss_mean"] - receding["oracle_loss_mean"]),
            "receding_macro_margin_vs_static": float(
                static["oracle_loss_macro_subtype_event_staticnorm"] - receding_macro
            ),
            "receding_action_coverage": int(receding["receding_action_coverage"]),
            "receding_always_on": int(receding["always_on_sensor_count"]),
            "receding_always_off": int(receding["always_off_sensor_count"]),
            "receding_mid_duty": int(receding["mid_duty_sensor_count"]),
            "receding_warmup_aborts": int(receding["warmup_abort_count"]),
        })

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    summary = pd.DataFrame([{
        "seed_count": len(table),
        "quality_ordinary_wins": int((table["quality_ordinary_margin_vs_static"] > 0.0).sum()),
        "quality_macro_wins": int((table["quality_macro_margin_vs_static"] > 0.0).sum()),
        "quality_mean_ordinary_margin": float(table["quality_ordinary_margin_vs_static"].mean()),
        "quality_mean_macro_margin": float(table["quality_macro_margin_vs_static"].mean()),
        "receding_ordinary_wins": int((table["receding_ordinary_margin_vs_static"] > 0.0).sum()),
        "receding_macro_wins": int((table["receding_macro_margin_vs_static"] > 0.0).sum()),
        "receding_mean_ordinary_margin": float(table["receding_ordinary_margin_vs_static"].mean()),
        "receding_mean_macro_margin": float(table["receding_macro_margin_vs_static"].mean()),
        "receding_behavior_passes": int(((table["receding_always_on"] == 0) & (table["receding_always_off"] == 0) & (table["receding_mid_duty"] == 5) & (table["receding_warmup_aborts"] == 0)).sum()),
    }])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
