#!/usr/bin/env python3
"""Aggregate the V358 receding-oracle structural diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def macro_from_candidate(row: pd.Series, normalizers: dict[str, float]) -> float:
    values = []
    for subtype in ("particle", "flux", "thermal"):
        values.append(
            float(row[f"oracle_loss_subtype_{subtype}"])
            / float(normalizers[f"oracle_loss_subtype_{subtype}"])
        )
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-glob", required=True)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for policy_dir in sorted(Path().glob(args.policy_glob)):
        seed = int(policy_dir.name.split("seed", 1)[1].split("_", 1)[0])
        control_dir = args.control_root / (
            f"v357_confirmation_scene_control_seed{seed}_b1p75_20260822"
        )
        metrics = pd.read_csv(policy_dir / "v2_custom_ppo_metrics.csv")
        custom = metrics.loc[metrics["policy"].eq("custom_ppo")].iloc[0]
        static = metrics.loc[
            metrics["policy"].isin(["validation_selected_static", "feasible_static_projected"])
        ]
        dynamic = metrics.loc[metrics["policy"].isin(["aoi", "round_robin", "random"])]
        oracle_dir = control_dir / "v358_receding_oracle"
        summary = json.loads((oracle_dir / "oracle_lift_summary.json").read_text())
        candidate = pd.read_csv(oracle_dir / "oracle_lift_candidate_table.csv")
        normalizers = json.loads(
            (policy_dir / "reward_staticnorm_normalizers.json").read_text()
        )["normalizers"]
        oracle = candidate.loc[candidate["sensor_ids"].eq("dynamic:receding_oracle_l8")].iloc[0]
        oracle_macro = macro_from_candidate(oracle, normalizers)
        rows.append(
            {
                "seed": seed,
                "pdppo_ordinary": float(custom["oracle_loss_mean"]),
                "pdppo_macro": float(custom["oracle_loss_macro_subtype_event_staticnorm"]),
                "best_static_ordinary": float(static["oracle_loss_mean"].min()),
                "best_static_macro": float(static["oracle_loss_macro_subtype_event_staticnorm"].min()),
                "best_dynamic_ordinary": float(dynamic["oracle_loss_mean"].min()),
                "best_dynamic_macro": float(dynamic["oracle_loss_macro_subtype_event_staticnorm"].min()),
                "oracle_ordinary": float(oracle["oracle_loss_mean"]),
                "oracle_macro": oracle_macro,
                "oracle_minus_pdppo_ordinary": float(oracle["oracle_loss_mean"] - custom["oracle_loss_mean"]),
                "oracle_minus_pdppo_macro": float(oracle_macro - custom["oracle_loss_macro_subtype_event_staticnorm"]),
                "oracle_minus_static_ordinary": float(oracle["oracle_loss_mean"] - static["oracle_loss_mean"].min()),
                "oracle_minus_static_macro": float(oracle_macro - static["oracle_loss_macro_subtype_event_staticnorm"].min()),
                "oracle_action_coverage": float(oracle["receding_action_coverage"]),
                "oracle_switches_per_step": float(oracle["switches_per_step"]),
                "oracle_power_mean": float(oracle["power_mean"]),
                "oracle_peak_max": float(oracle["peak_max"]),
                "oracle_warmup_abort_count": int(oracle["warmup_abort_count"]),
                "scene_best_overall_check": float(summary["best_overall"]["oracle_loss_mean"]),
            }
        )

    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    families = []
    for family in ("pdppo", "best_static", "best_dynamic", "oracle"):
        families.append(
            {
                "family": family,
                "seed_count": len(frame),
                "mean_ordinary": float(frame[f"{family}_ordinary"].mean()),
                "mean_macro": float(frame[f"{family}_macro"].mean()),
            }
        )
    pd.DataFrame(families).to_csv(args.out_dir / "family_means.csv", index=False)
    lines = [
        "# V358 receding-oracle diagnostic",
        "",
        f"Four independent V357 scenes were replayed with an 8-step receding oracle over the same feasible action geometry.",
        "Lower forecast loss is better. The oracle is a privileged structural diagnostic and is not a deployable policy.",
        "",
        "| Seed | PD-PPO | Best static | Receding oracle | Oracle - PD-PPO | Oracle - static | Oracle action coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.pdppo_ordinary:.6f} | {row.best_static_ordinary:.6f} | "
            f"{row.oracle_ordinary:.6f} | {row.oracle_minus_pdppo_ordinary:+.6f} | "
            f"{row.oracle_minus_static_ordinary:+.6f} | {row.oracle_action_coverage:.0f} |"
        )
    lines.extend(
        [
            "",
            f"Mean ordinary loss: PD-PPO {frame.pdppo_ordinary.mean():.6f}; "
            f"best static {frame.best_static_ordinary.mean():.6f}; "
            f"receding oracle {frame.oracle_ordinary.mean():.6f}.",
            f"Oracle beats best static on ordinary loss in "
            f"{int((frame.oracle_minus_static_ordinary < 0).sum())}/{len(frame)} scenes "
            f"and beats PD-PPO in {int((frame.oracle_minus_pdppo_ordinary < 0).sum())}/{len(frame)} scenes.",
            "The diagnostic establishes available dynamic value in these scenes; it does not establish that PD-PPO has learned it.",
        ]
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
