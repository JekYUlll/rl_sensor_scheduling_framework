#!/usr/bin/env python3
"""Audit forecast-loss crossover among all feasible subset candidates.

This is a read-only diagnostic over frozen geometry artifacts.  It does not
select a policy, retrain an oracle, or use PPO outputs.  Relative regret is
reported because the unclipped oracle losses can differ substantially in
scale across seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit(root: Path, output: Path) -> None:
    rows: list[dict] = []
    details: list[dict] = []
    for csv_path in sorted(root.glob("seed*/subset_condition_losses_seed*.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        seed = int(frame["seed"].iloc[0])
        candidate_info = frame[["candidate", "selected_sensor_ids", "steady_cost"]].drop_duplicates()
        conditions = sorted(str(x) for x in frame["operating_condition"].dropna().unique())
        means = (
            frame.groupby(["candidate", "selected_sensor_ids", "steady_cost", "operating_condition"], as_index=False)
            .oracle_loss.mean()
        )
        winners: dict[str, str] = {}
        near_sets: dict[str, dict[str, list[str]]] = {}
        for condition in conditions:
            part = means[means.operating_condition == condition].copy()
            best = float(part.oracle_loss.min())
            part["relative_regret"] = part.oracle_loss / max(best, 1.0e-12) - 1.0
            winners[condition] = str(part.sort_values("oracle_loss").iloc[0].candidate)
            near_sets[condition] = {
                "1pct": sorted(str(x) for x in part.loc[part.relative_regret <= 0.01, "candidate"]),
                "5pct": sorted(str(x) for x in part.loc[part.relative_regret <= 0.05, "candidate"]),
            }
            for _, row in part.iterrows():
                rows.append({
                    "seed": seed,
                    "condition": condition,
                    "candidate": str(row.candidate),
                    "selected_sensor_ids": str(row.selected_sensor_ids),
                    "steady_cost": float(row.steady_cost),
                    "mean_oracle_loss": float(row.oracle_loss),
                    "relative_regret": float(row.relative_regret),
                })
        static = (
            frame.groupby(["candidate", "selected_sensor_ids", "steady_cost"], as_index=False)
            .oracle_loss.mean()
            .sort_values("oracle_loss")
        )
        intersection_1 = set.intersection(*[
            set(near_sets[c]["1pct"]) for c in conditions
        ]) if conditions else set()
        intersection_5 = set.intersection(*[
            set(near_sets[c]["5pct"]) for c in conditions
        ]) if conditions else set()
        details.append({
            "seed": seed,
            "candidate_count": int(candidate_info.shape[0]),
            "conditions": conditions,
            "condition_best_candidates": winners,
            "best_static_candidate": str(static.iloc[0].candidate),
            "best_static_sensors": str(static.iloc[0].selected_sensor_ids),
            "best_static_cost": float(static.iloc[0].steady_cost),
            "best_static_loss": float(static.iloc[0].oracle_loss),
            "conditionwise_1pct_intersection": sorted(intersection_1),
            "conditionwise_5pct_intersection": sorted(intersection_5),
        })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "subset_forecast_crossover_long.csv", index=False)
    (output / "subset_forecast_crossover_summary.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Subset forecast crossover audit", "", "| Seed | Best static | Cost | Condition winners | 1% intersection |", "|---:|---|---:|---|---|"]
    for item in details:
        lines.append(
            f"| {item['seed']} | `{item['best_static_candidate']}` "
            f"({item['best_static_sensors']}) | {item['best_static_cost']:.4f} | "
            f"{item['condition_best_candidates']} | {item['conditionwise_1pct_intersection']} |"
        )
    (output / "subset_forecast_crossover_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.geometry_root, args.output)


if __name__ == "__main__":
    main()
