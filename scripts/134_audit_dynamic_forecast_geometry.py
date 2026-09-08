"""Filter complete-subset forecast losses by the V602 dynamic frontier.

This is a diagnostic replay. The forecast losses come from independently
evaluated fixed-mask rollouts; the resource trace only determines whether a
mask is feasible at the corresponding time index. It is not policy evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--resource-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--effective-budget", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    losses = pd.read_csv(args.losses)
    trace = pd.read_csv(args.resource_trace).set_index("time_idx")
    if "time_idx" not in losses:
        raise ValueError("loss artifact must preserve time_idx")
    resource_columns = {c.removeprefix("resource_effective_power_"): c for c in trace.columns if c.startswith("resource_effective_power_")}
    if not resource_columns:
        raise ValueError("resource trace has no effective power columns")
    records = []
    for _, row in losses.iterrows():
        time_idx = int(row.time_idx)
        if time_idx not in trace.index:
            continue
        selected = [x for x in str(row.selected_sensor_ids).split(";") if x and x != "cr1000xe_backbone"]
        cost = sum(float(trace.loc[time_idx, resource_columns[channel]]) for channel in selected)
        item = row.to_dict()
        item["dynamic_effective_cost"] = cost
        item["dynamic_feasible"] = bool(cost <= args.effective_budget)
        records.append(item)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("no loss rows matched resource trace time_idx")
    feasible = frame[frame.dynamic_feasible].copy()
    fixed = (
        frame.groupby(["candidate", "selected_sensor_ids"])
        .dynamic_feasible.all()
        .rename("feasible_all_rows")
        .reset_index()
    )
    fixed = fixed[fixed.feasible_all_rows]
    fixed_losses = (
        frame.merge(fixed[["candidate", "selected_sensor_ids"]], on=["candidate", "selected_sensor_ids"])
        .groupby(["candidate", "selected_sensor_ids"], as_index=False)
        .oracle_loss.mean()
        .sort_values("oracle_loss")
    )
    per_time = feasible.groupby("time_idx", as_index=False).oracle_loss.min()
    summary = {
        "seed": args.seed,
        "loss_rows": int(len(losses)),
        "matched_rows": int(len(frame)),
        "candidate_count": int(frame.candidate.nunique()),
        "dynamic_feasible_row_fraction": float(frame.dynamic_feasible.mean()),
        "dynamic_feasible_candidate_count": int(feasible.candidate.nunique()),
        "common_static_candidate_count": int(len(fixed)),
        "best_common_static_loss": float(fixed_losses.iloc[0].oracle_loss) if len(fixed_losses) else None,
        "dynamic_rowwise_best_loss": float(per_time.oracle_loss.mean()) if len(per_time) else None,
        "dynamic_opportunity_gap": (
            float(fixed_losses.iloc[0].oracle_loss - per_time.oracle_loss.mean())
            if len(fixed_losses) and len(per_time) else None
        ),
        "diagnostic_status": "fixed_mask_replay_filtered_by_dynamic_resource; not_executable_policy_evidence",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / f"dynamic_filtered_losses_seed{args.seed}.csv", index=False)
    fixed_losses.to_csv(args.output_dir / f"common_static_losses_seed{args.seed}.csv", index=False)
    (args.output_dir / f"dynamic_forecast_geometry_seed{args.seed}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
