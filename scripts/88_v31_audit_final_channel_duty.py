#!/usr/bin/env python
"""Audit final PD-PPO channel duty from saved held-out rollouts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SENSOR_IDS = (
    "met_station_core",
    "radiometer_basic",
    "shielded_thermo_hygro",
    "surface_temp_ir",
    "laser_disdrometer",
    "fc4_flux",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-metrics-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    seed_rows = pd.read_csv(args.seed_metrics_csv)
    rows: list[dict[str, object]] = []
    for item in seed_rows.itertuples(index=False):
        run_dir = Path(str(item.run_dir))
        rollout = run_dir / str(item.router_eval_dir) / "rollout_custom_ppo.npz"
        with np.load(rollout, allow_pickle=False) as data:
            masks = np.asarray(data["selected_masks"], dtype=float)
        if masks.ndim != 2 or masks.shape[1] != len(SENSOR_IDS):
            raise ValueError(f"Unexpected mask shape {masks.shape} in {rollout}")
        row: dict[str, object] = {"seed": int(item.seed), "run_dir": str(run_dir), "steps": int(masks.shape[0])}
        for idx, sensor_id in enumerate(SENSOR_IDS):
            row[f"duty__{sensor_id}"] = float(masks[:, idx].mean())
        rows.append(row)
    by_seed = pd.DataFrame(rows).sort_values("seed")
    summary_rows = []
    for sensor_id in SENSOR_IDS:
        values = by_seed[f"duty__{sensor_id}"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "sensor_id": sensor_id,
                "min_duty": float(values.min()),
                "max_duty": float(values.max()),
                "mean_duty": float(values.mean()),
                "always_on_seed_count": int(np.sum(values >= 1.0 - 1.0e-12)),
                "always_off_seed_count": int(np.sum(values <= 1.0e-12)),
                "mid_duty_seed_count": int(np.sum((values > 1.0e-12) & (values < 1.0 - 1.0e-12))),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(args.out_dir / "channel_duty_by_seed.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "channel_duty_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
