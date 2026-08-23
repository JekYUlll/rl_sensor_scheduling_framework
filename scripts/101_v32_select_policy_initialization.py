#!/usr/bin/env python
"""Select one policy initialization per scene using frozen validation scores."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", action="append", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for pattern in args.run_glob:
        for run_str in sorted(glob.glob(pattern)):
            run = Path(run_str)
            metadata = json.loads((run / "v2_ppo_metadata.json").read_text())
            checkpoint = dict(metadata["checkpoint_selection"])
            metrics = pd.read_csv(run / "v2_custom_ppo_metrics.csv")
            custom = metrics[metrics["policy"] == "custom_ppo"].iloc[0]
            static = metrics[metrics["policy"] == "validation_selected_static"].iloc[0]
            dynamic = metrics[metrics["policy"].isin(["round_robin", "aoi", "random"])].sort_values(
                "oracle_loss_mean"
            ).iloc[0]
            rows.append({
                "seed": int(metadata["seed"]),
                "policy_seed": int(metadata["policy_seed"]),
                "run_dir": str(run),
                "selected_update": int(checkpoint["selected_update"]),
                "validation_score": float(checkpoint["selected_score"]),
                "pdppo_loss": float(custom["oracle_loss_mean"]),
                "static_loss": float(static["oracle_loss_mean"]),
                "ordinary_margin_vs_static": float(static["oracle_loss_mean"] - custom["oracle_loss_mean"]),
                "pdppo_macro": float(custom["oracle_loss_macro_subtype_event_staticnorm"]),
                "static_macro": float(static["oracle_loss_macro_subtype_event_staticnorm"]),
                "macro_margin_vs_static": float(
                    static["oracle_loss_macro_subtype_event_staticnorm"]
                    - custom["oracle_loss_macro_subtype_event_staticnorm"]
                ),
                "ordinary_margin_vs_dynamic": float(dynamic["oracle_loss_mean"] - custom["oracle_loss_mean"]),
                "always_on": int(custom["always_on_sensor_count"]),
                "always_off": int(custom["always_off_sensor_count"]),
                "mid_duty": int(custom["mid_duty_sensor_count"]),
                "switches_per_step": float(custom["switches_per_step"]),
                "warmup_abort_count": int(custom["warmup_abort_count"]),
            })

    candidates = pd.DataFrame(rows).drop_duplicates(["seed", "policy_seed"])
    if candidates.empty:
        raise SystemExit("No complete policy initializations found")
    candidates = candidates.sort_values(["seed", "validation_score", "policy_seed"])
    selected = candidates.groupby("seed", as_index=False).first()
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "policy_initialization_candidates.csv", index=False)
    selected.to_csv(out / "selected_seed_metrics.csv", index=False)


if __name__ == "__main__":
    main()
