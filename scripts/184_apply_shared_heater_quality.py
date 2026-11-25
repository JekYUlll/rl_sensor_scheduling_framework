#!/usr/bin/env python3
"""Apply the declared heater-quality relation to a shared-driver scene.

This is a truth-only transformation.  Resource heater demand and heater state
are generated independently by the shared-driver controller; the relation
below maps the same operating risk to observation quality.  It does not use
future targets, event labels, or policy outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUALITY_LEVELS = {
    "met_station_core": ("resource_heater_demand_met_station_core", "resource_heater_on_met_station_core", 0.35),
    "laser_disdrometer": ("resource_heater_demand_laser_disdrometer", "resource_heater_on_laser_disdrometer", 0.25),
    "radiometer_basic": ("resource_shared_exposure_load", None, 0.75),
    "surface_temp_ir": ("resource_shared_exposure_load", None, 0.80),
    "fc4_flux": ("resource_shared_exposure_load", None, 0.95),
    "cr1000xe_backbone": (None, None, 1.0),
}


def build(truth: pd.DataFrame, resource: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not truth["time_idx"].reset_index(drop=True).equals(resource["time_idx"].reset_index(drop=True)):
        raise ValueError("truth and resource rows must have identical time_idx order")
    out = truth.copy()
    risk_threshold = 0.35
    for channel, (demand_col, heater_col, unheated_quality) in QUALITY_LEVELS.items():
        if channel == "cr1000xe_backbone":
            out[f"agent_context_quality_{channel}"] = 1.0
            continue
        demand = resource[demand_col].to_numpy(float)
        risk = demand >= risk_threshold
        quality = np.where(risk, unheated_quality, 1.0)
        if heater_col is not None:
            quality = np.where(resource[heater_col].to_numpy(bool), 1.0, quality)
        out[f"agent_context_quality_{channel}"] = quality
        out[f"generator_heater_quality_risk_{channel}"] = risk.astype(np.int8)
    return out, {
        "generator": Path(__file__).name,
        "relation": "heater_quality_relation_v1 semantics",
        "risk_threshold": risk_threshold,
        "future_targets_used": False,
        "event_labels_used": False,
        "quality_convention": "higher score means lower observation noise",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out, metadata = build(pd.read_csv(args.truth), pd.read_csv(args.resource))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(out),
        "quality_means": {c: float(out[f"agent_context_quality_{c}"].mean()) for c in QUALITY_LEVELS},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
