"""Derive deployment-observable heater/resource traces from existing truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def dew_point_c(temp: pd.Series, rh: pd.Series) -> pd.Series:
    rh_safe = rh.clip(lower=1.0, upper=100.0)
    gamma = np.log(rh_safe / 100.0) + (17.62 * temp / (243.12 + temp))
    return 243.12 * gamma / (17.62 - gamma)


def hysteresis(on_condition: pd.Series, off_condition: pd.Series) -> pd.Series:
    state = False
    result: list[bool] = []
    for on, off in zip(on_condition.fillna(False), off_condition.fillna(False)):
        if not state and bool(on):
            state = True
        elif state and bool(off):
            state = False
        result.append(state)
    return pd.Series(result, index=on_condition.index, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = pd.read_csv(args.input)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    required = {"air_temperature_c", "relative_humidity", "snow_surface_temperature_c", "solar_radiation_wm2"}
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"missing runtime-observable truth columns: {missing}")
    out = truth.copy()
    dp = dew_point_c(out["air_temperature_c"], out["relative_humidity"])
    margin = out["air_temperature_c"] - dp
    out["resource_dew_point_c"] = dp
    out["resource_dew_point_margin_c"] = margin
    gmx_on = (out["air_temperature_c"] <= -25.0) | ((margin <= 1.0) & (out["air_temperature_c"] <= -5.0))
    gmx_off = (out["air_temperature_c"] >= -23.0) & (margin >= 2.0)
    parsivel_on = (out["air_temperature_c"] <= -20.0) | ((margin <= 1.0) & (out["snow_surface_temperature_c"] <= -5.0))
    parsivel_off = (out["air_temperature_c"] >= -18.0) & (margin >= 2.0)
    states = {
        "met_station_core": hysteresis(gmx_on, gmx_off),
        "laser_disdrometer": hysteresis(parsivel_on, parsivel_off),
    }
    rows = {row["channel_id"]: row for row in manifest["channels"]}
    for channel, state in states.items():
        row = rows[channel]
        out[f"resource_heater_on_{channel}"] = state.astype(int)
        out[f"resource_power_w_{channel}"] = float(row["base_power_w"]) + state.astype(float) * float(row["heater_increment_w"])
        out[f"resource_effective_power_{channel}"] = float(row["effective_base"]) + state.astype(float) * float(row["heater_increment_effective"])
    for channel, row in rows.items():
        if channel not in states:
            out[f"resource_heater_on_{channel}"] = 0
            out[f"resource_power_w_{channel}"] = float(row["base_power_w"])
            out[f"resource_effective_power_{channel}"] = float(row["effective_base"])
    optional = list(rows)
    out["resource_optional_power_w"] = out[[f"resource_power_w_{c}" for c in optional]].sum(axis=1)
    out["resource_total_power_w"] = out["resource_optional_power_w"] + float(manifest["mandatory_backbone"]["steady_power_w"])
    out["resource_optional_effective_power"] = out[[f"resource_effective_power_{c}" for c in optional]].sum(axis=1)
    out["resource_absolute_feasible_all_optional"] = out["resource_total_power_w"] <= float(manifest["absolute_controller_budget_w"])
    out["resource_effective_feasible_all_optional"] = out["resource_optional_effective_power"] <= float(manifest["development_effective_budget"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    metadata = {
        "input": str(args.input), "manifest": str(args.manifest), "output": str(args.output),
        "truth_targets_changed": False, "event_labels_changed": False,
        "policy_observation_columns_added": [c for c in out.columns if c.startswith("resource_")],
        "rules_are": "development_proxy_hysteresis unless manufacturer threshold is explicitly documented",
    }
    args.output.with_suffix(".derivation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out), "gmx_on_fraction": float(states["met_station_core"].mean()), "parsivel_on_fraction": float(states["laser_disdrometer"].mean())}, sort_keys=True))


if __name__ == "__main__":
    main()
