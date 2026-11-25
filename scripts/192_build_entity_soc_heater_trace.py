"""Build a documented PV/SOC trace with entity heater loads.

The heater states come from the deployment-observable hysteresis rules in
``131_build_heater_resource_trace.py``. The trace is for policy-free resource
screening: it does not alter targets or event labels and does not use policy
outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2.entity_energy import system_power_w
from v2.entity_resource_envelope import FIXED_AUXILIARY_POWER_W


def _dew_point_c(temp: pd.Series, rh: pd.Series) -> pd.Series:
    rh_safe = rh.clip(lower=1.0, upper=100.0)
    gamma = np.log(rh_safe / 100.0) + (17.62 * temp / (243.12 + temp))
    return 243.12 * gamma / (17.62 - gamma)


def _hysteresis(on_condition: pd.Series, off_condition: pd.Series) -> pd.Series:
    state = False
    result: list[bool] = []
    for on, off in zip(on_condition.fillna(False), off_condition.fillna(False), strict=True):
        if not state and bool(on):
            state = True
        elif state and bool(off):
            state = False
        result.append(state)
    return pd.Series(result, index=on_condition.index, dtype=bool)


def _build_heater_trace(truth: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    required = {
        "air_temperature_c",
        "relative_humidity",
        "snow_surface_temperature_c",
        "solar_radiation_wm2",
    }
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"missing runtime-observable truth columns: {missing}")
    out = truth.copy()
    dp = _dew_point_c(out["air_temperature_c"], out["relative_humidity"])
    margin = out["air_temperature_c"] - dp
    gmx_on = (out["air_temperature_c"] <= -25.0) | ((margin <= 1.0) & (out["air_temperature_c"] <= -5.0))
    gmx_off = (out["air_temperature_c"] >= -23.0) & (margin >= 2.0)
    parsivel_on = (out["air_temperature_c"] <= -20.0) | (
        (margin <= 1.0) & (out["snow_surface_temperature_c"] <= -5.0)
    )
    parsivel_off = (out["air_temperature_c"] >= -18.0) & (margin >= 2.0)
    states = {
        "met_station_core": _hysteresis(gmx_on, gmx_off),
        "laser_disdrometer": _hysteresis(parsivel_on, parsivel_off),
    }
    rows = {row["channel_id"]: row for row in manifest["channels"]}
    for channel, state in states.items():
        row = rows[channel]
        out[f"resource_heater_on_{channel}"] = state.astype(int)
        out[f"resource_power_w_{channel}"] = float(row["base_power_w"]) + state.astype(float) * float(row["heater_increment_w"])
    for channel, row in rows.items():
        if channel not in states:
            out[f"resource_heater_on_{channel}"] = 0
            out[f"resource_power_w_{channel}"] = float(row["base_power_w"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pv-derating", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.0 <= float(args.pv_derating) <= 1.0:
        raise ValueError("pv derating must be in [0, 1]")

    truth = pd.read_csv(args.input)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out = _build_heater_trace(truth, manifest)
    irradiance = np.clip(out["solar_radiation_wm2"].to_numpy(dtype=float), 0.0, None)
    pv_w = np.minimum(600.0, 600.0 * irradiance / 1000.0 * float(args.pv_derating))
    out["energy_harvest_wh"] = pv_w
    out["generation_pv_w"] = pv_w
    out["resource_power_w_cr1000xe_backbone"] = float(
        system_power_w([])["steady_power_w"]
    )
    fixed_load = float(FIXED_AUXILIARY_POWER_W)
    baseline_soc = np.empty(len(out), dtype=float)
    soc = 8640.0
    for index, harvest in enumerate(pv_w):
        soc = float(np.clip(soc + float(harvest) - fixed_load, 0.0, 8640.0))
        baseline_soc[index] = soc
    out["baseline_soc_wh"] = baseline_soc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    metadata = {
        "generator": Path(__file__).name,
        "heater_builder": "131_build_heater_resource_trace.py",
        "manifest": str(args.manifest),
        "pv_derating": float(args.pv_derating),
        "fixed_baseline_load_w": fixed_load,
        "battery_capacity_wh": 8640.0,
        "truth_targets_changed": False,
        "event_labels_changed": False,
        "policy_outputs_used": False,
    }
    args.output.with_suffix(".derivation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "rows": len(out),
        "mean_pv_w": float(np.mean(pv_w)),
        "zero_pv_fraction": float(np.mean(pv_w <= 1.0e-12)),
        "gmx_heater_fraction": float(out["resource_heater_on_met_station_core"].mean()),
        "parsivel_heater_fraction": float(out["resource_heater_on_laser_disdrometer"].mean()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
