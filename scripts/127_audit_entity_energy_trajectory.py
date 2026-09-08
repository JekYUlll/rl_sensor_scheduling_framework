"""Audit finite-energy trajectories for the documented entity configuration.

This is a truth/energy audit, not a policy experiment.  It evaluates fixed,
predeclared schedules under the documented device loads and an optional
non-controllable external load.  The report keeps installed-load evidence
separate from design-envelope stress assumptions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from v2.entity_resource_envelope import FIXED_AUXILIARY_POWER_W
from v2.entity_external_load import build_hysteretic_heater_profile
from v2.env import WarmupEnvConfig, WarmupSchedulingEnv
from v2.power_projector import PowerConstraintsV2
from v2.sensor_spec import load_sensor_specs


STATE_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
    "wind_dir_sin",
    "wind_dir_cos",
)
SELECTABLE = (
    "met_station_core",
    "radiometer_basic",
    "surface_temp_ir",
    "laser_disdrometer",
    "fc4_flux",
)
LOGGER = "cr1000xe_backbone"


def _sample_hourly(
    truth: pd.DataFrame,
    source_period_s: int,
    *,
    heater_power_w: float,
) -> pd.DataFrame:
    stride = max(1, int(round(3600.0 / float(source_period_s))))
    sampled = truth.iloc[::stride].reset_index(drop=True).copy()
    if sampled.empty:
        raise ValueError("hourly sampling produced no rows")
    sampled["fixed_external_power_w"] = build_hysteretic_heater_profile(
        sampled["snow_surface_temperature_c"].to_numpy(dtype=float),
        heater_power_w=float(heater_power_w),
    )
    return sampled


def _run_schedule(
    truth: pd.DataFrame,
    sensor_specs,
    schedule: tuple[str, ...],
    *,
    horizon: int,
    capacity_wh: float,
    reserve_wh: float,
    fixed_auxiliary_power_w: float,
) -> dict[str, object]:
    constraints = PowerConstraintsV2(
        per_step_budget=1200.0,
        startup_peak_budget=1200.0,
        required_sensor_ids=(LOGGER,),
    )
    cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        reward_target_columns=STATE_COLUMNS,
        episode_len=min(int(horizon), len(truth)),
        base_freq_s=3600,
        energy_account_enabled=True,
        energy_capacity=float(capacity_wh),
        initial_energy=float(capacity_wh),
        reserve_energy=float(reserve_wh),
        energy_step_hours=1.0,
        fixed_external_power_w=float(fixed_auxiliary_power_w),
        fixed_external_power_column="fixed_external_power_w",
        min_dwell_steps=1,
    )
    env = WarmupSchedulingEnv(truth, sensor_specs, constraints, cfg)
    env.reset()
    mask = [spec.sensor_id in set(schedule) for spec in sensor_specs]
    rows: list[dict[str, object]] = []
    done = False
    while not done:
        _, _, done, info = env.step_mask(mask)
        rows.append(
            {
                "step": len(rows),
                "soc_wh": float(info["soc"]),
                "energy_consumption_wh": float(info["energy_consumption_wh"]),
                "external_power_w": float(info["fixed_external_power_w"]),
                "sensor_power_w": float(info["power"]),
                "energy_guard_dropped": int(info["energy_guard_dropped"]),
                "energy_deficit": float(info["energy_deficit"]),
                "selected_sensor_ids": ",".join(info["selected_sensor_ids"]),
            }
        )
    frame = pd.DataFrame(rows)
    heater_power = (frame["external_power_w"] - float(fixed_auxiliary_power_w)).clip(lower=0.0)
    return {
        "schedule": list(schedule),
        "horizon_hours": int(len(frame)),
        "capacity_wh": float(capacity_wh),
        "reserve_wh": float(reserve_wh),
        "fixed_auxiliary_power_w": float(fixed_auxiliary_power_w),
        "initial_soc_wh": float(capacity_wh),
        "final_soc_wh": float(frame["soc_wh"].iloc[-1]),
        "min_soc_wh": float(frame["soc_wh"].min()),
        "energy_guard_dropped_steps": int((frame["energy_guard_dropped"] > 0).sum()),
        "energy_deficit_steps": int((frame["energy_deficit"] > 0).sum()),
        "heater_on_fraction": float((heater_power > 0.0).mean()),
        "mean_heater_power_w": float(heater_power.mean()),
        "mean_external_power_w": float(frame["external_power_w"].mean()),
        "mean_sensor_power_w": float(frame["sensor_power_w"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-truth", type=Path, required=True)
    parser.add_argument("--sensor-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-period-s", type=int, default=1)
    parser.add_argument("--capacity-wh", type=float, default=8640.0)
    parser.add_argument("--reserve-wh", type=float, default=0.0)
    parser.add_argument("--fixed-auxiliary-power-w", type=float, default=FIXED_AUXILIARY_POWER_W)
    parser.add_argument("--heater-power-w", type=float, default=600.0)
    args = parser.parse_args()

    truth = _sample_hourly(
        pd.read_csv(args.input_truth),
        args.source_period_s,
        heater_power_w=float(args.heater_power_w),
    )
    sensor_specs = load_sensor_specs(args.sensor_config)
    schedules = {
        "all_selectable": tuple((*SELECTABLE, LOGGER)),
        "without_parsivel": tuple(("met_station_core", "radiometer_basic", "surface_temp_ir", "fc4_flux", LOGGER)),
        "low_power_only": tuple(("surface_temp_ir", "fc4_flux", LOGGER)),
        "backbone_only": (LOGGER,),
    }
    scenarios = [24, 72, 168]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for horizon in scenarios:
        for name, schedule in schedules.items():
            result = _run_schedule(
                truth,
                sensor_specs,
                schedule,
                horizon=horizon,
                capacity_wh=args.capacity_wh,
                reserve_wh=args.reserve_wh,
                fixed_auxiliary_power_w=args.fixed_auxiliary_power_w,
            )
            result["scenario"] = name
            result["requested_horizon_hours"] = horizon
            result["horizon_truncated"] = bool(len(truth) < horizon)
            result["valid_for_requested_horizon"] = bool(len(truth) >= horizon)
            results.append(result)
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "entity_energy_trajectory_summary.csv", index=False)
    summary = {
        "input_truth": str(args.input_truth),
        "sensor_config": str(args.sensor_config),
        "source_period_s": int(args.source_period_s),
        "hourly_rows_available": int(len(truth)),
        "capacity_wh": float(args.capacity_wh),
        "reserve_wh": float(args.reserve_wh),
        "fixed_auxiliary_power_w": float(args.fixed_auxiliary_power_w),
        "heater_power_w": float(args.heater_power_w),
        "external_load_policy": "fixed hysteretic 600 W profile derived from sampled surface temperature; not policy controllable",
        "schedule_count": len(schedules),
        "horizons_hours": scenarios,
        "horizon_truncation_note": (
            "Requested horizons longer than the available sampled truth are retained as diagnostics "
            "but are not valid multi-day evidence."
        ),
        "interpretation": "Resource-envelope audit only; no PPO, forecaster selection, or scene tuning.",
    }
    (output_dir / "entity_energy_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
