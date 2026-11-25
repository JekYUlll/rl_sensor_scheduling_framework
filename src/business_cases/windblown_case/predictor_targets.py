from __future__ import annotations


def default_reward_target_columns() -> list[str]:
    return [
        "air_temperature_c",
        "snow_surface_temperature_c",
        "wind_speed_ms",
    ]


def default_forecast_target_columns() -> list[str]:
    return [
        "air_temperature_c",
        "snow_surface_temperature_c",
        "wind_speed_ms",
        "wind_dir_sin",
        "wind_dir_cos",
        "snow_mass_flux_kg_m2_s",
        "snow_particle_mean_velocity_ms",
    ]
