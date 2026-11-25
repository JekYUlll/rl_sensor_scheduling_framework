from __future__ import annotations


def detect_event(latent: dict[str, float]) -> bool:
    return bool(latent.get("wind_speed_ms", 0.0) >= 10.0 or latent.get("snow_mass_flux_kg_m2_s", 0.0) >= 2.5e-5)
