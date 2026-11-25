from __future__ import annotations

import pytest

from src.v2.entity_energy import system_power_w, validate_channel_mapping


def test_logger_backbone_is_always_included() -> None:
    result = system_power_w(["fc4_flux"])
    assert result["backbone_power_w"] == pytest.approx(0.4104)
    assert result["steady_power_w"] == pytest.approx(0.4704)


def test_parsivel_heater_is_a_peak_load() -> None:
    result = system_power_w(["laser_disdrometer"], heater_on=True)
    assert result["steady_power_w"] == pytest.approx(1.9104)
    assert result["peak_power_w"] >= 100.0


def test_unmapped_legacy_channel_is_rejected() -> None:
    with pytest.raises(ValueError, match="shielded_thermo_hygro"):
        validate_channel_mapping(["shielded_thermo_hygro"])
