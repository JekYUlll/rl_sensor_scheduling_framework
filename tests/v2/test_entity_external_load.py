import numpy as np
import pytest

from v2.entity_external_load import build_hysteretic_heater_profile


def test_hysteretic_heater_profile_does_not_chatter_at_intermediate_temperature():
    temperatures = np.asarray([-24.0, -26.0, -24.0, -22.5, -24.5, -26.0])
    profile = build_hysteretic_heater_profile(temperatures, heater_power_w=600.0)
    assert np.array_equal(profile, np.asarray([0.0, 600.0, 600.0, 0.0, 0.0, 600.0]))


def test_hysteretic_heater_rejects_invalid_threshold_order():
    with pytest.raises(ValueError):
        build_hysteretic_heater_profile(np.asarray([-20.0]), on_threshold_c=-20.0, off_threshold_c=-25.0)
