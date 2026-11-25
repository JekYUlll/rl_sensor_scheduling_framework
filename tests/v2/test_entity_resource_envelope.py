import math

from v2 import entity_resource_envelope as audit


def test_enumerates_all_selectable_subsets_and_documents_controller_result():
    rows = audit.rows()
    assert len(rows) == 2 ** len(audit.CHANNEL_DEVICE)
    all_channels = max(rows, key=lambda row: int(row["selected_count"]))
    assert math.isclose(float(all_channels["system_steady_power_w"]), 8.656)
    assert all_channels["controller_steady_feasible"] is True
    assert all_channels["controller_heater_peak_feasible"] is True


def test_design_envelope_is_not_mistaken_for_renewable_harvest_trace():
    assert audit.BATTERY_CAPACITY_WH == 8640.0
    assert audit.CONTROLLER_POWER_W == 1200.0
    assert audit.PV_RATED_W + audit.WIND_RATED_W == 1000.0
