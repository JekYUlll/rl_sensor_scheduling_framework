from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_training_module():
    path = Path(__file__).parents[1] / "scripts" / "25_v2_train_custom_ppo.py"
    spec = importlib.util.spec_from_file_location("train_custom_ppo_units", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_trace_prefers_explicit_physical_watt_columns(tmp_path: Path) -> None:
    module = _load_training_module()
    trace_path = tmp_path / "trace.csv"
    pd.DataFrame(
        {
            "time_idx": [0, 1],
            "resource_power_w_met_station_core": [0.30, 4.80],
            "resource_effective_power_met_station_core": [0.72, 2.83],
        }
    ).to_csv(trace_path, index=False)
    truth = pd.DataFrame({"time_idx": [0, 1], "air_temperature_c": [-20.0, -19.0]})

    merged, mapping = module.merge_dynamic_resource_trace(truth, str(trace_path))

    assert mapping == (("met_station_core", "resource_power_w_met_station_core"),)
    assert merged["resource_power_w_met_station_core"].tolist() == [0.30, 4.80]


def test_dynamic_trace_keeps_legacy_effective_prefix_compatibility(tmp_path: Path) -> None:
    module = _load_training_module()
    trace_path = tmp_path / "legacy_trace.csv"
    pd.DataFrame(
        {
            "time_idx": [0],
            "resource_effective_power_met_station_core": [4.80],
        }
    ).to_csv(trace_path, index=False)
    truth = pd.DataFrame({"time_idx": [0], "air_temperature_c": [-20.0]})

    _, mapping = module.merge_dynamic_resource_trace(truth, str(trace_path))

    assert mapping == (("met_station_core", "resource_effective_power_met_station_core"),)
