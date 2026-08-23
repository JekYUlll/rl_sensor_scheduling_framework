from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "framework_baseline_supplements",
    ROOT / "scripts" / "81_v31_framework_baseline_supplements.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_continuity_guarded_context_actions_preserve_calm_channels() -> None:
    sensors = [SimpleNamespace(sensor_id=name) for name in ("broad_a", "broad_b", "laser", "flux")]
    masks = np.asarray(
        [
            [1, 1, 0, 0],  # calibration-selected calm
            [0, 0, 1, 0],  # lower particle loss but no continuity
            [1, 0, 1, 0],
            [0, 1, 1, 0],  # tied continuity, lower particle loss
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    )
    table = pd.DataFrame(
        {
            "action_idx": range(len(masks)),
            "oracle_loss_mean": [0.2, 0.8, 0.4, 0.3, 0.4, 0.5],
            "oracle_loss_non_event": [0.1, 0.9, 0.5, 0.5, 0.5, 0.5],
            "oracle_loss_subtype_particle": [0.8, 0.1, 0.3, 0.2, 0.9, 0.9],
            "oracle_loss_subtype_flux": [0.8, 0.9, 0.9, 0.9, 0.2, 0.3],
            "oracle_loss_subtype_thermal": [0.8, 0.9, 0.9, 0.9, 0.2, 0.3],
        }
    )
    metadata = {
        "oracle_subtype_teacher_sensors": {
            "calm": ["broad_a", "broad_b"],
            "particle": ["broad_a", "laser"],
            "flux": ["broad_a", "flux"],
            "thermal": ["broad_a", "flux"],
        }
    }

    selected = MODULE.build_continuity_guarded_context_action_indices(
        metadata, sensors, masks, table
    )

    assert selected == {"calm": 0, "particle": 3, "flux": 4, "thermal": 4}


def test_coordinate_selector_optimizes_complete_mapping() -> None:
    table = pd.DataFrame(
        {
            "action_idx": [0, 1, 2],
            "oracle_loss_mean": [0.1, 0.2, 0.3],
            "oracle_loss_non_event": [0.1, 0.2, 0.3],
            "oracle_loss_subtype_particle": [0.3, 0.1, 0.2],
            "oracle_loss_subtype_flux": [0.3, 0.2, 0.1],
            "oracle_loss_subtype_thermal": [0.3, 0.2, 0.1],
        }
    )

    def evaluate(mapping: dict[str, int]) -> tuple[float, float]:
        target = {"calm": 1, "particle": 2, "flux": 0, "thermal": 1}
        distance = sum(int(mapping[label] != action) for label, action in target.items())
        return float(distance), float(sum(mapping.values()))

    selected, ledger = MODULE.coordinate_select_context_actions(
        table,
        evaluate,
        pool_size=3,
        passes=2,
    )

    assert selected == {"calm": 1, "particle": 2, "flux": 0, "thermal": 1}
    assert not ledger.empty
    assert {"selection_primary", "selection_secondary"}.issubset(ledger.columns)


def test_intensity_policy_uses_calm_low_and_high_actions() -> None:
    sensors = [SimpleNamespace(sensor_id="a")]
    masks = np.asarray([[0], [1]], dtype=bool)
    mapping = {
        "calm": 0,
        "particle_low": 0,
        "particle_high": 1,
        "flux_low": 0,
        "flux_high": 1,
        "thermal_low": 0,
        "thermal_high": 1,
    }
    policy = MODULE.IntensityBinnedContextPolicy(
        sensors=sensors,
        candidate_masks=masks,
        action_indices=mapping,
        threshold=0.5,
        high_threshold=0.75,
        name="test",
    )
    env = SimpleNamespace(current_idx=0)
    for value, expected in ((0.4, False), (0.6, False), (0.8, True)):
        env.truth_df = pd.DataFrame({
            "agent_context_particle_alert": [value],
            "agent_context_flux_alert": [0.0],
            "agent_context_thermal_alert": [0.0],
        })
        assert bool(policy.act_mask(env)[0]) is expected


def test_intensity_coordinate_selector_can_separate_levels() -> None:
    table = pd.DataFrame({
        "action_idx": [0, 1],
        "oracle_loss_mean": [0.1, 0.2],
        "oracle_loss_non_event": [0.1, 0.2],
        "oracle_loss_subtype_particle": [0.1, 0.2],
        "oracle_loss_subtype_flux": [0.1, 0.2],
        "oracle_loss_subtype_thermal": [0.1, 0.2],
    })
    target = {
        "calm": 0,
        "particle_low": 0,
        "particle_high": 1,
        "flux_low": 0,
        "flux_high": 1,
        "thermal_low": 0,
        "thermal_high": 1,
    }

    def evaluate(mapping: dict[str, int]) -> tuple[float, float]:
        distance = sum(int(mapping[label] != action) for label, action in target.items())
        return float(distance), float(sum(mapping.values()))

    selected, ledger = MODULE.coordinate_select_intensity_actions(table, evaluate, pool_size=2, passes=2)
    assert selected == target
    assert not ledger.empty
