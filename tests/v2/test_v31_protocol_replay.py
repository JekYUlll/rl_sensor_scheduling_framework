from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_overlapping_random_starts_are_deterministic_and_disjoint() -> None:
    module = _load_script("57_v31_independent_replay.py")
    starts = module.non_overlapping_random_starts(
        sequence_steps=30000,
        eval_steps=1024,
        horizon=8,
        n_rollouts=6,
        seed=11818,
    )
    repeated = module.non_overlapping_random_starts(
        sequence_steps=30000,
        eval_steps=1024,
        horizon=8,
        n_rollouts=6,
        seed=11818,
    )

    assert starts == repeated
    assert len(starts) == 6
    for idx, left in enumerate(starts):
        for right in starts[idx + 1 :]:
            assert left + 1024 <= right or right + 1024 <= left


def test_interval_overlap_pairs_detect_cross_window_overlap() -> None:
    module = _load_script("56_v31_protocol_audit.py")

    pairs = module.interval_overlap_pairs((100, 1000), 200, (250, 1200), 200)

    assert pairs == [(0, 0)]


def test_split_protocol_reserves_non_overlapping_final_test_windows() -> None:
    module = _load_script("58_v31_split_protocol_run.py")
    bounds = module.partition_bounds(90000, (0.35, 0.50, 0.075, 0.075))
    final_starts = module.non_overlapping_starts(
        bounds=bounds["final_test"],
        window_steps=1024,
        horizon=8,
        count=6,
        seed=1818,
    )

    assert bounds["final_test"] == (83250, 90000)
    assert len(final_starts) == 6
    assert min(final_starts) >= bounds["final_test"][0]
    assert max(final_starts) + 1024 + 8 < bounds["final_test"][1]


def test_control_source_files_follow_reward_normalization_mode() -> None:
    module = _load_script("25_v2_train_custom_ppo.py")

    unnormalized = module.control_source_required_files("none")
    staticnorm = module.control_source_required_files("staticnorm_subtype")

    assert "reward_staticnorm_candidates.csv" not in unnormalized
    assert "reward_staticnorm_normalizers.json" not in unnormalized
    assert "reward_staticnorm_candidates.csv" in staticnorm
    assert "reward_staticnorm_normalizers.json" in staticnorm


def test_flexible_behavior_gate_preserves_frozen_six_channel_rule() -> None:
    module = _load_script("25_v2_train_custom_ppo.py")
    valid = {
        "always_on_sensor_count": 0,
        "always_off_sensor_count": 1,
        "switches_per_step": 0.01,
        "warmup_abort_count": 0,
    }

    assert module.flexible_six_channel_behavior_valid(valid)
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "always_on_sensor_count": 1}
    )
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "always_off_sensor_count": 2}
    )
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "switches_per_step": 0.0}
    )
    assert not module.flexible_six_channel_behavior_valid(
        {**valid, "warmup_abort_count": 1}
    )


def test_multiscene_collector_uses_same_behavior_gate() -> None:
    module = _load_script("102_v32_collect_multiscene_loo.py")
    import pandas as pd

    valid = pd.Series({
        "always_on_sensor_count": 0,
        "always_off_sensor_count": 1,
        "switches_per_step": 0.01,
        "warmup_abort_count": 0,
    })
    invalid = valid.copy()
    invalid["always_off_sensor_count"] = 2

    assert module.behavior_valid(valid)
    assert not module.behavior_valid(invalid)


def test_channel_quality_generation_is_deterministic_and_bounded() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    frame = pd.DataFrame({"x": range(200)})
    kwargs = dict(
        sensor_ids=("a", "b"),
        seed=17,
        coverage=0.2,
        min_duration=8,
        max_duration=16,
        min_gap=4,
        degraded_quality=0.2,
        report_noise_std=0.0,
    )
    first = module.add_channel_quality_dynamics(frame, **kwargs)
    second = module.add_channel_quality_dynamics(frame, **kwargs)

    assert first.equals(second)
    assert first["agent_context_quality_a"].between(0.2, 1.0).all()
    assert (first["agent_context_quality_a"] < 1.0).any()
    assert not first["agent_context_quality_a"].equals(first["agent_context_quality_b"])


def test_receding_diagnostic_propagates_frozen_sensor_quality_config() -> None:
    module = _load_script("99_v32_receding_upper.py")
    command = ["python", "diagnostic.py"]
    metadata = {
        "sensor_quality": {
            "columns": ["quality_a", "quality_b"],
            "max_noise_multiplier": 6.0,
            "availability_floor": 0.2,
        }
    }

    module.append_sensor_quality_args(command, metadata)

    assert command[-7:] == [
        "--sensor-quality-columns",
        "quality_a",
        "quality_b",
        "--sensor-quality-max-noise-multiplier",
        "6.0",
        "--sensor-quality-availability-floor",
        "0.2",
    ]
