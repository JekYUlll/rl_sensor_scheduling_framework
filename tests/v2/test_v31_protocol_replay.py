from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


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


def test_score_policy_replay_preserves_all_online_state_configuration() -> None:
    module = _load_script("23_v2_train_ppo.py")
    cfg = module.WarmupEnvConfig(
        state_columns=("x",),
        reward_target_columns=("x",),
        lookback=4,
        episode_len=8,
        seed=7,
        sensor_quality_columns=("quality_a", "quality_b"),
        include_alert_context_features=True,
        alert_context_columns=("particle_alert", "flux_alert", "thermal_alert"),
        alert_context_threshold=0.6,
        alert_context_trend_lookback=9,
    )
    replay_cfg = module.rollout_config_with_seed(cfg, offset=3)

    assert replay_cfg.seed == 10
    assert replay_cfg.sensor_quality_columns == cfg.sensor_quality_columns
    assert replay_cfg.include_alert_context_features is True
    assert replay_cfg.alert_context_columns == cfg.alert_context_columns
    assert replay_cfg.alert_context_threshold == 0.6
    assert replay_cfg.alert_context_trend_lookback == 9


def test_receding_command_preserves_alert_context_contract() -> None:
    module = _load_script("99_v32_receding_upper.py")
    command: list[str] = []
    module.append_alert_context_args(
        command,
        {
            "agent_alert_context": {
                "columns": ["particle_alert", "flux_alert", "thermal_alert"],
                "threshold": 0.6,
                "trend_lookback": 9,
                "include_alert_context_features": True,
                "include_event_flag_in_state": False,
            }
        },
    )

    assert command == [
        "--alert-context-columns", "particle_alert", "flux_alert", "thermal_alert",
        "--alert-context-threshold", "0.6",
        "--alert-context-trend-lookback", "9",
        "--include-alert-context-features",
        "--no-include-event-flag-in-state",
    ]


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
        transition_steps=0,
        report_noise_std=0.0,
    )
    first = module.add_channel_quality_dynamics(frame, **kwargs)
    second = module.add_channel_quality_dynamics(frame, **kwargs)

    assert first.equals(second)
    assert first["agent_context_quality_a"].between(0.2, 1.0).all()
    assert (first["agent_context_quality_a"] < 1.0).any()
    assert not first["agent_context_quality_a"].equals(first["agent_context_quality_b"])


def test_channel_quality_transition_is_gradual() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    frame = pd.DataFrame({"x": range(240)})
    out = module.add_channel_quality_dynamics(
        frame,
        sensor_ids=("a",),
        seed=29,
        coverage=0.2,
        min_duration=24,
        max_duration=24,
        min_gap=8,
        degraded_quality=0.2,
        transition_steps=6,
        report_noise_std=0.0,
    )
    quality = out["agent_context_quality_a"].to_numpy()

    assert np.any((quality > 0.2) & (quality < 1.0))
    assert np.max(np.abs(np.diff(quality))) < 0.8


def test_condition_dependent_quality_is_bounded_and_exogenous_to_availability() -> None:
    module = _load_script("20_build_public_weather_truth.py")
    import pandas as pd

    frame = pd.DataFrame({
        "wind_speed_ms": [8.0, 10.0, 15.0, 19.0, 21.0],
        "relative_humidity": [60.0, 65.0, 70.0, 78.0, 86.0],
        "event_subtype_particle_latent": [0.0, 0.2, 1.0, 0.0, 0.0],
        "event_subtype_flux_latent": [0.0, 0.0, 0.1, 1.0, 0.0],
        "event_subtype_thermal_latent": [0.0, 0.0, 0.0, 0.0, 1.0],
    })
    out = module.add_channel_quality_dynamics(
        frame,
        sensor_ids=("surface_temp_ir", "laser_disdrometer", "fc4_flux"),
        seed=31,
        coverage=0.0,
        min_duration=1,
        max_duration=1,
        min_gap=0,
        degraded_quality=0.35,
        transition_steps=0,
        report_noise_std=0.0,
        mode="condition_dependent",
    )

    for column in out.filter(like="agent_context_quality_"):
        assert out[column].between(0.35, 1.0).all()
    assert out.loc[2, "agent_context_quality_fc4_flux"] < out.loc[2, "agent_context_quality_laser_disdrometer"]
    assert out.loc[3, "agent_context_quality_laser_disdrometer"] < out.loc[3, "agent_context_quality_fc4_flux"]


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


def test_quality_gate_reconstructs_static_normalized_macro() -> None:
    module = _load_script("103_v32_collect_quality_scene_gate.py")
    import pandas as pd

    static = pd.Series({
        "oracle_loss_subtype_particle": 0.4,
        "oracle_loss_subtype_particle_staticnorm": 0.8,
        "oracle_loss_subtype_flux": 0.6,
        "oracle_loss_subtype_flux_staticnorm": 1.2,
        "oracle_loss_subtype_thermal": 0.5,
        "oracle_loss_subtype_thermal_staticnorm": 1.0,
    })
    candidate = pd.Series({
        "oracle_loss_subtype_particle": 0.25,
        "oracle_loss_subtype_flux": 0.50,
        "oracle_loss_subtype_thermal": 0.75,
    })

    assert module.staticnorm_macro(candidate, static) == 1.0
