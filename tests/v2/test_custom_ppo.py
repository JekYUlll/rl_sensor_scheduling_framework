from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from v2.custom_ppo import (  # noqa: E402
    ActionEmbedding,
    CustomPPO,
    CustomPPOConfig,
    MaskedActor,
    advantage_weighted_bc_loss,
    channel_marginal_distribution_entropy,
    feasible_candidate_mask,
    full_rollout_schedule,
    masked_soft_target_cross_entropy,
    soft_forecast_value_targets,
    restore_env,
    snapshot_env,
    standardized_negative_cost_targets,
)
from v2.env import WarmupEnvConfig, WarmupSchedulingEnv  # noqa: E402
from v2.oracle import LinearFrozenForecastOracle, OracleConfig, build_supervised_windows  # noqa: E402
from v2.power_projector import PowerConstraintsV2  # noqa: E402
from v2.sensor_spec import SensorSpecV2  # noqa: E402


STATE_COLUMNS = (
    "wind_speed_ms",
    "wind_direction_deg",
    "wind_dir_sin",
    "wind_dir_cos",
    "air_temperature_c",
    "relative_humidity",
    "air_pressure_pa",
    "solar_radiation_wm2",
    "snow_surface_temperature_c",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
    "snow_mass_flux_kg_m2_s",
)


def test_full_rollout_schedule_never_emits_short_tail_batch() -> None:
    assert full_rollout_schedule(40_000, 1_024) == (1_024,) * 40
    assert sum(full_rollout_schedule(40_000, 1_024)) == 40_960
    assert full_rollout_schedule(0, 1_024) == ()


def test_channel_marginal_distribution_entropy_distinguishes_channel_coverage() -> None:
    candidate_masks = torch.eye(3)
    balanced_logits = torch.zeros((4, 3), requires_grad=True)
    balanced = channel_marginal_distribution_entropy(
        torch.softmax(balanced_logits, dim=1), candidate_masks
    )
    collapsed = channel_marginal_distribution_entropy(
        torch.tensor([[1.0, 0.0, 0.0]]), candidate_masks
    )

    assert balanced.item() == pytest.approx(1.0)
    assert collapsed.item() == pytest.approx(0.0)
    (-balanced).backward()
    assert balanced_logits.grad is not None


def test_soft_forecast_value_targets_normalize_per_state_and_mask_invalid_actions() -> None:
    targets = soft_forecast_value_targets(
        np.asarray([[3.0, 1.0, 2.0], [5.0, 5.0, 9.0]]),
        np.asarray([[True, True, False], [True, True, False]]),
        temperature=1.0,
    )

    assert np.allclose(targets.sum(axis=1), 1.0)
    assert np.allclose(targets[:, 2], 0.0)
    assert targets[0, 1] > targets[0, 0]
    assert targets[1, 0] == pytest.approx(targets[1, 1])


def test_standardized_negative_cost_targets_preserve_feasible_ranking() -> None:
    targets = standardized_negative_cost_targets(
        np.asarray([[3.0, 1.0, 2.0], [5.0, np.inf, 9.0]]),
        np.asarray([[True, True, True], [True, False, True]]),
    )

    assert int(np.argmax(targets[0])) == 1
    assert targets[1, 1] == 0.0
    assert targets[1, 0] > targets[1, 2]
    assert np.mean(targets[0]) == pytest.approx(0.0, abs=1.0e-6)


def _truth(rows: int = 64) -> pd.DataFrame:
    t = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "wind_speed_ms": 8.0 + 0.1 * t,
            "wind_direction_deg": 90.0 + t,
            "wind_dir_sin": np.sin(np.deg2rad(90.0 + t)),
            "wind_dir_cos": np.cos(np.deg2rad(90.0 + t)),
            "air_temperature_c": -20.0 + 0.05 * t,
            "relative_humidity": 65.0,
            "air_pressure_pa": 70000.0,
            "solar_radiation_wm2": 100.0,
            "snow_surface_temperature_c": -21.0,
            "snow_particle_mean_diameter_mm": 0.2,
            "snow_particle_mean_velocity_ms": 2.0,
            "snow_mass_flux_kg_m2_s": 1e-5 * t,
            "agent_context_particle_alert": np.clip(t / max(rows - 1, 1), 0.0, 1.0),
            "agent_context_flux_alert": np.zeros(rows),
            "agent_context_thermal_alert": np.clip(1.0 - t / max(rows - 1, 1), 0.0, 1.0),
            "event_flag": t > rows // 2,
        }
    )


def _sensors() -> list[SensorSpecV2]:
    return [
        SensorSpecV2("met", ("wind_speed_ms", "air_temperature_c"), 0.5, 0.8, warmup_steps=0),
        SensorSpecV2("snow", ("snow_mass_flux_kg_m2_s",), 1.2, 1.8, warmup_steps=2),
        SensorSpecV2("rad", ("solar_radiation_wm2",), 0.4, 0.5, warmup_steps=0),
    ]


def _oracle(truth: pd.DataFrame) -> LinearFrozenForecastOracle:
    truth_values = truth[list(STATE_COLUMNS)].to_numpy(dtype=float)
    masks = np.ones_like(truth_values)
    x, y = build_supervised_windows(truth_values, masks, truth_values, lookback=4, horizon=2)
    return LinearFrozenForecastOracle(OracleConfig(lookback=4, horizon=2, ridge_alpha=0.1)).fit(x, y)


def test_masked_actor_feasible_only() -> None:
    actor = MaskedActor(obs_dim=5, n_sensors=3, embed_dim=8, hidden_dim=16)
    obs = torch.zeros((2, 5), dtype=torch.float32)
    candidate_masks = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    action_mask = torch.tensor([[True, False, True], [False, True, False]])

    probs = actor.dist(obs, candidate_masks, action_mask).probs

    assert torch.allclose(probs[0, 1], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs[1, 0], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs[1, 2], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_masked_actor_can_disable_state_independent_action_prior() -> None:
    actor = MaskedActor(
        obs_dim=5,
        n_sensors=3,
        embed_dim=8,
        hidden_dim=16,
        n_actions=3,
        trainable_action_prior=False,
    )
    assert actor.action_prior is None


def test_forecast_value_head_is_masked_and_gradient_isolated_from_policy_logits() -> None:
    actor = MaskedActor(
        obs_dim=5,
        n_sensors=3,
        embed_dim=8,
        hidden_dim=16,
        n_actions=3,
        forecast_value_head_enabled=True,
    )
    obs = torch.randn((2, 5), dtype=torch.float32)
    candidate_masks = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    feasible = torch.tensor([[True, False, True], [True, True, False]])

    actor.logits(obs, candidate_masks, feasible).sum().backward()
    forecast_parameters = [
        parameter
        for name, parameter in actor.named_parameters()
        if name.startswith("forecast_")
    ]
    assert forecast_parameters
    assert all(parameter.grad is None for parameter in forecast_parameters)

    actor.zero_grad(set_to_none=True)
    forecast_logits = actor.forecast_value_logits(obs, candidate_masks, feasible)
    assert torch.all(forecast_logits[~feasible] < -1.0e8)
    forecast_logits[feasible].sum().backward()
    assert any(parameter.grad is not None for parameter in forecast_parameters)


def test_nonlinear_action_embedding_encodes_subset_interactions() -> None:
    actor = MaskedActor(
        obs_dim=5,
        n_sensors=3,
        embed_dim=8,
        hidden_dim=16,
        n_actions=4,
        nonlinear_action_embedding=True,
    )
    assert not isinstance(actor.action_embedding.subset_encoder, torch.nn.Identity)
    masks = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    embeddings = actor._action_embeddings(masks)
    assert embeddings.shape == (4, 8)
    assert not torch.allclose(embeddings[2], embeddings[0] + embeddings[1])


def test_custom_ppo_episode_env_preserves_complete_config() -> None:
    truth = _truth(64)
    cfg = WarmupEnvConfig(
        state_columns=STATE_COLUMNS,
        reward_target_columns=("air_temperature_c",),
        reward_proxy_mode="uncertainty",
        lookback=4,
        episode_len=8,
        seed=9,
        event_subtype_particle_reward_multiplier=1.7,
        oracle_loss_reward_normalizers=(1.1, 2.2, 3.3),
        min_dwell_steps=6,
        agent_context_columns=(
            "agent_context_particle_alert",
            "agent_context_flux_alert",
            "agent_context_thermal_alert",
        ),
        include_event_flag_in_state=False,
        uncertainty_process_variance=tuple(0.02 for _ in STATE_COLUMNS),
        measurement_update_mode="variance_weighted",
    )
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=_sensors(),
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        env_cfg=cfg,
        oracle=_oracle(truth),
        candidate_masks=np.asarray([[True, False, False], [True, False, True]], dtype=bool),
        cfg=CustomPPOConfig(total_timesteps=1, n_steps=1, batch_size=1, n_epochs=1, device="cpu"),
    )

    episode_env = trainer._make_env(seed_offset=4)

    assert episode_env.cfg.reward_proxy_mode == "uncertainty"
    assert episode_env.cfg.event_subtype_particle_reward_multiplier == pytest.approx(1.7)
    assert episode_env.cfg.oracle_loss_reward_normalizers == (1.1, 2.2, 3.3)
    assert episode_env.cfg.min_dwell_steps == 6
    assert not episode_env.cfg.include_event_flag_in_state
    assert episode_env.cfg.measurement_update_mode == "variance_weighted"
    assert episode_env.cfg.seed == 13


def test_forecast_gain_reward_uses_same_epoch_no_measurement_counterfactual() -> None:
    truth = _truth(64)
    env = WarmupSchedulingEnv(
        truth,
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            reward_proxy_mode="forecast_gain",
            lookback=4,
            episode_len=8,
            lambda_warmup_abort=0.0,
            lambda_switch=0.0,
        ),
        oracle=_oracle(truth),
    )
    env.reset(start_idx=12)

    _, reward, _, info = env.step_mask(np.asarray([True, False, True]))

    expected_loss = float(info["oracle_loss"]) - float(info["counterfactual_oracle_loss"])
    assert float(info["reward_proxy_loss"]) == pytest.approx(expected_loss)
    assert reward == pytest.approx(-expected_loss)


def test_separate_gradient_clipping_limits_actor_and_critic_independently() -> None:
    truth = _truth(64)
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=_sensors(),
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        env_cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=("air_temperature_c",),
            lookback=4,
            episode_len=8,
            seed=9,
        ),
        oracle=_oracle(truth),
        candidate_masks=np.asarray([[True, False, False], [True, False, True]], dtype=bool),
        cfg=CustomPPOConfig(
            total_timesteps=1,
            n_steps=1,
            batch_size=1,
            n_epochs=1,
            max_grad_norm=0.5,
            separate_actor_critic_grad_clip=True,
            device="cpu",
        ),
    )
    for parameter in trainer.model.actor.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)
    for parameter in trainer.model.critic.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    trainer._clip_update_gradients()

    actor_norm = torch.sqrt(
        sum(torch.sum(parameter.grad**2) for parameter in trainer.model.actor.parameters())
    )
    critic_norm = torch.sqrt(
        sum(torch.sum(parameter.grad**2) for parameter in trainer.model.critic.parameters())
    )
    assert actor_norm <= 0.50001
    assert critic_norm <= 0.50001


def test_oracle_lookahead_snapshot_restores_uncertainty_and_constraint_state() -> None:
    truth = _truth(64)
    env = WarmupSchedulingEnv(
        truth,
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            reward_proxy_mode="uncertainty",
            lookback=4,
            episode_len=8,
            min_dwell_steps=3,
        ),
        oracle=_oracle(truth),
    )
    env.reset()
    snapshot = snapshot_env(env)
    env.step_mask(np.asarray([True, False, False]))

    restore_env(env, snapshot)

    assert np.allclose(env.posterior_variance, snapshot["posterior_variance"])
    assert env.dwell_hold_remaining == snapshot["dwell_hold_remaining"]
    assert env.current_idx == snapshot["current_idx"]


def test_masked_actor_context_encoder_preserves_feasible_mask() -> None:
    actor = MaskedActor(
        obs_dim=25,
        n_sensors=3,
        embed_dim=8,
        hidden_dim=16,
        context_encoder_enabled=True,
        context_feature_dim=20,
        context_hidden_dim=8,
    )
    obs = torch.zeros((2, 25), dtype=torch.float32)
    candidate_masks = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    action_mask = torch.tensor([[True, False, True], [False, True, False]])

    probs = actor.dist(obs, candidate_masks, action_mask).probs

    assert torch.allclose(probs[0, 1], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs[1, 0], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs[1, 2], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_masked_actor_gated_context_fusion_preserves_feasible_mask() -> None:
    actor = MaskedActor(
        obs_dim=25,
        n_sensors=3,
        embed_dim=8,
        hidden_dim=16,
        context_encoder_enabled=True,
        context_feature_dim=20,
        context_hidden_dim=8,
        context_fusion_mode="gated_add",
        context_layer_norm=True,
    )
    obs = torch.zeros((2, 25), dtype=torch.float32)
    candidate_masks = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    action_mask = torch.tensor([[True, False, True], [False, True, False]])

    probs = actor.dist(obs, candidate_masks, action_mask).probs

    assert torch.allclose(probs[0, 1], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs[1, 0], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs[1, 2], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_masked_actor_temporal_encoder_uses_history_and_preserves_feasible_mask() -> None:
    history_steps = 4
    state_dim = 3
    context_dim = 2
    runtime_dim = 5
    actor = MaskedActor(
        obs_dim=2 * history_steps * state_dim + runtime_dim + context_dim,
        n_sensors=3,
        embed_dim=8,
        hidden_dim=16,
        context_encoder_enabled=True,
        context_feature_dim=context_dim,
        context_hidden_dim=8,
        temporal_encoder_enabled=True,
        temporal_history_steps=history_steps,
        temporal_state_dim=state_dim,
        temporal_hidden_dim=7,
    )
    obs = torch.zeros((2, 2 * history_steps * state_dim + runtime_dim + context_dim))
    obs[1, : history_steps * state_dim] = torch.arange(
        history_steps * state_dim, dtype=torch.float32
    )
    candidate_masks = torch.eye(3)
    action_mask = torch.tensor([[True, False, True], [True, True, False]])

    contexts = actor.encode_context(obs)
    probs = actor.dist(obs, candidate_masks, action_mask).probs

    assert not torch.allclose(contexts[0], contexts[1])
    assert probs[0, 1] == 0
    assert probs[1, 2] == 0
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_masked_actor_subtype_moe_routes_online_context_and_preserves_mask() -> None:
    actor = MaskedActor(
        obs_dim=25,
        n_sensors=3,
        embed_dim=8,
        hidden_dim=16,
        context_encoder_enabled=True,
        context_feature_dim=20,
        context_hidden_dim=8,
        context_fusion_mode="subtype_moe",
        context_layer_norm=True,
        subtype_aux_classes=4,
    )
    obs = torch.zeros((2, 25), dtype=torch.float32)
    obs[1, -20:] = 1.0
    candidate_masks = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    action_mask = torch.tensor([[True, False, True], [False, True, True]])

    subtype_logits = actor.subtype_logits(obs)
    probs = actor.dist(obs, candidate_masks, action_mask).probs
    loss = subtype_logits.square().mean() + probs.square().mean()
    loss.backward()

    assert subtype_logits.shape == (2, 4)
    assert actor.context_router.weight.grad is not None
    assert all(expert[0].weight.grad is not None for expert in actor.context_experts)
    assert torch.allclose(probs[0, 1], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs[1, 0], torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_action_embedding_structure() -> None:
    action_embedding = ActionEmbedding(n_sensors=3, embed_dim=3)
    with torch.no_grad():
        action_embedding.sensor_embedding.embedding.weight.copy_(torch.eye(3))
    masks = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    emb = action_embedding(masks)
    sim_shared = torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0)
    sim_less_shared = torch.nn.functional.cosine_similarity(emb[0], emb[2], dim=0)

    assert float(sim_shared.detach()) > float(sim_less_shared.detach())


def test_awbc_loss_prefers_high_advantage_greedy_action() -> None:
    good_dist = torch.distributions.Categorical(logits=torch.tensor([[3.0, -1.0], [3.0, -1.0]]))
    bad_dist = torch.distributions.Categorical(logits=torch.tensor([[-1.0, 3.0], [-1.0, 3.0]]))
    greedy = torch.tensor([0, 0], dtype=torch.long)
    advantages = torch.tensor([2.0, 1.0], dtype=torch.float32)

    assert float(advantage_weighted_bc_loss(good_dist, greedy, advantages)) < float(
        advantage_weighted_bc_loss(bad_dist, greedy, advantages)
    )


def test_feasible_candidate_mask_respects_current_projector() -> None:
    truth = _truth(16)
    env = WarmupSchedulingEnv(
        truth,
        _sensors(),
        PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        WarmupEnvConfig(state_columns=STATE_COLUMNS, lookback=4, episode_len=8, seed=1),
    )
    env.reset()
    candidate_masks = np.asarray(
        [
            [True, False, False],
            [True, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )

    mask = feasible_candidate_mask(env, candidate_masks)

    assert mask.tolist() == [True, False, True]


def test_agent_cycle_phase_is_optional_and_preserves_event_tail() -> None:
    truth = _truth(16)
    sensors = _sensors()
    constraints = PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0)
    base_env = WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        WarmupEnvConfig(state_columns=STATE_COLUMNS, lookback=4, episode_len=8, seed=1),
    )
    phase_env = WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=8,
            seed=1,
            include_agent_cycle_phase=True,
            agent_cycle_period_steps=10,
            agent_cycle_dwell_steps=4,
        ),
    )

    base_obs, _ = base_env.reset(start_idx=3)
    phase_obs, _ = phase_env.reset(start_idx=3)

    assert phase_obs.shape[0] == base_obs.shape[0] + 4
    np.testing.assert_allclose(phase_obs[-5:-1], np.asarray([0.0, 1.0, 0.0, 1.0]), atol=1e-7)
    assert phase_obs[-1] == base_obs[-1]


def test_alert_context_tail_can_replace_event_flag_in_state() -> None:
    truth = _truth(16)
    sensors = _sensors()
    constraints = PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0)
    base_env = WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        WarmupEnvConfig(state_columns=STATE_COLUMNS, lookback=4, episode_len=8, seed=1),
    )
    alert_env = WarmupSchedulingEnv(
        truth,
        sensors,
        constraints,
        WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=8,
            seed=1,
            include_event_flag_in_state=False,
            include_alert_context_features=True,
            alert_context_threshold=0.5,
            alert_context_trend_lookback=4,
        ),
    )

    base_obs, _ = base_env.reset(start_idx=10)
    alert_obs, _ = alert_env.reset(start_idx=10)
    alert_tail = alert_obs[-20:]

    assert alert_env.alert_context_feature_dim == 20
    assert alert_obs.shape[0] == base_obs.shape[0] + 20 - 1
    np.testing.assert_allclose(
        alert_tail[:3],
        truth.loc[10, ["agent_context_particle_alert", "agent_context_flux_alert", "agent_context_thermal_alert"]].to_numpy(dtype=float),
        atol=1e-7,
    )
    assert alert_tail[3:6].tolist() == [1.0, 0.0, 0.0]
    assert alert_tail[6:10].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_custom_ppo_short_run(tmp_path: Path) -> None:
    truth = _truth(64)
    sensors = _sensors()
    constraints = PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0)
    candidate_masks = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=constraints,
        env_cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=12,
            seed=3,
        ),
        oracle=_oracle(truth),
        candidate_masks=candidate_masks,
        cfg=CustomPPOConfig(
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            embed_dim=8,
            hidden_dim=16,
            greedy_lookahead_steps=2,
            forecast_value_aux_coef=0.05,
            forecast_value_aux_stride=2,
            forecast_value_aux_lookahead_steps=1,
            forecast_value_head_enabled=True,
            forecast_value_head_hidden_dim=16,
            device="cpu",
            seed=5,
        ),
    )

    trainer.train()
    trainer.save(tmp_path / "custom_ppo.pt")
    saved = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
    with torch.no_grad():
        next(trainer.model.parameters()).add_(1.0)
    trainer.load_policy_checkpoint(tmp_path / "custom_ppo.pt")

    assert trainer.history
    assert np.isfinite(float(trainer.history[-1]["loss"]))
    assert np.isfinite(float(trainer.history[-1]["forecast_value_aux_loss"]))
    assert float(trainer.history[-1]["forecast_value_label_rate"]) == 0.5
    assert (tmp_path / "custom_ppo.pt").exists()
    assert all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in saved.items())


def test_custom_ppo_short_run_with_soft_forecast_value_auxiliary(tmp_path: Path) -> None:
    truth = _truth(64)
    sensors = _sensors()
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        env_cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=12,
            seed=3,
        ),
        oracle=_oracle(truth),
        candidate_masks=np.asarray(
            [[True, False, False], [False, True, False], [True, False, True]], dtype=bool
        ),
        cfg=CustomPPOConfig(
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            embed_dim=8,
            hidden_dim=16,
            greedy_lookahead_steps=2,
            forecast_value_aux_coef=0.5,
            forecast_value_aux_stride=2,
            forecast_value_aux_lookahead_steps=1,
            forecast_value_aux_loss="soft_ce",
            forecast_value_aux_temperature=1.0,
            device="cpu",
            seed=6,
        ),
    )

    trainer.train()

    assert np.isfinite(float(trainer.history[-1]["forecast_value_aux_loss"]))
    assert float(trainer.history[-1]["forecast_value_label_rate"]) == 0.5


def test_masked_soft_target_cross_entropy_respects_preferences_and_logit_shifts() -> None:
    targets = torch.tensor([[2.0, 0.0, -8.0]], dtype=torch.float32)
    feasible = torch.tensor([[True, True, False]])
    rows = torch.tensor([True])
    aligned = torch.tensor([[3.0, -1.0, 100.0]], dtype=torch.float32)
    reversed_logits = torch.tensor([[-1.0, 3.0, -100.0]], dtype=torch.float32)

    aligned_loss = masked_soft_target_cross_entropy(
        aligned, targets, feasible, rows, temperature=0.5
    )
    shifted_loss = masked_soft_target_cross_entropy(
        aligned + 17.0, targets, feasible, rows, temperature=0.5
    )
    reversed_loss = masked_soft_target_cross_entropy(
        reversed_logits, targets, feasible, rows, temperature=0.5
    )

    assert torch.allclose(aligned_loss, shifted_loss, atol=1.0e-6)
    assert aligned_loss < reversed_loss


def test_custom_ppo_samples_only_inside_configured_training_partition() -> None:
    truth = _truth(64)
    sensors = _sensors()
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=sensors,
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        env_cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=8,
            seed=3,
        ),
        oracle=_oracle(truth),
        candidate_masks=np.asarray([[True, False, False]], dtype=bool),
        cfg=CustomPPOConfig(
            total_timesteps=8,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            embed_dim=8,
            hidden_dim=16,
            device="cpu",
            seed=5,
            train_start_min=20,
            train_start_max=25,
        ),
    )

    starts = [trainer._sample_start_idx(8, seed_offset=idx) for idx in range(20)]

    assert all(20 <= start <= 25 for start in starts)


def test_ppo_actor_inputs_use_online_alert_in_rollout_and_teacher_batch() -> None:
    truth = _truth(32)
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=_sensors(),
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        env_cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=4,
            seed=3,
            include_event_flag_in_state=False,
            agent_context_columns=(
                "agent_context_particle_alert",
                "agent_context_flux_alert",
                "agent_context_thermal_alert",
            ),
        ),
        oracle=_oracle(truth),
        candidate_masks=np.asarray([[True, False, False]], dtype=bool),
        cfg=CustomPPOConfig(
            total_timesteps=4,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            embed_dim=8,
            hidden_dim=16,
            device="cpu",
            seed=5,
            train_start_min=20,
            train_start_max=20,
        ),
    )

    rollout = trainer.collect_rollout(4)
    teacher = trainer.collect_teacher_batch(4)
    expected_online = truth.loc[20:23, [
        "agent_context_particle_alert",
        "agent_context_flux_alert",
        "agent_context_thermal_alert",
    ]].max(axis=1).to_numpy(dtype=np.float32)
    exact_labels = truth.loc[20:23, "event_flag"].to_numpy(dtype=np.float32)

    np.testing.assert_allclose(rollout["event_flags"], expected_online, atol=1e-7)
    np.testing.assert_allclose(teacher["event_flags"], expected_online, atol=1e-7)
    assert not np.allclose(expected_online, exact_labels)


def test_awbc_coefficient_can_decay_to_zero_without_changing_default() -> None:
    constant = CustomPPO.__new__(CustomPPO)
    constant.cfg = CustomPPOConfig(awbc_coef=0.2)
    assert constant._effective_awbc_coef(0) == 0.2
    assert constant._effective_awbc_coef(50_000) == 0.2

    decayed = CustomPPO.__new__(CustomPPO)
    decayed.cfg = CustomPPOConfig(awbc_coef=0.2, awbc_decay_timesteps=10_000)
    assert decayed._effective_awbc_coef(0) == 0.2
    assert decayed._effective_awbc_coef(5_000) == 0.1
    assert decayed._effective_awbc_coef(10_000) == 0.0
    assert decayed._effective_awbc_coef(20_000) == 0.0


def test_awbc_event_only_filters_calm_labels() -> None:
    trainer = CustomPPO.__new__(CustomPPO)
    trainer.cfg = CustomPPOConfig(awbc_event_only=True)
    assert not trainer._awbc_label_allowed(0, 1.0)
    assert not trainer._awbc_label_allowed(1, 0.0)
    assert trainer._awbc_label_allowed(1, 1.0)

    trainer.cfg = CustomPPOConfig(awbc_event_only=False)
    assert trainer._awbc_label_allowed(0, 0.0)


def test_train_update_callback_receives_completed_updates() -> None:
    truth = _truth(32)
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=_sensors(),
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        env_cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=2,
            seed=3,
        ),
        oracle=_oracle(truth),
        candidate_masks=np.asarray([[True, False, False]], dtype=bool),
        cfg=CustomPPOConfig(
            total_timesteps=2,
            n_steps=1,
            batch_size=1,
            n_epochs=1,
            embed_dim=8,
            hidden_dim=16,
            device="cpu",
            seed=5,
            train_start_min=20,
            train_start_max=20,
        ),
    )
    observed: list[tuple[int, int]] = []
    trainer.train(on_update=lambda _trainer, update, steps, _metrics: observed.append((update, steps)))
    assert observed == [(1, 1), (2, 2)]


def test_train_update_callback_includes_bc_checkpoint_at_step_zero() -> None:
    truth = _truth(32)
    trainer = CustomPPO(
        truth_df=truth,
        sensor_specs=_sensors(),
        constraints=PowerConstraintsV2(max_active=3, per_step_budget=1.7, startup_peak_budget=2.0),
        env_cfg=WarmupEnvConfig(
            state_columns=STATE_COLUMNS,
            reward_target_columns=STATE_COLUMNS,
            lookback=4,
            episode_len=2,
            seed=3,
        ),
        oracle=_oracle(truth),
        candidate_masks=np.asarray([[True, False, False]], dtype=bool),
        cfg=CustomPPOConfig(
            total_timesteps=1,
            n_steps=1,
            batch_size=1,
            n_epochs=1,
            bc_pretrain_steps=2,
            bc_pretrain_epochs=1,
            embed_dim=8,
            hidden_dim=16,
            device="cpu",
            seed=5,
            train_start_min=20,
            train_start_max=20,
        ),
    )
    observed: list[tuple[int, int]] = []
    trainer.train(on_update=lambda _trainer, update, steps, _metrics: observed.append((update, steps)))
    assert observed == [(0, 0), (1, 1)]


def test_subtype_action_event_only_excludes_calm_samples() -> None:
    trainer = CustomPPO.__new__(CustomPPO)
    trainer.cfg = CustomPPOConfig(
        subtype_action_ce_coef=1.0,
        subtype_action_event_only=True,
        awbc_teacher_subtype_calm_action=0,
        awbc_teacher_subtype_particle_action=1,
        awbc_teacher_subtype_flux_action=2,
        awbc_teacher_subtype_thermal_action=3,
    )
    logits = torch.tensor([[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]])
    dist = torch.distributions.Categorical(logits=logits)
    masks = torch.ones((2, 4), dtype=torch.bool)
    labels = torch.tensor([0, 1], dtype=torch.long)
    valid = torch.ones(2)

    ce_loss, margin_loss, valid_rate = trainer._subtype_action_losses(dist, masks, labels, valid)

    expected = -torch.log_softmax(logits[1], dim=0)[1]
    assert torch.allclose(ce_loss, expected)
    assert float(margin_loss) == 0.0
    assert valid_rate == 0.5


def test_subtype_action_positive_inclusion_does_not_penalize_extra_sensors() -> None:
    trainer = CustomPPO.__new__(CustomPPO)
    trainer.cfg = CustomPPOConfig(
        subtype_action_ce_coef=1.0,
        subtype_action_supervision_mode="positive_sensor_inclusion",
        awbc_teacher_subtype_calm_action=0,
        awbc_teacher_subtype_particle_action=1,
        awbc_teacher_subtype_flux_action=2,
        awbc_teacher_subtype_thermal_action=3,
    )
    trainer.candidate_masks_t = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=torch.float32
    )
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.6, 0.1]], dtype=torch.float32))
    dist = torch.distributions.Categorical(logits=logits)
    feasible = torch.ones((1, 4), dtype=torch.bool)
    labels = torch.tensor([1], dtype=torch.long)
    valid = torch.ones(1)

    inclusion_loss, margin_loss, valid_rate = trainer._subtype_action_losses(
        dist, feasible, labels, valid
    )

    assert torch.allclose(inclusion_loss, -torch.log(torch.tensor(0.8)))
    assert float(margin_loss) == 0.0
    assert valid_rate == 1.0
