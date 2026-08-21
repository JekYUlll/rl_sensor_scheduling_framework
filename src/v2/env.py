from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from v2.power_projector import PowerConstraintsV2, PowerProjector
from v2.oracle import LinearFrozenForecastOracle, make_oracle_feature
from v2.sensor_spec import SensorSpecV2
from v2.warmup_state import SensorRuntime


@dataclass(frozen=True)
class WarmupEnvConfig:
    state_columns: tuple[str, ...]
    reward_target_columns: tuple[str, ...] | None = None
    reward_proxy_mode: str = "forecast"
    lookback: int = 20
    episode_len: int | None = None
    seed: int = 42
    base_freq_s: int = 1
    event_column: str = "event_flag"
    normalize_agent_state: bool = True
    normalization_mean: tuple[float, ...] | None = None
    normalization_std: tuple[float, ...] | None = None
    lambda_warmup_abort: float = 0.02
    lambda_switch: float = 0.002
    event_reward_multiplier: float = 1.0
    event_subtype_particle_reward_multiplier: float = 1.0
    event_subtype_flux_reward_multiplier: float = 1.0
    event_subtype_thermal_reward_multiplier: float = 1.0
    oracle_loss_reward_normalizers: tuple[float, float, float] | None = None
    oracle_loss_reward_default_normalizer: float = 1.0
    energy_account_enabled: bool = False
    energy_capacity: float = 0.0
    initial_energy: float = 0.0
    harvest_per_step: float = 0.0
    reserve_energy: float = 0.0
    lambda_energy_deficit: float = 1.0
    soc_soft_penalty_buffer: float = 0.0
    lambda_soc_soft_penalty: float = 0.0
    lambda_duty_balance: float = 0.0
    duty_balance_low: float = 0.05
    duty_balance_high: float = 0.95
    duty_balance_grace_steps: int = 64
    duty_score_feedback: float = 0.0
    duty_score_target: float = 0.40
    duty_hard_guard: bool = False
    duty_hard_low: float = 0.08
    duty_hard_high: float = 0.92
    duty_hard_score: float = 8.0
    min_dwell_steps: int = 1
    common_random_numbers: bool = False
    include_agent_cycle_phase: bool = False
    agent_cycle_period_steps: int = 0
    agent_cycle_dwell_steps: int = 1
    include_observable_regime_belief: bool = False
    regime_belief_lookback: int = 6
    agent_context_columns: tuple[str, ...] = ()
    include_event_flag_in_state: bool = True
    include_alert_context_features: bool = False
    alert_context_columns: tuple[str, ...] = (
        "agent_context_particle_alert",
        "agent_context_flux_alert",
        "agent_context_thermal_alert",
    )
    alert_context_threshold: float = 0.5
    alert_context_trend_lookback: int = 6
    uncertainty_process_variance: tuple[float, ...] | None = None
    uncertainty_initial_variance: float = 1.0
    uncertainty_max_variance: float = 25.0


class WarmupSchedulingEnv:
    """Minimal v2 Gym-like environment for warmup-aware sensor scheduling."""

    def __init__(
        self,
        truth_df: pd.DataFrame,
        sensor_specs: list[SensorSpecV2],
        constraints: PowerConstraintsV2,
        cfg: WarmupEnvConfig,
        oracle: LinearFrozenForecastOracle | None = None,
    ) -> None:
        missing = [col for col in cfg.state_columns if col not in truth_df.columns]
        if missing:
            raise ValueError(f"truth_df is missing state columns: {missing}")
        missing_context = [col for col in cfg.agent_context_columns if col not in truth_df.columns]
        if missing_context:
            raise ValueError(f"truth_df is missing agent context columns: {missing_context}")
        missing_alert_context = [
            col
            for col in cfg.alert_context_columns
            if bool(cfg.include_alert_context_features) and col not in truth_df.columns
        ]
        if missing_alert_context:
            raise ValueError(f"truth_df is missing alert context columns: {missing_alert_context}")
        self.truth_df = truth_df.reset_index(drop=True)
        self.sensor_specs = list(sensor_specs)
        self.sensor_ids = tuple(spec.sensor_id for spec in self.sensor_specs)
        self.cfg = cfg
        self.oracle = oracle
        self.rng = np.random.default_rng(int(cfg.seed))
        self.projector = PowerProjector(self.sensor_specs, constraints)
        self.action_to_sensor_mask = self._build_action_to_sensor_mask()
        self.runtimes = {spec.sensor_id: SensorRuntime(spec) for spec in self.sensor_specs}
        self.state_columns = tuple(cfg.state_columns)
        self.state_index = {name: idx for idx, name in enumerate(self.state_columns)}
        configured_targets = tuple(cfg.reward_target_columns or self.state_columns)
        missing_targets = [name for name in configured_targets if name not in self.state_index]
        if missing_targets:
            raise ValueError(f"reward_target_columns are not in state_columns: {missing_targets}")
        self.reward_target_columns = configured_targets
        self.reward_target_indices = np.asarray([self.state_index[name] for name in self.reward_target_columns], dtype=int)
        self.truth_values = self.truth_df[list(self.state_columns)].to_numpy(dtype=float)
        self.agent_context_columns = tuple(str(col) for col in cfg.agent_context_columns)
        self.agent_context_values = (
            self.truth_df[list(self.agent_context_columns)].to_numpy(dtype=float)
            if self.agent_context_columns
            else np.zeros((len(self.truth_df), 0), dtype=float)
        )
        self.alert_context_columns = tuple(str(col) for col in cfg.alert_context_columns)
        self.alert_context_values = (
            self.truth_df[list(self.alert_context_columns)].to_numpy(dtype=float)
            if bool(cfg.include_alert_context_features)
            else np.zeros((len(self.truth_df), 0), dtype=float)
        )
        # scores, binary flags, calm/particle/flux/thermal one-hot, max confidence,
        # elapsed alert age, rolling trends, previous specialist one-hot, dwell remainder.
        self.alert_context_feature_dim = 20 if bool(cfg.include_alert_context_features) else 0
        state_mean = np.mean(self.truth_values, axis=0)
        state_std = np.maximum(np.std(self.truth_values, axis=0), 1e-6)
        if cfg.normalization_mean is not None or cfg.normalization_std is not None:
            if cfg.normalization_mean is None or cfg.normalization_std is None:
                raise ValueError("normalization_mean and normalization_std must be configured together")
            state_mean = np.asarray(cfg.normalization_mean, dtype=float).reshape(-1)
            state_std = np.asarray(cfg.normalization_std, dtype=float).reshape(-1)
            if state_mean.shape[0] != len(self.state_columns) or state_std.shape[0] != len(self.state_columns):
                raise ValueError("normalization statistics must contain one value per state column")
            if np.any(~np.isfinite(state_mean)) or np.any(~np.isfinite(state_std)) or np.any(state_std <= 0):
                raise ValueError("normalization statistics must be finite and standard deviations positive")
        self.state_mean = state_mean
        self.state_std = np.maximum(state_std, 1e-6)
        if cfg.uncertainty_process_variance is None:
            process_variance = np.full(len(self.state_columns), 0.01, dtype=float)
        else:
            process_variance = np.asarray(cfg.uncertainty_process_variance, dtype=float).reshape(-1)
            if process_variance.shape[0] != len(self.state_columns):
                raise ValueError("uncertainty_process_variance must contain one value per state column")
            if np.any(~np.isfinite(process_variance)) or np.any(process_variance < 0.0):
                raise ValueError("uncertainty_process_variance must be finite and non-negative")
        self.uncertainty_process_variance = np.maximum(process_variance, 1e-8)
        self.posterior_variance = np.full(
            len(self.state_columns),
            max(float(cfg.uncertainty_initial_variance), 1e-8),
            dtype=float,
        )
        self.event_flags = (
            self.truth_df[cfg.event_column].astype(bool).to_numpy()
            if cfg.event_column in self.truth_df.columns
            else np.zeros(len(self.truth_df), dtype=bool)
        )
        self.event_subtype_ids = (
            self.truth_df["event_subtype_id"].astype(int).to_numpy()
            if "event_subtype_id" in self.truth_df.columns
            else np.zeros(len(self.truth_df), dtype=int)
        )
        self.episode_len = int(cfg.episode_len or len(self.truth_df))
        self.episode_start_idx = 0
        self.episode_end_idx = min(len(self.truth_df), self.episode_len)
        self.current_idx = 0
        self.last_observation = np.array(self.state_mean, dtype=float, copy=True)
        self.observed_mask = np.zeros(len(self.state_columns), dtype=float)
        self.history = np.repeat(self.state_mean.reshape(1, -1), int(cfg.lookback), axis=0).astype(float)
        self.mask_history = np.zeros_like(self.history)
        self.previous_action_mask = np.zeros(len(self.sensor_specs), dtype=float)
        self.sensor_on_counts = np.zeros(len(self.sensor_specs), dtype=float)
        self.elapsed_steps = 0
        self.dwell_hold_remaining = 0
        self.current_energy = float(cfg.initial_energy if cfg.energy_account_enabled else 0.0)
        self.energy_deficit_steps = 0
        self.energy_deficit_total = 0.0
        self.last_info: dict[str, object] = {}

    def reset(self, *, start_idx: int = 0) -> tuple[np.ndarray, dict[str, object]]:
        if start_idx < 0 or start_idx >= len(self.truth_df):
            raise ValueError(f"start_idx out of range: {start_idx}")
        self.current_idx = int(start_idx)
        self.episode_start_idx = int(start_idx)
        self.episode_end_idx = min(len(self.truth_df), self.episode_start_idx + self.episode_len)
        for runtime in self.runtimes.values():
            runtime.reset()
        self.last_observation = np.array(self.state_mean, dtype=float, copy=True)
        self.observed_mask = np.zeros(len(self.state_columns), dtype=float)
        self.history = np.repeat(self.state_mean.reshape(1, -1), int(self.cfg.lookback), axis=0).astype(float)
        self.mask_history = np.zeros_like(self.history)
        self.posterior_variance = np.full(
            len(self.state_columns),
            max(float(self.cfg.uncertainty_initial_variance), 1e-8),
            dtype=float,
        )
        self.previous_action_mask = np.zeros(len(self.sensor_specs), dtype=float)
        self.sensor_on_counts = np.zeros(len(self.sensor_specs), dtype=float)
        self.elapsed_steps = 0
        self.dwell_hold_remaining = 0
        self.current_energy = self._initial_energy()
        self.energy_deficit_steps = 0
        self.energy_deficit_total = 0.0
        self.last_info = {
            "power": 0.0,
            "peak_power": 0.0,
            "event": bool(self.event_flags[self.current_idx]),
            "soc": float(self.current_energy),
            "soc_ratio": self._soc_ratio(),
            "energy_deficit": 0.0,
            "energy_deficit_steps": 0,
            "duty_balance_penalty": 0.0,
            "sensor_duty_estimate": self._sensor_duty_estimate().tolist(),
            "min_dwell_steps": int(max(1, int(self.cfg.min_dwell_steps))),
            "dwell_hold_remaining": int(self.dwell_hold_remaining),
            "dwell_hold_applied": 0,
        }
        return self._state(), dict(self.last_info)

    def step_scores(self, scores: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        projection = self.projector.project_scores(self._duty_adjusted_scores(scores), self.runtimes)
        return self._step_projection(projection.selected_mask)

    def step_mask(self, desired_mask: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        mask = np.asarray(desired_mask, dtype=bool).reshape(-1)
        if self._duty_score_feedback_enabled():
            scores = np.where(mask, 1.0, -1.0)
            projection = self.projector.project_scores(self._duty_adjusted_scores(scores), self.runtimes)
        else:
            projection = self.projector.project_mask(mask, self.runtimes)
        return self._step_projection(projection.selected_mask)

    def get_feasible_actions(self) -> list[int]:
        """Return action indices whose sensor subset is feasible at the current warmup state."""
        feasible: list[int] = []
        for action_idx, mask in self.action_to_sensor_mask.items():
            indices = [int(idx) for idx in np.flatnonzero(mask)]
            if self.projector._is_feasible(indices, self.runtimes):
                feasible.append(int(action_idx))
        return feasible

    def _step_projection(self, selected_mask: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        selected_mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        selected_mask, dwell_hold_applied = self._apply_min_dwell_guard(selected_mask)
        selected_mask, energy_guard_dropped = self._apply_energy_guard(selected_mask)
        previous_action_mask = np.asarray(self.previous_action_mask, dtype=float).reshape(-1)
        self._update_min_dwell_state(selected_mask, previous_action_mask)
        abort_count_before = int(sum(rt.warmup_abort_count for rt in self.runtimes.values()))
        selected_ids = {self.sensor_specs[idx].sensor_id for idx in np.flatnonzero(selected_mask)}
        is_event = bool(self.event_flags[self.current_idx])
        event_subtype_id = int(self.event_subtype_ids[self.current_idx]) if is_event else 0
        statuses = {}
        for spec in self.sensor_specs:
            statuses[spec.sensor_id] = self.runtimes[spec.sensor_id].begin_step(
                spec.sensor_id in selected_ids,
                step_idx=self.current_idx,
            )

        truth = self.truth_values[self.current_idx]
        observed_this_step = np.zeros(len(self.state_columns), dtype=float)
        observation_variance = np.full(len(self.state_columns), np.inf, dtype=float)
        prior_variance = np.minimum(
            self.posterior_variance + self.uncertainty_process_variance,
            max(float(self.cfg.uncertainty_max_variance), 1e-8),
        )
        observation = np.array(self.last_observation, dtype=float, copy=True)
        common_draws = (
            self._draw_common_observation_randomness(
                is_event=is_event,
                event_subtype_id=event_subtype_id,
            )
            if bool(self.cfg.common_random_numbers)
            else None
        )
        observation_candidates: dict[str, list[tuple[float, float]]] = {}
        for spec in self.sensor_specs:
            runtime = self.runtimes[spec.sensor_id]
            if not runtime.can_observe(self.current_idx):
                continue
            for variable in spec.observed_variables:
                if variable not in self.state_index:
                    continue
                noise_std = self._observation_noise_std(
                    spec,
                    variable,
                    is_event=is_event,
                    event_subtype_id=event_subtype_id,
                )
                if common_draws is None:
                    if not self._variable_available(
                        spec, variable, is_event=is_event, event_subtype_id=event_subtype_id
                    ):
                        continue
                    noise = self.rng.normal(0.0, noise_std)
                else:
                    available, noise = common_draws[
                        (str(spec.sensor_id), str(variable))
                    ]
                    if not available:
                        continue
                col_idx = self.state_index[variable]
                value = float(truth[col_idx] + noise)
                observation_candidates.setdefault(str(variable), []).append((value, float(noise_std)))
            runtime.mark_observed(self.current_idx)

        for variable, candidates in observation_candidates.items():
            if variable not in self.state_index:
                continue
            values = np.asarray([value for value, _ in candidates], dtype=float)
            noise_stds = np.asarray([std for _, std in candidates], dtype=float)
            weights = 1.0 / np.maximum(np.square(noise_stds), 1.0e-9)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                fused_value = float(np.mean(values))
            elif variable == "wind_direction_deg":
                angles = np.deg2rad(values)
                sin_mean = float(np.sum(weights * np.sin(angles)) / weight_sum)
                cos_mean = float(np.sum(weights * np.cos(angles)) / weight_sum)
                fused_value = float((np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)
            else:
                fused_value = float(np.sum(weights * values) / weight_sum)
            col_idx = self.state_index[variable]
            observation[col_idx] = fused_value
            observed_this_step[col_idx] = 1.0
            physical_variance = float(
                1.0 / np.sum(1.0 / np.maximum(np.square(noise_stds), 1.0e-12))
            )
            observation_variance[col_idx] = physical_variance / max(float(self.state_std[col_idx]) ** 2, 1.0e-12)
            if variable == "wind_direction_deg":
                theta = np.deg2rad(fused_value)
                angular_variance = physical_variance * (np.pi / 180.0) ** 2
                if "wind_dir_sin" in self.state_index:
                    sin_idx = self.state_index["wind_dir_sin"]
                    observation[sin_idx] = np.sin(theta)
                    observed_this_step[sin_idx] = 1.0
                    observation_variance[sin_idx] = angular_variance / max(
                        float(self.state_std[sin_idx]) ** 2,
                        1.0e-12,
                    )
                if "wind_dir_cos" in self.state_index:
                    cos_idx = self.state_index["wind_dir_cos"]
                    observation[cos_idx] = np.cos(theta)
                    observed_this_step[cos_idx] = 1.0
                    observation_variance[cos_idx] = angular_variance / max(
                        float(self.state_std[cos_idx]) ** 2,
                        1.0e-12,
                    )

        observed_indices = np.flatnonzero(np.isfinite(observation_variance))
        posterior_variance = np.array(prior_variance, copy=True)
        if observed_indices.size:
            prior_precision = 1.0 / np.maximum(prior_variance[observed_indices], 1.0e-12)
            measurement_precision = 1.0 / np.maximum(observation_variance[observed_indices], 1.0e-12)
            posterior_variance[observed_indices] = 1.0 / (prior_precision + measurement_precision)
        self.posterior_variance = np.minimum(
            posterior_variance,
            max(float(self.cfg.uncertainty_max_variance), 1e-8),
        )

        for spec in self.sensor_specs:
            self.runtimes[spec.sensor_id].end_step(spec.sensor_id in selected_ids)
        mode_ids_after = {
            spec.sensor_id: int(self.runtimes[spec.sensor_id].mode)
            for spec in self.sensor_specs
        }

        self.last_observation = observation
        self.observed_mask = observed_this_step
        self.history = np.vstack([self.history[1:], observation.reshape(1, -1)])
        self.mask_history = np.vstack([self.mask_history[1:], observed_this_step.reshape(1, -1)])
        self.previous_action_mask = selected_mask.astype(float)

        steady_power = float(sum(float(status["power_cost"]) for status in statuses.values()))
        peak_power = float(sum(float(status["peak_power"]) for status in statuses.values()))
        energy_before = float(self.current_energy)
        energy_harvest = self._energy_harvest()
        energy_after_unclipped = energy_before + energy_harvest - steady_power
        energy_deficit = max(0.0, float(self.cfg.reserve_energy) - energy_after_unclipped) if self._energy_enabled() else 0.0
        if self._energy_enabled():
            self.current_energy = float(np.clip(energy_after_unclipped, 0.0, max(float(self.cfg.energy_capacity), 1e-6)))
            if energy_deficit > 1e-12:
                self.energy_deficit_steps += 1
                self.energy_deficit_total += float(energy_deficit)
        soc_soft_penalty = self._soc_soft_penalty()
        error = float(np.mean(np.abs(observation - truth)))
        oracle_loss = self._oracle_loss()
        abort_count_after = int(sum(rt.warmup_abort_count for rt in self.runtimes.values()))
        warmup_abort_delta = max(0, abort_count_after - abort_count_before)
        switch_rate = float(np.mean(np.abs(selected_mask.astype(float) - previous_action_mask)))
        projected_on_counts = self.sensor_on_counts + selected_mask.astype(float)
        projected_elapsed_steps = int(self.elapsed_steps) + 1
        duty_balance_penalty = self._duty_balance_penalty(
            projected_on_counts,
            elapsed_steps=projected_elapsed_steps,
        )
        sensor_duty_estimate = self._sensor_duty_estimate(
            projected_on_counts,
            elapsed_steps=projected_elapsed_steps,
        )
        event_multiplier = self._event_loss_multiplier(
            is_event=is_event,
            event_subtype_id=event_subtype_id,
        )
        raw_reward_loss, reward_loss = self._training_reward_loss(
            oracle_loss=oracle_loss,
            instant_error=error,
            event_subtype_id=event_subtype_id,
        )
        base_loss = float(event_multiplier * float(reward_loss))
        shaping_penalty = (
            float(self.cfg.lambda_warmup_abort) * float(warmup_abort_delta)
            + float(self.cfg.lambda_switch) * switch_rate
            + float(self.cfg.lambda_energy_deficit) * float(energy_deficit)
            + float(soc_soft_penalty)
            + float(self.cfg.lambda_duty_balance) * float(duty_balance_penalty)
        )
        reward = -float(base_loss + shaping_penalty)
        self.sensor_on_counts = projected_on_counts
        self.elapsed_steps = projected_elapsed_steps

        done = self.current_idx >= (self.episode_end_idx - 1)
        info = {
            "selected_sensor_ids": tuple(sorted(selected_ids)),
            "selected_mask": selected_mask.astype(int).tolist(),
            "sensor_status": statuses,
            "mode_ids_after_step": mode_ids_after,
            "power": steady_power,
            "peak_power": peak_power,
            "soc": float(self.current_energy),
            "soc_ratio": self._soc_ratio(),
            "energy_before": float(energy_before),
            "energy_harvest": float(energy_harvest),
            "energy_deficit": float(energy_deficit),
            "energy_deficit_steps": int(self.energy_deficit_steps),
            "energy_deficit_total": float(self.energy_deficit_total),
            "soc_soft_penalty": float(soc_soft_penalty),
            "energy_guard_dropped": int(energy_guard_dropped),
            "event": is_event,
            "event_subtype_id": int(event_subtype_id),
            "event_loss_multiplier": float(event_multiplier),
            "warmup_abort_count": abort_count_after,
            "warmup_abort_delta": int(warmup_abort_delta),
            "switch_rate": float(switch_rate),
            "duty_balance_penalty": float(duty_balance_penalty),
            "sensor_duty_estimate": sensor_duty_estimate.tolist(),
            "min_dwell_steps": int(max(1, int(self.cfg.min_dwell_steps))),
            "dwell_hold_remaining": int(self.dwell_hold_remaining),
            "dwell_hold_applied": int(dwell_hold_applied),
            "shaping_penalty": float(shaping_penalty),
            "instant_abs_error": error,
            "oracle_loss": float(oracle_loss) if oracle_loss is not None else float("nan"),
            "oracle_loss_reward": float(reward_loss),
            "reward_proxy_mode": str(self.cfg.reward_proxy_mode),
            "reward_proxy_loss": float(raw_reward_loss),
            "uncertainty_proxy_mean": float(self._target_uncertainty_loss()),
        }
        self.last_info = info
        if not done:
            self.current_idx += 1
        return self._state(), reward, done, info

    def _training_reward_loss(
        self,
        *,
        oracle_loss: float | None,
        instant_error: float,
        event_subtype_id: int,
    ) -> tuple[float, float]:
        mode = str(self.cfg.reward_proxy_mode or "forecast")
        if mode == "forecast":
            raw_loss = float(oracle_loss if oracle_loss is not None else instant_error)
            shaped_loss = (
                self._reward_oracle_loss(raw_loss, event_subtype_id=event_subtype_id)
                if oracle_loss is not None
                else raw_loss
            )
            return raw_loss, float(shaped_loss)
        if mode == "aoi":
            raw_loss = self._target_aoi_loss()
            return raw_loss, raw_loss
        if mode == "coverage":
            raw_loss = self._target_coverage_loss()
            return raw_loss, raw_loss
        if mode == "uncertainty":
            raw_loss = self._target_uncertainty_loss()
            return raw_loss, raw_loss
        if mode == "instant_error":
            raw_loss = float(instant_error)
            return raw_loss, raw_loss
        raise ValueError(f"Unsupported reward_proxy_mode={mode!r}")

    def _target_aoi_loss(self) -> float:
        masks = np.asarray(self.mask_history[:, self.reward_target_indices], dtype=float)
        if masks.size == 0:
            return 1.0
        lookback = int(masks.shape[0])
        denom = float(max(lookback - 1, 1))
        ages = []
        for col_idx in range(int(masks.shape[1])):
            hits = np.flatnonzero(masks[:, col_idx] > 0.5)
            if hits.size == 0:
                ages.append(1.0)
            else:
                ages.append(float(lookback - 1 - int(hits[-1])) / denom)
        return float(np.clip(np.mean(ages), 0.0, 1.0))

    def _target_coverage_loss(self) -> float:
        masks = np.asarray(self.mask_history[:, self.reward_target_indices], dtype=float)
        if masks.size == 0:
            return 1.0
        coverage = float(np.mean(masks > 0.5))
        return float(np.clip(1.0 - coverage, 0.0, 1.0))

    def _target_uncertainty_loss(self) -> float:
        variance = np.asarray(self.posterior_variance[self.reward_target_indices], dtype=float)
        if variance.size == 0:
            return 1.0
        bounded = variance / (1.0 + np.maximum(variance, 0.0))
        return float(np.clip(np.mean(bounded), 0.0, 1.0))

    def online_event_context(self) -> float:
        """Prefer an online alert proxy, with the exact event flag as a legacy fallback."""
        if self.agent_context_values.shape[1] > 0:
            context = np.nan_to_num(
                self.agent_context_values[int(self.current_idx)],
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            if "agent_context_event_alert" in self.agent_context_columns:
                idx = self.agent_context_columns.index("agent_context_event_alert")
                return float(np.clip(context[idx], 0.0, 1.0))
            alert_indices = [
                idx
                for idx, name in enumerate(self.agent_context_columns)
                if str(name).endswith("_alert")
            ]
            if alert_indices:
                return float(np.clip(np.max(context[alert_indices]), 0.0, 1.0))
        if bool(self.cfg.include_event_flag_in_state):
            return float(bool(self.event_flags[int(self.current_idx)]))
        return 0.0

    def _event_loss_multiplier(self, *, is_event: bool, event_subtype_id: int) -> float:
        if not bool(is_event):
            return 1.0
        base = max(0.0, float(self.cfg.event_reward_multiplier))
        subtype = int(event_subtype_id)
        if subtype == 1:
            return float(base * max(0.0, float(self.cfg.event_subtype_particle_reward_multiplier)))
        if subtype == 2:
            return float(base * max(0.0, float(self.cfg.event_subtype_flux_reward_multiplier)))
        if subtype == 3:
            return float(base * max(0.0, float(self.cfg.event_subtype_thermal_reward_multiplier)))
        return float(base)

    def _reward_oracle_loss(self, loss: float, *, event_subtype_id: int) -> float:
        if not np.isfinite(float(loss)):
            return float(loss)
        denom = float(self.cfg.oracle_loss_reward_default_normalizer)
        normalizers = self.cfg.oracle_loss_reward_normalizers
        subtype = int(event_subtype_id)
        if normalizers is not None and subtype in (1, 2, 3):
            values = np.asarray(normalizers, dtype=float).reshape(-1)
            if values.size >= subtype:
                denom = float(values[subtype - 1])
        if not np.isfinite(denom) or denom <= 0.0:
            return float(loss)
        return float(loss) / denom

    def _apply_min_dwell_guard(self, desired_mask: np.ndarray) -> tuple[np.ndarray, int]:
        dwell = max(1, int(self.cfg.min_dwell_steps))
        desired = np.asarray(desired_mask, dtype=bool).reshape(-1).copy()
        if dwell <= 1 or int(self.elapsed_steps) <= 0 or int(self.dwell_hold_remaining) <= 0:
            return desired, 0
        previous = np.asarray(self.previous_action_mask, dtype=bool).reshape(-1)
        if previous.shape == desired.shape and not np.array_equal(previous, desired):
            return previous.copy(), 1
        return desired, 0

    def _update_min_dwell_state(self, selected_mask: np.ndarray, previous_action_mask: np.ndarray) -> None:
        dwell = max(1, int(self.cfg.min_dwell_steps))
        if dwell <= 1:
            self.dwell_hold_remaining = 0
            return
        selected = np.asarray(selected_mask, dtype=bool).reshape(-1)
        previous = np.asarray(previous_action_mask, dtype=bool).reshape(-1)
        if selected.shape == previous.shape and not np.array_equal(selected, previous):
            self.dwell_hold_remaining = dwell - 1
        elif int(self.dwell_hold_remaining) > 0:
            self.dwell_hold_remaining -= 1

    def _energy_enabled(self) -> bool:
        return bool(self.cfg.energy_account_enabled) and float(self.cfg.energy_capacity) > 0.0

    def _initial_energy(self) -> float:
        if not self._energy_enabled():
            return 0.0
        capacity = max(float(self.cfg.energy_capacity), 1e-6)
        initial = float(self.cfg.initial_energy)
        if initial <= 0.0:
            initial = capacity
        return float(np.clip(initial, 0.0, capacity))

    def _energy_harvest(self) -> float:
        return float(max(0.0, float(self.cfg.harvest_per_step))) if self._energy_enabled() else 0.0

    def _soc_ratio(self) -> float:
        if not self._energy_enabled():
            return 1.0
        return float(np.clip(self.current_energy / max(float(self.cfg.energy_capacity), 1e-6), 0.0, 1.0))

    def _soc_soft_penalty(self) -> float:
        if not self._energy_enabled():
            return 0.0
        penalty = max(0.0, float(self.cfg.lambda_soc_soft_penalty))
        if penalty <= 0.0:
            return 0.0
        threshold = float(self.cfg.reserve_energy) + max(0.0, float(self.cfg.soc_soft_penalty_buffer))
        return float(penalty if float(self.current_energy) < threshold else 0.0)

    def _sensor_duty_estimate(
        self,
        on_counts: np.ndarray | None = None,
        *,
        elapsed_steps: int | None = None,
    ) -> np.ndarray:
        counts = self.sensor_on_counts if on_counts is None else np.asarray(on_counts, dtype=float).reshape(-1)
        elapsed = int(self.elapsed_steps if elapsed_steps is None else elapsed_steps)
        if elapsed <= 0:
            return np.zeros(len(self.sensor_specs), dtype=float)
        return np.clip(counts / float(elapsed), 0.0, 1.0)

    def _duty_balance_penalty(self, on_counts: np.ndarray, *, elapsed_steps: int) -> float:
        weight = max(0.0, float(self.cfg.lambda_duty_balance))
        if weight <= 0.0:
            return 0.0
        elapsed = int(elapsed_steps)
        if elapsed < max(1, int(self.cfg.duty_balance_grace_steps)):
            return 0.0
        low = float(np.clip(float(self.cfg.duty_balance_low), 0.0, 1.0))
        high = float(np.clip(float(self.cfg.duty_balance_high), 0.0, 1.0))
        if high <= low:
            high = min(1.0, low + 1.0e-6)
        duty = self._sensor_duty_estimate(on_counts, elapsed_steps=elapsed)
        low_violation = np.maximum(low - duty, 0.0)
        high_violation = np.maximum(duty - high, 0.0)
        return float(np.mean(low_violation + high_violation))

    def _duty_adjusted_scores(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float).reshape(-1)
        if not self._duty_score_feedback_enabled():
            return scores_arr
        adjusted = np.array(scores_arr, dtype=float, copy=True)
        strength = max(0.0, float(self.cfg.duty_score_feedback))
        duty = self._sensor_duty_estimate()
        if strength > 0.0:
            target = float(np.clip(float(self.cfg.duty_score_target), 0.0, 1.0))
            adjusted = adjusted - strength * (duty - target)
        if bool(self.cfg.duty_hard_guard):
            low = float(np.clip(float(self.cfg.duty_hard_low), 0.0, 1.0))
            high = float(np.clip(float(self.cfg.duty_hard_high), 0.0, 1.0))
            if high <= low:
                high = min(1.0, low + 1.0e-6)
            force = max(0.0, float(self.cfg.duty_hard_score))
            if force > 0.0:
                adjusted = np.where(duty < low, np.maximum(adjusted, force), adjusted)
                adjusted = np.where(duty > high, np.minimum(adjusted, -force), adjusted)
        return adjusted

    def _duty_score_feedback_enabled(self) -> bool:
        return (
            (
                max(0.0, float(self.cfg.duty_score_feedback)) > 0.0
                or bool(self.cfg.duty_hard_guard)
            )
            and int(self.elapsed_steps) >= max(1, int(self.cfg.duty_balance_grace_steps))
        )

    def _apply_energy_guard(self, selected_mask: np.ndarray) -> tuple[np.ndarray, int]:
        mask = np.asarray(selected_mask, dtype=bool).reshape(-1).copy()
        if not self._energy_enabled():
            return mask, 0
        available = float(self.current_energy) + self._energy_harvest() - float(self.cfg.reserve_energy)
        required = {int(idx) for idx in self.projector.required_indices}

        def selected_power() -> float:
            return float(sum(float(self.sensor_specs[idx].power_cost) for idx in np.flatnonzero(mask)))

        dropped = 0
        while selected_power() > available + 1e-12:
            optional = [int(idx) for idx in np.flatnonzero(mask) if int(idx) not in required]
            if not optional:
                break
            # Drop the largest optional load first; this is a deterministic
            # account guard, not a learned scheduling heuristic.
            idx = max(optional, key=lambda item: float(self.sensor_specs[item].power_cost))
            mask[idx] = False
            dropped += 1
        return mask, dropped

    @staticmethod
    def _observation_noise_std(
        spec: SensorSpecV2,
        variable: str,
        *,
        is_event: bool,
        event_subtype_id: int = 0,
    ) -> float:
        base = float(spec.noise_std.get(variable, 0.0))
        if not is_event:
            if variable in spec.calm_noise_std:
                return max(0.0, float(spec.calm_noise_std[variable]))
            multiplier = float(spec.calm_noise_multiplier.get(variable, 1.0))
            return max(0.0, base * multiplier)
        subtype_noise = spec.event_subtype_noise_std.get(int(event_subtype_id), {})
        if variable in subtype_noise:
            return max(0.0, float(subtype_noise[variable]))
        if variable in spec.event_noise_std:
            return max(0.0, float(spec.event_noise_std[variable]))
        multiplier = float(spec.event_noise_multiplier.get(variable, 1.0))
        return max(0.0, base * multiplier)

    def _variable_available(
        self,
        spec: SensorSpecV2,
        variable: str,
        *,
        is_event: bool,
        event_subtype_id: int = 0,
    ) -> bool:
        if is_event:
            subtype_probability = spec.event_subtype_observation_probability.get(int(event_subtype_id), {})
            if variable in subtype_probability:
                probability = float(subtype_probability[variable])
            else:
                probability = float(spec.event_observation_probability.get(variable, 1.0))
        else:
            probability = float(spec.calm_observation_probability.get(variable, 1.0))
        probability = float(np.clip(probability, 0.0, 1.0))
        return bool(self.rng.random() <= probability)

    def _draw_common_observation_randomness(
        self,
        *,
        is_event: bool,
        event_subtype_id: int = 0,
    ) -> dict[tuple[str, str], tuple[bool, float]]:
        draws: dict[tuple[str, str], tuple[bool, float]] = {}
        for spec in self.sensor_specs:
            for variable in spec.observed_variables:
                if variable not in self.state_index:
                    continue
                available = self._variable_available(
                    spec,
                    variable,
                    is_event=is_event,
                    event_subtype_id=event_subtype_id,
                )
                noise_std = self._observation_noise_std(
                    spec,
                    variable,
                    is_event=is_event,
                    event_subtype_id=event_subtype_id,
                )
                noise = float(self.rng.normal(0.0, noise_std))
                draws[(str(spec.sensor_id), str(variable))] = (
                    available,
                    noise,
                )
        return draws

    def _oracle_loss(self) -> float | None:
        if self.oracle is None or not self.oracle.is_fitted:
            return None
        horizon = int(self.oracle.cfg.horizon)
        start = self.current_idx + 1
        end = start + horizon
        if end > len(self.truth_values):
            return None
        feature = make_oracle_feature(self.history, self.mask_history)
        future = self.truth_values[start:end]
        future = future[:, self.reward_target_indices]
        loss_with_context = getattr(self.oracle, "loss_with_context", None)
        if callable(loss_with_context):
            context = {
                "event_flag": bool(self.event_flags[self.current_idx]) if self.event_flags.size else False,
                "event_subtype_id": int(self.event_subtype_ids[self.current_idx]) if self.event_subtype_ids.size else 0,
                "time_index": int(self.current_idx),
            }
            return loss_with_context(feature, future, context=context)
        return self.oracle.loss(feature, future)

    def _observable_regime_belief_features(self) -> np.ndarray:
        span = max(1, min(int(self.cfg.regime_belief_lookback), int(self.history.shape[0])))
        hist_z = (self.history - self.state_mean.reshape(1, -1)) / self.state_std.reshape(1, -1)
        recent = hist_z[-span:]
        latest = recent[-1]
        masks = np.asarray(self.mask_history[-span:], dtype=float)

        def value(name: str) -> float:
            idx = self.state_index.get(str(name))
            if idx is None:
                return 0.0
            return float(np.clip(latest[int(idx)], -5.0, 5.0))

        def slope(name: str) -> float:
            idx = self.state_index.get(str(name))
            if idx is None or recent.shape[0] <= 1:
                return 0.0
            return float(np.clip(recent[-1, int(idx)] - recent[0, int(idx)], -5.0, 5.0))

        def coverage(names: tuple[str, ...]) -> float:
            idxs = [int(self.state_index[name]) for name in names if name in self.state_index]
            if not idxs:
                return 0.0
            return float(np.clip(np.max(np.mean(masks[:, idxs], axis=0)), 0.0, 1.0))

        wind = value("wind_speed_ms")
        flux = value("snow_mass_flux_kg_m2_s")
        particle_velocity = value("snow_particle_mean_velocity_ms")
        particle_diameter = value("snow_particle_mean_diameter_mm")
        surface_air_gap = value("air_temperature_c") - value("snow_surface_temperature_c")
        particle_signal = float(np.tanh(0.45 * particle_velocity + 0.25 * particle_diameter + 0.15 * flux))
        flux_signal = float(np.tanh(0.55 * flux + 0.25 * wind + 0.10 * slope("snow_mass_flux_kg_m2_s")))
        thermal_signal = float(np.tanh(0.45 * surface_air_gap - 0.10 * slope("snow_surface_temperature_c")))
        return np.asarray(
            [
                particle_signal,
                flux_signal,
                thermal_signal,
                float(np.clip(wind, -5.0, 5.0)),
                float(np.clip(flux, -5.0, 5.0)),
                float(np.clip(particle_velocity, -5.0, 5.0)),
                float(np.clip(surface_air_gap, -5.0, 5.0)),
                coverage(("snow_particle_mean_velocity_ms", "snow_particle_mean_diameter_mm")),
                coverage(("snow_mass_flux_kg_m2_s", "wind_speed_ms")),
                coverage(("snow_surface_temperature_c", "air_temperature_c")),
            ],
            dtype=float,
        )

    def _state(self) -> np.ndarray:
        if self.cfg.normalize_agent_state:
            history_for_agent = (self.history - self.state_mean.reshape(1, -1)) / self.state_std.reshape(1, -1)
        else:
            history_for_agent = self.history
        max_warm = max(1, max((int(spec.warmup_steps) for spec in self.sensor_specs), default=1))
        sensor_modes = np.asarray([self.runtimes[sid].mode for sid in self.sensor_ids], dtype=float) / 2.0
        warm_remaining = np.asarray([self.runtimes[sid].warm_remaining for sid in self.sensor_ids], dtype=float) / float(max_warm)
        freshness = np.asarray([self.runtimes[sid].freshness(self.current_idx) for sid in self.sensor_ids], dtype=float)
        sensor_duty = self._sensor_duty_estimate()
        duty_state = (
            [sensor_duty]
            if (
                float(self.cfg.lambda_duty_balance) > 0.0
                or float(self.cfg.duty_score_feedback) > 0.0
                or bool(self.cfg.duty_hard_guard)
            )
            else []
        )
        power_budget = self.projector.constraints.per_step_budget or 1.0
        power_ratio = float(self.last_info.get("power", 0.0)) / max(float(power_budget), 1e-6)
        seconds = float(self.current_idx * int(self.cfg.base_freq_s))
        theta = 2.0 * np.pi * ((seconds % 86400.0) / 86400.0)
        cycle_tail: list[float] = []
        if bool(self.cfg.include_agent_cycle_phase):
            elapsed = max(0, int(self.current_idx) - int(self.episode_start_idx))
            period = int(self.cfg.agent_cycle_period_steps)
            if period <= 0:
                period = max(1, int(self.episode_len))
            dwell = max(1, int(self.cfg.agent_cycle_dwell_steps))
            cycle_theta = 2.0 * np.pi * float(elapsed % period) / float(max(period, 1))
            dwell_theta = 2.0 * np.pi * float(elapsed % dwell) / float(max(dwell, 1))
            cycle_tail = [
                float(np.sin(cycle_theta)),
                float(np.cos(cycle_theta)),
                float(np.sin(dwell_theta)),
                float(np.cos(dwell_theta)),
            ]
        regime_tail = (
            self._observable_regime_belief_features().tolist()
            if bool(self.cfg.include_observable_regime_belief)
            else []
        )
        context_tail = (
            np.nan_to_num(
                self.agent_context_values[int(self.current_idx)],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).tolist()
            if self.agent_context_values.shape[1] > 0
            else []
        )
        event_tail = [float(self.event_flags[self.current_idx])] if bool(self.cfg.include_event_flag_in_state) else []
        alert_context_tail = (
            self._alert_context_features().tolist()
            if bool(self.cfg.include_alert_context_features)
            else []
        )
        tail = np.asarray(
            [
                power_ratio,
                np.sin(theta),
                np.cos(theta),
                *cycle_tail,
                *regime_tail,
                *context_tail,
                *event_tail,
                *([self._soc_ratio()] if self._energy_enabled() else []),
                *alert_context_tail,
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                history_for_agent.reshape(-1),
                self.mask_history.reshape(-1),
                sensor_modes,
                warm_remaining,
                freshness,
                self.previous_action_mask,
                *duty_state,
                tail,
            ]
        ).astype(float)

    def _alert_context_scores(self, idx: int) -> np.ndarray:
        if self.alert_context_values.shape[1] == 0:
            return np.zeros(3, dtype=float)
        idx = int(np.clip(int(idx), 0, max(0, len(self.alert_context_values) - 1)))
        values = np.nan_to_num(
            self.alert_context_values[idx],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(float)
        if values.size < 3:
            values = np.pad(values, (0, 3 - values.size), constant_values=0.0)
        return np.clip(values[:3], 0.0, 1.0)

    def _alert_context_features(self) -> np.ndarray:
        scores = self._alert_context_scores(int(self.current_idx))
        threshold = float(self.cfg.alert_context_threshold)
        flags = (scores >= threshold).astype(float)
        max_idx = int(np.argmax(scores)) if scores.size else 0
        max_conf = float(scores[max_idx]) if scores.size else 0.0
        argmax_one_hot = np.zeros(4, dtype=float)
        argmax_one_hot[0 if max_conf < threshold else max_idx + 1] = 1.0

        lookback = max(1, int(self.cfg.alert_context_trend_lookback))
        past_idx = max(0, int(self.current_idx) - lookback)
        trend = np.clip(scores - self._alert_context_scores(past_idx), -1.0, 1.0)

        alert_age = 0
        if max_conf >= threshold:
            start_idx = max(0, int(self.current_idx) - lookback)
            for idx in range(int(self.current_idx), start_idx - 1, -1):
                if float(np.max(self._alert_context_scores(idx))) < threshold:
                    break
                alert_age = int(self.current_idx) - int(idx)
        alert_age_norm = float(np.clip(alert_age / float(lookback), 0.0, 1.0))

        previous_specialist = self._previous_specialist_one_hot()
        dwell_norm = float(
            np.clip(
                float(self.dwell_hold_remaining) / float(max(1, int(self.cfg.min_dwell_steps))),
                0.0,
                1.0,
            )
        )
        return np.concatenate(
            [
                scores.astype(float),
                flags.astype(float),
                argmax_one_hot,
                np.asarray([max_conf, alert_age_norm], dtype=float),
                trend.astype(float),
                previous_specialist,
                np.asarray([dwell_norm], dtype=float),
            ]
        ).astype(float)

    def _previous_specialist_one_hot(self) -> np.ndarray:
        out = np.zeros(4, dtype=float)
        label = 0
        active = np.flatnonzero(np.asarray(self.previous_action_mask, dtype=float).reshape(-1) > 0.5)
        for sensor_idx in active:
            spec = self.sensor_specs[int(sensor_idx)]
            sensor_id = str(spec.sensor_id).lower()
            variables = {str(name).lower() for name in spec.observed_variables}
            if (
                "particle" in sensor_id
                or "laser" in sensor_id
                or "event_subtype_particle_latent" in variables
                or "snow_particle_mean_velocity_ms" in variables
            ):
                label = 1
                break
            if (
                "flux" in sensor_id
                or "fc4" in sensor_id
                or "event_subtype_flux_latent" in variables
                or "snow_mass_flux_kg_m2_s" in variables
            ):
                label = 2
                break
            if (
                "thermal" in sensor_id
                or "surface" in sensor_id
                or "event_subtype_thermal_latent" in variables
                or "snow_surface_temperature_c" in variables
            ):
                label = 3
                break
        out[int(label)] = 1.0
        return out

    def _build_action_to_sensor_mask(self) -> dict[int, np.ndarray]:
        n_sensors = len(self.sensor_specs)
        max_active = self.projector.constraints.max_active
        max_size = n_sensors if max_active is None else min(n_sensors, int(max_active))
        actions: dict[int, np.ndarray] = {}
        action_idx = 0
        for size in range(0, max_size + 1):
            for combo in combinations(range(n_sensors), size):
                mask = np.zeros(n_sensors, dtype=bool)
                if combo:
                    mask[list(combo)] = True
                actions[action_idx] = mask
                action_idx += 1
        return actions
