from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v2.env import WarmupSchedulingEnv
from v2.policies import V2Policy

MID_DUTY_LOW = 0.05
MID_DUTY_HIGH = 0.95
ALWAYS_OFF_DUTY = 0.01
ALWAYS_ON_DUTY = 0.99


@dataclass
class RolloutResult:
    policy_name: str
    observations: np.ndarray
    masks: np.ndarray
    truth: np.ndarray
    rewards: np.ndarray
    scores: np.ndarray
    powers: np.ndarray
    peaks: np.ndarray
    selected_masks: np.ndarray
    mode_ids: np.ndarray
    event_flags: np.ndarray
    oracle_losses: np.ndarray
    step_indices: np.ndarray
    warmup_abort_count: int
    warmup_abort_deltas: np.ndarray
    energy_guard_dropped: np.ndarray
    soc: np.ndarray


def run_policy_rollout(
    env: WarmupSchedulingEnv,
    policy: V2Policy,
    *,
    steps: int,
    start_idx: int = 0,
) -> RolloutResult:
    policy.reset()
    env.reset(start_idx=int(start_idx))
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    rewards: list[float] = []
    score_rows: list[np.ndarray] = []
    powers: list[float] = []
    peaks: list[float] = []
    selected_masks: list[np.ndarray] = []
    mode_ids: list[np.ndarray] = []
    event_flags: list[float] = []
    oracle_losses: list[float] = []
    step_indices: list[int] = []
    warmup_abort_deltas: list[int] = []
    energy_guard_dropped: list[int] = []
    soc: list[float] = []
    warmup_abort_count = 0

    for _ in range(int(steps)):
        step_idx = int(env.current_idx)
        truth_at_step = np.array(env.truth_values[env.current_idx], dtype=float, copy=True)
        act_mask = getattr(policy, "act_mask", None)
        desired = act_mask(env) if callable(act_mask) else None
        if desired is not None:
            desired_mask = np.asarray(desired, dtype=bool).reshape(-1)
            scores = np.where(desired_mask, 1.0, -1.0)
            _, reward, done, info = env.step_mask(desired_mask)
        else:
            scores = policy.act_scores(env)
            _, reward, done, info = env.step_scores(scores)
        observations.append(np.array(env.last_observation, dtype=float, copy=True))
        masks.append(np.array(env.observed_mask, dtype=float, copy=True))
        truth.append(truth_at_step)
        rewards.append(float(reward))
        score_rows.append(np.asarray(scores, dtype=float).reshape(-1))
        powers.append(float(info["power"]))
        peaks.append(float(info["peak_power"]))
        selected_masks.append(np.asarray(info["selected_mask"], dtype=int))
        mode_after = info.get("mode_ids_after_step", {})
        mode_ids.append(np.asarray([mode_after.get(sid, info["sensor_status"][sid]["mode_id"]) for sid in env.sensor_ids], dtype=int))
        event_flags.append(float(info["event"]))
        oracle_losses.append(float(info["oracle_loss"]))
        step_indices.append(step_idx)
        warmup_abort_deltas.append(int(info.get("warmup_abort_delta", 0)))
        energy_guard_dropped.append(int(info.get("energy_guard_dropped", 0)))
        soc.append(float(info.get("soc", float("nan"))))
        warmup_abort_count = int(info["warmup_abort_count"])
        if done:
            break

    return RolloutResult(
        policy_name=policy.name,
        observations=np.asarray(observations, dtype=float),
        masks=np.asarray(masks, dtype=float),
        truth=np.asarray(truth, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        scores=np.asarray(score_rows, dtype=float),
        powers=np.asarray(powers, dtype=float),
        peaks=np.asarray(peaks, dtype=float),
        selected_masks=np.asarray(selected_masks, dtype=int),
        mode_ids=np.asarray(mode_ids, dtype=int),
        event_flags=np.asarray(event_flags, dtype=float),
        oracle_losses=np.asarray(oracle_losses, dtype=float),
        step_indices=np.asarray(step_indices, dtype=int),
        warmup_abort_count=warmup_abort_count,
        warmup_abort_deltas=np.asarray(warmup_abort_deltas, dtype=int),
        energy_guard_dropped=np.asarray(energy_guard_dropped, dtype=int),
        soc=np.asarray(soc, dtype=float),
    )


def concat_rollout_results(results: list[RolloutResult], *, policy_name: str | None = None) -> RolloutResult:
    if not results:
        raise ValueError("concat_rollout_results requires at least one rollout")
    name = str(policy_name or results[0].policy_name)
    return RolloutResult(
        policy_name=name,
        observations=np.concatenate([result.observations for result in results], axis=0),
        masks=np.concatenate([result.masks for result in results], axis=0),
        truth=np.concatenate([result.truth for result in results], axis=0),
        rewards=np.concatenate([result.rewards for result in results], axis=0),
        scores=np.concatenate([result.scores for result in results], axis=0),
        powers=np.concatenate([result.powers for result in results], axis=0),
        peaks=np.concatenate([result.peaks for result in results], axis=0),
        selected_masks=np.concatenate([result.selected_masks for result in results], axis=0),
        mode_ids=np.concatenate([result.mode_ids for result in results], axis=0),
        event_flags=np.concatenate([result.event_flags for result in results], axis=0),
        oracle_losses=np.concatenate([result.oracle_losses for result in results], axis=0),
        step_indices=np.concatenate([result.step_indices for result in results], axis=0),
        warmup_abort_count=int(sum(result.warmup_abort_count for result in results)),
        warmup_abort_deltas=np.concatenate([result.warmup_abort_deltas for result in results], axis=0),
        energy_guard_dropped=np.concatenate([result.energy_guard_dropped for result in results], axis=0),
        soc=np.concatenate([result.soc for result in results], axis=0),
    )


def rollout_metrics(result: RolloutResult) -> dict[str, float | str | int]:
    err = np.abs(result.observations - result.truth)
    oracle_losses = result.oracle_losses[np.isfinite(result.oracle_losses)]
    metrics = {
        "policy": result.policy_name,
        "steps": int(result.rewards.size),
        "reward_mean": float(np.mean(result.rewards)) if result.rewards.size else float("nan"),
        "instant_mae": float(np.mean(err)) if err.size else float("nan"),
        "oracle_loss_mean": float(np.mean(oracle_losses)) if oracle_losses.size else float("nan"),
        "dtw_mean": mean_dtw(result.observations, result.truth),
        "power_mean": float(np.mean(result.powers)) if result.powers.size else float("nan"),
        "peak_power_max": float(np.max(result.peaks)) if result.peaks.size else float("nan"),
        "event_rate": float(np.mean(result.event_flags)) if result.event_flags.size else float("nan"),
        "warmup_abort_count": int(result.warmup_abort_count),
    }
    metrics.update(selected_mask_diagnostics(result.selected_masks))
    return metrics


def selected_mask_diagnostics(selected_masks: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(selected_masks, dtype=float)
    if selected.ndim != 2 or selected.shape[0] == 0 or selected.shape[1] == 0:
        return {
            "switches_per_step": float("nan"),
            "always_on_sensor_count": 0,
            "always_off_sensor_count": 0,
            "mid_duty_sensor_count": 0,
            "duty_entropy": float("nan"),
            "duty_min": float("nan"),
            "duty_max": float("nan"),
            "duty_std": float("nan"),
        }
    duties = np.mean(selected, axis=0)
    if selected.shape[0] > 1:
        switches = np.mean(np.abs(np.diff(selected, axis=0)), axis=1)
        switches_per_step = float(np.mean(switches))
    else:
        switches_per_step = 0.0
    duty_entropy = float(
        -np.mean(
            duties * np.log(np.clip(duties, 1e-9, 1.0))
            + (1.0 - duties) * np.log(np.clip(1.0 - duties, 1e-9, 1.0))
        )
        / np.log(2.0)
    )
    return {
        "switches_per_step": switches_per_step,
        "always_on_sensor_count": int(np.sum(duties >= ALWAYS_ON_DUTY)),
        "always_off_sensor_count": int(np.sum(duties <= ALWAYS_OFF_DUTY)),
        "mid_duty_sensor_count": int(np.sum((duties >= MID_DUTY_LOW) & (duties <= MID_DUTY_HIGH))),
        "duty_entropy": duty_entropy,
        "duty_min": float(np.min(duties)),
        "duty_max": float(np.max(duties)),
        "duty_std": float(np.std(duties)),
    }


def save_rollout_npz(
    path: str | Path,
    result: RolloutResult,
    *,
    sensor_ids: list[str] | tuple[str, ...],
    state_columns: list[str] | tuple[str, ...],
) -> None:
    np.savez(
        path,
        observations=result.observations,
        masks=result.masks,
        truth=result.truth,
        rewards=result.rewards,
        scores=result.scores,
        powers=result.powers,
        peaks=result.peaks,
        selected_masks=result.selected_masks,
        mode_ids=result.mode_ids,
        event_flags=result.event_flags,
        oracle_losses=result.oracle_losses,
        step_indices=result.step_indices,
        warmup_abort_count=np.asarray([int(result.warmup_abort_count)], dtype=int),
        warmup_abort_deltas=result.warmup_abort_deltas,
        energy_guard_dropped=result.energy_guard_dropped,
        soc=result.soc,
        sensor_ids=np.asarray([str(sensor_id) for sensor_id in sensor_ids]),
        state_columns=np.asarray([str(name) for name in state_columns]),
        policy=np.asarray([str(result.policy_name)]),
    )


def mean_dtw(observed: np.ndarray, truth: np.ndarray, *, window: int = 50) -> float:
    obs = np.asarray(observed, dtype=float)
    ref = np.asarray(truth, dtype=float)
    if obs.shape != ref.shape or obs.ndim != 2 or obs.shape[0] == 0:
        return float("nan")
    distances = []
    for col in range(obs.shape[1]):
        distances.append(_dtw_1d(obs[:, col], ref[:, col], window=int(window)) / float(obs.shape[0]))
    return float(np.mean(distances)) if distances else float("nan")


def _dtw_1d(a: np.ndarray, b: np.ndarray, *, window: int) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    n = int(x.size)
    m = int(y.size)
    if n == 0 or m == 0:
        return float("nan")
    band = max(int(window), abs(n - m))
    previous = np.full(m + 1, np.inf, dtype=float)
    current = np.full(m + 1, np.inf, dtype=float)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current.fill(np.inf)
        start = max(1, i - band)
        end = min(m, i + band)
        for j in range(start, end + 1):
            cost = abs(float(x[i - 1] - y[j - 1]))
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous
    return float(previous[m])
