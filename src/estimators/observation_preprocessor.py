from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObservationPreprocessor:
    """Causal robust smoothing for sensor observations before Kalman updates.

    The preprocessor is intentionally scheduler-agnostic: every policy sees the
    same sensor-noise handling, so full-open can act as a clean upper baseline
    instead of a high-frequency noise injector.
    """

    enabled: bool = False
    alpha: float = 0.35
    clip_sigma: float | None = 4.0
    min_std: float = 1e-6
    gap_reset_steps: int | None = None
    effective_noise_scale: float | None = None
    _filtered: dict[tuple[str, str], float] = field(default_factory=dict, init=False)
    _last_seen_t: dict[tuple[str, str], int] = field(default_factory=dict, init=False)

    @classmethod
    def from_config(cls, cfg: Mapping[str, object] | None) -> "ObservationPreprocessor":
        raw = dict(cfg or {})
        enabled = bool(raw.get("enabled", False))
        mode = str(raw.get("type", raw.get("mode", "robust_ema"))).strip().lower()
        if mode in {"none", "disabled", "off"}:
            enabled = False
        if mode not in {"none", "disabled", "off", "robust_ema", "ema"}:
            raise ValueError(
                f"Unsupported observation_preprocessing type '{mode}'. "
                "Supported types: robust_ema, ema, none"
            )
        alpha = float(raw.get("alpha", 0.35))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("observation_preprocessing.alpha must be in (0, 1]")
        clip_raw = raw.get("clip_sigma", 4.0)
        clip_sigma = None if clip_raw is None else float(clip_raw)
        if clip_sigma is not None and clip_sigma <= 0.0:
            raise ValueError("observation_preprocessing.clip_sigma must be positive")
        gap_raw = raw.get("gap_reset_steps")
        gap_reset_steps = None if gap_raw is None else max(int(gap_raw), 0)
        min_std = max(float(raw.get("min_std", 1e-6)), 1e-12)
        scale_raw = raw.get("effective_noise_scale", "auto")
        if isinstance(scale_raw, str) and scale_raw.strip().lower() == "auto":
            effective_noise_scale = alpha / max(2.0 - alpha, 1e-12)
        elif scale_raw is None:
            effective_noise_scale = None
        else:
            effective_noise_scale = max(float(scale_raw), 1e-6)
        return cls(
            enabled=enabled,
            alpha=alpha,
            clip_sigma=clip_sigma,
            min_std=min_std,
            gap_reset_steps=gap_reset_steps,
            effective_noise_scale=effective_noise_scale,
        )

    def reset(self) -> None:
        self._filtered.clear()
        self._last_seen_t.clear()

    def process(self, observations: Sequence[Mapping[str, object]]) -> list[dict]:
        if not self.enabled:
            return [dict(obs) for obs in observations]
        return [self._process_one(obs) for obs in observations]

    def _process_one(self, obs: Mapping[str, object]) -> dict:
        out = dict(obs)
        if not bool(out.get("available", False)):
            return out
        y = np.asarray(out["y"], dtype=float).reshape(-1)
        r_mat = np.asarray(out["R"], dtype=float)
        variables = [str(v) for v in out.get("variables", [])]
        sensor_id = str(out.get("sensor_id", "sensor"))
        t_raw = out.get("t")
        t = None if t_raw is None else int(t_raw)
        y_out = np.array(y, dtype=float, copy=True)
        r_out = np.array(r_mat, dtype=float, copy=True)
        for idx in range(y.shape[0]):
            variable = variables[idx] if idx < len(variables) else f"obs_{idx}"
            key = (sensor_id, variable)
            meas = float(y[idx])
            variance = float(r_mat[idx, idx]) if r_mat.ndim == 2 else self.min_std**2
            std = max(float(np.sqrt(max(variance, 0.0))), self.min_std)
            if self._should_reset_key(key, t):
                self._filtered.pop(key, None)
                self._last_seen_t.pop(key, None)
            if key in self._filtered:
                prev = float(self._filtered[key])
                if self.clip_sigma is not None:
                    limit = self.clip_sigma * std
                    meas = float(prev + np.clip(meas - prev, -limit, limit))
                filt = self.alpha * meas + (1.0 - self.alpha) * prev
            else:
                filt = meas
            self._filtered[key] = float(filt)
            if t is not None:
                self._last_seen_t[key] = int(t)
            y_out[idx] = float(filt)
            if self.effective_noise_scale is not None and r_out.ndim == 2:
                r_out[idx, idx] = max(float(r_out[idx, idx]) * self.effective_noise_scale, self.min_std**2)
        out["y_raw"] = np.array(y, dtype=float, copy=True)
        out["R_raw"] = np.array(r_mat, dtype=float, copy=True)
        out["preprocessed"] = True
        out["y"] = y_out
        out["R"] = r_out
        return out

    def _should_reset_key(self, key: tuple[str, str], t: int | None) -> bool:
        if t is None or self.gap_reset_steps is None:
            return False
        last = self._last_seen_t.get(key)
        if last is None:
            return False
        return (int(t) - int(last)) > int(self.gap_reset_steps)
