from __future__ import annotations

import numpy as np

from estimators.base_estimator import BaseEstimator


class KalmanFilterEstimator(BaseEstimator):
    def __init__(
        self,
        A: np.ndarray,
        Q: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        sensor_ids: list[str],
        b: np.ndarray | None = None,
        state_scale: np.ndarray | None = None,
        uncertainty_weights: np.ndarray | None = None,
        sensor_warmup_steps: dict[str, int] | None = None,
        normalize_rl_state: bool = True,
        use_logdet: bool = False,
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.x0 = np.asarray(x0, dtype=float).reshape(-1)
        self.P0 = np.asarray(P0, dtype=float)
        self.b = np.zeros_like(self.x0) if b is None else np.asarray(b, dtype=float).reshape(-1)
        self.state_scale = (
            np.ones_like(self.x0, dtype=float)
            if state_scale is None
            else np.maximum(np.asarray(state_scale, dtype=float).reshape(-1), 1e-6)
        )
        self.uncertainty_weights = (
            np.ones_like(self.x0, dtype=float)
            if uncertainty_weights is None
            else np.asarray(uncertainty_weights, dtype=float).reshape(-1)
        )
        self.normalize_rl_state = bool(normalize_rl_state)
        self.use_logdet = bool(use_logdet)
        self.sensor_ids = list(sensor_ids)
        self.id_to_idx = {sid: i for i, sid in enumerate(self.sensor_ids)}
        warmup_map = sensor_warmup_steps or {}
        self.sensor_warmup_steps = np.asarray(
            [max(int(warmup_map.get(sid, 0)), 0) for sid in self.sensor_ids],
            dtype=float,
        )
        self.reset()

    def reset(self) -> None:
        self.x_hat = self.x0.copy()
        self.P = self.P0.copy()
        self.t = 0
        self.freshness = np.zeros(len(self.sensor_ids), dtype=float)
        self.coverage_hits = np.zeros(len(self.sensor_ids), dtype=float)
        self.coverage_total = np.zeros(len(self.sensor_ids), dtype=float)
        self.last_action = np.zeros(len(self.sensor_ids), dtype=float)
        self.current_budget_ratio = 1.0
        self.sensor_mode = np.zeros(len(self.sensor_ids), dtype=float)
        self.powered_mask = np.zeros(len(self.sensor_ids), dtype=float)
        self.warming_mask = np.zeros(len(self.sensor_ids), dtype=float)
        self.ready_mask = np.zeros(len(self.sensor_ids), dtype=float)
        self.warm_remaining = np.zeros(len(self.sensor_ids), dtype=float)
        self.warm_remaining_norm = np.zeros(len(self.sensor_ids), dtype=float)

    def predict(self) -> None:
        self.x_hat = self.A @ self.x_hat + self.b
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, observations: list[dict]) -> None:
        for obs in observations:
            if not obs.get("available", False):
                continue
            y = np.asarray(obs["y"], dtype=float).reshape(-1, 1)
            C = np.asarray(obs["C"], dtype=float)
            R = np.asarray(obs["R"], dtype=float)
            x_col = self.x_hat.reshape(-1, 1)
            S = C @ self.P @ C.T + R
            K = self.P @ C.T @ np.linalg.pinv(S)
            x_col = x_col + K @ (y - C @ x_col)
            I = np.eye(self.P.shape[0])
            residual_proj = I - K @ C
            self.P = residual_proj @ self.P @ residual_proj.T + K @ R @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            self.x_hat = x_col.reshape(-1)

    def on_step(
        self,
        selected_sensor_ids: list[str],
        power_ratio: float = 1.0,
        observed_sensor_ids: list[str] | None = None,
        sensor_status: dict[str, dict[str, float | int | bool | str]] | None = None,
    ) -> None:
        self.t += 1
        self.freshness += 1.0
        self.last_action = np.zeros(len(self.sensor_ids), dtype=float)
        self.sensor_mode = np.zeros(len(self.sensor_ids), dtype=float)
        self.powered_mask = np.zeros(len(self.sensor_ids), dtype=float)
        self.warming_mask = np.zeros(len(self.sensor_ids), dtype=float)
        self.ready_mask = np.zeros(len(self.sensor_ids), dtype=float)
        self.warm_remaining = np.zeros(len(self.sensor_ids), dtype=float)
        self.warm_remaining_norm = np.zeros(len(self.sensor_ids), dtype=float)
        observed_ids = list(observed_sensor_ids) if observed_sensor_ids is not None else list(selected_sensor_ids)
        selected_set = {str(sid) for sid in selected_sensor_ids}
        observed_set = {str(sid) for sid in observed_ids}
        for sid in self.sensor_ids:
            idx = self.id_to_idx[sid]
            self.coverage_total[idx] += 1.0
            status = None if sensor_status is None else sensor_status.get(sid)
            if status is None:
                powered = sid in selected_set
                ready = sid in observed_set
                warming = powered and not ready
                warm_remaining = 0.0
            else:
                powered = bool(status.get("powered", False))
                warming = bool(status.get("warming", False))
                ready = bool(status.get("ready", False))
                warm_remaining = float(status.get("warm_remaining_steps", 0.0))
            self.powered_mask[idx] = 1.0 if powered else 0.0
            self.warming_mask[idx] = 1.0 if warming else 0.0
            self.ready_mask[idx] = 1.0 if ready else 0.0
            self.sensor_mode[idx] = 2.0 if ready else (1.0 if warming else 0.0)
            self.warm_remaining[idx] = warm_remaining
            denom = max(float(self.sensor_warmup_steps[idx]), 1.0)
            self.warm_remaining_norm[idx] = float(warm_remaining) / denom
        for sid in selected_sensor_ids:
            idx = self.id_to_idx.get(sid)
            if idx is None:
                continue
            self.last_action[idx] = 1.0
        for sid in observed_ids:
            idx = self.id_to_idx.get(sid)
            if idx is None:
                continue
            self.freshness[idx] = 0.0
            self.coverage_hits[idx] += 1.0
        self.current_budget_ratio = float(power_ratio)

    def get_state_estimate(self) -> np.ndarray:
        return self.x_hat.copy()

    def _diag_p(self) -> np.ndarray:
        return np.diag(self.P).astype(float)

    def _diag_p_norm(self) -> np.ndarray:
        return self._diag_p() / np.maximum(self.state_scale**2, 1e-12)

    def get_uncertainty_summary(self) -> dict:
        trace_p = float(np.trace(self.P))
        diag_p = self._diag_p()
        diag_p_norm = self._diag_p_norm()
        out = {
            "trace_P": trace_p,
            "diag_P": diag_p.tolist(),
            "trace_P_norm": float(np.sum(diag_p_norm)),
            "diag_P_norm": diag_p_norm.tolist(),
            "weighted_trace_P_norm": float(np.sum(self.uncertainty_weights * diag_p_norm)),
        }
        if self.use_logdet:
            sign, logdet = np.linalg.slogdet(self.P + 1e-8 * np.eye(self.P.shape[0]))
            out["logdet_P"] = float(logdet if sign > 0 else np.nan)
        return out

    def get_rl_state_features(self) -> dict:
        unc = self.get_uncertainty_summary()
        coverage = np.divide(
            self.coverage_hits,
            np.maximum(self.coverage_total, 1.0),
        )
        state = {
            "x_hat": self.x_hat.astype(float).tolist(),
            "diag_P": unc["diag_P"],
            "trace_P": unc["trace_P"],
            "freshness": self.freshness.astype(float).tolist(),
            "coverage_ratio": coverage.astype(float).tolist(),
            "budget_ratio": float(self.current_budget_ratio),
            "previous_action": self.last_action.astype(float).tolist(),
            "sensor_mode": self.sensor_mode.astype(float).tolist(),
            "powered_mask": self.powered_mask.astype(float).tolist(),
            "warming_mask": self.warming_mask.astype(float).tolist(),
            "ready_mask": self.ready_mask.astype(float).tolist(),
            "warm_remaining": self.warm_remaining.astype(float).tolist(),
            "warm_remaining_norm": self.warm_remaining_norm.astype(float).tolist(),
        }
        if self.normalize_rl_state:
            state["x_hat_scaled"] = (self.x_hat / self.state_scale).astype(float).tolist()
            state["diag_P_norm"] = unc["diag_P_norm"]
            state["trace_P_norm"] = unc["trace_P_norm"]
            state["weighted_trace_P_norm"] = unc["weighted_trace_P_norm"]
        return state

    def normalized_state_error(self, truth_state: np.ndarray, dims: list[int] | None = None) -> float:
        truth = np.asarray(truth_state, dtype=float).reshape(-1)
        err = (self.x_hat - truth) / self.state_scale
        if dims is not None:
            err = err[np.asarray(dims, dtype=int)]
        return float(np.mean(err**2))
