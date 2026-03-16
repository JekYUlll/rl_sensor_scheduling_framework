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
        use_logdet: bool = False,
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.x0 = np.asarray(x0, dtype=float).reshape(-1)
        self.P0 = np.asarray(P0, dtype=float)
        self.use_logdet = bool(use_logdet)
        self.sensor_ids = list(sensor_ids)
        self.id_to_idx = {sid: i for i, sid in enumerate(self.sensor_ids)}
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

    def predict(self) -> None:
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q

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
            self.P = (I - K @ C) @ self.P
            self.x_hat = x_col.reshape(-1)

    def on_step(self, selected_sensor_ids: list[str], power_ratio: float = 1.0) -> None:
        self.t += 1
        self.freshness += 1.0
        self.last_action = np.zeros(len(self.sensor_ids), dtype=float)
        for sid in self.sensor_ids:
            idx = self.id_to_idx[sid]
            self.coverage_total[idx] += 1.0
        for sid in selected_sensor_ids:
            idx = self.id_to_idx.get(sid)
            if idx is None:
                continue
            self.freshness[idx] = 0.0
            self.coverage_hits[idx] += 1.0
            self.last_action[idx] = 1.0
        self.current_budget_ratio = float(power_ratio)

    def get_state_estimate(self) -> np.ndarray:
        return self.x_hat.copy()

    def get_uncertainty_summary(self) -> dict:
        trace_p = float(np.trace(self.P))
        diag_p = np.diag(self.P).astype(float)
        out = {"trace_P": trace_p, "diag_P": diag_p.tolist()}
        if self.use_logdet:
            sign, logdet = np.linalg.slogdet(self.P + 1e-8 * np.eye(self.P.shape[0]))
            out["logdet_P"] = float(logdet if sign > 0 else np.nan)
        return out

    def get_rl_state_features(self) -> dict:
        coverage = np.divide(
            self.coverage_hits,
            np.maximum(self.coverage_total, 1.0),
        )
        return {
            "x_hat": self.x_hat.astype(float).tolist(),
            "diag_P": np.diag(self.P).astype(float).tolist(),
            "trace_P": float(np.trace(self.P)),
            "freshness": self.freshness.astype(float).tolist(),
            "coverage_ratio": coverage.astype(float).tolist(),
            "budget_ratio": float(self.current_budget_ratio),
            "previous_action": self.last_action.astype(float).tolist(),
        }
