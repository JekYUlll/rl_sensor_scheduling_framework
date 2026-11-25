from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forecasting.base_predictor import BasePredictor
from forecasting.lstm import _LSTM
from forecasting.physics_constraints import PhysicsContext, build_physics_constraints, resolve_event_weighting


class PINNPredictor(BasePredictor):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        physics_weight: float = 0.1,
        constraints: list[dict] | None = None,
        event_weighting: dict | None = None,
        device: str | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.physics_weight = physics_weight
        self.constraints_cfg = list(constraints or [])
        self.event_weighting_cfg = dict(event_weighting or {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: _LSTM | None = None
        self.horizon: int | None = None
        self.n_features: int | None = None
        self.target_dim: int | None = None
        self.context: PhysicsContext | None = None
        self.physics_constraints = []
        self.event_weighting = None
        self.history: dict[str, list[float]] | None = None

    def set_context(
        self,
        input_feature_names: list[str],
        target_feature_names: list[str],
        stats: dict[str, np.ndarray],
    ) -> None:
        self.context = PhysicsContext(
            input_feature_names=list(input_feature_names),
            target_feature_names=list(target_feature_names),
            x_mean=torch.as_tensor(stats["x_mean"], dtype=torch.float32, device=self.device),
            x_std=torch.as_tensor(stats["x_std"], dtype=torch.float32, device=self.device),
            y_mean=torch.as_tensor(stats["y_mean"], dtype=torch.float32, device=self.device),
            y_std=torch.as_tensor(stats["y_std"], dtype=torch.float32, device=self.device),
        )
        self.physics_constraints = build_physics_constraints(self.constraints_cfg, self.context)

    def fit(self, train_data: Any, val_data: Any = None) -> None:
        X, Y = train_data.X, train_data.Y
        n, _, f = X.shape
        h = Y.shape[1]
        target_dim = Y.shape[2]
        self.horizon = h
        self.n_features = f
        self.target_dim = target_dim
        self.model = _LSTM(f, self.hidden_dim, h * target_dim, num_layers=self.num_layers).to(self.device)
        if self.context is None:
            raise ValueError("PINNPredictor.set_context must be called before fit")
        self.event_weighting = resolve_event_weighting(self.event_weighting_cfg, self.context, Y)

        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        train_loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_data is not None and val_data.X.shape[0] > 0:
            x_val = torch.as_tensor(val_data.X, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(val_data.Y, dtype=torch.float32, device=self.device)
            val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        best_metric = float("inf")
        self.history = {
            "train_loss": [],
            "train_data_loss": [],
            "train_physics_loss": [],
            "val_loss": [],
        }

        for _ in range(self.epochs):
            self.model.train()
            train_total = 0.0
            train_data = 0.0
            train_phys = 0.0
            train_count = 0

            for xb, yb in train_loader:
                pred = self.model(xb).reshape(xb.shape[0], h, target_dim)
                data_term = self._data_loss(pred, yb)
                physics_term = torch.zeros((), dtype=torch.float32, device=self.device)
                for constraint in self.physics_constraints:
                    physics_term = physics_term + float(constraint.weight) * constraint.compute(xb, pred)
                loss = data_term + float(self.physics_weight) * physics_term

                opt.zero_grad()
                loss.backward()
                opt.step()

                batch_n = xb.shape[0]
                train_total += float(loss.detach().item()) * batch_n
                train_data += float(data_term.detach().item()) * batch_n
                train_phys += float(physics_term.detach().item()) * batch_n
                train_count += batch_n

            self.history["train_loss"].append(train_total / max(train_count, 1))
            self.history["train_data_loss"].append(train_data / max(train_count, 1))
            self.history["train_physics_loss"].append(train_phys / max(train_count, 1))

            self.model.eval()
            if val_loader is None:
                metric = self.history["train_loss"][-1]
            else:
                val_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred = self.model(xb).reshape(xb.shape[0], h, target_dim)
                        data_term = self._data_loss(pred, yb)
                        physics_term = torch.zeros((), dtype=torch.float32, device=self.device)
                        for constraint in self.physics_constraints:
                            physics_term = physics_term + float(constraint.weight) * constraint.compute(xb, pred)
                        loss = data_term + float(self.physics_weight) * physics_term
                        batch_n = xb.shape[0]
                        val_sum += float(loss.detach().item()) * batch_n
                        val_count += batch_n
                metric = val_sum / max(val_count, 1)
            self.history["val_loss"].append(metric)

            if metric < best_metric:
                best_metric = metric
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)

    def _data_loss(self, y_pred_norm: torch.Tensor, y_true_norm: torch.Tensor) -> torch.Tensor:
        err_sq = (y_pred_norm - y_true_norm) ** 2
        if self.event_weighting is None or self.context is None:
            return err_sq.mean()
        y_true_denorm = self.context.denormalize_y(y_true_norm)
        weights = self.event_weighting.weights(y_true_denorm)
        return (weights * err_sq).mean()

    def predict(self, test_data: Any) -> np.ndarray:
        X = test_data.X
        n = X.shape[0]
        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        assert self.model is not None
        assert self.horizon is not None
        assert self.target_dim is not None
        self.model.eval()
        with torch.no_grad():
            y = self.model(x_t).cpu().numpy()
        return y.reshape(n, self.horizon, self.target_dim)
