from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from forecasting.base_predictor import BasePredictor
from forecasting.torch_utils import train_regressor


class _S4MLikeRegressor(nn.Module):
    def __init__(self, in_dim: int, state_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, state_dim)
        self.gate_proj = nn.Linear(in_dim, state_dim)
        self.logit_decay = nn.Parameter(torch.zeros(state_dim))
        self.out = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        state_dim = self.logit_decay.shape[0]
        state = torch.zeros(batch, state_dim, dtype=x.dtype, device=x.device)
        decay = torch.sigmoid(self.logit_decay).unsqueeze(0)
        for t in range(x.shape[1]):
            xt = x[:, t, :]
            u = torch.tanh(self.in_proj(xt))
            g = torch.sigmoid(self.gate_proj(xt))
            state = decay * state + g * u
        return self.out(state)


class S4MLikePredictor(BasePredictor):
    """Missing-aware state-space-like baseline inspired by S4M-style modeling.

    This is a scoped baseline, not a paper-faithful reimplementation.
    It expects the caller to append mask / delta channels to the input.
    """

    def __init__(
        self,
        state_dim: int = 96,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: _S4MLikeRegressor | None = None
        self.horizon: int | None = None
        self.target_dim: int | None = None
        self.history: dict[str, list[float]] | None = None

    def fit(self, train_data: Any, val_data: Any = None) -> None:
        X, Y = train_data.X, train_data.Y
        n, _, f = X.shape
        h = Y.shape[1]
        target_dim = Y.shape[2]
        self.horizon = h
        self.target_dim = target_dim
        self.model = _S4MLikeRegressor(f, self.state_dim, h * target_dim).to(self.device)
        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(Y.reshape(n, -1), dtype=torch.float32, device=self.device)
        x_val = y_val = None
        if val_data is not None and val_data.X.shape[0] > 0:
            x_val = torch.as_tensor(val_data.X, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(val_data.Y.reshape(val_data.Y.shape[0], -1), dtype=torch.float32, device=self.device)
        self.history = train_regressor(
            self.model,
            x_t,
            y_t,
            x_val=x_val,
            y_val=y_val,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
        )

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
