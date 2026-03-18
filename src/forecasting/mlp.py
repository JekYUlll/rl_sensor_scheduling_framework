from __future__ import annotations

import numpy as np
import torch
from torch import nn

from forecasting.base_predictor import BasePredictor
from forecasting.torch_utils import train_regressor


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPPredictor(BasePredictor):
    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.lookback = None
        self.n_features = None
        self.horizon = None
        self.history = None

    def fit(self, train_data, val_data=None) -> None:
        X = train_data.X
        Y = train_data.Y
        n, lb, f = X.shape
        h = Y.shape[1]
        self.lookback = lb
        self.n_features = f
        self.horizon = h
        in_dim = lb * f
        out_dim = h * f
        self.model = _MLP(in_dim, out_dim, self.hidden_dim).to(self.device)
        x_t = torch.as_tensor(X.reshape(n, -1), dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(Y.reshape(n, -1), dtype=torch.float32, device=self.device)
        x_val = y_val = None
        if val_data is not None and val_data.X.shape[0] > 0:
            x_val = torch.as_tensor(val_data.X.reshape(val_data.X.shape[0], -1), dtype=torch.float32, device=self.device)
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

    def predict(self, test_data):
        X = test_data.X
        n = X.shape[0]
        x_t = torch.as_tensor(X.reshape(n, -1), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y = self.model(x_t).cpu().numpy()
        return y.reshape(n, self.horizon, self.n_features)
