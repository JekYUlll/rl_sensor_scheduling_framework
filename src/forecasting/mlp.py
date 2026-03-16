from __future__ import annotations

import numpy as np
import torch
from torch import nn

from forecasting.base_predictor import BasePredictor


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
    def __init__(self, hidden_dim: int = 128, epochs: int = 10, lr: float = 1e-3, device: str = "cpu") -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device)
        self.model = None
        self.lookback = None
        self.n_features = None
        self.horizon = None

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
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        x_t = torch.as_tensor(X.reshape(n, -1), dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(Y.reshape(n, -1), dtype=torch.float32, device=self.device)
        for _ in range(self.epochs):
            pred = self.model(x_t)
            loss = loss_fn(pred, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, test_data):
        X = test_data.X
        n = X.shape[0]
        x_t = torch.as_tensor(X.reshape(n, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(x_t).cpu().numpy()
        return y.reshape(n, self.horizon, self.n_features)
