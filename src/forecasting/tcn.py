from __future__ import annotations

import torch
from torch import nn

from forecasting.base_predictor import BasePredictor


class _TCNRegressor(nn.Module):
    def __init__(self, in_dim: int, channels: list[int], kernel_size: int, out_dim: int) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for c in channels:
            layers.append(nn.Conv1d(prev, c, kernel_size=kernel_size, padding=kernel_size - 1))
            layers.append(nn.ReLU())
            prev = c
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        z = self.backbone(x)
        z = z[:, :, -1]
        return self.head(z)


class TCNPredictor(BasePredictor):
    def __init__(self, channels: list[int] | None = None, kernel_size: int = 3, epochs: int = 10, lr: float = 1e-3, device: str = "cpu") -> None:
        self.channels = channels or [32, 32]
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device)
        self.model = None
        self.horizon = None
        self.n_features = None

    def fit(self, train_data, val_data=None) -> None:
        X, Y = train_data.X, train_data.Y
        n, _, f = X.shape
        h = Y.shape[1]
        self.horizon = h
        self.n_features = f
        self.model = _TCNRegressor(f, self.channels, self.kernel_size, h * f).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
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
        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(x_t).cpu().numpy()
        return y.reshape(n, self.horizon, self.n_features)
