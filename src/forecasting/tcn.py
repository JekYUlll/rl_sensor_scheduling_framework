from __future__ import annotations

import torch
from torch import nn

from forecasting.base_predictor import BasePredictor
from forecasting.torch_utils import train_regressor


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
    def __init__(
        self,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.channels = channels or [32, 32]
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.horizon = None
        self.n_features = None
        self.history = None

    def fit(self, train_data, val_data=None) -> None:
        X, Y = train_data.X, train_data.Y
        n, _, f = X.shape
        h = Y.shape[1]
        self.horizon = h
        self.n_features = f
        self.model = _TCNRegressor(f, self.channels, self.kernel_size, h * f).to(self.device)
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

    def predict(self, test_data):
        X = test_data.X
        n = X.shape[0]
        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y = self.model(x_t).cpu().numpy()
        return y.reshape(n, self.horizon, self.n_features)
