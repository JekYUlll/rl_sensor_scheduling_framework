from __future__ import annotations

import torch
from torch import nn

from forecasting.base_predictor import BasePredictor
from forecasting.torch_utils import train_regressor


class _LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMPredictor(BasePredictor):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        self.model = _LSTM(f, self.hidden_dim, h * f, num_layers=self.num_layers).to(self.device)
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
