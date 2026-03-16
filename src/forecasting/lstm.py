from __future__ import annotations

import torch
from torch import nn

from forecasting.base_predictor import BasePredictor


class _LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMPredictor(BasePredictor):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 1, epochs: int = 10, lr: float = 1e-3, device: str = "cpu") -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        self.model = _LSTM(f, self.hidden_dim, h * f, num_layers=self.num_layers).to(self.device)
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
