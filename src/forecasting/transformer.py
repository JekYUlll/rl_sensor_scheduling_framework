from __future__ import annotations

import torch
from torch import nn

from forecasting.base_predictor import BasePredictor


class _TransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class TransformerPredictor(BasePredictor):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2, epochs: int = 10, lr: float = 1e-3, device: str = "cpu") -> None:
        self.d_model = d_model
        self.nhead = nhead
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
        self.model = _TransformerRegressor(f, self.d_model, self.nhead, self.num_layers, h * f).to(self.device)
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
