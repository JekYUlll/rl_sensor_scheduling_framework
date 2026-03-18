from __future__ import annotations

import torch
from torch import nn

from forecasting.base_predictor import BasePredictor
from forecasting.torch_utils import train_regressor


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class _TransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.pos_enc(z)
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class TransformerPredictor(BasePredictor):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.d_model = d_model
        self.nhead = nhead
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
        self.model = _TransformerRegressor(f, self.d_model, self.nhead, self.num_layers, h * f).to(self.device)
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
