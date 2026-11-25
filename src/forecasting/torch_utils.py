from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_regressor(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> dict[str, list[float]]:
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = None
    if x_val is not None and y_val is not None and len(x_val) > 0:
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("inf")

    for _ in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_n = xb.shape[0]
            train_loss_sum += float(loss.detach().item()) * batch_n
            train_count += batch_n

        train_loss = train_loss_sum / max(train_count, 1)
        history["train_loss"].append(train_loss)

        model.eval()
        if val_loader is None:
            metric = train_loss
            history["val_loss"].append(train_loss)
        else:
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    batch_n = xb.shape[0]
                    val_loss_sum += float(loss.detach().item()) * batch_n
                    val_count += batch_n
            metric = val_loss_sum / max(val_count, 1)
            history["val_loss"].append(metric)

        if metric < best_metric:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return history
