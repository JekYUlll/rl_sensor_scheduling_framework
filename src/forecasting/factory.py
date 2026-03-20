from __future__ import annotations

from forecasting.baselines import NaivePredictor
from forecasting.informer import InformerPredictor
from forecasting.lstm import LSTMPredictor
from forecasting.mlp import MLPPredictor
from forecasting.pinn import PINNPredictor
from forecasting.tcn import TCNPredictor
from forecasting.transformer import TransformerPredictor


def build_predictor(cfg: dict):
    name = cfg.get("predictor_name", "naive")
    device = cfg.get("device")
    batch_size = int(cfg.get("batch_size", 128))
    if name == "naive":
        return NaivePredictor()
    if name == "mlp":
        return MLPPredictor(
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            epochs=int(cfg.get("epochs", 10)),
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=batch_size,
            device=device,
        )
    if name == "lstm":
        return LSTMPredictor(
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 1)),
            epochs=int(cfg.get("epochs", 10)),
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=batch_size,
            device=device,
        )
    if name == "transformer":
        return TransformerPredictor(
            d_model=int(cfg.get("d_model", 64)),
            nhead=int(cfg.get("nhead", 4)),
            num_layers=int(cfg.get("num_layers", 2)),
            epochs=int(cfg.get("epochs", 10)),
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=batch_size,
            device=device,
        )
    if name == "informer":
        return InformerPredictor(
            d_model=int(cfg.get("d_model", 64)),
            nhead=int(cfg.get("nhead", 4)),
            num_layers=int(cfg.get("num_layers", 2)),
            epochs=int(cfg.get("epochs", 10)),
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=batch_size,
            device=device,
        )
    if name == "tcn":
        return TCNPredictor(
            channels=list(cfg.get("channels", [32, 32])),
            kernel_size=int(cfg.get("kernel_size", 3)),
            epochs=int(cfg.get("epochs", 10)),
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=batch_size,
            device=device,
        )
    if name == "pinn":
        return PINNPredictor(
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 1)),
            epochs=int(cfg.get("epochs", 10)),
            lr=float(cfg.get("lr", 1e-3)),
            batch_size=batch_size,
            physics_weight=float(cfg.get("physics_weight", 0.1)),
            constraints=list(cfg.get("constraints", [])),
            device=device,
        )
    raise ValueError(f"Unsupported predictor_name: {name}")
