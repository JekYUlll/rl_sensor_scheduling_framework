from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.config import load_yaml
from evaluation.forecast_metrics import compute_forecast_metrics
from forecasting.baselines import NaivePredictor
from forecasting.dataset_builder import ForecastDataset, build_window_dataset, split_dataset
from forecasting.factory import build_predictor
from forecasting.series_preparation import prepare_input_and_targets, select_target_columns
from forecasting.informer import InformerPredictor
from forecasting.lstm import LSTMPredictor, _LSTM
from forecasting.mlp import MLPPredictor, _MLP
from forecasting.pinn import PINNPredictor
from forecasting.tcn import TCNPredictor, _TCNRegressor
from forecasting.transformer import TransformerPredictor, _TransformerRegressor


@dataclass
class FrozenForecastRewardOracle:
    predictor: Any
    lookback: int
    horizon: int
    base_feature_names: list[str]
    input_columns: list[str]
    target_columns: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    loss_name: str
    loss_delta: float
    use_observed_mask: bool
    use_time_delta: bool

    @property
    def target_dim(self) -> int:
        return len(self.target_columns)

    def ready(self, history_length: int, future_length: int) -> bool:
        return history_length >= self.lookback and future_length >= self.horizon

    def score(
        self,
        history_window: np.ndarray,
        future_truth: np.ndarray,
        observed_mask_window: np.ndarray | None = None,
    ) -> float:
        history = np.asarray(history_window, dtype=float)
        future = np.asarray(future_truth, dtype=float)
        if not self.ready(history.shape[0], future.shape[0]):
            return 0.0

        history_window = history[-self.lookback :]
        mask_window = None
        if observed_mask_window is not None:
            mask_hist = np.asarray(observed_mask_window, dtype=float)
            if mask_hist.shape[0] >= self.lookback:
                mask_window = mask_hist[-self.lookback :]

        input_prepared, input_names, _, _, _ = prepare_input_and_targets(
            input_series=history_window,
            target_series=history_window,
            feature_names=list(self.base_feature_names),
            observed_mask=mask_window,
            use_observed_mask=self.use_observed_mask,
            use_time_delta=self.use_time_delta,
            target_columns=[],
        )
        future_selected, future_names, _ = select_target_columns(
            target_series=future[: self.horizon],
            feature_names=list(self.base_feature_names),
            target_columns=list(self.target_columns),
        )
        if list(future_names) != list(self.target_columns):
            raise ValueError(
                "Reward oracle target column mismatch: "
                f"expected={self.target_columns}, got={future_names}"
            )
        if input_prepared.shape[1] != len(self.input_columns):
            raise ValueError(
                "Reward oracle input feature mismatch: "
                f"expected={len(self.input_columns)}, got={input_prepared.shape[1]}"
            )

        x = input_prepared.reshape(1, self.lookback, len(self.input_columns))
        y = future_selected.reshape(1, self.horizon, self.target_dim)
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        ds = ForecastDataset(
            X=x_norm.astype(np.float32),
            Y=y_norm.astype(np.float32),
        )
        y_pred_norm = np.asarray(self.predictor.predict(ds), dtype=float)
        err = y_pred_norm - y_norm
        if self.loss_name == "mae":
            return float(np.mean(np.abs(err)))
        if self.loss_name == "huber":
            abs_err = np.abs(err)
            quad = np.minimum(abs_err, self.loss_delta)
            lin = abs_err - quad
            return float(np.mean(0.5 * quad**2 + self.loss_delta * lin))
        return float(np.mean(err**2))


def _normalize_split(train: ForecastDataset, val: ForecastDataset, test: ForecastDataset):
    x_mean = train.X.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(train.X.std(axis=(0, 1), keepdims=True), 1e-6)
    y_mean = train.Y.mean(axis=(0, 1), keepdims=True)
    y_std = np.maximum(train.Y.std(axis=(0, 1), keepdims=True), 1e-6)

    def _apply(ds: ForecastDataset) -> ForecastDataset:
        return ForecastDataset(
            X=(ds.X - x_mean) / x_std,
            Y=(ds.Y - y_mean) / y_std,
            target_indices=ds.target_indices,
        )

    stats = {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }
    return _apply(train), _apply(val), _apply(test), stats


def _build_model_for_payload(predictor: Any, payload: dict) -> Any:
    name = str(payload["predictor_cfg"].get("predictor_name", "naive"))
    n_features = int(payload["n_features"])
    horizon = int(payload["horizon"])
    target_dim = int(payload["target_dim"])

    predictor.horizon = horizon
    predictor.n_features = n_features
    predictor.target_dim = target_dim

    if name == "naive":
        return predictor
    if name == "mlp":
        assert isinstance(predictor, MLPPredictor)
        predictor.lookback = int(payload["lookback"])
        predictor.model = _MLP(int(payload["lookback"]) * n_features, horizon * target_dim, predictor.hidden_dim).to(predictor.device)
    elif name in {"lstm", "pinn"}:
        assert isinstance(predictor, (LSTMPredictor, PINNPredictor))
        predictor.model = _LSTM(n_features, predictor.hidden_dim, horizon * target_dim, num_layers=predictor.num_layers).to(predictor.device)
    elif name == "tcn":
        assert isinstance(predictor, TCNPredictor)
        predictor.model = _TCNRegressor(n_features, predictor.channels, predictor.kernel_size, horizon * target_dim).to(predictor.device)
    elif name in {"transformer", "informer"}:
        assert isinstance(predictor, (TransformerPredictor, InformerPredictor))
        predictor.model = _TransformerRegressor(n_features, predictor.d_model, predictor.nhead, predictor.num_layers, horizon * target_dim).to(predictor.device)
    else:
        raise ValueError(f"Unsupported reward predictor_name: {name}")

    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Missing model_state_dict for reward predictor '{name}'")
    predictor.model.load_state_dict(state_dict)
    predictor.model.eval()
    return predictor


def save_reward_oracle_artifact(
    predictor: Any,
    predictor_cfg: dict,
    stats: dict,
    lookback: int,
    horizon: int,
    base_feature_names: list[str],
    input_columns: list[str],
    target_columns: list[str],
    loss_name: str,
    loss_delta: float,
    metrics: dict,
    artifact_path: str | Path,
) -> str:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "predictor_cfg": predictor_cfg,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "n_features": int(len(input_columns)),
        "target_dim": int(len(target_columns)),
        "base_feature_names": list(base_feature_names),
        "input_columns": list(input_columns),
        "target_columns": list(target_columns),
        "loss_name": str(loss_name),
        "loss_delta": float(loss_delta),
        "x_mean": np.asarray(stats["x_mean"], dtype=np.float32),
        "x_std": np.asarray(stats["x_std"], dtype=np.float32),
        "y_mean": np.asarray(stats["y_mean"], dtype=np.float32),
        "y_std": np.asarray(stats["y_std"], dtype=np.float32),
        "use_observed_mask": bool(predictor_cfg.get("use_observed_mask", False)),
        "use_time_delta": bool(predictor_cfg.get("use_time_delta", False)),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "model_state_dict": None,
    }
    if hasattr(predictor, "model") and getattr(predictor, "model") is not None:
        payload["model_state_dict"] = getattr(predictor, "model").state_dict()
    torch.save(payload, path)
    return str(path)


def load_reward_oracle(artifact_path: str | Path) -> FrozenForecastRewardOracle:
    payload = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
    predictor_cfg = dict(payload["predictor_cfg"])
    predictor = build_predictor(predictor_cfg)
    predictor = _build_model_for_payload(predictor, payload)
    return FrozenForecastRewardOracle(
        predictor=predictor,
        lookback=int(payload["lookback"]),
        horizon=int(payload["horizon"]),
        base_feature_names=[str(v) for v in payload.get("base_feature_names", payload["input_columns"])],
        input_columns=[str(v) for v in payload["input_columns"]],
        target_columns=[str(v) for v in payload["target_columns"]],
        x_mean=np.asarray(payload["x_mean"], dtype=np.float32),
        x_std=np.asarray(payload["x_std"], dtype=np.float32),
        y_mean=np.asarray(payload["y_mean"], dtype=np.float32),
        y_std=np.asarray(payload["y_std"], dtype=np.float32),
        loss_name=str(payload.get("loss_name", "huber")),
        loss_delta=float(payload.get("loss_delta", 1.0)),
        use_observed_mask=bool(payload.get("use_observed_mask", False)),
        use_time_delta=bool(payload.get("use_time_delta", False)),
    )


def train_reward_oracle_from_series(
    input_series: np.ndarray,
    target_series: np.ndarray,
    input_columns: list[str],
    target_columns: list[str],
    reward_cfg: dict,
    artifact_path: str | Path,
    observed_mask: np.ndarray | None = None,
) -> dict:
    lookback = int(reward_cfg.get("lookback", 20))
    horizon = int(reward_cfg.get("horizon", 1))
    train_ratio = float(reward_cfg.get("train_ratio_within_pretrain", 0.8))
    val_ratio = float(reward_cfg.get("val_ratio_within_pretrain", 0.1))
    predictor_cfg_path = str(reward_cfg["predictor_cfg"])
    predictor_cfg = load_yaml(predictor_cfg_path)
    use_observed_mask = bool(predictor_cfg.get("use_observed_mask", False))
    use_time_delta = bool(predictor_cfg.get("use_time_delta", False))

    input_series_prepared, input_columns_prepared, target_series_prepared, target_columns_prepared, _ = prepare_input_and_targets(
        input_series=np.asarray(input_series, dtype=float),
        target_series=np.asarray(target_series, dtype=float),
        feature_names=[str(name) for name in input_columns],
        observed_mask=None if observed_mask is None else np.asarray(observed_mask, dtype=float),
        use_observed_mask=use_observed_mask,
        use_time_delta=use_time_delta,
        target_columns=target_columns,
    )

    ds = build_window_dataset(
        series=np.asarray(input_series_prepared, dtype=float),
        lookback=lookback,
        horizon=horizon,
        target_series=np.asarray(target_series_prepared, dtype=float),
    )
    train, val, test = split_dataset(ds, train_ratio=train_ratio, val_ratio=val_ratio)
    train_norm, val_norm, test_norm, stats = _normalize_split(train, val, test)

    predictor = build_predictor(predictor_cfg)
    predictor.fit(train_norm, val_norm)
    pred_norm = np.asarray(predictor.predict(test_norm), dtype=float)
    pred = pred_norm * stats["y_std"] + stats["y_mean"]
    metrics = compute_forecast_metrics(test.Y, pred)
    artifact = save_reward_oracle_artifact(
        predictor=predictor,
        predictor_cfg=predictor_cfg,
        stats=stats,
        lookback=lookback,
        horizon=horizon,
        base_feature_names=[str(name) for name in input_columns],
        input_columns=input_columns_prepared,
        target_columns=target_columns_prepared,
        loss_name=str(reward_cfg.get("loss", "huber")),
        loss_delta=float(reward_cfg.get("loss_delta", 1.0)),
        metrics=metrics,
        artifact_path=artifact_path,
    )
    return {
        "artifact_path": artifact,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "lookback": lookback,
        "horizon": horizon,
        "predictor_name": str(predictor_cfg.get("predictor_name", "unknown")),
        "target_columns": list(target_columns_prepared),
    }
