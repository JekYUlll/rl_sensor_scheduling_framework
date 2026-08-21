from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TCNOracleConfig:
    lookback: int = 20
    horizon: int = 5
    channels: int = 64
    levels: int = 3
    kernel_size: int = 3
    dropout: float = 0.05
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    normalized_loss: bool = True
    loss_clip: float = 10.0
    target_weights: tuple[float, ...] | None = None
    target_scales: tuple[float, ...] | None = None
    use_mask_channels: bool = True
    train_weighted_loss: bool = True
    subtype_loss_weighting: bool = False
    subtype_particle_target_weights: tuple[float, ...] | None = None
    subtype_flux_target_weights: tuple[float, ...] | None = None
    subtype_thermal_target_weights: tuple[float, ...] | None = None


class TCNFrozenForecastOracle:
    """Frozen nonlinear sequence oracle used as the v2 forecast-loss reward.

    The environment passes the same flattened feature used by the previous
    linear oracle: lookback observations followed by lookback observation masks.
    Internally this oracle reshapes it into a temporal tensor and trains a small
    causal-ish TCN to predict the future truth window.
    """

    def __init__(self, cfg: TCNOracleConfig) -> None:
        self.cfg = cfg
        self.model_: Any | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.y_std_: np.ndarray | None = None
        self.n_flat_features_: int | None = None
        self.n_step_features_: int | None = None
        self.n_targets_: int | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.device_: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        sample_contexts: Any | None = None,
    ) -> "TCNFrozenForecastOracle":
        torch, nn, DataLoader, TensorDataset = _torch_modules()
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            raise ValueError(f"x and y must be 2D, got {x_arr.shape=} {y_arr.shape=}")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x/y sample mismatch: {x_arr.shape[0]} != {y_arr.shape[0]}")
        self._set_shapes(x_arr, y_arr)

        self.x_mean_ = x_arr.mean(axis=0)
        self.x_std_ = np.maximum(x_arr.std(axis=0), 1e-6).astype(np.float32)
        self.y_mean_ = y_arr.mean(axis=0)
        self.y_std_ = np.maximum(y_arr.std(axis=0), 1e-6).astype(np.float32)
        x_norm = ((x_arr - self.x_mean_) / self.x_std_).astype(np.float32)
        y_norm = ((y_arr - self.y_mean_) / self.y_std_).astype(np.float32)
        x_seq = self._reshape_flat_features(x_norm)

        rng = np.random.default_rng(int(self.cfg.seed))
        order = rng.permutation(x_seq.shape[0])
        val_count = int(round(float(self.cfg.val_fraction) * x_seq.shape[0]))
        val_count = min(max(val_count, 0), max(0, x_seq.shape[0] - 1))
        val_idx = order[:val_count]
        train_idx = order[val_count:]
        if train_idx.size == 0:
            train_idx = order
            val_idx = np.asarray([], dtype=int)

        device = _select_device(torch, self.cfg.device)
        self.device_ = str(device)
        torch.manual_seed(int(self.cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.cfg.seed))

        model = _TinyTCN(
            in_channels=self._in_channels(),
            out_dim=int(y_arr.shape[1]),
            channels=int(self.cfg.channels),
            levels=int(self.cfg.levels),
            kernel_size=int(self.cfg.kernel_size),
            dropout=float(self.cfg.dropout),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.cfg.learning_rate),
            weight_decay=float(self.cfg.weight_decay),
        )
        loss_fn = nn.SmoothL1Loss(reduction="none")
        sample_weights_np = self._sample_training_weights(
            int(y_arr.shape[1]),
            sample_contexts=sample_contexts,
            n_samples=int(y_arr.shape[0]),
        )

        x_tensor = torch.as_tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_norm, dtype=torch.float32)
        weight_tensor = torch.as_tensor(sample_weights_np, dtype=torch.float32)
        dataset = TensorDataset(x_tensor[train_idx], y_tensor[train_idx], weight_tensor[train_idx])
        loader = DataLoader(
            dataset,
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            drop_last=False,
        )
        val_x = x_tensor[val_idx].to(device) if val_idx.size else None
        val_y = y_tensor[val_idx].to(device) if val_idx.size else None
        val_weights = weight_tensor[val_idx].to(device) if val_idx.size else None

        self.history_ = {"train_loss": [], "val_loss": []}
        for _ in range(int(self.cfg.epochs)):
            model.train()
            batch_losses: list[float] = []
            for xb, yb, wb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = _weighted_loss(loss_fn(model(xb), yb), wb)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            self.history_["train_loss"].append(float(np.mean(batch_losses)) if batch_losses else float("nan"))
            if val_x is not None and val_y is not None and val_weights is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = _weighted_loss(loss_fn(model(val_x), val_y), val_weights)
                self.history_["val_loss"].append(float(val_loss.detach().cpu().item()))

        self.model_ = model.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.x_mean_ is None or self.x_std_ is None or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("Oracle is not fitted")
        torch, _, _, _ = _torch_modules()
        x_arr = np.asarray(x, dtype=np.float32)
        was_1d = x_arr.ndim == 1
        if was_1d:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 1D or 2D, got {x_arr.shape}")
        x_norm = ((x_arr - self.x_mean_) / self.x_std_).astype(np.float32)
        x_seq = self._reshape_flat_features(x_norm)
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            pred_norm = self.model_(torch.as_tensor(x_seq, dtype=torch.float32, device=device)).detach().cpu().numpy()
        pred = pred_norm * self.y_std_ + self.y_mean_
        return pred[0] if was_1d else pred

    def loss(self, feature: np.ndarray, future: np.ndarray) -> float:
        return self.loss_with_context(feature, future, context=None)

    def loss_with_context(self, feature: np.ndarray, future: np.ndarray, *, context: dict[str, Any] | None = None) -> float:
        pred_flat = self.predict(feature).reshape(-1)
        true_flat = np.asarray(future, dtype=np.float32).reshape(-1)
        if pred_flat.shape != true_flat.shape:
            raise ValueError(f"Prediction/target shape mismatch: {pred_flat.shape} != {true_flat.shape}")
        errors = np.abs(pred_flat - true_flat)
        if self.cfg.normalized_loss:
            errors = errors / self._flat_scales(len(errors))
        if float(self.cfg.loss_clip) > 0:
            errors = np.minimum(errors, float(self.cfg.loss_clip))
        weights = self._flat_weights(len(errors), context=context)
        return float(np.sum(errors * weights) / np.sum(weights))

    def save(self, path: str | Path) -> None:
        if self.model_ is None or self.x_mean_ is None or self.x_std_ is None or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("Cannot save an unfitted oracle")
        torch, _, _, _ = _torch_modules()
        payload = {
            "cfg": self.cfg.__dict__,
            "state_dict": self.model_.state_dict(),
            "x_mean": self.x_mean_,
            "x_std": self.x_std_,
            "y_mean": self.y_mean_,
            "y_std": self.y_std_,
            "n_flat_features": self.n_flat_features_,
            "n_step_features": self.n_step_features_,
            "n_targets": self.n_targets_,
            "history": self.history_,
        }
        torch.save(payload, str(path))

    def to_device(self, device: str) -> "TCNFrozenForecastOracle":
        if self.model_ is None:
            raise RuntimeError("Oracle is not fitted")
        torch, _, _, _ = _torch_modules()
        selected = _select_device(torch, device)
        self.model_ = self.model_.to(selected).eval()
        self.device_ = str(selected)
        return self

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> "TCNFrozenForecastOracle":
        torch, _, _, _ = _torch_modules()
        map_location = _select_device(torch, device)
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        cfg_data = dict(payload["cfg"])
        cfg_data["device"] = str(device)
        oracle = cls(TCNOracleConfig(**cfg_data))
        oracle.x_mean_ = np.asarray(payload["x_mean"], dtype=np.float32)
        oracle.x_std_ = np.asarray(payload["x_std"], dtype=np.float32)
        oracle.y_mean_ = np.asarray(payload["y_mean"], dtype=np.float32)
        oracle.y_std_ = np.asarray(payload["y_std"], dtype=np.float32)
        oracle.n_flat_features_ = int(payload["n_flat_features"])
        oracle.n_step_features_ = int(payload["n_step_features"])
        oracle.n_targets_ = int(payload["n_targets"])
        oracle.history_ = payload.get("history", {"train_loss": [], "val_loss": []})
        model = _TinyTCN(
            in_channels=oracle._in_channels(),
            out_dim=int(oracle.n_targets_) * int(oracle.cfg.horizon),
            channels=int(oracle.cfg.channels),
            levels=int(oracle.cfg.levels),
            kernel_size=int(oracle.cfg.kernel_size),
            dropout=float(oracle.cfg.dropout),
        ).to(map_location)
        model.load_state_dict(payload["state_dict"])
        oracle.model_ = model.eval()
        oracle.device_ = str(map_location)
        return oracle

    def _set_shapes(self, x: np.ndarray, y: np.ndarray) -> None:
        lookback = int(self.cfg.lookback)
        horizon = int(self.cfg.horizon)
        divisor = 2 * lookback
        if x.shape[1] % divisor != 0:
            raise ValueError(f"x feature dimension {x.shape[1]} is not divisible by 2*lookback={divisor}")
        if y.shape[1] % horizon != 0:
            raise ValueError(f"y target dimension {y.shape[1]} is not divisible by horizon={horizon}")
        self.n_flat_features_ = int(x.shape[1])
        self.n_step_features_ = int(x.shape[1] // divisor)
        self.n_targets_ = int(y.shape[1] // horizon)

    def _reshape_flat_features(self, x: np.ndarray) -> np.ndarray:
        if self.n_step_features_ is None:
            raise RuntimeError("Oracle shapes are not initialized")
        batch = int(x.shape[0])
        lookback = int(self.cfg.lookback)
        n_step = int(self.n_step_features_)
        obs_len = lookback * n_step
        observed = x[:, :obs_len].reshape(batch, lookback, n_step)
        mask = x[:, obs_len:].reshape(batch, lookback, n_step)
        seq = np.concatenate([observed, mask], axis=2) if bool(self.cfg.use_mask_channels) else observed
        return np.transpose(seq, (0, 2, 1)).astype(np.float32)

    def _in_channels(self) -> int:
        if self.n_step_features_ is None:
            return 1
        multiplier = 2 if bool(self.cfg.use_mask_channels) else 1
        return int(self.n_step_features_) * multiplier

    def _flat_weights(self, n: int, *, context: dict[str, Any] | None = None) -> np.ndarray:
        target_weights = self._target_weights_for_context(context)
        if target_weights is None:
            return np.ones(n, dtype=np.float32)
        base = np.asarray(target_weights, dtype=np.float32).reshape(-1)
        if base.size == 0:
            return np.ones(n, dtype=np.float32)
        reps = int(np.ceil(n / base.size))
        return np.tile(base, reps)[:n].astype(np.float32)

    def _flat_training_weights(self, n: int) -> np.ndarray:
        if not bool(self.cfg.train_weighted_loss):
            return np.ones(n, dtype=np.float32)
        weights = self._flat_weights(n, context=None).astype(np.float32)
        mean_weight = float(np.mean(weights)) if weights.size else 1.0
        if mean_weight <= 0.0 or not np.isfinite(mean_weight):
            return np.ones(n, dtype=np.float32)
        return np.maximum(weights / mean_weight, 0.0).astype(np.float32)

    def _sample_training_weights(
        self,
        n: int,
        *,
        sample_contexts: Any | None,
        n_samples: int,
    ) -> np.ndarray:
        sample_count = int(n_samples)
        if sample_count <= 0:
            return np.ones((0, int(n)), dtype=np.float32)
        if not bool(self.cfg.train_weighted_loss):
            return np.ones((sample_count, int(n)), dtype=np.float32)
        if sample_contexts is None:
            base = self._flat_weights(int(n), context=None).reshape(1, -1)
            rows = np.repeat(base, sample_count, axis=0)
        else:
            contexts = list(self._iter_sample_contexts(sample_contexts, expected=sample_count))
            rows = np.vstack([self._flat_weights(int(n), context=context) for context in contexts])
        mean_weight = float(np.mean(rows)) if rows.size else 1.0
        if mean_weight <= 0.0 or not np.isfinite(mean_weight):
            return np.ones((sample_count, int(n)), dtype=np.float32)
        return np.maximum(rows / mean_weight, 0.0).astype(np.float32)

    @staticmethod
    def _iter_sample_contexts(sample_contexts: Any, *, expected: int) -> list[dict[str, Any] | None]:
        if isinstance(sample_contexts, np.ndarray):
            raw_values = sample_contexts.reshape(-1).tolist()
        else:
            raw_values = list(sample_contexts)
        if len(raw_values) != int(expected):
            raise ValueError(f"sample_contexts length mismatch: {len(raw_values)} != {expected}")
        contexts: list[dict[str, Any] | None] = []
        for value in raw_values:
            if value is None:
                contexts.append(None)
            elif isinstance(value, dict):
                contexts.append(value)
            else:
                contexts.append({"event_subtype_id": int(value)})
        return contexts

    def _target_weights_for_context(self, context: dict[str, Any] | None) -> tuple[float, ...] | None:
        if not bool(self.cfg.subtype_loss_weighting):
            return self.cfg.target_weights
        subtype_id = 0
        if context is not None:
            subtype_id = int(context.get("event_subtype_id", 0) or 0)
        if subtype_id == 1 and self.cfg.subtype_particle_target_weights is not None:
            return self.cfg.subtype_particle_target_weights
        if subtype_id == 2 and self.cfg.subtype_flux_target_weights is not None:
            return self.cfg.subtype_flux_target_weights
        if subtype_id == 3 and self.cfg.subtype_thermal_target_weights is not None:
            return self.cfg.subtype_thermal_target_weights
        return self.cfg.target_weights

    def _flat_scales(self, n: int) -> np.ndarray:
        if self.cfg.target_scales is not None:
            base = np.asarray(self.cfg.target_scales, dtype=np.float32).reshape(-1)
        elif self.y_std_ is not None:
            base = np.asarray(self.y_std_, dtype=np.float32).reshape(-1)
        else:
            base = np.ones(n, dtype=np.float32)
        if base.size == 0:
            return np.ones(n, dtype=np.float32)
        reps = int(np.ceil(n / base.size))
        return np.maximum(np.tile(base, reps)[:n], 1e-6).astype(np.float32)


def _torch_modules() -> tuple[Any, Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def _select_device(torch: Any, requested: str) -> Any:
    requested = str(requested)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _weighted_loss(loss_values: Any, weights: Any) -> Any:
    expanded_weights = loss_values.new_ones(loss_values.shape) * weights
    return (loss_values * weights).sum() / (expanded_weights.sum() + 1.0e-8)


class _TinyTCN:
    def __new__(
        cls,
        *,
        in_channels: int,
        out_dim: int,
        channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ) -> Any:
        _, nn, _, _ = _torch_modules()

        class ResidualBlock(nn.Module):
            def __init__(self, input_channels: int, output_channels: int, dilation: int) -> None:
                super().__init__()
                padding = dilation * (int(kernel_size) - 1)
                self.net = nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, int(kernel_size), padding=padding, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Conv1d(output_channels, output_channels, int(kernel_size), padding=padding, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                )
                self.skip = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

            def forward(self, x: Any) -> Any:
                y = self.net(x)
                y = y[..., : x.shape[-1]]
                return y + self.skip(x)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                blocks = []
                current = int(in_channels)
                for level in range(int(levels)):
                    blocks.append(ResidualBlock(current, int(channels), dilation=2**level))
                    current = int(channels)
                self.tcn = nn.Sequential(*blocks)
                self.head = nn.Linear(int(channels), int(out_dim))

            def forward(self, x: Any) -> Any:
                y = self.tcn(x)
                return self.head(y[..., -1])

        return Model()
