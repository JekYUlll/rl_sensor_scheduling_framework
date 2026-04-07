from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch
from core.config import load_yaml, save_yaml
from evaluation.forecast_metrics import compute_forecast_metrics
from forecasting.baselines import NaivePredictor
from forecasting.dataset_builder import ForecastDataset, build_window_dataset, split_dataset
from forecasting.factory import build_predictor
from forecasting.series_preparation import (
    extract_context_series,
    prepare_input_and_targets,
    select_target_columns,
)
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
    horizon_weights: np.ndarray
    context_features: list[str] = field(default_factory=list)
    base_freq_s: int = 1
    score_scale: float = 1.0
    score_clip: float = 50.0

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
        time_index_window: np.ndarray | None = None,
        context_series_window: dict[str, np.ndarray] | None = None,
    ) -> float:
        history = np.asarray(history_window, dtype=float)
        future = np.asarray(future_truth, dtype=float)
        if not self.ready(history.shape[0], future.shape[0]):
            return 0.0
        history_window = history[-self.lookback:]
        mask_window = None
        if observed_mask_window is not None:
            mask_hist = np.asarray(observed_mask_window, dtype=float)
            if mask_hist.shape[0] >= self.lookback:
                mask_window = mask_hist[-self.lookback:]
        time_window = None if time_index_window is None else np.asarray(time_index_window, dtype=int)[-self.lookback:]
        input_prepared, input_names, _, _, _ = prepare_input_and_targets(
            input_series=history_window,
            target_series=history_window,
            feature_names=list(self.base_feature_names),
            observed_mask=mask_window,
            use_observed_mask=self.use_observed_mask,
            use_time_delta=self.use_time_delta,
            target_columns=[],
            time_index=time_window,
            base_freq_s=int(self.base_freq_s),
            context_series=context_series_window,
            context_features=list(self.context_features),
        )
        future_time = None
        if time_window is not None and time_window.size > 0:
            step_stride = max(1, int(time_window[-1] - time_window[-2])) if time_window.size >= 2 else 1
            future_time = time_window[-1] + step_stride * np.arange(1, self.horizon + 1, dtype=int)
        future_selected, future_names, _ = select_target_columns(target_series=future[:self.horizon], feature_names=list(self.base_feature_names), target_columns=list(self.target_columns), time_index=future_time, base_freq_s=int(self.base_freq_s))
        if list(future_names) != list(self.target_columns):
            raise ValueError(f'Reward oracle target column mismatch: expected={self.target_columns}, got={future_names}')
        if input_prepared.shape[1] != len(self.input_columns):
            raise ValueError(f'Reward oracle input feature mismatch: expected={len(self.input_columns)}, got={input_prepared.shape[1]}')
        x = input_prepared.reshape(1, self.lookback, len(self.input_columns))
        y = future_selected.reshape(1, self.horizon, self.target_dim)
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        ds = ForecastDataset(X=x_norm.astype(np.float32), Y=y_norm.astype(np.float32))
        y_pred_norm = np.asarray(self.predictor.predict(ds), dtype=float)
        err = y_pred_norm - y_norm
        if self.loss_name == 'mae':
            loss = np.abs(err)
        elif self.loss_name == 'huber':
            abs_err = np.abs(err)
            quad = np.minimum(abs_err, self.loss_delta)
            lin = abs_err - quad
            loss = 0.5 * quad ** 2 + self.loss_delta * lin
        else:
            loss = err ** 2
        per_horizon = np.mean(loss, axis=(0, 2))
        weights = np.asarray(self.horizon_weights, dtype=float).reshape(-1)
        if weights.shape[0] != self.horizon:
            raise ValueError(
                f"horizon_weights length mismatch: expected={self.horizon}, got={weights.shape[0]}"
            )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights = np.ones(self.horizon, dtype=float) / float(self.horizon)
        else:
            weights = weights / weight_sum
        raw = float(np.sum(weights * per_horizon))
        scaled = raw / max(float(self.score_scale), 1e-06)
        return float(min(scaled, float(self.score_clip)))

@dataclass
class FrozenForecastRewardEnsemble:
    oracles: list[FrozenForecastRewardOracle]
    weights: list[float]
    model_names: list[str]

    def __post_init__(self) -> None:
        if not self.oracles:
            raise ValueError('Reward oracle ensemble requires at least one oracle')
        if len(self.oracles) != len(self.weights):
            raise ValueError('Reward oracle ensemble length mismatch between oracles and weights')
        if len(self.oracles) != len(self.model_names):
            raise ValueError('Reward oracle ensemble length mismatch between oracles and model_names')
        ref = self.oracles[0]
        for oracle in self.oracles[1:]:
            if oracle.lookback != ref.lookback or oracle.horizon != ref.horizon:
                raise ValueError('All reward oracles in the ensemble must share lookback/horizon')
            if list(oracle.target_columns) != list(ref.target_columns):
                raise ValueError('All reward oracles in the ensemble must share target columns')
        weights = np.asarray(self.weights, dtype=float)
        if np.any(~np.isfinite(weights)):
            raise ValueError('Reward oracle ensemble weights must be finite numbers')
        if np.any(weights < 0.0):
            raise ValueError('Reward oracle ensemble weights must be non-negative')
        total = float(np.sum(weights))
        if total <= 0.0:
            weights = np.ones_like(weights, dtype=float)
            total = float(np.sum(weights))
        self.weights = (weights / total).tolist()

    @property
    def lookback(self) -> int:
        return int(self.oracles[0].lookback)

    @property
    def horizon(self) -> int:
        return int(self.oracles[0].horizon)

    @property
    def target_columns(self) -> list[str]:
        return list(self.oracles[0].target_columns)

    def ready(self, history_length: int, future_length: int) -> bool:
        return all((oracle.ready(history_length, future_length) for oracle in self.oracles))

    def score(
        self,
        history_window: np.ndarray,
        future_truth: np.ndarray,
        observed_mask_window: np.ndarray | None = None,
        time_index_window: np.ndarray | None = None,
        context_series_window: dict[str, np.ndarray] | None = None,
    ) -> float:
        if not self.ready(history_window.shape[0], future_truth.shape[0]):
            return 0.0
        total = 0.0
        for oracle, weight in zip(self.oracles, self.weights, strict=True):
            total += float(weight) * float(
                oracle.score(
                    history_window,
                    future_truth,
                    observed_mask_window,
                    time_index_window,
                    context_series_window,
                )
            )
        return float(total)

def _normalize_split(train: ForecastDataset, val: ForecastDataset, test: ForecastDataset):
    x_mean = train.X.mean(axis=(0, 1), keepdims=True)
    x_std_raw = train.X.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std_raw < 1e-08, 1.0, x_std_raw)
    y_mean = train.Y.mean(axis=(0, 1), keepdims=True)
    y_std_raw = train.Y.std(axis=(0, 1), keepdims=True)
    y_std = np.where(y_std_raw < 1e-08, 1.0, y_std_raw)

    def _apply(ds: ForecastDataset) -> ForecastDataset:
        return ForecastDataset(X=(ds.X - x_mean) / x_std, Y=(ds.Y - y_mean) / y_std, target_indices=ds.target_indices)
    stats = {'x_mean': x_mean.astype(np.float32), 'x_std': x_std.astype(np.float32), 'y_mean': y_mean.astype(np.float32), 'y_std': y_std.astype(np.float32)}
    return (_apply(train), _apply(val), _apply(test), stats)

def _concat_datasets(datasets: list[ForecastDataset]) -> ForecastDataset:
    valid = [ds for ds in datasets if ds.X.shape[0] > 0]
    if not valid:
        raise ValueError('No non-empty datasets to concatenate')
    target_indices = valid[0].target_indices
    x = np.concatenate([ds.X for ds in valid], axis=0)
    y = np.concatenate([ds.Y for ds in valid], axis=0)
    return ForecastDataset(X=x, Y=y, target_indices=target_indices)


def _resolve_horizon_weights(reward_cfg: dict, horizon: int) -> np.ndarray:
    raw = reward_cfg.get('horizon_weights')
    if raw is None:
        return np.ones(horizon, dtype=np.float32)
    weights = np.asarray(raw, dtype=np.float32).reshape(-1)
    if weights.shape[0] == 1:
        return np.full(horizon, float(weights[0]), dtype=np.float32)
    if weights.shape[0] != horizon:
        raise ValueError(
            f"horizon_weights length mismatch: expected={horizon}, got={weights.shape[0]}"
        )
    return weights.astype(np.float32)

def _split_rollouts_then_concat(datasets: list[ForecastDataset], *, train_ratio: float, val_ratio: float) -> tuple[ForecastDataset, ForecastDataset, ForecastDataset]:
    train_parts: list[ForecastDataset] = []
    val_parts: list[ForecastDataset] = []
    test_parts: list[ForecastDataset] = []
    for ds in datasets:
        train_ds, val_ds, test_ds = split_dataset(ds, train_ratio=train_ratio, val_ratio=val_ratio)
        if train_ds.X.shape[0] > 0:
            train_parts.append(train_ds)
        if val_ds.X.shape[0] > 0:
            val_parts.append(val_ds)
        if test_ds.X.shape[0] > 0:
            test_parts.append(test_ds)
    if not train_parts or not test_parts:
        merged = _concat_datasets(datasets)
        return split_dataset(merged, train_ratio=train_ratio, val_ratio=val_ratio)
    train = _concat_datasets(train_parts)
    val = _concat_datasets(val_parts) if val_parts else ForecastDataset(X=np.empty((0, train.X.shape[1], train.X.shape[2])), Y=np.empty((0, train.Y.shape[1], train.Y.shape[2])), target_indices=train.target_indices)
    test = _concat_datasets(test_parts)
    return (train, val, test)

def _build_model_for_payload(predictor: Any, payload: dict) -> Any:
    name = str(payload['predictor_cfg'].get('predictor_name', 'naive'))
    n_features = int(payload['n_features'])
    horizon = int(payload['horizon'])
    target_dim = int(payload['target_dim'])
    predictor.horizon = horizon
    predictor.n_features = n_features
    predictor.target_dim = target_dim
    if name == 'naive':
        return predictor
    if name == 'mlp':
        assert isinstance(predictor, MLPPredictor)
        predictor.lookback = int(payload['lookback'])
        predictor.model = _MLP(int(payload['lookback']) * n_features, horizon * target_dim, predictor.hidden_dim).to(predictor.device)
    elif name in {'lstm', 'pinn'}:
        assert isinstance(predictor, (LSTMPredictor, PINNPredictor))
        predictor.model = _LSTM(n_features, predictor.hidden_dim, horizon * target_dim, num_layers=predictor.num_layers).to(predictor.device)
    elif name == 'tcn':
        assert isinstance(predictor, TCNPredictor)
        predictor.model = _TCNRegressor(n_features, predictor.channels, predictor.kernel_size, horizon * target_dim).to(predictor.device)
    elif name in {'transformer', 'informer'}:
        assert isinstance(predictor, (TransformerPredictor, InformerPredictor))
        predictor.model = _TransformerRegressor(n_features, predictor.d_model, predictor.nhead, predictor.num_layers, horizon * target_dim).to(predictor.device)
    else:
        raise ValueError(f'Unsupported reward predictor_name: {name}')
    state_dict = payload.get('model_state_dict')
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
    score_scale: float,
    score_clip: float,
    metrics: dict,
    artifact_path: str | Path,
    *,
    base_freq_s: int,
    context_features: list[str] | None = None,
) -> str:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'predictor_cfg': predictor_cfg, 'lookback': int(lookback), 'horizon': int(horizon), 'n_features': int(len(input_columns)), 'target_dim': int(len(target_columns)), 'base_feature_names': list(base_feature_names), 'input_columns': list(input_columns), 'target_columns': list(target_columns), 'loss_name': str(loss_name), 'loss_delta': float(loss_delta), 'score_scale': float(score_scale), 'score_clip': float(score_clip), 'horizon_weights': np.asarray(stats['horizon_weights'], dtype=np.float32), 'base_freq_s': int(base_freq_s), 'context_features': [str(name) for name in (context_features or [])], 'x_mean': np.asarray(stats['x_mean'], dtype=np.float32), 'x_std': np.asarray(stats['x_std'], dtype=np.float32), 'y_mean': np.asarray(stats['y_mean'], dtype=np.float32), 'y_std': np.asarray(stats['y_std'], dtype=np.float32), 'use_observed_mask': bool(predictor_cfg.get('use_observed_mask', False)), 'use_time_delta': bool(predictor_cfg.get('use_time_delta', False)), 'metrics': {k: float(v) for k, v in metrics.items()}, 'model_state_dict': None}
    if hasattr(predictor, 'model') and getattr(predictor, 'model') is not None:
        payload['model_state_dict'] = getattr(predictor, 'model').state_dict()
    torch.save(payload, path)
    return str(path)

def _load_single_reward_oracle(artifact_path: str | Path) -> FrozenForecastRewardOracle:
    payload = torch.load(Path(artifact_path), map_location='cpu', weights_only=False)
    predictor_cfg = dict(payload['predictor_cfg'])
    predictor = build_predictor(predictor_cfg)
    predictor = _build_model_for_payload(predictor, payload)
    horizon = int(payload['horizon'])
    weights = np.asarray(payload.get('horizon_weights', np.ones(horizon, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if weights.shape[0] != horizon:
        weights = np.ones(horizon, dtype=np.float32)
    return FrozenForecastRewardOracle(predictor=predictor, lookback=int(payload['lookback']), horizon=horizon, base_feature_names=[str(v) for v in payload.get('base_feature_names', payload['input_columns'])], input_columns=[str(v) for v in payload['input_columns']], target_columns=[str(v) for v in payload['target_columns']], x_mean=np.asarray(payload['x_mean'], dtype=np.float32), x_std=np.asarray(payload['x_std'], dtype=np.float32), y_mean=np.asarray(payload['y_mean'], dtype=np.float32), y_std=np.asarray(payload['y_std'], dtype=np.float32), loss_name=str(payload.get('loss_name', 'huber')), loss_delta=float(payload.get('loss_delta', 1.0)), use_observed_mask=bool(payload.get('use_observed_mask', False)), use_time_delta=bool(payload.get('use_time_delta', False)), horizon_weights=weights, context_features=[str(v) for v in payload.get('context_features', [])], base_freq_s=int(payload.get('base_freq_s', 1)), score_scale=float(payload.get('score_scale', 1.0)), score_clip=float(payload.get('score_clip', 50.0)))


def _resolve_manifest_artifact_path(manifest_dir: Path, artifact_value: str) -> Path:
    raw_path = Path(str(artifact_value))
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
        candidates.append(manifest_dir / raw_path.name)
    else:
        candidates.append((manifest_dir / raw_path).resolve())
        candidates.append((manifest_dir / raw_path.name).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Reward oracle artifact not found. Tried: {[str(path) for path in candidates]}"
    )

def load_reward_oracle(artifact_path: str | Path) -> FrozenForecastRewardOracle | FrozenForecastRewardEnsemble:
    path = Path(artifact_path)
    if path.suffix.lower() in {'.yaml', '.yml'}:
        manifest = load_yaml(path)
        entries = list(manifest.get('entries', []))
        if not entries:
            raise ValueError(f'Reward oracle manifest has no entries: {path}')
        oracles: list[FrozenForecastRewardOracle] = []
        model_names: list[str] = []
        weights: list[float] = []
        for item in entries:
            item_path = _resolve_manifest_artifact_path(path.parent, str(item['artifact_path']))
            oracles.append(_load_single_reward_oracle(item_path))
            model_names.append(str(item.get('model_name', item_path.stem)))
            weights.append(float(item.get('weight', 1.0)))
        return FrozenForecastRewardEnsemble(oracles=oracles, weights=weights, model_names=model_names)
    return _load_single_reward_oracle(path)

def train_reward_oracle_from_series(input_series: np.ndarray, target_series: np.ndarray, input_columns: list[str], target_columns: list[str], reward_cfg: dict, artifact_path: str | Path, observed_mask: np.ndarray | None=None, time_index: np.ndarray | None=None, context_series: dict[str, np.ndarray] | None=None) -> dict:
    lookback = int(reward_cfg.get('lookback', 20))
    horizon = int(reward_cfg.get('horizon', 1))
    train_ratio = float(reward_cfg.get('train_ratio_within_pretrain', 0.8))
    val_ratio = float(reward_cfg.get('val_ratio_within_pretrain', 0.1))
    predictor_cfg_path = str(reward_cfg['predictor_cfg'])
    predictor_cfg = load_yaml(predictor_cfg_path)
    horizon_weights = _resolve_horizon_weights(reward_cfg, horizon)
    base_freq_s = int(reward_cfg.get('base_freq_s', 1))
    use_observed_mask = bool(predictor_cfg.get('use_observed_mask', False))
    use_time_delta = bool(predictor_cfg.get('use_time_delta', False))
    context_features = [str(name) for name in predictor_cfg.get('context_features', [])]
    input_series_prepared, input_columns_prepared, target_series_prepared, target_columns_prepared, _ = prepare_input_and_targets(input_series=np.asarray(input_series, dtype=float), target_series=np.asarray(target_series, dtype=float), feature_names=[str(name) for name in input_columns], observed_mask=None if observed_mask is None else np.asarray(observed_mask, dtype=float), use_observed_mask=use_observed_mask, use_time_delta=use_time_delta, target_columns=target_columns, time_index=None if time_index is None else np.asarray(time_index, dtype=int), base_freq_s=base_freq_s, context_series=extract_context_series(context_series), context_features=context_features)
    ds = build_window_dataset(series=np.asarray(input_series_prepared, dtype=float), lookback=lookback, horizon=horizon, target_series=np.asarray(target_series_prepared, dtype=float))
    train, val, test = split_dataset(ds, train_ratio=train_ratio, val_ratio=val_ratio)
    train_norm, val_norm, test_norm, stats = _normalize_split(train, val, test)
    stats['horizon_weights'] = horizon_weights
    predictor = build_predictor(predictor_cfg)
    predictor.fit(train_norm, val_norm)
    pred_norm = np.asarray(predictor.predict(test_norm), dtype=float)
    pred = pred_norm * stats['y_std'] + stats['y_mean']
    metrics = compute_forecast_metrics(test.Y, pred)
    metrics_norm = compute_forecast_metrics(test_norm.Y, pred_norm)
    score_scale = max(float(metrics_norm.get('rmse', 1.0)), 0.001)
    score_clip = float(reward_cfg.get('score_clip', 50.0))
    artifact = save_reward_oracle_artifact(predictor=predictor, predictor_cfg=predictor_cfg, stats=stats, lookback=lookback, horizon=horizon, base_feature_names=[str(name) for name in input_columns], input_columns=input_columns_prepared, target_columns=target_columns_prepared, loss_name=str(reward_cfg.get('loss', 'huber')), loss_delta=float(reward_cfg.get('loss_delta', 1.0)), score_scale=score_scale, score_clip=score_clip, metrics=metrics, artifact_path=artifact_path, base_freq_s=base_freq_s, context_features=context_features)
    return {'artifact_path': artifact, 'metrics': {k: float(v) for k, v in metrics.items()}, 'metrics_norm': {k: float(v) for k, v in metrics_norm.items()}, 'lookback': lookback, 'horizon': horizon, 'predictor_name': str(predictor_cfg.get('predictor_name', 'unknown')), 'target_columns': list(target_columns_prepared), 'score_scale': score_scale, 'score_clip': score_clip}

def train_reward_oracle_from_rollouts(rollouts: list[dict[str, np.ndarray]], input_columns: list[str], target_columns: list[str], reward_cfg: dict, artifact_path: str | Path) -> dict:
    if not rollouts:
        raise ValueError('rollouts must be non-empty')
    lookback = int(reward_cfg.get('lookback', 20))
    horizon = int(reward_cfg.get('horizon', 1))
    train_ratio = float(reward_cfg.get('train_ratio_within_pretrain', 0.8))
    val_ratio = float(reward_cfg.get('val_ratio_within_pretrain', 0.1))
    predictor_cfg_path = str(reward_cfg['predictor_cfg'])
    predictor_cfg = load_yaml(predictor_cfg_path)
    horizon_weights = _resolve_horizon_weights(reward_cfg, horizon)
    base_freq_s = int(reward_cfg.get('base_freq_s', 1))
    use_observed_mask = bool(predictor_cfg.get('use_observed_mask', False))
    use_time_delta = bool(predictor_cfg.get('use_time_delta', False))
    context_features = [str(name) for name in predictor_cfg.get('context_features', [])]
    datasets: list[ForecastDataset] = []
    input_columns_prepared: list[str] | None = None
    target_columns_prepared: list[str] | None = None
    for rollout in rollouts:
        input_series_prepared, rollout_input_names, target_series_prepared, rollout_target_names, target_indices = prepare_input_and_targets(input_series=np.asarray(rollout['input_series'], dtype=float), target_series=np.asarray(rollout['target_series'], dtype=float), feature_names=[str(name) for name in input_columns], observed_mask=None if rollout.get('observed_mask') is None else np.asarray(rollout['observed_mask'], dtype=float), use_observed_mask=use_observed_mask, use_time_delta=use_time_delta, target_columns=target_columns, time_index=None if rollout.get('time_index') is None else np.asarray(rollout['time_index'], dtype=int), base_freq_s=base_freq_s, context_series=extract_context_series(rollout), context_features=context_features)
        ds = build_window_dataset(series=np.asarray(input_series_prepared, dtype=float), lookback=lookback, horizon=horizon, target_series=np.asarray(target_series_prepared, dtype=float), target_indices=target_indices)
        datasets.append(ds)
        if input_columns_prepared is None:
            input_columns_prepared = list(rollout_input_names)
            target_columns_prepared = list(rollout_target_names)
        else:
            if list(rollout_input_names) != input_columns_prepared:
                raise ValueError('Reward rollout input feature names are inconsistent across pretraining rollouts')
            if list(rollout_target_names) != target_columns_prepared:
                raise ValueError('Reward rollout target names are inconsistent across pretraining rollouts')
    assert input_columns_prepared is not None
    assert target_columns_prepared is not None
    train, val, test = _split_rollouts_then_concat(datasets, train_ratio=train_ratio, val_ratio=val_ratio)
    train_norm, val_norm, test_norm, stats = _normalize_split(train, val, test)
    stats['horizon_weights'] = horizon_weights
    predictor = build_predictor(predictor_cfg)
    predictor.fit(train_norm, val_norm)
    pred_norm = np.asarray(predictor.predict(test_norm), dtype=float)
    pred = pred_norm * stats['y_std'] + stats['y_mean']
    metrics = compute_forecast_metrics(test.Y, pred)
    metrics_norm = compute_forecast_metrics(test_norm.Y, pred_norm)
    score_scale = max(float(metrics_norm.get('rmse', 1.0)), 0.001)
    score_clip = float(reward_cfg.get('score_clip', 50.0))
    artifact = save_reward_oracle_artifact(predictor=predictor, predictor_cfg=predictor_cfg, stats=stats, lookback=lookback, horizon=horizon, base_feature_names=[str(name) for name in input_columns], input_columns=input_columns_prepared, target_columns=target_columns_prepared, loss_name=str(reward_cfg.get('loss', 'huber')), loss_delta=float(reward_cfg.get('loss_delta', 1.0)), score_scale=score_scale, score_clip=score_clip, metrics=metrics, artifact_path=artifact_path, base_freq_s=base_freq_s, context_features=context_features)
    return {'artifact_path': artifact, 'metrics': {k: float(v) for k, v in metrics.items()}, 'metrics_norm': {k: float(v) for k, v in metrics_norm.items()}, 'lookback': lookback, 'horizon': horizon, 'predictor_name': str(predictor_cfg.get('predictor_name', 'unknown')), 'target_columns': list(target_columns_prepared), 'score_scale': score_scale, 'score_clip': score_clip}

def _sanitize_name(value: str) -> str:
    return ''.join((ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(value)))

def _resolve_suite_entries(reward_cfg: dict) -> list[dict]:
    entries_raw = reward_cfg.get('predictor_cfgs')
    if entries_raw is None:
        legacy = reward_cfg.get('predictor_cfg')
        if legacy is None:
            raise ValueError('reward_cfg must define `predictor_cfgs` (or legacy `predictor_cfg`)')
        entries_raw = [legacy]
    model_weights_cfg = reward_cfg.get('model_weights', {})
    entries: list[dict] = []
    for i, item in enumerate(entries_raw):
        if isinstance(item, str):
            predictor_cfg_path = item
            predictor_cfg = load_yaml(predictor_cfg_path)
            model_name = str(predictor_cfg.get('predictor_name', f'model_{i}'))
            weight = float(model_weights_cfg.get(model_name, 1.0))
        elif isinstance(item, dict):
            predictor_cfg_path = str(item.get('predictor_cfg', item.get('path', '')))
            if not predictor_cfg_path:
                raise ValueError(f'Invalid predictor_cfgs entry at index {i}: {item}')
            predictor_cfg = load_yaml(predictor_cfg_path)
            model_name = str(item.get('model_name', predictor_cfg.get('predictor_name', f'model_{i}')))
            weight = float(item.get('weight', model_weights_cfg.get(model_name, 1.0)))
        else:
            raise TypeError(f'Unsupported predictor_cfgs entry type: {type(item)!r}')
        entries.append({'predictor_cfg_path': predictor_cfg_path, 'predictor_cfg': predictor_cfg, 'model_name': model_name, 'weight': weight})
    return entries

def train_reward_oracle_suite_from_rollouts(rollouts: list[dict[str, np.ndarray]], input_columns: list[str], target_columns: list[str], reward_cfg: dict, artifact_dir: str | Path) -> dict:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    entries = _resolve_suite_entries(reward_cfg)
    if not entries:
        raise ValueError('Reward oracle suite requires at least one predictor config entry')
    trained_entries: list[dict] = []
    for item in entries:
        model_name = str(item['model_name'])
        single_cfg = dict(reward_cfg)
        single_cfg['predictor_cfg'] = str(item['predictor_cfg_path'])
        artifact_path = artifact_root / f'reward_predictor_{_sanitize_name(model_name)}.pt'
        out = train_reward_oracle_from_rollouts(rollouts=rollouts, input_columns=input_columns, target_columns=target_columns, reward_cfg=single_cfg, artifact_path=artifact_path)
        trained_entries.append({'model_name': model_name, 'weight': float(item['weight']), 'artifact_path': str(Path(out['artifact_path']).resolve()), 'predictor_cfg': str(item['predictor_cfg_path']), 'metrics': dict(out['metrics']), 'metrics_norm': dict(out.get('metrics_norm', {})), 'lookback': int(out['lookback']), 'horizon': int(out['horizon']), 'target_columns': list(out['target_columns']), 'score_scale': float(out.get('score_scale', 1.0)), 'score_clip': float(out.get('score_clip', 50.0))})
    weights = np.asarray([float(item['weight']) for item in trained_entries], dtype=float)
    if np.any(weights < 0.0):
        raise ValueError('Reward oracle suite weights must be non-negative')
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights, dtype=float)
    weights = weights / float(np.sum(weights))
    for item, weight in zip(trained_entries, weights.tolist(), strict=True):
        item['weight'] = float(weight)
    manifest = {'type': 'forecast_oracle_ensemble', 'lookback': int(trained_entries[0]['lookback']), 'horizon': int(trained_entries[0]['horizon']), 'target_columns': list(trained_entries[0]['target_columns']), 'entries': trained_entries}
    manifest_path = artifact_root / 'reward_oracles.yaml'
    save_yaml(manifest, manifest_path)
    csv_rows: list[dict[str, float | str]] = []
    for item in trained_entries:
        row: dict[str, float | str] = {'model_name': str(item['model_name']), 'weight': float(item['weight']), 'artifact_path': str(item['artifact_path']), 'score_scale': float(item.get('score_scale', 1.0)), 'score_clip': float(item.get('score_clip', 50.0))}
        for key, value in dict(item['metrics']).items():
            row[str(key)] = float(value)
        for key, value in dict(item.get('metrics_norm', {})).items():
            row[f'{key}_norm'] = float(value)
        csv_rows.append(row)
    if csv_rows:
        import pandas as pd
        pd.DataFrame(csv_rows).to_csv(artifact_root / 'reward_oracles_metrics.csv', index=False)
    return {'artifact_path': str(manifest_path), 'entries': trained_entries, 'lookback': int(manifest['lookback']), 'horizon': int(manifest['horizon']), 'target_columns': list(manifest['target_columns'])}
