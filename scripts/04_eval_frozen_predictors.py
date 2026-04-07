from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core.config import load_yaml
from evaluation.forecast_metrics import compute_forecast_metrics
from forecasting.dataset_builder import ForecastDataset, build_window_dataset
from forecasting.series_preparation import extract_context_series, prepare_input_and_targets
from reward.forecast_reward import FrozenForecastRewardEnsemble, FrozenForecastRewardOracle, load_reward_oracle

def _load_dataset_meta(series_npz: str) -> dict:
    meta_path = Path(series_npz).with_suffix('.meta.yaml')
    if meta_path.exists():
        return load_yaml(meta_path)
    return {}

def _select_oracle(reward_oracle: FrozenForecastRewardOracle | FrozenForecastRewardEnsemble, model_name: str) -> FrozenForecastRewardOracle:
    if isinstance(reward_oracle, FrozenForecastRewardEnsemble):
        names = [str(name) for name in reward_oracle.model_names]
        if model_name not in names:
            raise ValueError(f"Requested model_name '{model_name}' not found in reward ensemble. Available: {names}")
        idx = names.index(model_name)
        return reward_oracle.oracles[idx]
    return reward_oracle

def _normalize_with_oracle(oracle: FrozenForecastRewardOracle, ds: ForecastDataset) -> ForecastDataset:
    x_mean = np.asarray(oracle.x_mean, dtype=np.float32)
    x_std = np.asarray(oracle.x_std, dtype=np.float32)
    y_mean = np.asarray(oracle.y_mean, dtype=np.float32)
    y_std = np.asarray(oracle.y_std, dtype=np.float32)
    return ForecastDataset(X=((ds.X - x_mean) / x_std).astype(np.float32), Y=((ds.Y - y_mean) / y_std).astype(np.float32), target_indices=ds.target_indices)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--series_npz', required=True)
    parser.add_argument('--reward_artifact', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--run_id', required=True)
    args = parser.parse_args()
    data = np.load(args.series_npz, allow_pickle=True)
    input_series = data['input_series'] if 'input_series' in data else data['series']
    raw_target_series = data['target_series'] if 'target_series' in data else input_series
    base_feature_names = data['feature_names'].tolist() if 'feature_names' in data else [f'f{i}' for i in range(input_series.shape[1])]
    observed_mask = data['observed_mask'] if 'observed_mask' in data else None
    time_index = data['time_index'] if 'time_index' in data else None
    context_series = extract_context_series(data)
    reward_oracle = load_reward_oracle(args.reward_artifact)
    oracle = _select_oracle(reward_oracle, args.model_name)
    input_prepared, input_feature_names, target_prepared, target_feature_names, target_indices = prepare_input_and_targets(input_series=np.asarray(input_series, dtype=float), target_series=np.asarray(raw_target_series, dtype=float), feature_names=[str(name) for name in base_feature_names], observed_mask=None if observed_mask is None else np.asarray(observed_mask, dtype=float), use_observed_mask=bool(oracle.use_observed_mask), use_time_delta=bool(oracle.use_time_delta), target_columns=list(oracle.target_columns), time_index=None if time_index is None else np.asarray(time_index, dtype=int), base_freq_s=int(getattr(oracle, 'base_freq_s', 1)), context_series=context_series, context_features=list(getattr(oracle, 'context_features', [])))
    ds = build_window_dataset(series=input_prepared, lookback=int(oracle.lookback), horizon=int(oracle.horizon), target_series=target_prepared, target_indices=target_indices)
    ds_norm = _normalize_with_oracle(oracle, ds)
    y_pred_norm = np.asarray(oracle.predictor.predict(ds_norm), dtype=float)
    y_pred = y_pred_norm * np.asarray(oracle.y_std, dtype=np.float32) + np.asarray(oracle.y_mean, dtype=np.float32)
    metrics = compute_forecast_metrics(ds.Y, y_pred)
    metrics_norm = compute_forecast_metrics(ds_norm.Y, y_pred_norm)
    meta = _load_dataset_meta(args.series_npz)
    out_dir = ROOT / 'reports' / 'runs' / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {'model': str(args.model_name), 'scheduler': meta.get('scheduler_name', 'unknown'), 'source_run_id': meta.get('run_id', ''), 'dataset_npz': args.series_npz, 'lookback': int(oracle.lookback), 'horizon': int(oracle.horizon), 'n_features': int(len(input_feature_names)), 'n_targets': int(len(target_feature_names)), 'device': str(getattr(getattr(oracle.predictor, 'device', None), 'type', 'unknown')), 'target_columns': json.dumps(target_feature_names, ensure_ascii=False), 'context_features': json.dumps(list(getattr(oracle, 'context_features', [])), ensure_ascii=False), 'avg_power': float(meta.get('avg_power', float('nan'))), 'total_power': float(meta.get('total_power', float('nan'))), 'coverage_mean': float(meta.get('coverage_mean', float('nan'))), 'trace_P_mean': float(meta.get('trace_P_mean', float('nan'))), 'uncertainty_mean': float(meta.get('uncertainty_mean', float('nan'))), 'full_open_power': float(meta.get('full_open_power', float('nan'))), **metrics, 'rmse_norm': float(metrics_norm['rmse']), 'mae_norm': float(metrics_norm['mae']), 'mape_norm': float(metrics_norm['mape']), 'smape_norm': float(metrics_norm['smape']), 'pearson_h1_mean_norm': float(metrics_norm['pearson_h1_mean']), 'dtw_h1_mean_norm': float(metrics_norm['dtw_h1_mean'])}
    pd.DataFrame([row]).to_csv(out_dir / 'metrics_forecast.csv', index=False)
    np.savez(out_dir / 'forecast_predictions.npz', y_true=ds.Y, y_pred=y_pred, y_true_norm=ds_norm.Y, y_pred_norm=y_pred_norm, feature_names=np.asarray(target_feature_names), input_feature_names=np.asarray(input_feature_names), target_feature_names=np.asarray(target_feature_names))
    print(json.dumps(row, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
