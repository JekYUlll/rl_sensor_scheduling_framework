#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evaluation.sequence_metrics import dtw_distance_1d, smape_1d

SCHEDULER_ORDER = [
    'full_open',
    'info_priority',
    'periodic',
    'round_robin',
    'dqn',
    'cmdp_dqn',
    'ppo',
    'random',
]

SCHEDULER_LABELS = {
    'full_open': 'full_open',
    'info_priority': 'info_priority',
    'periodic': 'periodic',
    'round_robin': 'round_robin',
    'dqn': 'dqn',
    'cmdp_dqn': 'cmdp_dqn',
    'ppo': 'ppo',
    'random': 'random',
}


def _apply_style() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(
        {
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.figsize': (8, 4),
        }
    )


def _scheduler_color_map(names: Iterable[str]) -> dict[str, tuple[float, float, float, float]]:
    ordered = [n for n in SCHEDULER_ORDER if n in set(names)] + [n for n in names if n not in SCHEDULER_ORDER]
    cmap = plt.get_cmap('tab10', max(1, len(ordered)))
    return {name: cmap(i) for i, name in enumerate(ordered)}


def _sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def _discover_prediction_runs(run_tag: str) -> list[tuple[str, str, Path]]:
    runs_dir = ROOT / 'reports' / 'runs'
    discovered: list[tuple[str, str, Path]] = []
    pattern = re.compile(rf'^{re.escape(run_tag)}_(.+)_pred_(.+)$')
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        scheduler_name, model_name = match.groups()
        pred_path = path / 'forecast_predictions.npz'
        if pred_path.exists():
            discovered.append((scheduler_name, model_name, pred_path))
    return discovered


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _load_env_targets(env_cfg: Path) -> tuple[list[str], list[str]]:
    with env_cfg.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    primary = [str(v) for v in cfg.get('reward_target_columns', [])]
    forecast = [str(v) for v in cfg.get('forecast_target_columns', [])]
    return primary, forecast


def _build_long_metrics(run_tag: str, env_cfg: Path, out_dir: Path) -> pd.DataFrame:
    primary_targets, forecast_targets = _load_env_targets(env_cfg)
    discovered = _discover_prediction_runs(run_tag)
    if not discovered:
        raise FileNotFoundError(f'no forecast predictions found for run_tag={run_tag}')

    rows: list[dict[str, object]] = []
    metrics_map: dict[tuple[str, str, int], dict[str, dict[str, float]]] = {}

    for scheduler, model_name, pred_path in discovered:
        data = np.load(pred_path, allow_pickle=True)
        feature_names = [str(v) for v in data['target_feature_names'].tolist()]
        y_true = np.asarray(data['y_true'], dtype=float)
        y_pred = np.asarray(data['y_pred'], dtype=float)
        for target_idx, target in enumerate(feature_names):
            if target not in forecast_targets:
                continue
            for horizon_idx in range(y_true.shape[1]):
                seq_true = y_true[:, horizon_idx, target_idx]
                seq_pred = y_pred[:, horizon_idx, target_idx]
                key = (model_name, target, horizon_idx + 1)
                metrics_map.setdefault(key, {})[scheduler] = {
                    "rmse": _rmse(seq_true, seq_pred),
                    "dtw": dtw_distance_1d(seq_true, seq_pred),
                }

    for (model_name, target, horizon), scheduler_map in sorted(metrics_map.items()):
        baseline = scheduler_map.get("full_open")
        if baseline is None:
            continue
        for scheduler, metrics in scheduler_map.items():
            rows.append(
                {
                    'model': model_name,
                    'target': target,
                    'target_set': 'primary' if target in primary_targets else 'forecast',
                    'horizon': horizon,
                    'scheduler': scheduler,
                    'rmse': metrics["rmse"],
                    'dtw': metrics["dtw"],
                    'rmse_increase_pct_vs_full_open': 100.0 * (metrics["rmse"] - baseline["rmse"]) / max(baseline["rmse"], 1e-12),
                    'dtw_increase_pct_vs_full_open': 100.0 * (metrics["dtw"] - baseline["dtw"]) / max(baseline["dtw"], 1e-12),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('no long-form forecast metrics were computed')
    df.to_csv(out_dir / 'all_target_metrics_long.csv', index=False)
    return df


def _plot_boxplot_grid(df: pd.DataFrame, out_path: Path, target_set: str, metric_col: str, title: str, ylabel: str) -> None:
    _apply_style()
    subset = df.copy()
    if target_set == 'primary':
        subset = subset[subset['target_set'] == 'primary']
    elif target_set == 'forecast':
        pass
    else:
        raise ValueError(target_set)
    color_map = _scheduler_color_map(subset['scheduler'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, horizon in zip(axes, [1, 2, 3]):
        horizon_df = subset[subset['horizon'] == horizon]
        schedulers = [s for s in SCHEDULER_ORDER if s in set(horizon_df['scheduler'])]
        data = [horizon_df[horizon_df['scheduler'] == s][metric_col].dropna().to_numpy() for s in schedulers]
        if not any(len(vals) for vals in data):
            ax.set_axis_off()
            continue
        bp = ax.boxplot(data, tick_labels=[SCHEDULER_LABELS[s] for s in schedulers], showfliers=False, patch_artist=True)
        for patch, scheduler in zip(bp['boxes'], schedulers):
            patch.set_facecolor(color_map[scheduler])
            patch.set_alpha(0.20)
            patch.set_edgecolor('black')
        for median in bp['medians']:
            median.set_color('#ff8c00')
            median.set_linewidth(1.4)
        for i, (scheduler, vals) in enumerate(zip(schedulers, data), start=1):
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(42 + horizon + i).uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.65, color=color_map[scheduler], edgecolors='white', linewidths=0.25)
        ax.set_title(f'H={horizon}')
        ax.tick_params(axis='x', rotation=28)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        if horizon == 1:
            ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _choose_zoom_slice(seq: np.ndarray, width: int = 28) -> tuple[int, int]:
    x = np.asarray(seq, dtype=float).reshape(-1)
    if x.size <= width:
        return 0, x.size
    diff = np.abs(np.diff(x, prepend=x[0]))
    scores = np.convolve(diff, np.ones(width, dtype=float), mode='valid')
    start = int(np.argmax(scores))
    end = min(start + width, x.size)
    return start, end


def _radar(ax: plt.Axes, labels: list[str], values: list[float], color: tuple[float, float, float, float], alpha: float = 0.08) -> None:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]
    ax.plot(angles, values, linewidth=1.2, color=color)
    ax.fill(angles, values, color=color, alpha=alpha)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0.0, 1.05)


def _plot_summary_panel(run_tag: str, model: str, target: str, out_path: Path) -> None:
    _apply_style()
    discovered = [item for item in _discover_prediction_runs(run_tag) if item[1] == model]
    if not discovered:
        raise FileNotFoundError(f'no prediction runs found for model={model} run_tag={run_tag}')
    series_by_scheduler: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    feature_names_ref: list[str] | None = None
    for scheduler, _, pred_path in discovered:
        data = np.load(pred_path, allow_pickle=True)
        feature_names = [str(v) for v in data['target_feature_names'].tolist()]
        if target not in feature_names:
            continue
        feature_names_ref = feature_names
        target_idx = feature_names.index(target)
        y_true = np.asarray(data['y_true'], dtype=float)
        y_pred = np.asarray(data['y_pred'], dtype=float)
        horizon_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for h in range(y_true.shape[1]):
            horizon_map[h + 1] = (y_true[:, h, target_idx], y_pred[:, h, target_idx])
        series_by_scheduler[scheduler] = horizon_map
    if not series_by_scheduler:
        raise RuntimeError(f'target={target} not found for model={model}')

    schedulers = [s for s in SCHEDULER_ORDER if s in series_by_scheduler]
    color_map = _scheduler_color_map(schedulers)
    ref_true = series_by_scheduler[schedulers[0]][1][0]
    x = np.arange(ref_true.shape[0])
    zoom_start, zoom_end = _choose_zoom_slice(ref_true)

    fig = plt.figure(figsize=(14, 8))
    ax_main = fig.add_subplot(2, 2, 1)
    ax_zoom = fig.add_subplot(2, 2, 2)
    radar_axes = [fig.add_subplot(2, 3, 4 + i, polar=True) for i in range(3)]

    ax_main.plot(x, ref_true, label='true', linewidth=1.6, color='black')
    for scheduler in schedulers:
        _, seq_pred = series_by_scheduler[scheduler][1]
        ax_main.plot(x, seq_pred, label=scheduler, linewidth=1.1, alpha=0.95, color=color_map[scheduler])
    ax_main.axvspan(zoom_start, zoom_end - 1, color='gray', alpha=0.12)
    ax_main.set_title('Overlay (H=1)')
    ax_main.set_xlabel('test window index')
    ax_main.set_ylabel(target)

    ax_zoom.plot(x[zoom_start:zoom_end], ref_true[zoom_start:zoom_end], linewidth=1.6, color='black')
    for scheduler in schedulers:
        _, seq_pred = series_by_scheduler[scheduler][1]
        ax_zoom.plot(x[zoom_start:zoom_end], seq_pred[zoom_start:zoom_end], linewidth=1.1, alpha=0.95, color=color_map[scheduler])
    ax_zoom.set_title('Zoom-in (H=1)')
    ax_zoom.set_xlabel('test window index')

    metric_labels = ['RMSE', 'sMAPE', 'DTW']
    for horizon, ax_radar in zip([1, 2, 3], radar_axes):
        metrics_by_scheduler: dict[str, tuple[float, float, float]] = {}
        for scheduler in schedulers:
            seq_true, seq_pred = series_by_scheduler[scheduler][horizon]
            metrics_by_scheduler[scheduler] = (
                _rmse(seq_true, seq_pred),
                smape_1d(seq_true, seq_pred),
                dtw_distance_1d(seq_true, seq_pred),
            )
        best_rmse = min(v[0] for v in metrics_by_scheduler.values())
        best_smape = min(v[1] for v in metrics_by_scheduler.values())
        best_dtw = min(v[2] for v in metrics_by_scheduler.values())
        for scheduler in schedulers:
            rmse, smape, dtw = metrics_by_scheduler[scheduler]
            scores = [
                best_rmse / max(rmse, 1e-12),
                best_smape / max(smape, 1e-12),
                best_dtw / max(dtw, 1e-12),
            ]
            _radar(ax_radar, metric_labels, scores, color_map[scheduler], alpha=0.08)
        ax_radar.set_title(f'H={horizon} (higher is better)', fontsize=9)

    handles, labels = ax_main.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.54), frameon=False)
    title = textwrap.fill(f'Scheduler summary | model={model} | target={target}', width=70)
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.90, bottom=0.07, hspace=0.35, wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate legacy-style RMSE boxplots and summary panels for RL scheduling posthoc analysis')
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--env-cfg', default='configs/env/windblown_case.yaml')
    parser.add_argument('--model', default='informer')
    parser.add_argument('--targets', nargs='*', default=None, help='targets for summary panels; defaults to first primary target plus snow_mass_flux_kg_m2_s when available')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    env_cfg = ROOT / args.env_cfg
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / 'reports' / 'aggregate' / f'posthoc_{args.run_tag}' / 'legacy_style'
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_targets, forecast_targets = _load_env_targets(env_cfg)
    metrics_long = _build_long_metrics(args.run_tag, env_cfg, out_dir)

    _plot_boxplot_grid(
        metrics_long,
        out_dir / 'forecast_target_rmse_increase_boxplots.png',
        target_set='forecast',
        metric_col='rmse_increase_pct_vs_full_open',
        title='Forecast-target RMSE increase distribution vs full_open',
        ylabel='rRMSE increase (%)',
    )
    _plot_boxplot_grid(
        metrics_long,
        out_dir / 'primary_target_rmse_increase_boxplots.png',
        target_set='primary',
        metric_col='rmse_increase_pct_vs_full_open',
        title='Primary-target RMSE increase distribution vs full_open',
        ylabel='rRMSE increase (%)',
    )
    _plot_boxplot_grid(
        metrics_long,
        out_dir / 'forecast_target_dtw_increase_boxplots.png',
        target_set='forecast',
        metric_col='dtw_increase_pct_vs_full_open',
        title='Forecast-target DTW increase distribution vs full_open',
        ylabel='DTW increase (%)',
    )
    _plot_boxplot_grid(
        metrics_long,
        out_dir / 'primary_target_dtw_increase_boxplots.png',
        target_set='primary',
        metric_col='dtw_increase_pct_vs_full_open',
        title='Primary-target DTW increase distribution vs full_open',
        ylabel='DTW increase (%)',
    )

    panel_targets = list(args.targets) if args.targets else []
    if not panel_targets:
        if primary_targets:
            panel_targets.append(primary_targets[0])
        if 'snow_mass_flux_kg_m2_s' in forecast_targets and 'snow_mass_flux_kg_m2_s' not in panel_targets:
            panel_targets.append('snow_mass_flux_kg_m2_s')

    for target in panel_targets:
        _plot_summary_panel(args.run_tag, args.model, target, out_dir / f'summary_panel_{_sanitize(args.model)}_{_sanitize(target)}.png')


if __name__ == '__main__':
    main()
