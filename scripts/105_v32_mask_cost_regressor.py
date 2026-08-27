#!/usr/bin/env python3
"""Train and audit a mask-structured forecast-cost regressor on receding traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from v2.mask_cost_regressor import MaskCostRegressor, MaskCostRegressorConfig


def indexed_columns(table: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        (column for column in table.columns if column.startswith(prefix)),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )


def load_geometry(run_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    geometry = json.loads((run_dir / "action_geometry.json").read_text(encoding="utf-8"))
    sensor_ids = [str(value) for value in geometry["sensor_ids"]]
    sensor_index = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
    masks = np.zeros((len(geometry["masks"]), len(sensor_ids)), dtype=np.float32)
    action_features = np.zeros((len(geometry["masks"]), 2), dtype=np.float32)
    budget = max(float(geometry["budget"]), 1e-9)
    peak_budget = max(float(geometry["startup_peak_budget"]), 1e-9)
    for action_idx, item in enumerate(geometry["masks"]):
        for sensor_id in item["sensor_ids"]:
            masks[action_idx, sensor_index[str(sensor_id)]] = 1.0
        action_features[action_idx, 0] = float(item["steady_cost"]) / budget
        action_features[action_idx, 1] = float(item["cold_start_cost"]) / peak_budget
    return sensor_ids, masks, action_features


def trace_context(
    run_dir: Path,
    table: pd.DataFrame,
    sensor_ids: list[str],
    feature_set: str,
) -> np.ndarray:
    metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    quality_columns = [str(value) for value in dict(metadata.get("sensor_quality") or {}).get("columns") or []]
    if len(quality_columns) != len(sensor_ids):
        raise ValueError("Trace run must provide one online quality column per sensor")
    truth = pd.read_csv(run_dir / "truth_v31_split.csv", usecols=quality_columns)
    indices = table["truth_step_idx"].to_numpy(dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(truth)):
        raise ValueError("trace truth_step_idx is outside truth table")
    alerts = table[indexed_columns(table, "alert_feature_")].to_numpy(dtype=np.float32)
    if feature_set == "alert_context":
        context = alerts
    elif feature_set == "quality_alert_context":
        quality = truth.iloc[indices][quality_columns].to_numpy(dtype=np.float32)
        context = np.concatenate([quality, alerts], axis=1)
    else:
        raise ValueError(f"unsupported feature set: {feature_set}")
    return np.nan_to_num(context, nan=0.0, posinf=1.0, neginf=-1.0)


def training_costs(table: pd.DataFrame, target_mode: str) -> tuple[np.ndarray, np.ndarray]:
    raw = table[indexed_columns(table, "candidate_cost_")].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(raw)):
        raise ValueError("candidate costs must be finite")
    if target_mode == "row_standardized":
        mean = raw.mean(axis=1, keepdims=True)
        scale = raw.std(axis=1, keepdims=True)
    elif target_mode == "global_standardized":
        mean = np.asarray([[raw.mean()]], dtype=np.float32)
        scale = np.asarray([[raw.std()]], dtype=np.float32)
    else:
        raise ValueError(f"unsupported target mode: {target_mode}")
    return raw, (raw - mean) / np.maximum(scale, 1e-6)


def train_model(
    *,
    context: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    action_features: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ranking_weight: float,
    device: torch.device,
) -> tuple[MaskCostRegressor, list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MaskCostRegressor(
        MaskCostRegressorConfig(
            context_dim=context.shape[1],
            sensor_count=masks.shape[1],
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(context.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    masks_t = torch.from_numpy(masks).to(device)
    action_features_t = torch.from_numpy(action_features).to(device)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_context, batch_targets in loader:
            batch_context = batch_context.to(device)
            batch_targets = batch_targets.to(device)
            predicted = model(batch_context, masks_t, action_features_t)
            regression = nn.functional.smooth_l1_loss(predicted, batch_targets)
            best_action = torch.argmin(batch_targets, dim=1)
            ranking = nn.functional.cross_entropy(-predicted, best_action)
            loss = regression + float(ranking_weight) * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
    return model, history


@torch.no_grad()
def predict(
    model: MaskCostRegressor,
    context: np.ndarray,
    masks: np.ndarray,
    action_features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    masks_t = torch.from_numpy(masks).to(device)
    action_features_t = torch.from_numpy(action_features).to(device)
    outputs: list[np.ndarray] = []
    for start in range(0, len(context), batch_size):
        batch = torch.from_numpy(context[start : start + batch_size]).to(device)
        outputs.append(model(batch, masks_t, action_features_t).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate(
    *,
    seed: int,
    train_raw_costs: np.ndarray,
    test_raw_costs: np.ndarray,
    predictions: np.ndarray,
    masks: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    predicted_action = np.argmin(predictions, axis=1)
    oracle_action = np.argmin(test_raw_costs, axis=1)
    rows = np.arange(len(test_raw_costs))
    validation_static_action = int(np.argmin(train_raw_costs.mean(axis=0)))
    predicted_cost = test_raw_costs[rows, predicted_action]
    oracle_cost = test_raw_costs[rows, oracle_action]
    static_cost = test_raw_costs[:, validation_static_action]
    top3 = np.argsort(predictions, axis=1)[:, :3]
    oracle_gain = float(np.mean(static_cost - oracle_cost))
    model_gain = float(np.mean(static_cost - predicted_cost))
    predicted_duty = masks[predicted_action].mean(axis=0)
    metrics = {
        "seed": seed,
        "test_rows": len(test_raw_costs),
        "top1_action_accuracy": float(np.mean(predicted_action == oracle_action)),
        "top3_action_accuracy": float(np.mean([target in choices for target, choices in zip(oracle_action, top3)])),
        "mean_action_cost_regret": float(np.mean(predicted_cost - oracle_cost)),
        "mean_gain_vs_validation_static_action": model_gain,
        "receding_gain_vs_validation_static_action": oracle_gain,
        "fraction_of_receding_gain_recovered": model_gain / oracle_gain if oracle_gain > 0.0 else float("nan"),
        "validation_static_action_idx": validation_static_action,
        "predicted_action_coverage": int(np.unique(predicted_action).size),
        "predicted_always_off_sensor_count": int(np.sum(predicted_duty <= 0.01)),
        "predicted_always_on_sensor_count": int(np.sum(predicted_duty >= 0.99)),
        "predicted_mid_duty_sensor_count": int(np.sum((predicted_duty > 0.01) & (predicted_duty < 0.99))),
        "predicted_min_sensor_duty": float(np.min(predicted_duty)),
        "predicted_max_sensor_duty": float(np.max(predicted_duty)),
    }
    trace = np.column_stack(
        [predicted_action, oracle_action, np.full(len(predicted_action), validation_static_action), predicted_cost, oracle_cost, static_cost]
    )
    return metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument(
        "--feature-set",
        choices=("alert_context", "quality_alert_context"),
        default="quality_alert_context",
    )
    parser.add_argument(
        "--target-mode",
        choices=("row_standardized", "global_standardized"),
        default="row_standardized",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--training-partition", choices=("rl_train", "validation"), default="validation")
    parser.add_argument("--evaluation-partition", choices=("validation", "final_test"), default="final_test")
    args = parser.parse_args()
    device = torch.device(args.device)
    result_rows: list[dict[str, float | int]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    prediction_tables: list[pd.DataFrame] = []
    for run_dir in args.run_dirs:
        seed = int(run_dir.name.split("seed", 1)[1].split("_", 1)[0])
        sensor_ids, masks, action_features = load_geometry(run_dir)
        partition_dirs = {
            "rl_train": "receding_oracle_l8_rl_train_trace",
            "validation": "receding_oracle_l8_validation_trace",
            "final_test": "receding_oracle_l8_final_trace",
        }
        train_table = pd.read_csv(run_dir / partition_dirs[args.training_partition] / "receding_oracle_trace.csv")
        test_table = pd.read_csv(run_dir / partition_dirs[args.evaluation_partition] / "receding_oracle_trace.csv")
        train_context = trace_context(run_dir, train_table, sensor_ids, args.feature_set)
        test_context = trace_context(run_dir, test_table, sensor_ids, args.feature_set)
        train_raw, train_targets = training_costs(train_table, args.target_mode)
        test_raw = test_table[indexed_columns(test_table, "candidate_cost_")].to_numpy(dtype=np.float32)
        model, history = train_model(
            context=train_context,
            targets=train_targets,
            masks=masks,
            action_features=action_features,
            seed=seed,
            epochs=max(1, args.epochs),
            batch_size=max(1, args.batch_size),
            learning_rate=args.learning_rate,
            ranking_weight=args.ranking_weight,
            device=device,
        )
        predictions = predict(model, test_context, masks, action_features, device, args.batch_size)
        metrics, prediction_trace = evaluate(
            seed=seed,
            train_raw_costs=train_raw,
            test_raw_costs=test_raw,
            predictions=predictions,
            masks=masks,
        )
        metrics["training_partition"] = args.training_partition
        metrics["evaluation_partition"] = args.evaluation_partition
        result_rows.append(metrics)
        prediction_table = test_table[["rollout_idx", "rollout_step", "truth_step_idx"]].copy()
        prediction_table.insert(0, "seed", seed)
        prediction_table[[
            "predicted_action_idx",
            "oracle_action_idx",
            "training_selected_static_action_idx",
            "predicted_action_cost",
            "oracle_action_cost",
            "training_selected_static_action_cost",
        ]] = prediction_trace
        prediction_tables.append(prediction_table)
        histories[str(seed)] = history
    output = pd.DataFrame(result_rows).sort_values("seed")
    summary = pd.DataFrame([{
        "feature_set": args.feature_set,
        "target_mode": args.target_mode,
        "ranking_weight": args.ranking_weight,
        "training_partition": args.training_partition,
        "evaluation_partition": args.evaluation_partition,
        "seeds": len(output),
        "mean_top1_accuracy": output["top1_action_accuracy"].mean(),
        "mean_top3_accuracy": output["top3_action_accuracy"].mean(),
        "mean_action_cost_regret": output["mean_action_cost_regret"].mean(),
        "mean_gain_vs_static": output["mean_gain_vs_validation_static_action"].mean(),
        "mean_fraction_recovered": output["fraction_of_receding_gain_recovered"].mean(),
        "positive_gain_seeds": int((output["mean_gain_vs_validation_static_action"] > 0.0).sum()),
        "mean_predicted_action_coverage": output["predicted_action_coverage"].mean(),
        "behavior_pass_seeds": int(((output["predicted_always_off_sensor_count"] == 0) & (output["predicted_always_on_sensor_count"] == 0)).sum()),
    }])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(args.output_dir / "predicted_action_trace.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        "# Mask-structured forecast-cost regressor\n\n```text\n"
        + output.to_string(index=False, float_format=lambda value: f"{value:.6f}")
        + "\n\n"
        + summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")
        + "\n```\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
