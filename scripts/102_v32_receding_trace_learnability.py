#!/usr/bin/env python3
"""Audit whether online observations predict receding forecast-value actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feature_columns(table: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        (column for column in table.columns if column.startswith(prefix)),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )


def masked_predictions(
    model: object,
    features: np.ndarray,
    candidate_costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    scores = np.full((features.shape[0], candidate_costs.shape[1]), -np.inf, dtype=float)
    scores[:, classes] = probabilities
    scores[~np.isfinite(candidate_costs)] = -np.inf
    predictions = np.argmax(scores, axis=1)
    top3 = np.argsort(scores, axis=1)[:, -3:]
    return predictions, top3


def evaluate_model(
    *,
    seed: int,
    model_name: str,
    feature_set: str,
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    cost_columns: list[str],
) -> dict[str, object]:
    x_train = np.nan_to_num(train[columns].to_numpy(dtype=float))
    x_test = np.nan_to_num(test[columns].to_numpy(dtype=float))
    y_train = train["selected_action_idx"].to_numpy(dtype=int)
    y_test = test["selected_action_idx"].to_numpy(dtype=int)
    test_costs = test[cost_columns].to_numpy(dtype=float)
    model.fit(x_train, y_train)
    predicted, top3 = masked_predictions(model, x_test, test_costs)
    row_indices = np.arange(len(test))
    oracle_cost = test_costs[row_indices, y_test]
    predicted_cost = test_costs[row_indices, predicted]
    validation_costs = train[cost_columns].replace([np.inf, -np.inf], np.nan).mean(axis=0)
    static_action = int(np.nanargmin(validation_costs.to_numpy(dtype=float)))
    static_cost = test_costs[:, static_action]
    valid = np.isfinite(oracle_cost) & np.isfinite(predicted_cost) & np.isfinite(static_cost)
    oracle_gap = float(np.mean(static_cost[valid] - oracle_cost[valid]))
    model_gain = float(np.mean(static_cost[valid] - predicted_cost[valid]))
    return {
        "seed": int(seed),
        "model": model_name,
        "feature_set": feature_set,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_action_coverage": int(np.unique(y_train).size),
        "test_action_coverage": int(np.unique(y_test).size),
        "top1_action_accuracy": float(np.mean(predicted == y_test)),
        "top3_action_accuracy": float(np.mean([target in choices for target, choices in zip(y_test, top3)])),
        "mean_action_cost_regret": float(np.mean(predicted_cost[valid] - oracle_cost[valid])),
        "mean_gain_vs_validation_static_action": model_gain,
        "receding_gain_vs_validation_static_action": oracle_gap,
        "fraction_of_receding_gain_recovered": model_gain / oracle_gap if oracle_gap > 0.0 else float("nan"),
        "validation_static_action_idx": static_action,
    }


def evaluate_cost_regressor(
    *,
    seed: int,
    model_name: str,
    feature_set: str,
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    cost_columns: list[str],
) -> dict[str, object]:
    x_train = np.nan_to_num(train[columns].to_numpy(dtype=float))
    x_test = np.nan_to_num(test[columns].to_numpy(dtype=float))
    train_costs = train[cost_columns].to_numpy(dtype=float)
    test_costs = test[cost_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(train_costs)):
        raise ValueError("Training candidate costs must be finite for cost regression")
    model.fit(x_train, train_costs)
    predicted_costs = np.asarray(model.predict(x_test), dtype=float)
    predicted_costs[~np.isfinite(test_costs)] = np.inf
    predicted = np.argmin(predicted_costs, axis=1)
    y_test = test["selected_action_idx"].to_numpy(dtype=int)
    row_indices = np.arange(len(test))
    oracle_cost = test_costs[row_indices, y_test]
    predicted_cost = test_costs[row_indices, predicted]
    static_action = int(np.argmin(np.mean(train_costs, axis=0)))
    static_cost = test_costs[:, static_action]
    valid = np.isfinite(oracle_cost) & np.isfinite(predicted_cost) & np.isfinite(static_cost)
    oracle_gap = float(np.mean(static_cost[valid] - oracle_cost[valid]))
    model_gain = float(np.mean(static_cost[valid] - predicted_cost[valid]))
    return {
        "seed": int(seed),
        "model": model_name,
        "feature_set": feature_set,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_action_coverage": int(train["selected_action_idx"].nunique()),
        "test_action_coverage": int(test["selected_action_idx"].nunique()),
        "top1_action_accuracy": float(np.mean(predicted == y_test)),
        "top3_action_accuracy": float("nan"),
        "mean_action_cost_regret": float(np.mean(predicted_cost[valid] - oracle_cost[valid])),
        "mean_gain_vs_validation_static_action": model_gain,
        "receding_gain_vs_validation_static_action": oracle_gap,
        "fraction_of_receding_gain_recovered": model_gain / oracle_gap if oracle_gap > 0.0 else float("nan"),
        "validation_static_action_idx": static_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for run in args.run_dirs:
        seed = int(run.name.split("seed", 1)[1].split("_", 1)[0])
        train = pd.read_csv(run / "receding_oracle_l8_validation_trace" / "receding_oracle_trace.csv")
        test = pd.read_csv(run / "receding_oracle_l8_final_trace" / "receding_oracle_trace.csv")
        cost_columns = feature_columns(test, "candidate_cost_")
        feature_sets = {
            "alert_context": feature_columns(train, "alert_feature_"),
            "complete_online_state": feature_columns(train, "online_state_"),
        }
        for feature_set, columns in feature_sets.items():
            models = {
                "multinomial_logistic": make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
                ),
                "hist_gradient_boosting": HistGradientBoostingClassifier(
                    max_iter=200,
                    max_leaf_nodes=31,
                    learning_rate=0.08,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            }
            for model_name, model in models.items():
                rows.append(evaluate_model(
                    seed=seed,
                    model_name=model_name,
                    feature_set=feature_set,
                    model=model,
                    train=train,
                    test=test,
                    columns=columns,
                    cost_columns=cost_columns,
                ))
            regressors = {
                "ridge_cost_regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                "extra_trees_cost_regression": ExtraTreesRegressor(
                    n_estimators=200,
                    min_samples_leaf=4,
                    max_features=0.7,
                    n_jobs=-1,
                    random_state=seed,
                ),
            }
            for model_name, model in regressors.items():
                rows.append(evaluate_cost_regressor(
                    seed=seed,
                    model_name=model_name,
                    feature_set=feature_set,
                    model=model,
                    train=train,
                    test=test,
                    columns=columns,
                    cost_columns=cost_columns,
                ))
    output = pd.DataFrame(rows).sort_values(["feature_set", "model", "seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary = output.groupby(["feature_set", "model"], as_index=False).agg(
        seeds=("seed", "count"),
        mean_top1_accuracy=("top1_action_accuracy", "mean"),
        mean_top3_accuracy=("top3_action_accuracy", "mean"),
        mean_action_cost_regret=("mean_action_cost_regret", "mean"),
        mean_gain_vs_static=("mean_gain_vs_validation_static_action", "mean"),
        mean_fraction_recovered=("fraction_of_receding_gain_recovered", "mean"),
        positive_gain_seeds=("mean_gain_vs_validation_static_action", lambda values: int(np.sum(values > 0.0))),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    summary_lines = [
        "# Receding-action online learnability audit",
        "",
        "```text",
        summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
        "",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
