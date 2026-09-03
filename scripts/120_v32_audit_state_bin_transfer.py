#!/usr/bin/env python3
"""Audit whether forecast-demand bins transfer to expected subset value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEMAND_SLICE = slice(7, 10)  # power/cycle prefix + four nowcasts precede three demands


def bin_ids(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    high = np.asarray(values[:, DEMAND_SLICE] >= thresholds.reshape(1, 3), dtype=np.int64)
    return high[:, 0] * 4 + high[:, 1] * 2 + high[:, 2]


def action_cost_table(
    costs: np.ndarray,
    masks: np.ndarray,
    bins: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    table = np.full((8, candidate_count), np.inf, dtype=float)
    global_cost = np.full(candidate_count, np.inf, dtype=float)
    for action in range(candidate_count):
        valid = masks[:, action] & np.isfinite(costs[:, action])
        if np.any(valid):
            global_cost[action] = float(np.mean(costs[valid, action]))
        for state in range(8):
            rows = valid & (bins == state)
            if np.any(rows):
                table[state, action] = float(np.mean(costs[rows, action]))
    missing = ~np.isfinite(table)
    table[missing] = np.broadcast_to(global_cost, table.shape)[missing]
    return table


def evaluate(
    table: np.ndarray,
    bins: np.ndarray,
    costs: np.ndarray,
    masks: np.ndarray,
    static_index: int,
) -> dict[str, float | int]:
    predicted = np.where(masks, table[bins], np.inf)
    selected = np.argmin(predicted, axis=1)
    row = np.arange(len(costs))
    selected_cost = costs[row, selected]
    best_cost = np.min(np.where(masks, costs, np.inf), axis=1)
    static_valid = masks[:, static_index] & np.isfinite(costs[:, static_index])
    if not np.all(static_valid):
        raise ValueError("frozen validation static action is not feasible on every test row")
    static_cost = costs[:, static_index]
    return {
        "rows": int(len(costs)),
        "lookup_cost_mean": float(np.mean(selected_cost)),
        "static_cost_mean": float(np.mean(static_cost)),
        "static_minus_lookup_cost": float(np.mean(static_cost - selected_cost)),
        "lookup_regret_mean": float(np.mean(selected_cost - best_cost)),
        "static_regret_mean": float(np.mean(static_cost - best_cost)),
        "lookup_exact_top1": float(np.mean(selected == np.argmin(np.where(masks, costs, np.inf), axis=1))),
        "lookup_unique_actions": int(np.unique(selected).size),
    }


def finite_ranking(values: np.ndarray) -> list[int]:
    """Return finite action indices ordered by ascending expected cost."""
    finite = np.flatnonzero(np.isfinite(values))
    return finite[np.argsort(values[finite])].astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    archive = np.load(args.dataset)
    seeds = [int(x) for x in archive["scene_seeds"]]
    if len(seeds) != len(args.run_dir):
        raise ValueError("run-dir count does not match saved scene seeds")
    view = "context_quality_tail"
    train_x = np.vstack([archive[f"train_x__seed{seed}__{view}"] for seed in seeds])
    train_costs = np.vstack([archive[f"train_costs__seed{seed}"] for seed in seeds])
    train_masks = np.vstack([archive[f"train_masks__seed{seed}"] for seed in seeds])
    if train_x.shape[1] < DEMAND_SLICE.stop:
        raise ValueError("context-quality view does not contain the three forecast-demand features")
    thresholds = np.median(train_x[:, DEMAND_SLICE], axis=0)
    train_bins = bin_ids(train_x, thresholds)
    table = action_cost_table(train_costs, train_masks, train_bins, train_costs.shape[1])

    rows = []
    bin_rows = []
    for seed, run_dir in zip(seeds, args.run_dir, strict=True):
        ledger = pd.read_csv(run_dir / "validation_static_candidates.csv")
        static_index = int(ledger.iloc[0]["action_idx"])
        test_x = archive[f"test_x__seed{seed}__{view}"]
        test_costs = archive[f"test_costs__seed{seed}"]
        test_masks = archive[f"test_masks__seed{seed}"]
        test_bins = bin_ids(test_x, thresholds)
        metrics = evaluate(table, test_bins, test_costs, test_masks, static_index)
        rows.append({"seed": seed, "static_index": static_index, **metrics})
        for state in range(8):
            tr = train_bins == state
            te = test_bins == state
            test_means = np.full(test_costs.shape[1], np.nan, dtype=float)
            for action in range(test_costs.shape[1]):
                valid = te & test_masks[:, action] & np.isfinite(test_costs[:, action])
                if np.any(valid):
                    test_means[action] = float(np.mean(test_costs[valid, action]))
            finite = np.isfinite(test_means) & np.isfinite(table[state])
            train_rank = finite_ranking(table[state])
            test_rank = finite_ranking(test_means)
            train_best = train_rank[0] if train_rank else -1
            train_second = train_rank[1] if len(train_rank) > 1 else -1
            test_best = test_rank[0] if test_rank else -1
            test_second = test_rank[1] if len(test_rank) > 1 else -1
            rank_corr = (
                float(pd.Series(table[state, finite]).rank().corr(pd.Series(test_means[finite]).rank()))
                if np.sum(finite) >= 2 and np.any(te)
                else float("nan")
            )
            bin_rows.append({
                "seed": seed,
                "bin_id": state,
                "train_rows_pooled": int(np.sum(tr)),
                "test_rows": int(np.sum(te)),
                "train_selected_action": train_best,
                "train_second_action": train_second,
                "train_best_cost": float(table[state, train_best]) if train_best >= 0 else float("nan"),
                "train_second_cost": float(table[state, train_second]) if train_second >= 0 else float("nan"),
                "train_second_minus_best": (
                    float(table[state, train_second] - table[state, train_best])
                    if train_second >= 0 else float("nan")
                ),
                "train_static_cost": float(table[state, static_index]),
                "train_static_minus_best": float(table[state, static_index] - table[state, train_best]),
                "test_best_action": test_best,
                "test_second_action": test_second,
                "test_best_cost": float(test_means[test_best]) if test_best >= 0 else float("nan"),
                "test_second_cost": float(test_means[test_second]) if test_second >= 0 else float("nan"),
                "test_second_minus_best": (
                    float(test_means[test_second] - test_means[test_best])
                    if test_second >= 0 else float("nan")
                ),
                "test_train_action_cost": float(test_means[train_best]) if train_best >= 0 else float("nan"),
                "test_train_action_minus_best": (
                    float(test_means[train_best] - test_means[test_best])
                    if train_best >= 0 and test_best >= 0 else float("nan")
                ),
                "test_static_cost": float(test_means[static_index]),
                "test_static_minus_best": (
                    float(test_means[static_index] - test_means[test_best])
                    if test_best >= 0 else float("nan")
                ),
                "train_top3_actions": ",".join(str(x) for x in train_rank[:3]),
                "test_top3_actions": ",".join(str(x) for x in test_rank[:3]),
                "action_rank_correlation": rank_corr,
            })

    args.out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "state_bin_transfer_summary.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(args.out_dir / "state_bin_transfer_bins.csv", index=False)
    payload = {
        "demand_feature_indices": [7, 8, 9],
        "demand_feature_order": ["flux", "particle", "thermal"],
        "pooled_train_medians": thresholds.tolist(),
        "rows": rows,
    }
    (args.out_dir / "state_bin_transfer_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
