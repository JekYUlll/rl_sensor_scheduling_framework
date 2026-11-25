#!/usr/bin/env python3
"""Fit chronological probes to Monte-Carlo expected candidate costs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_expected(
    meta,
    starts,
    *,
    steps,
    replicas,
    lookahead,
    seed_offset,
    extra_context_columns=(),
    steady_budget=None,
    startup_budget=None,
):
    obs = load_module("110_v32_audit_online_subset_observability.py", f"obs_{seed_offset}")
    noise = load_module("117_v32_audit_expected_noise_value.py", f"noise_{seed_offset}")
    from v2.custom_ppo import feasible_candidate_mask

    env, candidates = obs.build_env(
        meta,
        seed_offset=int(seed_offset),
        extra_context_columns=tuple(extra_context_columns),
        steady_budget=steady_budget,
        startup_budget=startup_budget,
    )
    action_rng = np.random.default_rng(int(meta.get("seed", 0)) + int(seed_offset) + 97)
    states, costs, masks = [], [], []
    for rollout, start in enumerate(starts):
        env.reset(start_idx=int(start))
        for _ in range(int(steps)):
            feasible = feasible_candidate_mask(env, candidates)
            if int(env.elapsed_steps) <= 0 or int(env.dwell_hold_remaining) <= 0:
                expected, _ = noise.expected_candidate_costs(
                    env, candidates, lookahead_steps=int(lookahead), replicas=int(replicas),
                    base_seed=int(meta.get("seed", 0)) * 1_000_003 + int(env.current_idx) * 1009 + rollout * 65537,
                )
                states.append(env._state().astype(np.float32))
                costs.append(expected.astype(np.float32))
                masks.append(feasible.astype(bool))
            action = int(action_rng.choice(np.flatnonzero(feasible)))
            _, _, done, _ = env.step_mask(candidates[action])
            if done:
                break
    return np.vstack(states), np.vstack(costs), np.vstack(masks), env, candidates


def fit_candidate_conditioned(
    train_x,
    train_costs,
    train_masks,
    test_x,
    candidate_masks,
    *,
    seed,
    epochs,
):
    """Fit one shared cost function of online state and candidate subset."""
    torch.manual_seed(int(seed))
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-5)
    x = (train_x - mean) / std
    tx = (test_x - mean) / std
    valid = np.asarray(train_masks, dtype=bool) & np.isfinite(train_costs)
    row_min = np.min(np.where(valid, train_costs, np.inf), axis=1, keepdims=True)
    relative = train_costs - row_min
    scale = max(float(np.std(relative[valid])), 1.0e-4)

    row_idx, action_idx = np.nonzero(valid)
    pair_x = np.concatenate(
        [x[row_idx], np.asarray(candidate_masks, dtype=np.float32)[action_idx]],
        axis=1,
    )
    pair_y = (relative[row_idx, action_idx] / scale).astype(np.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(int(pair_x.shape[1]), 96),
        torch.nn.GELU(),
        torch.nn.LayerNorm(96),
        torch.nn.Linear(96, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=5.0e-3)
    order = np.arange(len(pair_x))
    pair_x_t = torch.as_tensor(pair_x, dtype=torch.float32)
    pair_y_t = torch.as_tensor(pair_y, dtype=torch.float32)
    for epoch in range(int(epochs)):
        rng = np.random.default_rng(int(seed) + epoch)
        rng.shuffle(order)
        for begin in range(0, len(order), 256):
            idx = torch.as_tensor(order[begin : begin + 256], dtype=torch.long)
            prediction = model(pair_x_t[idx]).squeeze(-1)
            loss = torch.square(prediction - pair_y_t[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    test_pairs = np.concatenate(
        [
            np.repeat(tx, len(candidate_masks), axis=0),
            np.tile(np.asarray(candidate_masks, dtype=np.float32), (len(tx), 1)),
        ],
        axis=1,
    )
    with torch.no_grad():
        prediction = model(torch.as_tensor(test_pairs, dtype=torch.float32)).squeeze(-1)
    return prediction.numpy().reshape(len(tx), len(candidate_masks)) * scale


def validation_static_index(run_dir: Path, candidate_count: int) -> int:
    """Return the frozen validation-selected static action from an asset bundle."""
    table = pd.read_csv(run_dir / "validation_static_candidates.csv")
    if table.empty or "action_idx" not in table.columns:
        raise ValueError(f"invalid validation static ledger: {run_dir}")
    index = int(table.iloc[0]["action_idx"])
    if index < 0 or index >= int(candidate_count):
        raise ValueError(f"validation static action {index} is outside candidate geometry")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-rollouts", type=int, default=8)
    parser.add_argument("--test-rollouts", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--lookahead-steps", type=int, default=6)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--steady-budget", type=float)
    parser.add_argument("--startup-budget", type=float)
    parser.add_argument("--extra-context-column", action="append")
    parser.add_argument(
        "--static-comparator-source",
        choices=("expected_train", "validation_ledger"),
        default="expected_train",
    )
    parser.add_argument(
        "--save-datasets",
        action="store_true",
        help="Save frozen train/test states, expected costs, masks, and predictions for diagnostics.",
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(args.torch_threads)))
    except RuntimeError:
        pass
    obs = load_module("110_v32_audit_online_subset_observability.py", "obs_main")
    regret = load_module("115_v32_audit_regret_tolerant_observability.py", "regret_main")
    pooled = load_module("116_v32_audit_pooled_regret_observability.py", "pooled_main")
    datasets = []
    for run_dir in args.run_dir:
        meta = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
        train_starts = obs.starts_from_interval(
            int(meta.get("train_start_min", 0)), int(meta.get("train_start_max", 1)),
            count=int(args.train_rollouts), span=int(args.steps),
        )
        test_starts = [int(x) for x in meta.get("eval_start_indices", [])[: int(args.test_rollouts)]]
        train_x, train_costs, train_masks, train_env, train_candidates = collect_expected(
            meta, train_starts, steps=args.steps, replicas=args.replicas,
            lookahead=args.lookahead_steps, seed_offset=101_000,
            extra_context_columns=tuple(args.extra_context_column or ()),
            steady_budget=args.steady_budget, startup_budget=args.startup_budget,
        )
        test_x, test_costs, test_masks, test_env, test_candidates = collect_expected(
            meta, test_starts, steps=args.steps, replicas=args.replicas,
            lookahead=args.lookahead_steps, seed_offset=103_000,
            extra_context_columns=tuple(args.extra_context_column or ()),
            steady_budget=args.steady_budget, startup_budget=args.startup_budget,
        )
        raw_constraints = dict(meta.get("constraints", {}))
        effective_steady_budget = (
            float(args.steady_budget)
            if args.steady_budget is not None
            else float(raw_constraints.get("per_step_budget", 1.75))
        )
        effective_startup_budget = (
            float(args.startup_budget)
            if args.startup_budget is not None
            else float(raw_constraints.get("startup_peak_budget", 2.15))
        )
        if not np.array_equal(train_candidates, test_candidates):
            raise RuntimeError("train/test candidate masks differ")
        datasets.append({
            "seed": int(meta.get("seed", -1)),
            "run_dir": str(run_dir),
            "steady_budget": effective_steady_budget,
            "startup_budget": effective_startup_budget,
            "extra_context_columns": [str(x) for x in (args.extra_context_column or ())],
            "train_costs": train_costs, "train_masks": train_masks,
            "test_costs": test_costs, "test_masks": test_masks,
            "candidate_masks": train_candidates,
            "train_views": regret.feature_views(train_x, train_env),
            "test_views": regret.feature_views(test_x, test_env),
        })
    rows = []
    diagnostic_arrays: dict[str, np.ndarray] = {}
    for view in ("compact_runtime_context", "context_quality_tail"):
        train_x = np.vstack([d["train_views"][view] for d in datasets])
        train_costs = np.vstack([d["train_costs"] for d in datasets])
        train_masks = np.vstack([d["train_masks"] for d in datasets])
        test_x = np.vstack([d["test_views"][view] for d in datasets])
        offsets = np.cumsum([0] + [len(d["test_costs"]) for d in datasets])
        candidate_masks = datasets[0]["candidate_masks"]
        if any(not np.array_equal(candidate_masks, d["candidate_masks"]) for d in datasets[1:]):
            raise RuntimeError("pooled scenes have different candidate masks")
        predictions = {
            "multi_output": pooled.fit_model(
                train_x, train_costs, train_masks, test_x, model_kind="shallow",
                seed=107_000, epochs=int(args.epochs),
            ),
            "candidate_conditioned": fit_candidate_conditioned(
                train_x,
                train_costs,
                train_masks,
                test_x,
                candidate_masks,
                seed=109_000,
                epochs=int(args.epochs),
            ),
        }
        if bool(args.save_datasets):
            diagnostic_arrays[f"pooled_train_x__{view}"] = train_x.astype(np.float32)
            diagnostic_arrays[f"pooled_train_costs__{view}"] = train_costs.astype(np.float32)
            diagnostic_arrays[f"pooled_train_masks__{view}"] = train_masks.astype(bool)
            diagnostic_arrays[f"pooled_test_x__{view}"] = test_x.astype(np.float32)
            for model_name, prediction in predictions.items():
                diagnostic_arrays[f"pooled_prediction__{view}__{model_name}"] = prediction.astype(np.float32)
        for model_name, prediction in predictions.items():
            for idx, dataset in enumerate(datasets):
                sl = slice(int(offsets[idx]), int(offsets[idx + 1]))
                model_metrics = regret.metrics(
                    prediction[sl], dataset["test_costs"], dataset["test_masks"]
                )
                masked_train_costs = np.where(
                    dataset["train_masks"], dataset["train_costs"], np.nan
                )
                static_index = (
                    validation_static_index(
                        Path(dataset["run_dir"]), int(dataset["train_costs"].shape[1])
                    )
                    if args.static_comparator_source == "validation_ledger"
                    else int(np.nanargmin(np.nanmean(masked_train_costs, axis=0)))
                )
                valid = (
                    dataset["test_masks"][:, static_index]
                    & np.isfinite(dataset["test_costs"][:, static_index])
                )
                static_prediction = np.full_like(dataset["test_costs"], np.inf)
                static_prediction[valid, static_index] = 0.0
                static_metrics = regret.metrics(
                    static_prediction, dataset["test_costs"], dataset["test_masks"]
                )
                rows.append({
                    "seed": dataset["seed"], "view": view, "model": model_name,
                    "steady_budget": dataset["steady_budget"],
                    "startup_budget": dataset["startup_budget"],
                    "candidate_count": int(dataset["train_costs"].shape[1]),
                    "extra_context_columns": ";".join(dataset["extra_context_columns"]),
                    "static_comparator_source": str(args.static_comparator_source),
                    "train_selected_static_index": static_index,
                    "feature_dim": int(train_x.shape[1]), "pooled_train_rows": int(len(train_x)),
                    **{f"probe_{k}": v for k, v in model_metrics.items()},
                    **{f"static_{k}": v for k, v in static_metrics.items()},
                    "probe_minus_static_mean_regret": float(model_metrics["mean_regret"] - static_metrics["mean_regret"]),
                })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "expected_cost_observability.csv", index=False)
    (args.out_dir / "expected_cost_observability.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if bool(args.save_datasets):
        diagnostic_arrays["test_offsets"] = offsets.astype(np.int64)
        diagnostic_arrays["scene_seeds"] = np.asarray(
            [int(dataset["seed"]) for dataset in datasets], dtype=np.int64
        )
        for idx, dataset in enumerate(datasets):
            diagnostic_arrays[f"train_costs__seed{dataset['seed']}"] = dataset["train_costs"].astype(np.float32)
            diagnostic_arrays[f"train_masks__seed{dataset['seed']}"] = dataset["train_masks"].astype(bool)
            diagnostic_arrays[f"test_costs__seed{dataset['seed']}"] = dataset["test_costs"].astype(np.float32)
            diagnostic_arrays[f"test_masks__seed{dataset['seed']}"] = dataset["test_masks"].astype(bool)
            diagnostic_arrays[f"candidate_masks__seed{dataset['seed']}"] = dataset["candidate_masks"].astype(bool)
            for view_name in ("compact_runtime_context", "context_quality_tail"):
                diagnostic_arrays[f"train_x__seed{dataset['seed']}__{view_name}"] = dataset["train_views"][view_name].astype(np.float32)
                diagnostic_arrays[f"test_x__seed{dataset['seed']}__{view_name}"] = dataset["test_views"][view_name].astype(np.float32)
        np.savez_compressed(args.out_dir / "expected_cost_observability_datasets.npz", **diagnostic_arrays)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
