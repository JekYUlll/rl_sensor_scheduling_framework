#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEEDS = (41, 42, 43)
DEFAULT_BUDGETS = (1.65, 1.70, 1.75)
POLICY_ORDER = (
    "full_open_unconstrained",
    "feasible_static_projected",
    "custom_ppo",
    "round_robin",
    "aoi",
    "dqn",
    "cmdp_dqn",
    "random",
)


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def format_mean_std(mean: float, std: float) -> str:
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def load_csv_grid(input_dir: Path, *, filename: str, budgets: list[float], seeds: list[int]) -> pd.DataFrame:
    frames = []
    for budget, seed in product(budgets, seeds):
        path = input_dir / f"budget{budget_tag(budget)}_seed{seed}" / "evaluation" / filename
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(seed))
        frame.insert(0, "budget", float(budget))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_csv_grids(input_dirs: list[Path], *, filename: str, budgets: list[float], seeds: list[int]) -> pd.DataFrame:
    frames = []
    for source_idx, input_dir in enumerate(input_dirs):
        frame = load_csv_grid(input_dir, filename=filename, budgets=budgets, seeds=seeds)
        if frame.empty:
            continue
        frame.insert(0, "source_idx", int(source_idx))
        frame.insert(0, "source_dir", str(input_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    key_cols = [name for name in ("budget", "seed", "policy", "variable", "condition", "group", "sensor") if name in combined.columns]
    if key_cols:
        combined = combined.sort_values("source_idx").drop_duplicates(key_cols, keep="last")
    return combined.drop(columns=["source_idx"], errors="ignore")


def generate_table2(overall: pd.DataFrame, *, budgets: list[float], order_budget: float) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    rows = []
    for budget in budgets:
        df_budget = overall[overall["budget"] == float(budget)]
        for policy, group in df_budget.groupby("policy"):
            values = group["forecast_weighted_mae_overall"].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "policy": str(policy),
                    "budget": float(budget),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(len(values)),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["mean_pm_std"] = [format_mean_std(m, s) for m, s in zip(summary["mean"], summary["std"])]
    wide = summary.pivot_table(index="policy", columns="budget", values="mean_pm_std", aggfunc="first")
    order_rows = summary[summary["budget"] == float(order_budget)].set_index("policy")["mean"]
    if not order_rows.empty:
        ordered = list(order_rows.sort_values().index)
    else:
        ordered = [policy for policy in POLICY_ORDER if policy in wide.index]
    ordered.extend(policy for policy in POLICY_ORDER if policy in wide.index and policy not in ordered)
    ordered.extend(policy for policy in wide.index if policy not in ordered)
    return wide.loc[ordered]


def generate_table3_by_variable(by_variable: pd.DataFrame, *, budget: float) -> pd.DataFrame:
    if by_variable.empty:
        return pd.DataFrame()
    subset = by_variable[by_variable["budget"] == float(budget)]
    if subset.empty or "forecast_mae" not in subset.columns:
        return pd.DataFrame()
    agg = subset.groupby(["policy", "variable"], as_index=False)["forecast_mae"].mean()
    wide = agg.pivot_table(index="policy", columns="variable", values="forecast_mae", aggfunc="first")
    return order_policy_index(wide)


def generate_table3_by_condition(by_condition: pd.DataFrame, *, budget: float) -> pd.DataFrame:
    if by_condition.empty:
        return pd.DataFrame()
    subset = by_condition[by_condition["budget"] == float(budget)]
    if subset.empty or "forecast_weighted_mae" not in subset.columns:
        return pd.DataFrame()
    agg = (
        subset.groupby(["policy", "condition"])["forecast_weighted_mae"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1) if len(x) > 1 else 0.0, n="count")
        .reset_index()
    )
    agg["mean_pm_std"] = [format_mean_std(m, s) for m, s in zip(agg["mean"], agg["std"])]
    wide = agg.pivot_table(index="policy", columns="condition", values="mean_pm_std", aggfunc="first")
    return order_policy_index(wide)


def order_policy_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = [policy for policy in POLICY_ORDER if policy in frame.index]
    ordered.extend(policy for policy in frame.index if policy not in ordered)
    return frame.loc[ordered]


def export_learning_curves(input_dir: Path, output_dir: Path, *, budgets: list[float], seeds: list[int], focus_budget: float) -> None:
    curve_dir = output_dir / "figure4_learning_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        run_dir = input_dir / f"budget{budget_tag(focus_budget)}_seed{seed}"
        source = run_dir / "custom_ppo_training_log.csv"
        if not source.exists():
            source = run_dir / "custom_ppo_training_history_live.json"
        if not source.exists():
            print(f"WARNING: no training curve source for seed={seed}: {run_dir}")
            continue
        target = curve_dir / f"seed{seed}_training_log{source.suffix}"
        shutil.copy(source, target)
        rows.append({"budget": float(focus_budget), "seed": int(seed), "path": str(target)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(curve_dir / "manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate v2 grid results into paper-ready CSV tables.")
    parser.add_argument("--input-dir", default="reports/v2_forecast_eval_grid")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=None,
        help=(
            "Optional ordered list of grid directories to merge. Later directories "
            "override duplicate budget/seed/policy rows, which is useful when adding "
            "a new policy such as DQN to an existing PPO grid."
        ),
    )
    parser.add_argument("--output-dir", default="reports/v2_paper_tables")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--focus-budget", type=float, default=1.70)
    args = parser.parse_args()

    input_dirs = [Path(path) for path in (args.input_dirs if args.input_dirs is not None else [args.input_dir])]
    input_dir = input_dirs[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds]
    budgets = [float(x) for x in args.budgets]

    overall = load_csv_grids(input_dirs, filename="v2_eval_overall.csv", budgets=budgets, seeds=seeds)
    by_variable = load_csv_grids(input_dirs, filename="v2_eval_by_variable.csv", budgets=budgets, seeds=seeds)
    by_condition = load_csv_grids(input_dirs, filename="v2_eval_by_condition.csv", budgets=budgets, seeds=seeds)

    if not overall.empty:
        overall.to_csv(output_dir / "overall_long.csv", index=False)
    if not by_variable.empty:
        by_variable.to_csv(output_dir / "by_variable_long.csv", index=False)
    if not by_condition.empty:
        by_condition.to_csv(output_dir / "by_condition_long.csv", index=False)

    table2 = generate_table2(overall, budgets=budgets, order_budget=float(args.focus_budget))
    table2.to_csv(output_dir / "table2_main_results.csv")
    print(output_dir / "table2_main_results.csv")
    if not table2.empty:
        print(table2.to_string())

    table3a = generate_table3_by_variable(by_variable, budget=float(args.focus_budget))
    if not table3a.empty:
        table3a.to_csv(output_dir / "table3_by_variable.csv")
        print(output_dir / "table3_by_variable.csv")

    table3b = generate_table3_by_condition(by_condition, budget=float(args.focus_budget))
    if not table3b.empty:
        table3b.to_csv(output_dir / "table3_by_condition.csv")
        print(output_dir / "table3_by_condition.csv")

    export_learning_curves(
        input_dir,
        output_dir,
        budgets=budgets,
        seeds=seeds,
        focus_budget=float(args.focus_budget),
    )
    print(f"Saved paper tables to {output_dir}")


if __name__ == "__main__":
    main()
