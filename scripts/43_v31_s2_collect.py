#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EVAL_NAME = "v2_eval_overall.csv"
PRIMARY_METRIC = "forecast_weighted_mae_overall"
PD_PPO_POLICY = "custom_ppo"
FULL_OPEN_POLICY = "full_open_unconstrained"
STATIC_POLICY = "feasible_static_projected"
NAIVE_BASELINES = ("round_robin", "aoi", "random")
SIGNIFICANCE_BASELINES = (
    "round_robin",
    "aoi",
    "random",
    "full_open_unconstrained",
)


def _budget_from_tag(tag: str) -> float | None:
    match = re.search(r"budget(?P<value>\d+p\d+)", tag)
    if not match:
        return None
    return float(match.group("value").replace("p", "."))


def _seed_from_tag(tag: str) -> int | None:
    match = re.search(r"seed(?P<value>\d+)", tag)
    if not match:
        return None
    return int(match.group("value"))


def _iter_eval_paths(out_dir: Path) -> Iterable[Path]:
    raw_dir = out_dir / "raw"
    if raw_dir.exists():
        yield from sorted(raw_dir.glob(f"*budget*_seed*/evaluation/{EVAL_NAME}"))
    yield from sorted(out_dir.glob(f"*budget*_seed*/evaluation/{EVAL_NAME}"))


def _read_long(out_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[pd.DataFrame] = []
    missing_from_done: list[str] = []
    done_dir = out_dir / "done"
    if done_dir.exists():
        for marker in sorted(done_dir.glob("budget*_seed*.done")):
            budget = _budget_from_tag(marker.stem)
            seed = _seed_from_tag(marker.stem)
            if budget is None or seed is None:
                continue
            expected = out_dir / "raw" / marker.stem / "evaluation" / EVAL_NAME
            if not expected.exists():
                missing_from_done.append(str(expected))

    seen: set[Path] = set()
    for eval_path in _iter_eval_paths(out_dir):
        eval_path = eval_path.resolve()
        if eval_path in seen:
            continue
        seen.add(eval_path)
        run_tag = eval_path.parents[1].name
        budget = _budget_from_tag(run_tag)
        seed = _seed_from_tag(run_tag)
        if budget is None or seed is None:
            continue
        df = pd.read_csv(eval_path)
        df.insert(0, "eval_path", str(eval_path))
        df.insert(0, "seed", seed)
        df.insert(0, "budget", budget)
        df.insert(0, "run_tag", run_tag)
        rows.append(df)

    if not rows:
        return pd.DataFrame(), missing_from_done
    return pd.concat(rows, ignore_index=True), missing_from_done


def _stats(long: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    grouped = long.groupby(["budget", "policy"], as_index=False)[metric_cols].agg(["mean", "std", "count"])
    grouped.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    return grouped.reset_index(drop=True)


def _condition_long(long: pd.DataFrame) -> pd.DataFrame:
    condition_map = {
        "overall": "forecast_weighted_mae_overall",
        "event": "forecast_weighted_mae_event",
        "non_event": "forecast_weighted_mae_non_event",
        "low_temp": "forecast_weighted_mae_low_temp",
        "normal_temp": "forecast_weighted_mae_normal",
    }
    rows = []
    base_cols = ["run_tag", "budget", "seed", "policy"]
    for condition, column in condition_map.items():
        if column not in long.columns:
            continue
        df = long[base_cols + [column]].copy()
        df.insert(4, "condition", condition)
        df.rename(columns={column: "forecast_weighted_mae"}, inplace=True)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _wilcoxon_pvalue(pdppo_values: np.ndarray, baseline_values: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(pdppo_values, baseline_values, alternative="two-sided", zero_method="wilcox")
        return float(result.pvalue)
    except Exception:
        diffs = pdppo_values - baseline_values
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[diffs != 0]
        if diffs.size == 0:
            return 1.0
        # Sign-test fallback: conservative and dependency-light.
        positives = int(np.sum(diffs > 0))
        negatives = int(np.sum(diffs < 0))
        k = min(positives, negatives)
        n = int(positives + negatives)
        prob = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
        return float(min(1.0, 2.0 * prob))


def _significance(
    long: pd.DataFrame,
    *,
    metric: str,
    bonferroni_family: int,
    static_policy: str = STATIC_POLICY,
) -> pd.DataFrame:
    if metric not in long.columns:
        return pd.DataFrame()
    rows = []
    for budget, budget_df in sorted(long.groupby("budget")):
        pivot = budget_df.pivot_table(index="seed", columns="policy", values=metric, aggfunc="mean")
        if PD_PPO_POLICY not in pivot.columns:
            continue
        for baseline in (*SIGNIFICANCE_BASELINES[:-1], str(static_policy), SIGNIFICANCE_BASELINES[-1]):
            if baseline not in pivot.columns:
                continue
            paired = pivot[[PD_PPO_POLICY, baseline]].dropna()
            if paired.empty:
                continue
            pdppo = paired[PD_PPO_POLICY].to_numpy(dtype=float)
            other = paired[baseline].to_numpy(dtype=float)
            p_value = _wilcoxon_pvalue(pdppo, other)
            diff = pdppo - other
            rows.append(
                {
                    "budget": float(budget),
                    "comparison": f"{PD_PPO_POLICY}_vs_{baseline}",
                    "baseline": baseline,
                    "n": int(paired.shape[0]),
                    "pdppo_mean": float(np.mean(pdppo)),
                    "baseline_mean": float(np.mean(other)),
                    "mean_diff_pdppo_minus_baseline": float(np.mean(diff)),
                    "relative_diff_vs_baseline": float(np.mean(diff) / np.mean(other)),
                    "pdppo_better_seed_count": int(np.sum(pdppo < other)),
                    "p_value_two_sided": p_value,
                    "p_value_bonferroni": float(min(1.0, p_value * int(bonferroni_family))),
                    "bonferroni_family": int(bonferroni_family),
                }
            )
    return pd.DataFrame(rows)


def _budget_check(
    long: pd.DataFrame,
    *,
    metric: str,
    static_gap_threshold: float,
    static_policy: str = STATIC_POLICY,
) -> pd.DataFrame:
    if metric not in long.columns:
        return pd.DataFrame()
    rows = []
    means = long.groupby(["budget", "policy"], as_index=False)[metric].mean()
    for budget, budget_df in sorted(means.groupby("budget")):
        values = dict(zip(budget_df["policy"], budget_df[metric], strict=False))
        pdppo = values.get(PD_PPO_POLICY, np.nan)
        static = values.get(str(static_policy), np.nan)
        best_policy = str(budget_df.sort_values(metric).iloc[0]["policy"]) if not budget_df.empty else ""
        static_gap = (pdppo - static) / static if np.isfinite(pdppo) and np.isfinite(static) and static != 0 else np.nan
        row = {
            "budget": float(budget),
            "best_policy": best_policy,
            "full_open_best": best_policy == FULL_OPEN_POLICY,
            "pdppo_mean": pdppo,
            "static_mean": static,
            "static_policy": str(static_policy),
            "pdppo_static_gap": static_gap,
            "pdppo_static_gap_ok": bool(np.isfinite(static_gap) and static_gap <= float(static_gap_threshold)),
        }
        for baseline in NAIVE_BASELINES:
            baseline_mean = values.get(baseline, np.nan)
            row[f"{baseline}_mean"] = baseline_mean
            row[f"pdppo_beats_{baseline}"] = bool(np.isfinite(pdppo) and np.isfinite(baseline_mean) and pdppo < baseline_mean)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V3.1 S2 outputs and compute summary checks.")
    parser.add_argument("--out-dir", default="reports/v31_s2_main")
    parser.add_argument("--metric", default=PRIMARY_METRIC)
    parser.add_argument("--bonferroni-family", type=int, default=6)
    parser.add_argument("--static-gap-threshold", type=float, default=0.03)
    parser.add_argument(
        "--static-policy",
        default=STATIC_POLICY,
        help="Policy name to use as the selected static comparator in checks and significance tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long, missing_from_done = _read_long(out_dir)
    if missing_from_done:
        missing_path = out_dir / "v31_s2_missing_from_done.txt"
        missing_path.write_text("\n".join(missing_from_done) + "\n", encoding="utf-8")
        print(f"[collect] missing done outputs: {len(missing_from_done)} -> {missing_path}", flush=True)
    if long.empty:
        print(f"[collect] no evaluation files found under {out_dir}", flush=True)
        return

    long_path = out_dir / "v31_s2_overall_long.csv"
    long.to_csv(long_path, index=False)

    metric_cols = [
        col
        for col in [
            "forecast_weighted_mae_overall",
            "forecast_raw_mae_overall",
            "forecast_weighted_mae_event",
            "forecast_weighted_mae_non_event",
            "forecast_weighted_mae_low_temp",
            "forecast_weighted_mae_normal",
            "obs_reconstruction_mae",
            "weighted_normalized_mae",
            "oracle_loss_mean",
            "power_mean",
            "warmup_abort_rate",
        ]
        if col in long.columns
    ]
    stats = _stats(long, metric_cols)
    stats_path = out_dir / "v31_s2_main_stats.csv"
    stats.to_csv(stats_path, index=False)

    condition = _condition_long(long)
    if not condition.empty:
        condition_path = out_dir / "v31_s2_condition_long.csv"
        condition.to_csv(condition_path, index=False)
        condition_stats = (
            condition.groupby(["budget", "condition", "policy"], as_index=False)["forecast_weighted_mae"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        condition_stats.to_csv(out_dir / "v31_s2_condition_stats.csv", index=False)

    significance = _significance(
        long,
        metric=str(args.metric),
        bonferroni_family=int(args.bonferroni_family),
        static_policy=str(args.static_policy),
    )
    if not significance.empty:
        significance.to_csv(out_dir / "v31_s2_significance.csv", index=False)

    budget_check = _budget_check(
        long,
        metric=str(args.metric),
        static_gap_threshold=float(args.static_gap_threshold),
        static_policy=str(args.static_policy),
    )
    if not budget_check.empty:
        budget_check.to_csv(out_dir / "v31_s2_budget_check.csv", index=False)

    print(long_path, flush=True)
    print(stats_path, flush=True)
    if not budget_check.empty:
        print((out_dir / "v31_s2_budget_check.csv"), flush=True)
        print(budget_check.to_string(index=False), flush=True)
    if not significance.empty:
        print((out_dir / "v31_s2_significance.csv"), flush=True)


if __name__ == "__main__":
    main()
