#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible v2 benchmark grid and aggregate evaluation tables."
    )
    parser.add_argument("--base-out-dir", default="reports/v2_benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.65, 1.75])
    parser.add_argument("--truth-steps", type=int, default=8192)
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-epochs", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=80000)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-device", default="cuda")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to scripts/23_v2_train_ppo.py after --.",
    )
    args = parser.parse_args()

    base_out = Path(args.base_out_dir)
    train_extra = list(args.train_args)
    if train_extra and train_extra[0] == "--":
        train_extra = train_extra[1:]

    aggregate_rows = []
    for budget in args.budgets:
        for seed in args.seeds:
            tag = f"budget{float(budget):.2f}_seed{int(seed)}".replace(".", "p")
            out_dir = base_out / tag
            train_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "23_v2_train_ppo.py"),
                "--out-dir",
                str(out_dir),
                "--truth-csv",
                str(base_out / f"truth_seed{int(seed)}.csv"),
                "--seed",
                str(int(seed)),
                "--per-step-budget",
                str(float(budget)),
                "--startup-peak-budget",
                str(float(args.startup_peak_budget)),
                "--truth-steps",
                str(int(args.truth_steps)),
                "--oracle-rollout-steps",
                str(int(args.oracle_rollout_steps)),
                "--oracle-epochs",
                str(int(args.oracle_epochs)),
                "--total-timesteps",
                str(int(args.total_timesteps)),
                "--eval-steps",
                str(int(args.eval_steps)),
                "--n-envs",
                str(int(args.n_envs)),
                "--device",
                str(args.device),
                "--oracle-device",
                str(args.oracle_device),
                "--oracle-inference-device",
                str(args.oracle_inference_device),
                "--diagnostic-freq",
                str(max(10000, int(args.total_timesteps) // 8)),
                *train_extra,
            ]
            run(train_cmd, dry_run=bool(args.dry_run))

            eval_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "24_v2_evaluate_rollouts.py"),
                "--run-dir",
                str(out_dir),
                "--per-step-budget",
                str(float(budget)),
                "--startup-peak-budget",
                str(float(args.startup_peak_budget)),
            ]
            run(eval_cmd, dry_run=bool(args.dry_run))

            figure_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "26_v2_make_figures.py"),
                "--run-dir",
                str(out_dir),
            ]
            run(figure_cmd, dry_run=bool(args.dry_run))

            diagnose_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "27_v2_diagnose_action_landscape.py"),
                "--run-dir",
                str(out_dir),
                "--steps",
                str(min(256, int(args.eval_steps))),
            ]
            run(diagnose_cmd, dry_run=bool(args.dry_run))

            eval_path = out_dir / "evaluation" / "v2_eval_overall.csv"
            if eval_path.exists():
                df = pd.read_csv(eval_path)
                df.insert(0, "seed", int(seed))
                df.insert(0, "per_step_budget", float(budget))
                df.insert(0, "run_tag", tag)
                aggregate_rows.append(df)

    if aggregate_rows and not args.dry_run:
        base_out.mkdir(parents=True, exist_ok=True)
        aggregate = pd.concat(aggregate_rows, ignore_index=True)
        aggregate.to_csv(base_out / "v2_benchmark_overall_long.csv", index=False)
        summary_cols = [
            col
            for col in [
                "forecast_weighted_mae_overall",
                "forecast_weighted_mae_event",
                "forecast_weighted_mae_non_event",
                "forecast_weighted_mae_low_temp",
                "forecast_weighted_mae_normal",
                "obs_reconstruction_mae",
                "weighted_normalized_mae",
                "mae",
                "rmse",
                "dtw",
                "oracle_loss_mean",
                "power_mean",
            ]
            if col in aggregate.columns
        ]
        summary = aggregate.groupby(["per_step_budget", "policy"], as_index=False)[summary_cols].agg(["mean", "std"])
        summary.to_csv(base_out / "v2_benchmark_summary.csv")
        print(base_out / "v2_benchmark_summary.csv")


if __name__ == "__main__":
    main()
