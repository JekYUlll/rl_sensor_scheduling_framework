#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGETS = (1.65, 1.70, 1.75)
DEFAULT_TRAIN_SEEDS = tuple(range(41, 51))


ABLATION_VARIANTS: dict[str, list[str]] = {
    "full_pd_ppo": [],
    "minus_oracle_prior": ["--no-use-oracle-candidate-prior", "--prior-kl-coef", "0.0"],
    "minus_masked_actor": ["--no-use-action-mask"],
    "minus_action_embedding": ["--no-use-action-embedding"],
    "minus_event_aware_critic": ["--no-event-aware-critic"],
    "minus_awbc": ["--awbc-coef", "0.0"],
}


DQN_DIAGNOSTIC_STAGES: dict[str, list[str]] = {
    # D1: masked feasible-subset actor and structured action embedding only.
    "D1_masked_actor": [
        "--no-event-aware-critic",
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    # D2: add EventAwareCritic while keeping AWBC and oracle prior disabled.
    "D2_event_critic": [
        "--event-aware-critic",
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    # D3: add AWBC, still no oracle-calibrated prior. This is the missing
    # diagnostic stage requested in docs/exp-sup.md.
    "D3_event_critic_awbc": [
        "--event-aware-critic",
        "--awbc-coef",
        "0.1",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    # D4: add the oracle-calibrated prior and KL regularizer.
    "D4_oracle_prior": [
        "--event-aware-critic",
        "--awbc-coef",
        "0.1",
        "--use-oracle-candidate-prior",
        "--prior-kl-coef",
        "1.0",
    ],
}


HYPERPARAM_SWEEPS: dict[str, list[tuple[str, list[str]]]] = {
    "lambda_awbc": [
        ("0.01", ["--awbc-coef", "0.01"]),
        ("0.05", ["--awbc-coef", "0.05"]),
        ("0.10", ["--awbc-coef", "0.10"]),
        ("0.20", ["--awbc-coef", "0.20"]),
        ("0.50", ["--awbc-coef", "0.50"]),
    ],
    "lambda_kl": [
        ("0.10", ["--prior-kl-coef", "0.10"]),
        ("0.50", ["--prior-kl-coef", "0.50"]),
        ("1.00", ["--prior-kl-coef", "1.00"]),
        ("2.00", ["--prior-kl-coef", "2.00"]),
        ("5.00", ["--prior-kl-coef", "5.00"]),
    ],
    "embed_dim": [
        ("16", ["--embed-dim", "16"]),
        ("32", ["--embed-dim", "32"]),
        ("64", ["--embed-dim", "64"]),
        ("128", ["--embed-dim", "128"]),
    ],
    "lambda_warm": [
        ("0.1x", ["--lambda-warmup-abort", "0.008"]),
        ("0.5x", ["--lambda-warmup-abort", "0.040"]),
        ("1.0x", ["--lambda-warmup-abort", "0.080"]),
        ("2.0x", ["--lambda-warmup-abort", "0.160"]),
        ("10.0x", ["--lambda-warmup-abort", "0.800"]),
    ],
}


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def value_tag(value: str) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("+", "p").replace("/", "_")


def run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def existing_eval(out_dir: Path) -> bool:
    return (out_dir / "evaluation" / "v2_eval_overall.csv").exists()


def training_common_args(args: argparse.Namespace, *, out_dir: Path, budget: float, seed: int) -> list[str]:
    truth_csv = Path(args.main_grid_dir) / f"truth_budget{budget_tag(budget)}_seed{int(seed)}.csv"
    return [
        "--output-dir",
        str(out_dir),
        "--checkpoint-path",
        str(out_dir / "custom_ppo_checkpoint.pt"),
        "--truth-csv",
        str(truth_csv),
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
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--eval-steps",
        str(int(args.eval_steps)),
        "--eval-rollouts",
        str(int(args.eval_rollouts)),
        "--eval-event-fraction",
        str(float(args.eval_event_fraction)),
        "--device",
        str(args.device),
        "--oracle-device",
        str(args.oracle_device),
        "--oracle-inference-device",
        str(args.oracle_inference_device),
        "--use-oracle-candidate-prior",
        "--candidate-prior-steps",
        str(int(args.candidate_prior_steps)),
        "--candidate-prior-rollouts",
        str(int(args.candidate_prior_rollouts)),
        "--candidate-prior-scale",
        str(float(args.candidate_prior_scale)),
    ]


def run_main_grid(args: argparse.Namespace) -> None:
    grid_args = [
        sys.executable,
        str(ROOT / "scripts" / "26_v2_grid_experiment.py"),
        "--base-out-dir",
        str(args.main_grid_dir),
        "--policy",
        "custom_ppo",
        "--budgets",
        *[str(float(budget)) for budget in args.budgets],
        "--seeds",
        *[str(int(seed)) for seed in args.seeds],
        "--truth-steps",
        str(int(args.truth_steps)),
        "--oracle-rollout-steps",
        str(int(args.oracle_rollout_steps)),
        "--oracle-epochs",
        str(int(args.oracle_epochs)),
        "--total-timesteps",
        str(int(args.total_timesteps)),
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--eval-steps",
        str(int(args.eval_steps)),
        "--eval-rollouts",
        str(int(args.eval_rollouts)),
        "--eval-event-fraction",
        str(float(args.eval_event_fraction)),
        "--device",
        str(args.device),
        "--oracle-device",
        str(args.oracle_device),
        "--oracle-inference-device",
        str(args.oracle_inference_device),
        "--startup-peak-budget",
        str(float(args.startup_peak_budget)),
    ]
    if args.force:
        grid_args.append("--force")
    run(grid_args, dry_run=bool(args.dry_run))

    if bool(args.include_dqn):
        dqn_args = list(grid_args)
        dqn_args[dqn_args.index("custom_ppo")] = "dqn"
        run(dqn_args, dry_run=bool(args.dry_run))


def run_custom_variant(
    args: argparse.Namespace,
    *,
    experiment: str,
    variant: str,
    extra: list[str],
    budget: float,
    seed: int,
) -> None:
    out_dir = Path(args.out_dir) / experiment / variant / f"budget{budget_tag(budget)}_seed{int(seed)}"
    if existing_eval(out_dir) and not bool(args.force):
        print(f"[skip] {experiment}/{variant} budget={budget:.2f} seed={seed}: {out_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25_v2_train_custom_ppo.py"),
        *training_common_args(args, out_dir=out_dir, budget=budget, seed=seed),
        *extra,
    ]
    run(cmd, dry_run=bool(args.dry_run))


def run_ablation(args: argparse.Namespace) -> None:
    for variant, extra in ABLATION_VARIANTS.items():
        for budget, seed in product(args.budgets, args.seeds):
            run_custom_variant(args, experiment="A1_ablation", variant=variant, extra=extra, budget=float(budget), seed=int(seed))


def run_diagnostic(args: argparse.Namespace) -> None:
    budgets = [float(args.focus_budget)]
    seeds = list(args.seeds[: int(args.diagnostic_seed_count)])
    for stage, extra in DQN_DIAGNOSTIC_STAGES.items():
        for budget, seed in product(budgets, seeds):
            run_custom_variant(args, experiment="A2_diagnostic", variant=stage, extra=extra, budget=float(budget), seed=int(seed))


def run_hyperparam(args: argparse.Namespace) -> None:
    seeds = list(args.seeds[: int(args.sensitivity_seed_count)])
    for param_name, choices in HYPERPARAM_SWEEPS.items():
        for value, extra in choices:
            variant = f"{param_name}_{value_tag(value)}"
            for seed in seeds:
                run_custom_variant(
                    args,
                    experiment=f"H1_{param_name}",
                    variant=variant,
                    extra=extra,
                    budget=float(args.focus_budget),
                    seed=int(seed),
                )


def run_extreme_eval(args: argparse.Namespace) -> None:
    for budget, seed in product(args.budgets, args.seeds):
        run_dir = Path(args.main_grid_dir) / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
        if not run_dir.exists():
            print(f"[warn] missing trained run for E1: {run_dir}", flush=True)
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "32_v2_condition_eval.py"),
            "--run-dir",
            str(run_dir),
            "--out-root",
            str(Path(args.out_dir) / "E1_condition_eval"),
            "--steps",
            str(int(args.eval_steps)),
            "--rollouts",
            str(int(args.eval_rollouts)),
            "--per-step-budget",
            str(float(budget)),
            "--startup-peak-budget",
            str(float(args.startup_peak_budget)),
            "--forecast-oracle-device",
            str(args.oracle_inference_device),
        ]
        run(cmd, dry_run=bool(args.dry_run))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 supplementary experiments from docs/exp-sup.md.")
    parser.add_argument("--experiments", nargs="+", default=["s1"], choices=["s1", "a1", "a2", "h1", "e1"])
    parser.add_argument("--main-grid-dir", default="reports/v2_forecast_eval_grid_prior_kl1")
    parser.add_argument("--out-dir", default="reports/v2_supplement_experiments")
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--focus-budget", type=float, default=1.70)
    parser.add_argument("--diagnostic-seed-count", type=int, default=5)
    parser.add_argument("--sensitivity-seed-count", type=int, default=5)
    parser.add_argument("--truth-steps", type=int, default=8192)
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-device", default="cuda")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--include-dqn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.budgets = [float(x) for x in args.budgets]
    args.seeds = [int(x) for x in args.seeds]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        if experiment == "s1":
            run_main_grid(args)
        elif experiment == "a1":
            run_ablation(args)
        elif experiment == "a2":
            run_diagnostic(args)
        elif experiment == "h1":
            run_hyperparam(args)
        elif experiment == "e1":
            run_extreme_eval(args)
        else:
            raise AssertionError(experiment)


if __name__ == "__main__":
    main()
