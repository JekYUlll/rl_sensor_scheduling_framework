#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (41, 42, 43)
DEFAULT_BUDGETS = (1.65, 1.70, 1.75)


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_single(args: argparse.Namespace, *, budget: float, seed: int, policy: str) -> None:
    tag = f"budget{budget_tag(float(budget))}_seed{int(seed)}"
    out_dir = Path(args.base_out_dir) / tag
    eval_path = out_dir / "evaluation" / "v2_eval_overall.csv"
    checkpoint_name = "custom_ppo_checkpoint.pt" if policy == "custom_ppo" else f"{policy}_checkpoint.pt"
    checkpoint_path = out_dir / checkpoint_name
    if eval_path.exists() and not bool(args.force):
        print(f"[skip] {tag}: {eval_path} already exists", flush=True)
        return

    truth_csv = Path(args.base_out_dir) / f"truth_budget{budget_tag(float(budget))}_seed{int(seed)}.csv"
    script_name = "25_v2_train_custom_ppo.py" if policy == "custom_ppo" else "29_v2_train_dqn.py"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / script_name),
        "--output-dir",
        str(out_dir),
        "--checkpoint-path",
        str(checkpoint_path),
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
        "--batch-size",
        str(int(args.batch_size)),
        "--device",
        str(args.device),
        "--oracle-device",
        str(args.oracle_device),
        "--oracle-inference-device",
        str(args.oracle_inference_device),
        "--oracle-full-open-repeat",
        str(int(args.oracle_full_open_repeat)),
        "--eval-steps",
        str(int(args.eval_steps)),
        "--eval-rollouts",
        str(int(args.eval_rollouts)),
        "--eval-event-fraction",
        str(float(args.eval_event_fraction)),
        "--learning-rate",
        str(float(args.learning_rate)),
    ]
    if policy == "custom_ppo":
        cmd.extend(
            [
                "--n-steps",
                str(int(args.n_steps)),
                "--n-epochs",
                str(int(args.n_epochs)),
                "--ppo-max-candidate-warmup",
                str(int(args.ppo_max_candidate_warmup)),
                "--awbc-coef",
                str(float(args.awbc_coef)),
                "--awbc-label-stride",
                str(int(args.awbc_label_stride)),
                "--prior-kl-coef",
                str(float(args.prior_kl_coef)),
                "--ent-coef",
                str(float(args.ent_coef)),
                "--greedy-lookahead-steps",
                str(int(args.greedy_lookahead_steps)),
                "--candidate-prior-steps",
                str(int(args.candidate_prior_steps)),
                "--candidate-prior-rollouts",
                str(int(args.candidate_prior_rollouts)),
                "--candidate-prior-scale",
                str(float(args.candidate_prior_scale)),
            ]
        )
        if bool(args.use_oracle_candidate_prior):
            cmd.append("--use-oracle-candidate-prior")
    elif policy == "dqn":
        cmd.extend(
            [
                "--replay-size",
                str(int(args.replay_size)),
                "--learning-starts",
                str(int(args.learning_starts)),
                "--train-freq",
                str(int(args.train_freq)),
                "--gradient-steps",
                str(int(args.gradient_steps)),
                "--target-update-interval",
                str(int(args.target_update_interval)),
                "--hidden-dim",
                str(int(args.hidden_dim)),
                "--n-step-return",
                str(int(args.n_step_return)),
                "--exploration-fraction",
                str(float(args.exploration_fraction)),
                "--exploration-final-eps",
                str(float(args.exploration_final_eps)),
                "--log-interval",
                str(int(args.log_interval)),
                "--dqn-max-candidate-warmup",
                str(int(args.dqn_max_candidate_warmup)),
                "--oracle-prefill-steps",
                str(int(args.oracle_prefill_steps)),
                "--oracle-prefill-lookahead-steps",
                str(int(args.oracle_prefill_lookahead_steps)),
            ]
        )
        if str(args.oracle_checkpoint):
            cmd.extend(["--oracle-checkpoint", str(args.oracle_checkpoint)])
    else:
        raise ValueError(f"Unsupported policy: {policy}")
    if bool(args.skip_evaluation):
        cmd.append("--skip-evaluation")
    cmd.extend(args.train_extra)
    run(cmd, dry_run=bool(args.dry_run))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v2 forecast-evaluation training grid.")
    parser.add_argument("--base-out-dir", default="reports/v2_forecast_eval_grid")
    parser.add_argument("--policy", choices=["custom_ppo", "dqn"], default="custom_ppo")
    parser.add_argument("--policies", nargs="+", choices=["custom_ppo", "dqn"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seed", type=int, default=None, help="Run only one seed.")
    parser.add_argument("--budget", type=float, default=None, help="Run only one budget.")
    parser.add_argument("--truth-steps", type=int, default=8192)
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-full-open-repeat", type=int, default=3)
    parser.add_argument("--oracle-checkpoint", default="")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-device", default="cuda")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--ppo-max-candidate-warmup", type=int, default=-1)
    parser.add_argument("--awbc-coef", type=float, default=0.1)
    parser.add_argument("--awbc-label-stride", type=int, default=8)
    parser.add_argument("--prior-kl-coef", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--greedy-lookahead-steps", type=int, default=2)
    parser.add_argument("--use-oracle-candidate-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.5)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-step-return", type=int, default=3)
    parser.add_argument("--exploration-fraction", type=float, default=0.20)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--dqn-max-candidate-warmup", type=int, default=-1)
    parser.add_argument("--oracle-prefill-steps", type=int, default=0)
    parser.add_argument("--oracle-prefill-lookahead-steps", type=int, default=2)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "train_extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to scripts/25_v2_train_custom_ppo.py after --.",
    )
    args = parser.parse_args()
    args.train_extra = list(args.train_extra)
    if args.train_extra and args.train_extra[0] == "--":
        args.train_extra = args.train_extra[1:]

    budgets = [float(args.budget)] if args.budget is not None else [float(x) for x in args.budgets]
    seeds = [int(args.seed)] if args.seed is not None else [int(x) for x in args.seeds]
    policies = [str(x) for x in (args.policies if args.policies is not None else [args.policy])]
    os.makedirs(args.base_out_dir, exist_ok=True)
    for policy in policies:
        for budget, seed in product(budgets, seeds):
            print(f"[grid] policy={policy} budget={budget:.2f} seed={seed}", flush=True)
            run_single(args, budget=budget, seed=seed, policy=policy)
    print("Grid experiment complete.", flush=True)


if __name__ == "__main__":
    main()
