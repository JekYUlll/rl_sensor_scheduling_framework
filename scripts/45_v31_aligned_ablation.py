#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS_A1_A2 = tuple(range(41, 51))
DEFAULT_SEEDS_H1 = tuple(range(41, 46))
DEFAULT_FOCUS_BUDGET = 1.70


A1_VARIANTS: dict[str, list[str]] = {
    "no_action_emb": ["--no-use-action-embedding"],
    "no_event_critic": ["--no-event-aware-critic"],
    "no_awbc": ["--awbc-coef", "0.0"],
    "no_oracle": ["--no-use-oracle-candidate-prior", "--prior-kl-coef", "0.0"],
    "no_awbc_oracle": [
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    "no_action_mask": ["--no-use-action-mask"],
    "masked_only": [
        "--no-use-action-embedding",
        "--no-event-aware-critic",
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
}


A2_STAGES: dict[str, list[str]] = {
    "D1_masked_actor_action_embedding": [
        "--no-event-aware-critic",
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    "D2_plus_event_critic": [
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    "D3_plus_awbc": [
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    "D4_plus_oracle_prior_full": [],
}


H1_GRID = tuple(
    (awbc, kl)
    for awbc in (0.05, 0.1, 0.2)
    for kl in (0.5, 1.0, 2.0)
    if not (abs(awbc - 0.1) < 1e-12 and abs(kl - 1.0) < 1e-12)
)


@dataclass(frozen=True)
class Task:
    experiment: str
    variant: str
    budget: float
    seed: int
    label: str
    cmd: tuple[str, ...]
    run_dir: Path
    eval_path: Path
    done_path: Path
    log_path: Path


def budget_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def value_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def truth_path(args: argparse.Namespace, *, budget: float, seed: int) -> Path:
    return (
        Path(args.truth_root)
        / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
        / "truth_v31.csv"
    )


def common_train_args(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    budget: float,
    seed: int,
) -> list[str]:
    return [
        "--output-dir",
        str(out_dir),
        "--checkpoint-path",
        str(out_dir / "custom_ppo_checkpoint.pt"),
        "--truth-csv",
        str(truth_path(args, budget=budget, seed=seed)),
        "--seed",
        str(int(seed)),
        "--per-step-budget",
        str(float(budget)),
        "--startup-peak-budget",
        str(float(args.startup_peak_budget)),
        "--truth-steps",
        str(int(args.truth_steps)),
        "--freq-s",
        str(int(args.freq_s)),
        "--stations",
        *[str(station) for station in args.stations],
        "--blowing-snow-event-coverage",
        str(float(args.event_coverage)),
        "--blowing-snow-event-model",
        "semi_markov",
        "--blowing-snow-min-duration-steps",
        str(int(args.min_duration)),
        "--blowing-snow-max-duration-steps",
        str(int(args.max_duration)),
        "--blowing-snow-min-gap-steps",
        str(int(args.min_gap)),
        "--blowing-snow-lead-steps",
        str(int(args.lead_steps)),
        "--blowing-snow-wind-margin-ms",
        str(float(args.wind_margin_ms)),
        "--cred-hysteresis-on",
        str(float(args.cred_hysteresis_on)),
        "--cred-hysteresis-off",
        str(float(args.cred_hysteresis_off)),
        "--flux-wind-exponent",
        str(float(args.flux_wind_exponent)),
        "--oracle-type",
        "tcn",
        "--oracle-rollout-steps",
        str(int(args.oracle_rollout_steps)),
        "--oracle-rollouts-per-policy",
        str(int(args.oracle_rollouts_per_policy)),
        "--oracle-event-fraction",
        str(float(args.oracle_event_fraction)),
        "--oracle-full-open-repeat",
        str(int(args.oracle_full_open_repeat)),
        "--oracle-epochs",
        str(int(args.oracle_epochs)),
        "--oracle-batch-size",
        str(int(args.oracle_batch_size)),
        "--oracle-device",
        str(args.oracle_device),
        "--oracle-inference-device",
        str(args.oracle_inference_device),
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
        "--awbc-coef",
        str(float(args.awbc_coef)),
        "--prior-kl-coef",
        str(float(args.prior_kl_coef)),
        "--use-oracle-candidate-prior",
        "--candidate-prior-steps",
        str(int(args.candidate_prior_steps)),
        "--candidate-prior-rollouts",
        str(int(args.candidate_prior_rollouts)),
        "--candidate-prior-scale",
        str(float(args.candidate_prior_scale)),
    ]


def make_task(
    args: argparse.Namespace,
    *,
    experiment: str,
    variant: str,
    extra: list[str],
    budget: float,
    seed: int,
) -> Task:
    run_dir = (
        Path(args.out_dir)
        / "raw"
        / experiment
        / variant
        / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
    )
    label = f"{experiment}_{variant}_budget{budget_tag(float(budget))}_seed{int(seed)}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25_v2_train_custom_ppo.py"),
        *common_train_args(args, out_dir=run_dir, budget=budget, seed=seed),
        *extra,
    ]
    return Task(
        experiment=experiment,
        variant=variant,
        budget=float(budget),
        seed=int(seed),
        label=label,
        cmd=tuple(cmd),
        run_dir=run_dir,
        eval_path=run_dir / "evaluation" / "v2_eval_overall.csv",
        done_path=Path(args.out_dir) / "done" / f"{safe_name(label)}.done",
        log_path=Path(args.out_dir) / "logs" / f"{safe_name(label)}.log",
    )


def build_tasks(args: argparse.Namespace) -> list[Task]:
    selected = set(args.experiments)
    if "all" in selected:
        selected = {"a1", "a2", "h1"}
    tasks: list[Task] = []
    budget = float(args.focus_budget)

    # Run A2 first because D4 is the matched full PD-PPO reference used by A1/H1.
    if "a2" in selected:
        for stage, extra in A2_STAGES.items():
            for seed in args.a2_seeds:
                tasks.append(
                    make_task(
                        args,
                        experiment="A2_staged_v31_aligned",
                        variant=stage,
                        extra=list(extra),
                        budget=budget,
                        seed=int(seed),
                    )
                )

    if "a1" in selected:
        for variant, extra in A1_VARIANTS.items():
            for seed in args.a1_seeds:
                tasks.append(
                    make_task(
                        args,
                        experiment="A1_remove_one_v31_aligned",
                        variant=variant,
                        extra=list(extra),
                        budget=budget,
                        seed=int(seed),
                    )
                )

    if "h1" in selected:
        for awbc, kl in H1_GRID:
            variant = f"awbc{value_tag(awbc)}_kl{value_tag(kl)}"
            extra = ["--awbc-coef", str(float(awbc)), "--prior-kl-coef", str(float(kl))]
            for seed in args.h1_seeds:
                tasks.append(
                    make_task(
                        args,
                        experiment="H1_hyperparam_v31_aligned",
                        variant=variant,
                        extra=extra,
                        budget=budget,
                        seed=int(seed),
                    )
                )
    return tasks


def pending_tasks(tasks: list[Task], *, force: bool) -> list[Task]:
    pending: list[Task] = []
    for task in tasks:
        if force or not (task.done_path.exists() and task.eval_path.exists()):
            pending.append(task)
    return pending


def run_tasks(tasks: list[Task], args: argparse.Namespace) -> None:
    pending = pending_tasks(tasks, force=bool(args.force))
    print(
        f"[v31-ablation] tasks={len(tasks)} skipped={len(tasks) - len(pending)} "
        f"pending={len(pending)} workers={int(args.workers)}",
        flush=True,
    )
    if bool(args.dry_run):
        for task in pending:
            print(f"[dry-run] {task.label}: {' '.join(task.cmd)}", flush=True)
        return
    if not pending:
        return

    task_queue: queue.Queue[Task] = queue.Queue()
    for task in pending:
        task_queue.put(task)

    failures: list[tuple[str, str, Path]] = []
    lock = threading.Lock()
    gpu_ids = [gpu.strip() for gpu in str(args.gpu_ids).split(",") if gpu.strip()]

    def worker(worker_id: int) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            task.done_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_id % len(gpu_ids)]
            print(f"[run] worker={worker_id} {task.label} -> {task.log_path}", flush=True)
            with task.log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(task.cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if proc.returncode != 0:
                with lock:
                    failures.append((task.label, f"exit={proc.returncode}", task.log_path))
            elif not task.eval_path.exists():
                with lock:
                    failures.append((task.label, f"missing_eval={task.eval_path}", task.log_path))
            else:
                task.done_path.write_text(
                    f"label={task.label}\n"
                    f"experiment={task.experiment}\n"
                    f"variant={task.variant}\n"
                    f"budget={task.budget:.6f}\n"
                    f"seed={task.seed}\n"
                    f"truth_csv={truth_path(args, budget=task.budget, seed=task.seed)}\n"
                    f"run_dir={task.run_dir}\n"
                    f"eval_path={task.eval_path}\n"
                    f"log_path={task.log_path}\n",
                    encoding="utf-8",
                )
            task_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(idx,), daemon=True)
        for idx in range(max(1, int(args.workers)))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        for label, reason, log_path in failures:
            print(f"[fail] {label} {reason} log={log_path}", flush=True)
        raise SystemExit(1)


def run_collector(args: argparse.Namespace) -> None:
    if bool(args.dry_run) or bool(args.skip_collect):
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "46_v31_aligned_ablation_collect.py"),
        "--run-root",
        str(args.out_dir),
        "--s2-main-root",
        str(args.s2_main_root),
        "--focus-budget",
        str(float(args.focus_budget)),
        "--a1-seeds",
        *[str(int(seed)) for seed in args.a1_seeds],
        "--a2-seeds",
        *[str(int(seed)) for seed in args.a2_seeds],
        "--h1-seeds",
        *[str(int(seed)) for seed in args.h1_seeds],
    ]
    print(f"[collect] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V3.1-aligned A1/A2/H1 ablations on the final S2 truth generator."
    )
    parser.add_argument("--experiments", nargs="+", choices=["all", "a1", "a2", "h1"], default=["all"])
    parser.add_argument("--out-dir", default="reports/v31_ablation_aligned")
    parser.add_argument("--truth-root", default="reports/v31_s2_main/raw")
    parser.add_argument("--s2-main-root", default="reports/v31_s2_main")
    parser.add_argument("--focus-budget", type=float, default=DEFAULT_FOCUS_BUDGET)
    parser.add_argument("--a1-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS_A1_A2))
    parser.add_argument("--a2-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS_A1_A2))
    parser.add_argument("--h1-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS_H1))
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--truth-steps", type=int, default=30000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--event-coverage", type=float, default=0.28)
    parser.add_argument("--min-duration", type=int, default=12)
    parser.add_argument("--max-duration", type=int, default=24)
    parser.add_argument("--min-gap", type=int, default=4)
    parser.add_argument("--lead-steps", type=int, default=6)
    parser.add_argument("--wind-margin-ms", type=float, default=1.2)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.6)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.3)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.0)
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-event-fraction", type=float, default=0.50)
    parser.add_argument("--oracle-full-open-repeat", type=int, default=3)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--awbc-coef", type=float, default=0.1)
    parser.add_argument("--prior-kl-coef", type=float, default=1.0)
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu-ids", default="1,4,5")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    out_dir = Path(parsed.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "done").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    tasks_to_run = build_tasks(parsed)
    run_tasks(tasks_to_run, parsed)
    run_collector(parsed)
