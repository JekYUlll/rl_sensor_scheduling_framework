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
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS_A1 = tuple(range(41, 51))
DEFAULT_SEEDS_H1 = tuple(range(41, 46))
FOCUS_BUDGET = 1.70


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


H1_GRID = [
    (awbc, kl)
    for awbc in (0.05, 0.1, 0.2)
    for kl in (0.5, 1.0, 2.0)
    if not (abs(awbc - 0.1) < 1e-12 and abs(kl - 1.0) < 1e-12)
]


@dataclass(frozen=True)
class Task:
    label: str
    cmd: tuple[str, ...]
    done_path: Path
    log_path: Path


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def value_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def truth_path(args: argparse.Namespace, *, budget: float, seed: int) -> Path:
    return Path(args.main_grid_dir) / f"truth_budget{budget_tag(budget)}_seed{int(seed)}.csv"


def common_train_args(args: argparse.Namespace, *, out_dir: Path, budget: float, seed: int) -> list[str]:
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
        "--oracle-rollout-steps",
        str(int(args.oracle_rollout_steps)),
        "--oracle-rollouts-per-policy",
        str(int(args.oracle_rollouts_per_policy)),
        "--oracle-epochs",
        str(int(args.oracle_epochs)),
        "--oracle-batch-size",
        str(int(args.oracle_batch_size)),
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
        "--awbc-coef",
        "0.1",
        "--prior-kl-coef",
        "1.0",
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
    out_dir = Path(args.out_dir) / experiment / variant / f"budget{budget_tag(budget)}_seed{int(seed)}"
    label = f"{experiment}_{variant}_budget{budget_tag(budget)}_seed{int(seed)}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25_v2_train_custom_ppo.py"),
        *common_train_args(args, out_dir=out_dir, budget=budget, seed=seed),
        *extra,
    ]
    return Task(
        label=label,
        cmd=tuple(cmd),
        done_path=out_dir / "evaluation" / "v2_eval_overall.csv",
        log_path=Path(args.log_dir) / f"{safe_name(label)}.log",
    )


def build_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    if args.experiment in ("a1", "all"):
        for variant, extra in A1_VARIANTS.items():
            for seed in args.a1_seeds:
                tasks.append(
                    make_task(
                        args,
                        experiment="A1_ablation_v31",
                        variant=variant,
                        extra=list(extra),
                        budget=float(args.focus_budget),
                        seed=int(seed),
                    )
                )
    if args.experiment in ("h1", "all"):
        for awbc, kl in H1_GRID:
            variant = f"awbc{value_tag(awbc)}_kl{value_tag(kl)}"
            extra = ["--awbc-coef", str(float(awbc)), "--prior-kl-coef", str(float(kl))]
            for seed in args.h1_seeds:
                tasks.append(
                    make_task(
                        args,
                        experiment="H1_hyperparam_v31",
                        variant=variant,
                        extra=extra,
                        budget=float(args.focus_budget),
                        seed=int(seed),
                    )
                )
    return tasks


def run_tasks(tasks: list[Task], *, workers: int, gpu_ids: str, dry_run: bool, force: bool) -> None:
    pending = [task for task in tasks if force or not task.done_path.exists()]
    skipped = len(tasks) - len(pending)
    print(f"[phase1] tasks={len(tasks)} skipped={skipped} pending={len(pending)} workers={workers}", flush=True)
    if dry_run:
        for task in pending:
            print(f"[dry-run] {task.label}: {' '.join(task.cmd)}", flush=True)
        return
    if not pending:
        return

    task_queue: queue.Queue[Task] = queue.Queue()
    for task in pending:
        task_queue.put(task)
    failures: list[tuple[str, int, Path]] = []
    lock = threading.Lock()

    def worker(worker_id: int) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            parsed_gpu_ids = [gpu.strip() for gpu in str(gpu_ids).split(",") if gpu.strip()]
            if parsed_gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = parsed_gpu_ids[worker_id % len(parsed_gpu_ids)]
            print(f"[run] worker={worker_id} {task.label} -> {task.log_path}", flush=True)
            with task.log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(task.cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if proc.returncode != 0:
                with lock:
                    failures.append((task.label, int(proc.returncode), task.log_path))
            task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(idx,), daemon=True) for idx in range(max(1, int(workers)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        for label, code, log_path in failures:
            print(f"[fail] {label} exit={code} log={log_path}", flush=True)
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.1 Phase-1 A1/H1 supplement experiments.")
    parser.add_argument("--experiment", choices=["a1", "h1", "all"], default="all")
    parser.add_argument("--main-grid-dir", default="reports/v2_forecast_eval_grid_prior_kl1")
    parser.add_argument("--out-dir", default="reports/v3_supplement_assets")
    parser.add_argument("--log-dir", default="reports/v3_supplement_assets/logs")
    parser.add_argument("--focus-budget", type=float, default=FOCUS_BUDGET)
    parser.add_argument("--a1-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS_A1))
    parser.add_argument("--h1-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS_H1))
    parser.add_argument("--truth-steps", type=int, default=8192)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tasks(
        build_tasks(args),
        workers=int(args.workers),
        gpu_ids=str(args.gpu_ids),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )
