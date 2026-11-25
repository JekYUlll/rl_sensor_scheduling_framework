#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGETS = (1.65, 1.70, 1.75)
DEFAULT_SEEDS = tuple(range(41, 51))


A2_DIAGNOSTIC_STAGES: dict[str, list[str]] = {
    "D1_masked_actor": [
        "--no-event-aware-critic",
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    "D2_event_critic": [
        "--event-aware-critic",
        "--awbc-coef",
        "0.0",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    "D3_event_critic_awbc": [
        "--event-aware-critic",
        "--awbc-coef",
        "0.1",
        "--no-use-oracle-candidate-prior",
        "--prior-kl-coef",
        "0.0",
    ],
    "D4_oracle_prior": [
        "--event-aware-critic",
        "--awbc-coef",
        "0.1",
        "--use-oracle-candidate-prior",
        "--prior-kl-coef",
        "1.0",
    ],
}


@dataclass(frozen=True)
class Task:
    label: str
    cmd: tuple[str, ...]
    done_path: Path
    log_path: Path


def budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}".replace(".", "p")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("_")


def truth_path(main_grid_dir: Path, budget: float, seed: int) -> Path:
    return main_grid_dir / f"truth_budget{budget_tag(budget)}_seed{int(seed)}.csv"


def train_common_args(args: argparse.Namespace, *, out_dir: Path, budget: float, seed: int) -> list[str]:
    return [
        "--output-dir",
        str(out_dir),
        "--checkpoint-path",
        str(out_dir / "custom_ppo_checkpoint.pt"),
        "--truth-csv",
        str(truth_path(Path(args.main_grid_dir), budget, seed)),
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
        "--use-oracle-candidate-prior",
    ]


def make_train_task(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    budget: float,
    seed: int,
    label: str,
    extra: list[str] | None = None,
) -> Task:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25_v2_train_custom_ppo.py"),
        *train_common_args(args, out_dir=out_dir, budget=budget, seed=seed),
        *(extra or []),
    ]
    return Task(
        label=label,
        cmd=tuple(cmd),
        done_path=out_dir / "evaluation" / "v2_eval_overall.csv",
        log_path=Path(args.log_dir) / f"{safe_name(label)}.log",
    )


def make_s1_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    main_grid = Path(args.main_grid_dir)
    for budget, seed in product(args.budgets, args.seeds):
        out_dir = main_grid / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
        label = f"S1_budget{budget_tag(float(budget))}_seed{int(seed)}"
        tasks.append(make_train_task(args, out_dir=out_dir, budget=float(budget), seed=int(seed), label=label))
    return tasks


def make_a2_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    seeds = list(args.seeds[: int(args.diagnostic_seed_count)])
    for stage, extra in A2_DIAGNOSTIC_STAGES.items():
        for seed in seeds:
            out_dir = (
                Path(args.out_dir)
                / "A2_diagnostic"
                / stage
                / f"budget{budget_tag(float(args.focus_budget))}_seed{int(seed)}"
            )
            label = f"A2_{stage}_budget{budget_tag(float(args.focus_budget))}_seed{int(seed)}"
            tasks.append(
                make_train_task(
                    args,
                    out_dir=out_dir,
                    budget=float(args.focus_budget),
                    seed=int(seed),
                    label=label,
                    extra=list(extra),
                )
            )
    return tasks


def make_e1_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    for budget, seed in product(args.budgets, args.seeds):
        run_dir = Path(args.main_grid_dir) / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
        if not run_dir.exists():
            print(f"[warn] E1 missing trained run: {run_dir}", flush=True)
            continue
        out_root = Path(args.out_dir) / str(args.e1_subdir)
        done_path = (
            out_root
            / str(args.e1_done_episode_type)
            / f"budget{budget_tag(float(budget))}_seed{int(seed)}"
            / "evaluation"
            / "v2_eval_overall.csv"
        )
        label = f"{str(args.e1_subdir)}_budget{budget_tag(float(budget))}_seed{int(seed)}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "32_v2_condition_eval.py"),
            "--run-dir",
            str(run_dir),
            "--out-root",
            str(out_root),
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
            "--policy-device",
            "cpu",
            "--condition-stride-divisor",
            str(int(args.condition_stride_divisor)),
            "--calm-max-event-rate",
            str(float(args.calm_max_event_rate)),
            "--mixed-min-event-rate",
            str(float(args.mixed_min_event_rate)),
            "--mixed-max-event-rate",
            str(float(args.mixed_max_event_rate)),
            "--event-min-event-rate",
            str(float(args.event_min_event_rate)),
        ]
        if bool(args.strict_condition_bands):
            cmd.append("--strict-condition-bands")
        tasks.append(
            Task(
                label=label,
                cmd=tuple(cmd),
                done_path=done_path,
                log_path=Path(args.log_dir) / f"{safe_name(label)}.log",
            )
        )
    return tasks


def run_tasks_parallel(tasks: list[Task], *, gpu_ids: list[str], dry_run: bool) -> None:
    pending = [task for task in tasks if not task.done_path.exists()]
    skipped = len(tasks) - len(pending)
    print(f"[parallel] tasks={len(tasks)} skipped={skipped} pending={len(pending)} gpus={','.join(gpu_ids)}", flush=True)
    if not pending:
        return
    if dry_run:
        for task in pending:
            print(f"[dry-run] {task.label}: {' '.join(task.cmd)}", flush=True)
        return

    task_queue: queue.Queue[Task] = queue.Queue()
    for task in pending:
        task_queue.put(task)
    failures: list[tuple[str, int, Path]] = []
    lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            if task.done_path.exists():
                print(f"[skip] {task.label}: {task.done_path}", flush=True)
                task_queue.task_done()
                continue
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env.setdefault(key, "1")
            started = time.time()
            print(f"[start gpu={gpu_id}] {task.label}", flush=True)
            with task.log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"$ {' '.join(task.cmd)}\n")
                log_file.flush()
                proc = subprocess.run(
                    list(task.cmd),
                    cwd=str(ROOT),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.time() - started
            if proc.returncode == 0 and task.done_path.exists():
                print(f"[done gpu={gpu_id}] {task.label} elapsed={elapsed/60:.1f}min", flush=True)
            else:
                print(
                    f"[fail gpu={gpu_id}] {task.label} rc={proc.returncode} elapsed={elapsed/60:.1f}min log={task.log_path}",
                    flush=True,
                )
                with lock:
                    failures.append((task.label, int(proc.returncode), task.log_path))
            task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=False) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        detail = "\n".join(f"{label}: rc={rc}, log={log}" for label, rc, log in failures)
        raise RuntimeError(f"{len(failures)} parallel tasks failed:\n{detail}")


def run_aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "31_v2_build_supplement_assets.py"),
        "--grid-dirs",
        str(args.main_grid_dir),
        "--table-dir",
        "reports/v2_paper_tables_prior_kl1",
        "--supp-root",
        str(args.out_dir),
        "--out-dir",
        str(args.asset_dir),
        "--budgets",
        *[str(float(budget)) for budget in args.budgets],
        "--seeds",
        *[str(int(seed)) for seed in args.seeds],
        "--bootstrap",
        str(int(args.bootstrap)),
    ]
    print(f"[aggregate] {' '.join(cmd)}", flush=True)
    if not bool(args.dry_run):
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel v2 supplementary experiment runner.")
    parser.add_argument("--stages", nargs="+", choices=["s1", "a2", "e1", "aggregate"], default=["s1", "a2", "e1", "aggregate"])
    parser.add_argument("--main-grid-dir", default="reports/v2_forecast_eval_grid_prior_kl1")
    parser.add_argument("--out-dir", default="reports/v2_supplement_experiments")
    parser.add_argument("--asset-dir", default="reports/v2_supplement_assets")
    parser.add_argument("--log-dir", default="reports/logs/v2_supplement_parallel")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--focus-budget", type=float, default=1.70)
    parser.add_argument("--diagnostic-seed-count", type=int, default=5)
    parser.add_argument("--truth-steps", type=int, default=8192)
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-full-open-repeat", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--e1-subdir", default="E1_condition_eval")
    parser.add_argument("--e1-done-episode-type", default="event")
    parser.add_argument("--condition-stride-divisor", type=int, default=8)
    parser.add_argument("--calm-max-event-rate", type=float, default=0.20)
    parser.add_argument("--mixed-min-event-rate", type=float, default=0.35)
    parser.add_argument("--mixed-max-event-rate", type=float, default=0.65)
    parser.add_argument("--event-min-event-rate", type=float, default=0.75)
    parser.add_argument("--strict-condition-bands", action="store_true")
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
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.budgets = [float(x) for x in args.budgets]
    args.seeds = [int(x) for x in args.seeds]
    gpus = [str(x) for x in args.gpus]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if "s1" in args.stages:
        print("[stage] S1 main grid", flush=True)
        run_tasks_parallel(make_s1_tasks(args), gpu_ids=gpus, dry_run=bool(args.dry_run))
    if "a2" in args.stages:
        print("[stage] A2 diagnostic", flush=True)
        run_tasks_parallel(make_a2_tasks(args), gpu_ids=gpus, dry_run=bool(args.dry_run))
    if "e1" in args.stages:
        print("[stage] E1 condition evaluation", flush=True)
        run_tasks_parallel(make_e1_tasks(args), gpu_ids=gpus, dry_run=bool(args.dry_run))
    if "aggregate" in args.stages:
        run_aggregate(args)
    print("[parallel] complete", flush=True)


if __name__ == "__main__":
    main()
