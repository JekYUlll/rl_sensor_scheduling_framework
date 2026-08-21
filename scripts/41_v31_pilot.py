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

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PilotTask:
    label: str
    cmd: tuple[str, ...]
    run_dir: Path
    done_path: Path
    log_path: Path


def _budget_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _resolve_sensor_cfg(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def _extend_optional_training_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.target_weights is not None:
        cmd.extend(["--target-weights", *[str(float(value)) for value in args.target_weights]])
    if args.target_scales is not None:
        cmd.extend(["--target-scales", *[str(float(value)) for value in args.target_scales]])
    if args.required_sensors is not None:
        cmd.extend(["--required-sensors", *[str(sensor_id) for sensor_id in args.required_sensors]])
    if bool(args.disable_coverage_groups):
        cmd.append("--disable-coverage-groups")
    if args.max_active is not None:
        cmd.extend(["--max-active", str(int(args.max_active))])


def _build_task(args: argparse.Namespace, *, seed: int) -> PilotTask:
    budget = float(args.budget)
    seed = int(seed)
    tag = f"v31_pilot_budget{_budget_tag(budget)}_seed{seed}"
    run_dir = Path(args.out_dir) / tag
    truth_csv = run_dir / "truth_v31.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25_v2_train_custom_ppo.py"),
        "--output-dir",
        str(run_dir),
        "--checkpoint-path",
        str(run_dir / "custom_ppo_checkpoint.pt"),
        "--truth-csv",
        str(truth_csv),
        "--antaws-root",
        str(args.antaws_root),
        "--sensor-cfg",
        str(_resolve_sensor_cfg(str(args.sensor_cfg))),
        "--seed",
        str(seed),
        "--per-step-budget",
        str(budget),
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
        "--event-microstructure-sigma",
        str(float(args.event_microstructure_sigma)),
        "--event-microstructure-alpha",
        str(float(args.event_microstructure_alpha)),
        "--event-microstructure-diameter-scale",
        str(float(args.event_microstructure_diameter_scale)),
        "--event-microstructure-velocity-scale",
        str(float(args.event_microstructure_velocity_scale)),
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
    _extend_optional_training_args(cmd, args)
    return PilotTask(
        label=tag,
        cmd=tuple(cmd),
        run_dir=run_dir,
        done_path=run_dir / "evaluation" / "v2_eval_overall.csv",
        log_path=Path(args.log_dir) / f"{_safe_name(tag)}.log",
    )


def _run_tasks(tasks: list[PilotTask], args: argparse.Namespace) -> None:
    pending = [task for task in tasks if bool(args.force) or not task.done_path.exists()]
    print(
        f"[v31-pilot] tasks={len(tasks)} skipped={len(tasks) - len(pending)} "
        f"pending={len(pending)} workers={int(args.workers)}",
        flush=True,
    )
    if bool(args.dry_run):
        for task in pending:
            print(f"[dry-run] {task.label}: {' '.join(task.cmd)}", flush=True)
        return
    task_queue: queue.Queue[PilotTask] = queue.Queue()
    for task in pending:
        task_queue.put(task)
    failures: list[tuple[str, int, Path]] = []
    lock = threading.Lock()
    gpu_ids = [gpu.strip() for gpu in str(args.gpu_ids).split(",") if gpu.strip()]

    def worker(worker_id: int) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_id % len(gpu_ids)]
            print(f"[run] worker={worker_id} {task.label} -> {task.log_path}", flush=True)
            with task.log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(task.cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if proc.returncode != 0:
                with lock:
                    failures.append((task.label, int(proc.returncode), task.log_path))
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
        for label, code, log_path in failures:
            print(f"[fail] {label} exit={code} log={log_path}", flush=True)
        raise SystemExit(1)


def _collect(tasks: list[PilotTask], args: argparse.Namespace) -> None:
    rows = []
    missing = []
    for task in tasks:
        if not task.done_path.exists():
            missing.append(str(task.done_path))
            continue
        df = pd.read_csv(task.done_path)
        seed_text = task.label.rsplit("_seed", 1)[-1]
        df.insert(0, "seed", int(seed_text))
        df.insert(0, "budget", float(args.budget))
        df.insert(0, "run_tag", task.label)
        rows.append(df)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        (out_dir / "v31_pilot_missing.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"[collect] missing={len(missing)}; see {out_dir / 'v31_pilot_missing.txt'}", flush=True)
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    long_path = out_dir / "v31_pilot_overall_long.csv"
    long.to_csv(long_path, index=False)
    metric_cols = [
        col
        for col in [
            "forecast_weighted_mae_overall",
            "forecast_weighted_mae_event",
            "forecast_weighted_mae_non_event",
            "forecast_weighted_mae_low_temp",
            "forecast_weighted_mae_normal",
            "obs_reconstruction_mae",
            "weighted_normalized_mae",
            "oracle_loss_mean",
            "power_mean",
        ]
        if col in long.columns
    ]
    summary = long.groupby(["budget", "policy"], as_index=False)[metric_cols].agg(["mean", "std", "count"])
    summary_path = out_dir / "v31_pilot_summary.csv"
    summary.to_csv(summary_path)
    print(long_path, flush=True)
    print(summary_path, flush=True)
    if "forecast_weighted_mae_overall" in long.columns:
        display = (
            long.groupby("policy", as_index=False)["forecast_weighted_mae_overall"]
            .agg(["mean", "std", "count"])
            .sort_values("mean")
        )
        print(display.to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small V3.1 semi-Markov pilot retraining experiment.")
    parser.add_argument("--out-dir", default="reports/v31_pilot")
    parser.add_argument("--log-dir", default="reports/v31_pilot/logs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--budget", type=float, default=1.70)
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
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
    parser.add_argument("--event-microstructure-sigma", type=float, default=0.0)
    parser.add_argument("--event-microstructure-alpha", type=float, default=0.18)
    parser.add_argument("--event-microstructure-diameter-scale", type=float, default=0.0)
    parser.add_argument("--event-microstructure-velocity-scale", type=float, default=0.0)
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
    parser.add_argument("--target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--target-scales", nargs="*", type=float, default=None)
    parser.add_argument("--required-sensors", nargs="*", default=None)
    parser.add_argument("--disable-coverage-groups", action="store_true")
    parser.add_argument("--max-active", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    pilot_tasks = [_build_task(parsed, seed=int(seed)) for seed in parsed.seeds]
    _run_tasks(pilot_tasks, parsed)
    if not bool(parsed.dry_run):
        _collect(pilot_tasks, parsed)
