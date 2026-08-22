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


@dataclass(frozen=True)
class SplitTask:
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def append_values(cmd: list[str], flag: str, values: list[object] | tuple[object, ...] | None) -> None:
    if values is not None:
        cmd.extend([flag, *[str(value) for value in values]])


def build_task(args: argparse.Namespace, *, budget: float, seed: int) -> SplitTask:
    label = f"budget{budget_tag(budget)}_seed{int(seed)}"
    run_dir = Path(args.out_dir) / "raw" / label
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "58_v31_split_protocol_run.py"),
        "--out-dir",
        str(run_dir),
        "--antaws-root",
        str(args.antaws_root),
        "--stations",
        *[str(station) for station in args.stations],
        "--sensor-cfg",
        str(args.sensor_cfg),
        "--seed",
        str(int(seed)),
        "--budget",
        str(float(budget)),
        "--startup-peak-budget",
        str(float(args.startup_peak_budget)),
        "--truth-steps",
        str(int(args.truth_steps)),
        "--freq-s",
        str(int(args.freq_s)),
        "--split-ratios",
        *[str(float(value)) for value in args.split_ratios],
        "--event-coverage",
        str(float(args.event_coverage)),
        "--min-duration",
        str(int(args.min_duration)),
        "--max-duration",
        str(int(args.max_duration)),
        "--min-gap",
        str(int(args.min_gap)),
        "--lead-steps",
        str(int(args.lead_steps)),
        "--wind-margin-ms",
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
        "--event-particle-microstructure-correlation",
        str(float(args.event_particle_microstructure_correlation)),
        "--event-subtype-particle-prob",
        str(float(args.event_subtype_particle_prob)),
        "--event-subtype-flux-prob",
        str(float(args.event_subtype_flux_prob)),
        "--event-subtype-thermal-prob",
        str(float(args.event_subtype_thermal_prob)),
        "--event-subtype-particle-flux-multiplier",
        str(float(args.event_subtype_particle_flux_multiplier)),
        "--event-subtype-flux-multiplier",
        str(float(args.event_subtype_flux_multiplier)),
        "--event-subtype-thermal-flux-multiplier",
        str(float(args.event_subtype_thermal_flux_multiplier)),
        "--event-subtype-particle-diameter-shift-mm",
        str(float(args.event_subtype_particle_diameter_shift_mm)),
        "--event-subtype-particle-velocity-boost-ms",
        str(float(args.event_subtype_particle_velocity_boost_ms)),
        "--event-subtype-flux-diameter-shift-mm",
        str(float(args.event_subtype_flux_diameter_shift_mm)),
        "--event-subtype-flux-velocity-boost-ms",
        str(float(args.event_subtype_flux_velocity_boost_ms)),
        "--event-subtype-thermal-surface-drop-c",
        str(float(args.event_subtype_thermal_surface_drop_c)),
        "--event-subtype-particle-humidity-boost-pct",
        str(float(args.event_subtype_particle_humidity_boost_pct)),
        "--event-subtype-flux-wind-boost-ms",
        str(float(args.event_subtype_flux_wind_boost_ms)),
        "--event-subtype-thermal-air-temp-drop-c",
        str(float(args.event_subtype_thermal_air_temp_drop_c)),
        "--event-subtype-latent-alpha",
        str(float(args.event_subtype_latent_alpha)),
        "--event-subtype-particle-latent-diameter-scale-mm",
        str(float(args.event_subtype_particle_latent_diameter_scale_mm)),
        "--event-subtype-particle-latent-velocity-scale-ms",
        str(float(args.event_subtype_particle_latent_velocity_scale_ms)),
        "--event-subtype-flux-latent-sigma",
        str(float(args.event_subtype_flux_latent_sigma)),
        "--event-subtype-thermal-latent-surface-scale-c",
        str(float(args.event_subtype_thermal_latent_surface_scale_c)),
        "--event-subtype-latent-target-lag-steps",
        str(int(args.event_subtype_latent_target_lag_steps)),
        "--event-subtype-context-lead-steps",
        str(int(args.event_subtype_context_lead_steps)),
        "--event-subtype-context-noise-std",
        str(float(args.event_subtype_context_noise_std)),
        "--oracle-rollout-steps",
        str(int(args.oracle_rollout_steps)),
        "--oracle-rollouts-per-policy",
        str(int(args.oracle_rollouts_per_policy)),
        "--oracle-epochs",
        str(int(args.oracle_epochs)),
        "--oracle-batch-size",
        str(int(args.oracle_batch_size)),
        "--oracle-loss-clip",
        str(float(args.oracle_loss_clip)),
        "--oracle-candidate-mask-repeat",
        str(int(args.oracle_candidate_mask_repeat)),
        "--oracle-candidate-mask-limit",
        str(int(args.oracle_candidate_mask_limit)),
        "--oracle-subtype-teacher-repeat",
        str(int(args.oracle_subtype_teacher_repeat)),
        "--oracle-subtype-teacher-lookahead-steps",
        str(int(args.oracle_subtype_teacher_lookahead_steps)),
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
        "--ent-coef",
        str(float(args.ent_coef)),
        "--awbc-coef",
        str(float(args.awbc_coef)),
        "--awbc-label-stride",
        str(int(args.awbc_label_stride)),
        "--bc-pretrain-steps",
        str(int(args.bc_pretrain_steps)),
        "--bc-pretrain-epochs",
        str(int(args.bc_pretrain_epochs)),
        "--bc-pretrain-batch-size",
        str(int(args.bc_pretrain_batch_size)),
        "--bc-pretrain-loss-coef",
        str(float(args.bc_pretrain_loss_coef)),
        "--subtype-aux-coef",
        str(float(args.subtype_aux_coef)),
        "--subtype-aux-classes",
        str(max(2, int(args.subtype_aux_classes))),
        "--subtype-aux-lookahead-steps",
        str(max(0, int(args.subtype_aux_lookahead_steps))),
        "--subtype-router" if bool(args.subtype_router) else "--no-subtype-router",
        "--subtype-router-min-confidence",
        str(float(args.subtype_router_min_confidence)),
        "--subtype-router-low-confidence-action",
        str(int(args.subtype_router_low_confidence_action)),
        "--awbc-teacher-mode",
        str(args.awbc_teacher_mode),
        "--awbc-teacher-event-lookahead-steps",
        str(int(args.awbc_teacher_event_lookahead_steps)),
        "--awbc-teacher-energy-mpc-horizon",
        str(int(args.awbc_teacher_energy_mpc_horizon)),
        "--awbc-teacher-energy-mpc-soc-bins",
        str(int(args.awbc_teacher_energy_mpc_soc_bins)),
        "--awbc-teacher-energy-mpc-low-soc-ratio",
        str(float(args.awbc_teacher_energy_mpc_low_soc_ratio)),
        "--awbc-teacher-energy-mpc-high-soc-ratio",
        str(float(args.awbc_teacher_energy_mpc_high_soc_ratio)),
        "--awbc-teacher-energy-mpc-terminal-soc-weight",
        str(float(args.awbc_teacher_energy_mpc_terminal_soc_weight)),
        "--awbc-teacher-energy-mpc-max-actions",
        str(int(args.awbc_teacher_energy_mpc_max_actions)),
        "--awbc-teacher-energy-mpc-low-power-action",
        str(int(args.awbc_teacher_energy_mpc_low_power_action)),
        "--awbc-teacher-dwell-steps",
        str(int(args.awbc_teacher_dwell_steps)),
        "--agent-cycle-period-steps",
        str(int(args.agent_cycle_period_steps)),
        "--agent-cycle-dwell-steps",
        str(int(args.agent_cycle_dwell_steps)),
        "--regime-belief-lookback",
        str(max(1, int(args.regime_belief_lookback))),
        "--prior-kl-coef",
        str(float(args.prior_kl_coef)),
        "--greedy-lookahead-steps",
        str(int(args.greedy_lookahead_steps)),
        "--event-start-prob",
        str(float(args.event_start_prob)),
        "--soc-aux-horizon",
        str(int(args.soc_aux_horizon)),
        "--soc-aux-coef",
        str(float(args.soc_aux_coef)),
        "--train-episode-len",
        str(int(args.train_episode_len)),
        "--use-candidate-prior" if bool(args.use_candidate_prior) else "--no-use-candidate-prior",
        "--candidate-prior-scale",
        str(float(args.candidate_prior_scale)),
        "--candidate-prior-steps",
        str(int(args.candidate_prior_steps)),
        "--candidate-prior-rollouts",
        str(int(args.candidate_prior_rollouts)),
        "--static-selection-steps",
        str(int(args.static_selection_steps)),
        "--static-selection-rollouts",
        str(int(args.static_selection_rollouts)),
        "--eval-steps",
        str(int(args.eval_steps)),
        "--eval-rollouts",
        str(int(args.eval_rollouts)),
        "--eval-start-selection",
        str(args.eval_start_selection),
        "--eval-event-fraction",
        str(float(args.eval_event_fraction)),
        "--eval-selection-stride",
        str(int(args.eval_selection_stride)),
        "--lambda-warmup-abort",
        str(float(args.lambda_warmup_abort)),
        "--lambda-switch",
        str(float(args.lambda_switch)),
        "--event-reward-multiplier",
        str(float(args.event_reward_multiplier)),
        "--lambda-energy-deficit",
        str(float(args.lambda_energy_deficit)),
        "--soc-soft-penalty-buffer",
        str(float(args.soc_soft_penalty_buffer)),
        "--lambda-soc-soft-penalty",
        str(float(args.lambda_soc_soft_penalty)),
        "--lambda-duty-balance",
        str(float(args.lambda_duty_balance)),
        "--duty-balance-low",
        str(float(args.duty_balance_low)),
        "--duty-balance-high",
        str(float(args.duty_balance_high)),
        "--duty-balance-grace-steps",
        str(int(args.duty_balance_grace_steps)),
        "--duty-score-feedback",
        str(float(args.duty_score_feedback)),
        "--duty-score-target",
        str(float(args.duty_score_target)),
        "--duty-hard-low",
        str(float(args.duty_hard_low)),
        "--duty-hard-high",
        str(float(args.duty_hard_high)),
        "--duty-hard-score",
        str(float(args.duty_hard_score)),
        "--min-dwell-steps",
        str(int(args.min_dwell_steps)),
        "--device",
        str(args.device),
    ]
    if bool(args.event_subtypes_enabled):
        cmd.append("--event-subtypes-enabled")
    if bool(args.duty_hard_guard):
        cmd.append("--duty-hard-guard")
    cmd.append("--event-aware-critic" if bool(args.event_aware_critic) else "--no-event-aware-critic")
    cmd.append("--event-gated-actor" if bool(args.event_gated_actor) else "--no-event-gated-actor")
    if bool(args.include_agent_cycle_phase):
        cmd.append("--include-agent-cycle-phase")
    if bool(args.include_observable_regime_belief):
        cmd.append("--include-observable-regime-belief")
    if bool(args.skip_rollout_evaluation):
        cmd.append("--skip-rollout-evaluation")
    if bool(args.eval_duty_constrained_baselines):
        cmd.append("--eval-duty-constrained-baselines")
    append_values(cmd, "--oracle-subtype-teacher-calm-sensors", args.oracle_subtype_teacher_calm_sensors)
    append_values(cmd, "--oracle-subtype-teacher-particle-sensors", args.oracle_subtype_teacher_particle_sensors)
    append_values(cmd, "--oracle-subtype-teacher-flux-sensors", args.oracle_subtype_teacher_flux_sensors)
    append_values(cmd, "--oracle-subtype-teacher-thermal-sensors", args.oracle_subtype_teacher_thermal_sensors)
    append_values(cmd, "--awbc-teacher-calm-sensors", args.awbc_teacher_calm_sensors)
    append_values(cmd, "--awbc-teacher-event-sensors", args.awbc_teacher_event_sensors)
    append_values(cmd, "--awbc-teacher-subtype-calm-sensors", args.awbc_teacher_subtype_calm_sensors)
    append_values(cmd, "--awbc-teacher-subtype-particle-sensors", args.awbc_teacher_subtype_particle_sensors)
    append_values(cmd, "--awbc-teacher-subtype-flux-sensors", args.awbc_teacher_subtype_flux_sensors)
    append_values(cmd, "--awbc-teacher-subtype-thermal-sensors", args.awbc_teacher_subtype_thermal_sensors)
    append_values(cmd, "--agent-context-columns", args.agent_context_columns)
    if args.awbc_teacher_calm_pool_spec is not None:
        cmd.extend(["--awbc-teacher-calm-pool-spec", str(args.awbc_teacher_calm_pool_spec)])
    if args.awbc_teacher_event_pool_spec is not None:
        cmd.extend(["--awbc-teacher-event-pool-spec", str(args.awbc_teacher_event_pool_spec)])
    if args.baseline_duty_hard_low is not None:
        cmd.extend(["--baseline-duty-hard-low", str(float(args.baseline_duty_hard_low))])
    if args.baseline_duty_hard_high is not None:
        cmd.extend(["--baseline-duty-hard-high", str(float(args.baseline_duty_hard_high))])
    if args.baseline_duty_hard_score is not None:
        cmd.extend(["--baseline-duty-hard-score", str(float(args.baseline_duty_hard_score))])
    if args.baseline_duty_score_feedback is not None:
        cmd.extend(["--baseline-duty-score-feedback", str(float(args.baseline_duty_score_feedback))])
    if bool(args.primary_eval_duty_guard):
        cmd.append("--primary-eval-duty-guard")
    append_values(cmd, "--target-weights", args.target_weights)
    append_values(cmd, "--target-scales", args.target_scales)
    append_values(cmd, "--additional-state-columns", args.additional_state_columns)
    cmd.append("--subtype-loss-weighting" if bool(args.subtype_loss_weighting) else "--no-subtype-loss-weighting")
    append_values(cmd, "--subtype-particle-target-weights", args.subtype_particle_target_weights)
    append_values(cmd, "--subtype-flux-target-weights", args.subtype_flux_target_weights)
    append_values(cmd, "--subtype-thermal-target-weights", args.subtype_thermal_target_weights)
    append_values(cmd, "--required-sensors", args.required_sensors)
    if args.energy_account:
        cmd.extend(
            [
                "--energy-account",
                "--energy-capacity",
                str(float(args.energy_capacity)),
                "--initial-energy",
                str(float(args.initial_energy)),
                "--harvest-per-step",
                str(float(args.harvest_per_step)),
                "--reserve-energy",
                str(float(args.reserve_energy)),
            ]
        )
    if args.disable_coverage_groups:
        cmd.append("--disable-coverage-groups")
    if args.max_active is not None:
        cmd.extend(["--max-active", str(int(args.max_active))])
    return SplitTask(
        budget=float(budget),
        seed=int(seed),
        label=label,
        cmd=tuple(cmd),
        run_dir=run_dir,
        eval_path=run_dir / (
            "v2_custom_ppo_metrics.csv"
            if bool(args.skip_rollout_evaluation)
            else "evaluation/v2_eval_overall.csv"
        ),
        done_path=Path(args.out_dir) / "done" / f"{label}.done",
        log_path=Path(args.out_dir) / "logs" / f"{safe_name(label)}.log",
    )


def pending_tasks(tasks: list[SplitTask], *, force: bool) -> list[SplitTask]:
    if force:
        return tasks
    return [task for task in tasks if not (task.done_path.exists() and task.eval_path.exists())]


def run_tasks(tasks: list[SplitTask], args: argparse.Namespace) -> None:
    pending = pending_tasks(tasks, force=bool(args.force))
    print(
        f"[split-grid] tasks={len(tasks)} skipped={len(tasks) - len(pending)} "
        f"pending={len(pending)} workers={int(args.workers)}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            print(f"[dry-run] {task.label}: {' '.join(task.cmd)}", flush=True)
        return
    task_queue: queue.Queue[SplitTask] = queue.Queue()
    for task in pending:
        task_queue.put(task)
    gpu_ids = [value.strip() for value in str(args.gpu_ids).split(",") if value.strip()]
    failures: list[tuple[str, str, Path]] = []
    lock = threading.Lock()

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
                result = subprocess.run(task.cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if result.returncode != 0:
                with lock:
                    failures.append((task.label, f"exit={result.returncode}", task.log_path))
            elif not task.eval_path.exists():
                with lock:
                    failures.append((task.label, f"missing_eval={task.eval_path}", task.log_path))
            else:
                task.done_path.write_text(
                    f"label={task.label}\nbudget={task.budget:.6f}\nseed={task.seed}\n"
                    f"eval_path={task.eval_path}\nlog_path={task.log_path}\n",
                    encoding="utf-8",
                )
            task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(idx,), daemon=True) for idx in range(max(1, int(args.workers)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        for label, reason, log_path in failures:
            print(f"[fail] {label} {reason} log={log_path}", flush=True)
        raise SystemExit(1)


def collect_results(args: argparse.Namespace) -> None:
    if args.dry_run or args.skip_collect:
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "43_v31_s2_collect.py"),
        "--out-dir",
        str(args.out_dir),
        "--static-policy",
        "validation_selected_static",
        "--bonferroni-family",
        str(int(args.bonferroni_family)),
    ]
    print(f"[collect] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.1 split-protocol budgets x seeds with resume markers.")
    parser.add_argument("--out-dir", default="reports/v31_split_protocol_main")
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.65, 1.70, 1.75])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(41, 51)))
    parser.add_argument("--startup-peak-budget", type=float, default=3.2)
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_balanced.yaml")
    parser.add_argument("--truth-steps", type=int, default=90000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--split-ratios", nargs=4, type=float, default=[0.35, 0.50, 0.075, 0.075])
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
    parser.add_argument("--event-particle-microstructure-correlation", type=float, default=1.0)
    parser.add_argument("--event-subtypes-enabled", action="store_true")
    parser.add_argument("--event-subtype-particle-prob", type=float, default=0.34)
    parser.add_argument("--event-subtype-flux-prob", type=float, default=0.33)
    parser.add_argument("--event-subtype-thermal-prob", type=float, default=0.33)
    parser.add_argument("--event-subtype-particle-flux-multiplier", type=float, default=0.72)
    parser.add_argument("--event-subtype-flux-multiplier", type=float, default=2.4)
    parser.add_argument("--event-subtype-thermal-flux-multiplier", type=float, default=0.55)
    parser.add_argument("--event-subtype-particle-diameter-shift-mm", type=float, default=0.10)
    parser.add_argument("--event-subtype-particle-velocity-boost-ms", type=float, default=1.3)
    parser.add_argument("--event-subtype-flux-diameter-shift-mm", type=float, default=-0.04)
    parser.add_argument("--event-subtype-flux-velocity-boost-ms", type=float, default=0.7)
    parser.add_argument("--event-subtype-thermal-surface-drop-c", type=float, default=2.0)
    parser.add_argument("--event-subtype-particle-humidity-boost-pct", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-wind-boost-ms", type=float, default=0.0)
    parser.add_argument("--event-subtype-thermal-air-temp-drop-c", type=float, default=0.0)
    parser.add_argument("--event-subtype-latent-alpha", type=float, default=0.18)
    parser.add_argument("--event-subtype-particle-latent-diameter-scale-mm", type=float, default=0.0)
    parser.add_argument("--event-subtype-particle-latent-velocity-scale-ms", type=float, default=0.0)
    parser.add_argument("--event-subtype-flux-latent-sigma", type=float, default=0.0)
    parser.add_argument("--event-subtype-thermal-latent-surface-scale-c", type=float, default=0.0)
    parser.add_argument("--event-subtype-latent-target-lag-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-lead-steps", type=int, default=0)
    parser.add_argument("--event-subtype-context-noise-std", type=float, default=0.08)
    parser.add_argument("--oracle-rollout-steps", type=int, default=7200)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-loss-clip", type=float, default=10.0)
    parser.add_argument("--oracle-candidate-mask-repeat", type=int, default=0)
    parser.add_argument("--oracle-candidate-mask-limit", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-repeat", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-lookahead-steps", type=int, default=0)
    parser.add_argument("--oracle-subtype-teacher-calm-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-particle-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-flux-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-subtype-teacher-thermal-sensors", nargs="*", default=None)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--awbc-coef", type=float, default=0.1)
    parser.add_argument("--awbc-label-stride", type=int, default=4)
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--bc-pretrain-epochs", type=int, default=4)
    parser.add_argument("--bc-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--bc-pretrain-loss-coef", type=float, default=1.0)
    parser.add_argument("--subtype-aux-coef", type=float, default=0.0)
    parser.add_argument("--subtype-aux-classes", type=int, default=4)
    parser.add_argument("--subtype-aux-lookahead-steps", type=int, default=0)
    parser.add_argument("--subtype-router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-router-min-confidence", type=float, default=0.0)
    parser.add_argument("--subtype-router-low-confidence-action", type=int, default=-1)
    parser.add_argument(
        "--awbc-teacher-mode",
        choices=["oracle_greedy", "event_pair", "event_cyclic", "subtype_auto", "energy_mpc"],
        default="oracle_greedy",
    )
    parser.add_argument("--awbc-teacher-calm-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-event-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-event-lookahead-steps", type=int, default=0)
    parser.add_argument("--awbc-teacher-energy-mpc-horizon", type=int, default=4)
    parser.add_argument("--awbc-teacher-energy-mpc-soc-bins", type=int, default=16)
    parser.add_argument("--awbc-teacher-energy-mpc-low-soc-ratio", type=float, default=0.25)
    parser.add_argument("--awbc-teacher-energy-mpc-high-soc-ratio", type=float, default=0.75)
    parser.add_argument("--awbc-teacher-energy-mpc-terminal-soc-weight", type=float, default=0.0)
    parser.add_argument("--awbc-teacher-energy-mpc-max-actions", type=int, default=0)
    parser.add_argument("--awbc-teacher-energy-mpc-low-power-action", type=int, default=-1)
    parser.add_argument("--awbc-teacher-calm-pool-spec", default=None)
    parser.add_argument("--awbc-teacher-event-pool-spec", default=None)
    parser.add_argument("--awbc-teacher-subtype-calm-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-particle-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-flux-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-subtype-thermal-sensors", nargs="*", default=None)
    parser.add_argument("--awbc-teacher-dwell-steps", type=int, default=1)
    parser.add_argument("--prior-kl-coef", type=float, default=1.0)
    parser.add_argument("--greedy-lookahead-steps", type=int, default=4)
    parser.add_argument("--event-start-prob", type=float, default=0.67)
    parser.add_argument("--event-aware-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-gated-actor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--soc-aux-horizon", type=int, default=0)
    parser.add_argument("--soc-aux-coef", type=float, default=0.0)
    parser.add_argument("--train-episode-len", type=int, default=512)
    parser.add_argument("--use-candidate-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-prior-scale", type=float, default=2.0)
    parser.add_argument("--candidate-prior-steps", type=int, default=512)
    parser.add_argument("--candidate-prior-rollouts", type=int, default=4)
    parser.add_argument("--static-selection-steps", type=int, default=512)
    parser.add_argument("--static-selection-rollouts", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument(
        "--eval-start-selection",
        choices=["uniform", "event_fraction", "event_rich", "event_transport_rich"],
        default="uniform",
    )
    parser.add_argument("--eval-event-fraction", type=float, default=0.67)
    parser.add_argument("--eval-selection-stride", type=int, default=64)
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--lambda-switch", type=float, default=0.002)
    parser.add_argument("--event-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--energy-account", action="store_true")
    parser.add_argument("--energy-capacity", type=float, default=0.0)
    parser.add_argument("--initial-energy", type=float, default=0.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.0)
    parser.add_argument("--reserve-energy", type=float, default=0.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--soc-soft-penalty-buffer", type=float, default=0.0)
    parser.add_argument("--lambda-soc-soft-penalty", type=float, default=0.0)
    parser.add_argument("--lambda-duty-balance", type=float, default=0.0)
    parser.add_argument("--duty-balance-low", type=float, default=0.05)
    parser.add_argument("--duty-balance-high", type=float, default=0.95)
    parser.add_argument("--duty-balance-grace-steps", type=int, default=64)
    parser.add_argument("--duty-score-feedback", type=float, default=0.0)
    parser.add_argument("--duty-score-target", type=float, default=0.40)
    parser.add_argument("--duty-hard-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--duty-hard-low", type=float, default=0.08)
    parser.add_argument("--duty-hard-high", type=float, default=0.92)
    parser.add_argument("--duty-hard-score", type=float, default=8.0)
    parser.add_argument("--min-dwell-steps", type=int, default=1)
    parser.add_argument("--include-agent-cycle-phase", action="store_true")
    parser.add_argument("--agent-cycle-period-steps", type=int, default=0)
    parser.add_argument("--agent-cycle-dwell-steps", type=int, default=1)
    parser.add_argument("--include-observable-regime-belief", action="store_true")
    parser.add_argument("--regime-belief-lookback", type=int, default=6)
    parser.add_argument("--agent-context-columns", nargs="*", default=None)
    parser.add_argument("--additional-state-columns", nargs="*", default=None)
    parser.add_argument("--eval-duty-constrained-baselines", action="store_true")
    parser.add_argument("--baseline-duty-hard-low", type=float, default=None)
    parser.add_argument("--baseline-duty-hard-high", type=float, default=None)
    parser.add_argument("--baseline-duty-hard-score", type=float, default=None)
    parser.add_argument("--baseline-duty-score-feedback", type=float, default=None)
    parser.add_argument("--primary-eval-duty-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--target-scales", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-loss-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subtype-particle-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-flux-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--subtype-thermal-target-weights", nargs="*", type=float, default=None)
    parser.add_argument("--required-sensors", nargs="*", default=None)
    parser.add_argument("--disable-coverage-groups", action="store_true")
    parser.add_argument("--max-active", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--bonferroni-family", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rollout-evaluation", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir)
    for name in ("raw", "done", "logs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    tasks = [build_task(args, budget=budget, seed=seed) for budget in args.budgets for seed in args.seeds]
    run_tasks(tasks, args)
    collect_results(args)


if __name__ == "__main__":
    main()
