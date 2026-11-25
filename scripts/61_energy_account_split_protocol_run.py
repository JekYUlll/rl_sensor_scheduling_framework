#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def partition_bounds(steps: int, ratios: tuple[float, float, float, float]) -> dict[str, tuple[int, int]]:
    total = float(sum(ratios))
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to one, got {total}")
    edges = [0]
    cumulative = 0.0
    for ratio in ratios[:-1]:
        cumulative += float(ratio)
        edges.append(int(round(int(steps) * cumulative)))
    edges.append(int(steps))
    names = ("oracle_pretrain", "rl_train", "validation", "final_test")
    bounds = {name: (int(edges[idx]), int(edges[idx + 1])) for idx, name in enumerate(names)}
    if any(end <= start for start, end in bounds.values()):
        raise ValueError(f"All partitions must be nonempty, got {bounds}")
    return bounds


def random_non_overlapping_starts(
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    start, end = (int(bounds[0]), int(bounds[1]))
    required_span = int(count) * int(window_steps) + int(horizon) + 1
    if end - start < required_span:
        raise ValueError(f"Partition [{start}, {end}) is too short for {count} evaluation windows")
    rng = np.random.default_rng(int(seed))
    slack = int(end - start - required_span)
    gaps = rng.multinomial(slack, np.full(int(count) + 1, 1.0 / float(int(count) + 1)))
    starts: list[int] = []
    cursor = start + int(gaps[0])
    for idx in range(int(count)):
        starts.append(int(cursor))
        cursor += int(window_steps) + int(gaps[idx + 1])
    return tuple(starts)


def event_rich_non_overlapping_starts(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    count: int,
    stride: int,
    event_column: str = "event_flag",
) -> tuple[tuple[int, ...], dict[str, object]]:
    start, end = (int(bounds[0]), int(bounds[1]))
    max_start = end - int(window_steps) - int(horizon) - 1
    if max_start < start:
        raise ValueError(f"Partition [{start}, {end}) cannot contain one requested window")
    if event_column not in truth.columns:
        raise ValueError(f"Truth data do not contain event column {event_column!r}")
    span = int(window_steps) + int(horizon) + 1
    candidate_starts = list(range(start, max_start + 1, max(1, int(stride))))
    if not candidate_starts or candidate_starts[-1] != max_start:
        candidate_starts.append(max_start)
    flags = truth[event_column].astype(bool).to_numpy()
    rates = np.asarray(
        [float(np.mean(flags[item : item + int(window_steps)])) for item in candidate_starts],
        dtype=float,
    )

    previous = [bisect.bisect_right(candidate_starts, value - span) - 1 for value in candidate_starts]
    n = len(candidate_starts)
    wanted = int(count)
    scores = np.full((n + 1, wanted + 1), -np.inf, dtype=float)
    selected = np.zeros((n + 1, wanted + 1), dtype=bool)
    scores[:, 0] = 0.0
    for idx in range(1, n + 1):
        for number in range(1, wanted + 1):
            skip = scores[idx - 1, number]
            take = rates[idx - 1] + scores[previous[idx - 1] + 1, number - 1]
            if take > skip:
                scores[idx, number] = take
                selected[idx, number] = True
            else:
                scores[idx, number] = skip
    if not np.isfinite(scores[n, wanted]):
        raise ValueError(f"Partition [{start}, {end}) cannot contain {wanted} non-overlapping windows")
    chosen: list[int] = []
    idx = n
    number = wanted
    while number > 0:
        if selected[idx, number]:
            chosen.append(int(candidate_starts[idx - 1]))
            idx = previous[idx - 1] + 1
            number -= 1
        else:
            idx -= 1
    chosen = sorted(chosen)
    selected_rates = [float(np.mean(flags[value : value + int(window_steps)])) for value in chosen]
    return tuple(chosen), {
        "selection": "maximum_total_event_rate_non_overlapping_within_declared_partition",
        "event_column": str(event_column),
        "stride": int(stride),
        "candidate_count": int(n),
        "selected_event_rates": selected_rates,
        "selected_event_rate_mean": float(np.mean(selected_rates)),
    }


def ensure_truth(args: argparse.Namespace, truth_path: Path) -> None:
    if truth_path.exists() and not bool(args.force_truth):
        return
    if truth_path.exists():
        truth_path.unlink()
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "20_build_public_weather_truth.py"),
        "--antaws-root",
        str(args.antaws_root),
        "--stations",
        *[str(station) for station in args.stations],
        "--steps",
        str(int(args.truth_steps)),
        "--freq-s",
        str(int(args.freq_s)),
        "--seed",
        str(int(args.seed)),
        "--blowing-snow-event-coverage",
        str(float(args.event_coverage)),
        "--blowing-snow-event-model",
        str(args.event_model),
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
        "--event-particle-microstructure-correlation",
        str(float(args.event_particle_microstructure_correlation)),
        "--out",
        str(truth_path),
        "--report-dir",
        str(Path(args.out_dir) / "dataset_validation"),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one split-protocol energy-account storm-conditional experiment.")
    parser.add_argument("--out-dir", default="reports/energy_account_split_protocol_gate/budget1p20_seed41")
    parser.add_argument("--truth-csv", default=None)
    parser.add_argument("--force-truth", action="store_true")
    parser.add_argument("--antaws-root", default="../data/AntAWS/3_hourly")
    parser.add_argument("--stations", nargs="+", default=["Panda100", "Panda200", "Taishan"])
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v4.yaml")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--truth-steps", type=int, default=90000)
    parser.add_argument("--freq-s", type=int, default=10800)
    parser.add_argument("--split-ratios", nargs=4, type=float, default=[0.30, 0.45, 0.125, 0.125])
    parser.add_argument("--event-coverage", type=float, default=0.30)
    parser.add_argument(
        "--event-model",
        default="semi_markov",
        help="Use the V3.1 semi-Markov generator by default; legacy clustered filling is not split-stationary.",
    )
    parser.add_argument("--min-duration", type=int, default=10)
    parser.add_argument("--max-duration", type=int, default=30)
    parser.add_argument("--min-gap", type=int, default=6)
    parser.add_argument("--lead-steps", type=int, default=5)
    parser.add_argument("--wind-margin-ms", type=float, default=1.5)
    parser.add_argument("--cred-hysteresis-on", type=float, default=0.6)
    parser.add_argument("--cred-hysteresis-off", type=float, default=0.3)
    parser.add_argument("--flux-wind-exponent", type=float, default=3.6)
    parser.add_argument("--event-microstructure-sigma", type=float, default=0.8)
    parser.add_argument("--event-microstructure-alpha", type=float, default=0.18)
    parser.add_argument("--event-microstructure-diameter-scale", type=float, default=0.05)
    parser.add_argument("--event-microstructure-velocity-scale", type=float, default=1.2)
    parser.add_argument("--event-particle-microstructure-correlation", type=float, default=1.0)
    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--energy-capacity", type=float, default=180.0)
    parser.add_argument("--initial-energy", type=float, default=180.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--reserve-energy", type=float, default=20.0)
    parser.add_argument("--lambda-energy-deficit", type=float, default=1.0)
    parser.add_argument("--lambda-warmup-abort", type=float, default=0.08)
    parser.add_argument("--soc-soft-penalty-buffer", type=float, default=0.0)
    parser.add_argument("--lambda-soc-soft-penalty", type=float, default=0.0)
    parser.add_argument("--soc-aux-horizon", type=int, default=0)
    parser.add_argument("--soc-aux-coef", type=float, default=0.0)
    parser.add_argument("--ppo-max-candidate-warmup", type=int, default=-1)
    parser.add_argument("--oracle-rollout-steps", type=int, default=2400)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=6)
    parser.add_argument("--oracle-epochs", type=int, default=18)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--train-episode-len", type=int, default=512)
    parser.add_argument("--curriculum-context-steps", type=int, default=1024)
    parser.add_argument("--curriculum-rollouts", type=int, default=6)
    parser.add_argument("--static-selection-steps", type=int, default=1024)
    parser.add_argument("--static-selection-rollouts", type=int, default=6)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    parser.add_argument("--selection-stride", type=int, default=64)
    parser.add_argument("--final-selection", choices=["event_rich", "uniform"], default="event_rich")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    truth_path = Path(args.truth_csv) if args.truth_csv else out_dir / "truth_energy_split.csv"
    ensure_truth(args, truth_path)
    truth = pd.read_csv(truth_path)
    if len(truth) != int(args.truth_steps):
        raise ValueError(f"--truth-steps declares {args.truth_steps}, but truth CSV contains {len(truth)} rows")
    bounds = partition_bounds(int(args.truth_steps), tuple(float(value) for value in args.split_ratios))
    horizon = 8
    curriculum_starts, curriculum_diag = event_rich_non_overlapping_starts(
        truth,
        bounds=bounds["rl_train"],
        window_steps=int(args.curriculum_context_steps),
        horizon=horizon,
        count=int(args.curriculum_rollouts),
        stride=int(args.selection_stride),
    )
    validation_starts, validation_diag = event_rich_non_overlapping_starts(
        truth,
        bounds=bounds["validation"],
        window_steps=int(args.static_selection_steps),
        horizon=horizon,
        count=int(args.static_selection_rollouts),
        stride=int(args.selection_stride),
    )
    if str(args.final_selection) == "event_rich":
        final_starts, final_diag = event_rich_non_overlapping_starts(
            truth,
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=horizon,
            count=int(args.eval_rollouts),
            stride=int(args.selection_stride),
        )
    else:
        final_starts = random_non_overlapping_starts(
            bounds=bounds["final_test"],
            window_steps=int(args.eval_steps),
            horizon=horizon,
            count=int(args.eval_rollouts),
            seed=int(args.seed) + 1777,
        )
        flags = truth["event_flag"].astype(bool).to_numpy()
        final_diag = {
            "selection": "uniform_random_non_overlapping_within_declared_partition",
            "selected_event_rates": [
                float(np.mean(flags[value : value + int(args.eval_steps)])) for value in final_starts
            ],
        }

    manifest = {
        "protocol": "chronological_split_energy_account_storm_conditional_v1",
        "evidence_role": "conditional_storm_test_using_truth_event_labels_not_operational_deployment",
        "truth_csv": str(truth_path),
        "truth_steps": int(args.truth_steps),
        "seed": int(args.seed),
        "split_ratios": [float(value) for value in args.split_ratios],
        "partitions": {name: [int(start), int(end)] for name, (start, end) in bounds.items()},
        "oracle_pretrain": {"range": list(bounds["oracle_pretrain"])},
        "rl_train": {
            "range": list(bounds["rl_train"]),
            "normalization_range": list(bounds["rl_train"]),
            "curriculum_starts": list(curriculum_starts),
            "episode_steps": int(args.train_episode_len),
            "context_selection_steps": int(args.curriculum_context_steps),
            **curriculum_diag,
        },
        "validation": {
            "static_selection_starts": list(validation_starts),
            "static_selection_steps": int(args.static_selection_steps),
            **validation_diag,
        },
        "final_test": {
            "eval_starts": list(final_starts),
            "eval_steps": int(args.eval_steps),
            **final_diag,
        },
        "energy_account": {
            "budget": float(args.budget),
            "startup_peak_budget": float(args.startup_peak_budget),
            "capacity": float(args.energy_capacity),
            "initial_energy": float(args.initial_energy),
            "harvest_per_step": float(args.harvest_per_step),
            "reserve_energy": float(args.reserve_energy),
            "lambda_energy_deficit": float(args.lambda_energy_deficit),
            "soc_soft_penalty_buffer": float(args.soc_soft_penalty_buffer),
            "lambda_soc_soft_penalty": float(args.lambda_soc_soft_penalty),
        },
        "ppo_controls": {
            "lambda_warmup_abort": float(args.lambda_warmup_abort),
            "soc_aux_horizon": int(args.soc_aux_horizon),
            "soc_aux_coef": float(args.soc_aux_coef),
            "ppo_max_candidate_warmup": int(args.ppo_max_candidate_warmup),
            "total_timesteps": int(args.total_timesteps),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25_v2_train_custom_ppo.py"),
        "--out-dir",
        str(out_dir),
        "--truth-csv",
        str(truth_path),
        "--sensor-cfg",
        str(args.sensor_cfg),
        "--seed",
        str(int(args.seed)),
        "--freq-s",
        str(int(args.freq_s)),
        "--blowing-snow-event-coverage",
        str(float(args.event_coverage)),
        "--blowing-snow-event-model",
        str(args.event_model),
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
        "--event-particle-microstructure-correlation",
        str(float(args.event_particle_microstructure_correlation)),
        "--per-step-budget",
        str(float(args.budget)),
        "--startup-peak-budget",
        str(float(args.startup_peak_budget)),
        "--max-active",
        str(int(args.max_active)),
        "--required-sensors",
        "met_station_core",
        "--disable-coverage-groups",
        "--energy-account",
        "--energy-capacity",
        str(float(args.energy_capacity)),
        "--initial-energy",
        str(float(args.initial_energy)),
        "--harvest-per-step",
        str(float(args.harvest_per_step)),
        "--reserve-energy",
        str(float(args.reserve_energy)),
        "--lambda-energy-deficit",
        str(float(args.lambda_energy_deficit)),
        "--lambda-warmup-abort",
        str(float(args.lambda_warmup_abort)),
        "--soc-soft-penalty-buffer",
        str(float(args.soc_soft_penalty_buffer)),
        "--lambda-soc-soft-penalty",
        str(float(args.lambda_soc_soft_penalty)),
        "--target-weights",
        "0.2",
        "0.3",
        "0.2",
        "0.1",
        "0.1",
        "0.1",
        "12.0",
        "8.0",
        "8.0",
        "--target-scales",
        "5.0",
        "5.0",
        "5.0",
        "1.0",
        "1.0",
        "100.0",
        "0.00005",
        "0.2",
        "5.0",
        "--oracle-type",
        "tcn",
        "--oracle-rollout-steps",
        str(int(args.oracle_rollout_steps)),
        "--oracle-rollouts-per-policy",
        str(int(args.oracle_rollouts_per_policy)),
        "--oracle-epochs",
        str(int(args.oracle_epochs)),
        "--oracle-start-idx",
        str(bounds["oracle_pretrain"][0]),
        "--oracle-end-idx",
        str(bounds["oracle_pretrain"][1]),
        "--normalization-start-idx",
        str(bounds["rl_train"][0]),
        "--normalization-end-idx",
        str(bounds["rl_train"][1]),
        "--train-episode-len",
        str(int(args.train_episode_len)),
        "--train-start-indices",
        *[str(value) for value in curriculum_starts],
        "--static-selection-steps",
        str(int(args.static_selection_steps)),
        "--static-selection-start-indices",
        *[str(value) for value in validation_starts],
        "--total-timesteps",
        str(int(args.total_timesteps)),
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--prior-kl-coef",
        "0.0",
        "--soc-aux-horizon",
        str(int(args.soc_aux_horizon)),
        "--soc-aux-coef",
        str(float(args.soc_aux_coef)),
        "--ppo-max-candidate-warmup",
        str(int(args.ppo_max_candidate_warmup)),
        "--event-start-prob",
        "0.85",
        "--eval-steps",
        str(int(args.eval_steps)),
        "--eval-rollouts",
        str(int(args.eval_rollouts)),
        "--eval-start-indices",
        *[str(value) for value in final_starts],
        "--device",
        str(args.device),
    ]
    if bool(args.dry_run):
        print(manifest_path)
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    print(manifest_path)


if __name__ == "__main__":
    main()
