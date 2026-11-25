#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


TARGET_PROFILES: dict[str, tuple[float, ...]] = {
    "physical_default": (0.8, 0.8, 1.2, 0.4, 0.4, 0.55, 4.0, 2.5, 2.5),
    "balanced_transport_v6": (0.4, 0.4, 0.6, 0.1, 0.1, 0.1, 6.0, 3.0, 3.0),
    "transport_v6": (0.2, 0.2, 0.3, 0.05, 0.05, 0.05, 8.0, 4.0, 4.0),
    "snow_task_v6": (0.1, 0.1, 0.25, 0.05, 0.05, 0.0, 10.0, 5.0, 5.0),
    "flux_task_v6": (0.05, 0.05, 0.15, 0.02, 0.02, 0.0, 24.0, 2.0, 2.0),
    "particle_flux_v6": (0.05, 0.05, 0.15, 0.02, 0.02, 0.0, 16.0, 6.0, 6.0),
    "micro_flux_v6": (0.03, 0.03, 0.10, 0.01, 0.01, 0.0, 18.0, 12.0, 12.0),
    "micro_particle_v6": (0.03, 0.03, 0.10, 0.01, 0.01, 0.0, 12.0, 16.0, 16.0),
    "flux_micro_v6": (0.03, 0.03, 0.10, 0.01, 0.01, 0.0, 24.0, 10.0, 10.0),
    "dual_flux_particle_v7": (0.03, 0.03, 0.10, 0.01, 0.01, 0.0, 22.0, 16.0, 16.0),
    "event_flux_particle_v7": (0.03, 0.03, 0.10, 0.01, 0.01, 0.0, 30.0, 12.0, 12.0),
    "particle_heavy_flux_v7": (0.03, 0.03, 0.10, 0.01, 0.01, 0.0, 16.0, 22.0, 22.0),
}


def _tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _append_required_sensors(cmd: list[str], required_sensors: list[str] | None) -> None:
    if required_sensors is None:
        return
    cmd.append("--required-sensors")
    cmd.extend(str(value) for value in required_sensors)


def _build_oracle_lift_cmd(
    args: argparse.Namespace,
    *,
    profile: str,
    budget: float,
    peak_budget: float,
    combo_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "49_v31_physical_event_oracle_lift.py"),
        "--truth-csv",
        str(Path(args.out_dir) / "truth_static_break_calibration.csv"),
        "--out-dir",
        str(combo_dir),
        "--sensor-cfg",
        str(args.sensor_cfg),
        "--budget",
        str(float(budget)),
        "--startup-peak-budget",
        str(float(peak_budget)),
        "--max-active",
        str(int(args.max_active)),
        "--target-weights",
        *[str(value) for value in TARGET_PROFILES[profile]],
        "--oracle-type",
        str(args.oracle_type),
        "--truth-steps",
        str(int(args.truth_steps)),
        "--freq-s",
        str(int(args.freq_s)),
        "--blowing-snow-event-coverage",
        str(float(args.event_coverage)),
        "--blowing-snow-min-duration-steps",
        str(int(args.min_duration)),
        "--blowing-snow-max-duration-steps",
        str(int(args.max_duration)),
        "--blowing-snow-min-gap-steps",
        str(int(args.min_gap)),
        "--blowing-snow-lead-steps",
        str(int(args.lead_steps)),
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
        "--oracle-rollout-steps",
        str(int(args.oracle_rollout_steps)),
        "--oracle-rollouts-per-policy",
        str(int(args.oracle_rollouts_per_policy)),
        "--oracle-epochs",
        str(int(args.oracle_epochs)),
        "--oracle-batch-size",
        str(int(args.oracle_batch_size)),
        "--oracle-device",
        str(args.oracle_device),
        "--oracle-inference-device",
        str(args.oracle_inference_device),
        "--eval-steps",
        str(int(args.eval_steps)),
        "--eval-rollouts",
        str(int(args.eval_rollouts)),
        "--eval-event-fraction",
        str(float(args.eval_event_fraction)),
        "--env-min-dwell-steps",
        str(int(max(1, int(args.env_min_dwell_steps)))),
        "--eval-start-selection",
        str(args.eval_start_selection),
        "--eval-selection-stride",
        str(int(args.eval_selection_stride)),
        "--schedule-diagnostics",
        "--schedule-family",
        str(args.schedule_family),
        "--schedule-lead-steps",
        str(int(args.schedule_lead_steps)),
        "--auto-schedule-top-k",
        str(int(args.auto_schedule_top_k)),
        "--diverse-schedule-dwell-steps",
        str(int(args.diverse_schedule_dwell_steps)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.deployable_static_diagnostics:
        cmd.extend(
            [
                "--deployable-static-diagnostics",
                "--deployable-static-top-k",
                str(int(args.deployable_static_top_k)),
                "--deployable-static-duty-low",
                str(float(args.deployable_static_duty_low)),
                "--deployable-static-duty-high",
                str(float(args.deployable_static_duty_high)),
                "--deployable-static-duty-score",
                str(float(args.deployable_static_duty_score)),
                "--deployable-static-duty-feedback",
                str(float(args.deployable_static_duty_feedback)),
            ]
        )
    if args.target_diagnostics:
        cmd.append("--target-diagnostics")
    if args.coverage_groups:
        cmd.append("--coverage-groups")
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
    if args.force_truth:
        cmd.append("--force-truth")
    if bool(args.event_subtypes_enabled):
        cmd.append("--event-subtypes-enabled")
    _append_required_sensors(cmd, args.required_sensors)
    return cmd


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summarize_combo(
    *,
    table: pd.DataFrame,
    profile: str,
    budget: float,
    peak_budget: float,
    required_sensors: list[str] | None,
    min_dynamic_margin: float,
    min_mid_duty_sensors: int,
    max_always_on_sensors: int,
    max_always_off_sensors: int,
    min_switches_per_step: float,
    max_switches_per_step: float,
    require_diverse_dynamic: bool,
    compare_deployable_static: bool,
    require_raw_static_margin: bool,
) -> dict[str, object]:
    deployable_mask = (
        table.get("is_deployable_static", pd.Series(False, index=table.index))
        .fillna(False)
        .astype(bool)
    )
    static = table[(table["action_idx"].astype(int) >= 0) & ~deployable_mask].copy()
    deployable_static = table[deployable_mask].copy()
    dynamic = table[(table["action_idx"].astype(int) < 0) & ~deployable_mask].copy()
    finite_static = static[np.isfinite(static["oracle_loss_mean"].to_numpy(dtype=float))]
    finite_deployable_static = deployable_static[
        np.isfinite(deployable_static["oracle_loss_mean"].to_numpy(dtype=float))
    ]
    finite_dynamic = dynamic[np.isfinite(dynamic["oracle_loss_mean"].to_numpy(dtype=float))]
    if finite_static.empty:
        raise ValueError("No finite static candidates in oracle-lift table")
    best_static = finite_static.sort_values("oracle_loss_mean").iloc[0]
    best_deployable_static = (
        finite_deployable_static.sort_values("oracle_loss_mean").iloc[0]
        if not finite_deployable_static.empty
        else None
    )
    reference_static = best_static
    reference_name = "raw_static"
    if bool(compare_deployable_static) and best_deployable_static is not None:
        reference_static = best_deployable_static
        reference_name = "deployable_static"
    best_any_dynamic = finite_dynamic.sort_values("oracle_loss_mean").iloc[0] if not finite_dynamic.empty else None
    eligible_dynamic = finite_dynamic
    if bool(require_diverse_dynamic) and not finite_dynamic.empty:
        eligible_dynamic = finite_dynamic[
            (finite_dynamic.get("mid_duty_sensor_count", 0).astype(float) >= float(min_mid_duty_sensors))
            & (finite_dynamic.get("always_on_sensor_count", 99).astype(float) <= float(max_always_on_sensors))
            & (finite_dynamic.get("always_off_sensor_count", 99).astype(float) <= float(max_always_off_sensors))
            & (finite_dynamic.get("switches_per_step", 0.0).astype(float) >= float(min_switches_per_step))
            & (finite_dynamic.get("switches_per_step", 0.0).astype(float) <= float(max_switches_per_step))
        ]
    best_dynamic = eligible_dynamic.sort_values("oracle_loss_mean").iloc[0] if not eligible_dynamic.empty else best_any_dynamic
    topk = finite_static.sort_values("oracle_loss_mean").head(5)
    static_loss = _safe_float(reference_static["oracle_loss_mean"])
    raw_static_loss = _safe_float(best_static["oracle_loss_mean"])
    dynamic_loss = _safe_float(best_dynamic["oracle_loss_mean"]) if best_dynamic is not None else float("nan")
    any_dynamic_loss = _safe_float(best_any_dynamic["oracle_loss_mean"]) if best_any_dynamic is not None else float("nan")
    event_static_loss = _safe_float(reference_static["oracle_loss_event"])
    event_dynamic_loss = _safe_float(best_dynamic["oracle_loss_event"]) if best_dynamic is not None else float("nan")
    dynamic_margin = (static_loss - dynamic_loss) / static_loss if static_loss > 0 and np.isfinite(dynamic_loss) else float("nan")
    raw_static_margin = (
        (raw_static_loss - dynamic_loss) / raw_static_loss
        if raw_static_loss > 0 and np.isfinite(dynamic_loss)
        else float("nan")
    )
    event_dynamic_margin = (
        (event_static_loss - event_dynamic_loss) / event_static_loss
        if event_static_loss > 0 and np.isfinite(event_dynamic_loss)
        else float("nan")
    )
    top5_laser_frac = float(topk["has_laser"].astype(bool).mean()) if not topk.empty else float("nan")
    laser_shortcut_broken = (not bool(best_static["has_laser"])) and top5_laser_frac <= 0.40
    dynamic_headroom = bool(np.isfinite(dynamic_margin) and dynamic_margin >= float(min_dynamic_margin))
    raw_static_headroom = bool(np.isfinite(raw_static_margin) and raw_static_margin >= float(min_dynamic_margin))
    diversity_ok = bool(
        best_dynamic is not None
        and _safe_float(best_dynamic.get("mid_duty_sensor_count"), -1) >= float(min_mid_duty_sensors)
        and _safe_float(best_dynamic.get("always_on_sensor_count"), 99) <= float(max_always_on_sensors)
        and _safe_float(best_dynamic.get("always_off_sensor_count"), 99) <= float(max_always_off_sensors)
        and _safe_float(best_dynamic.get("switches_per_step"), -1) >= float(min_switches_per_step)
        and _safe_float(best_dynamic.get("switches_per_step"), 99) <= float(max_switches_per_step)
    )
    gate_static_ok = bool(laser_shortcut_broken) if not bool(compare_deployable_static) else True
    strict_static_gate_ok = bool(raw_static_headroom) if bool(require_raw_static_margin) else True
    return {
        "profile": profile,
        "budget": float(budget),
        "startup_peak_budget": float(peak_budget),
        "required_sensors": "|".join(required_sensors or []),
        "static_reference": reference_name,
        "candidate_count": int(len(static)),
        "deployable_static_count": int(len(deployable_static)),
        "dynamic_count": int(len(dynamic)),
        "best_static_sensor_ids": str(best_static["sensor_ids"]),
        "best_static_loss": static_loss,
        "best_static_event_loss": event_static_loss,
        "best_static_non_event_loss": _safe_float(best_static["oracle_loss_non_event"]),
        "best_static_has_laser": bool(best_static["has_laser"]),
        "best_static_has_fc4": bool(best_static["has_fc4"]),
        "best_static_has_spc": bool(best_static["has_snow_particle_counter"]),
        "best_raw_static_sensor_ids": str(best_static["sensor_ids"]),
        "best_raw_static_loss": raw_static_loss,
        "best_raw_static_event_loss": _safe_float(best_static["oracle_loss_event"]),
        "best_deployable_static_sensor_ids": str(best_deployable_static["sensor_ids"])
        if best_deployable_static is not None
        else "",
        "best_deployable_static_source_sensor_ids": str(best_deployable_static.get("source_static_sensor_ids", ""))
        if best_deployable_static is not None
        else "",
        "best_deployable_static_loss": _safe_float(best_deployable_static["oracle_loss_mean"])
        if best_deployable_static is not None
        else float("nan"),
        "best_deployable_static_event_loss": _safe_float(best_deployable_static["oracle_loss_event"])
        if best_deployable_static is not None
        else float("nan"),
        "best_deployable_static_mid_duty_sensor_count": int(
            _safe_float(best_deployable_static.get("mid_duty_sensor_count"), -1)
        )
        if best_deployable_static is not None
        else -1,
        "best_deployable_static_always_on_sensor_count": int(
            _safe_float(best_deployable_static.get("always_on_sensor_count"), -1)
        )
        if best_deployable_static is not None
        else -1,
        "best_deployable_static_always_off_sensor_count": int(
            _safe_float(best_deployable_static.get("always_off_sensor_count"), -1)
        )
        if best_deployable_static is not None
        else -1,
        "top5_static_laser_frac": top5_laser_frac,
        "top5_static_fc4_frac": float(topk["has_fc4"].astype(bool).mean()) if not topk.empty else float("nan"),
        "top5_static_spc_frac": float(topk["has_snow_particle_counter"].astype(bool).mean())
        if not topk.empty
        else float("nan"),
        "best_dynamic_sensor_ids": str(best_dynamic["sensor_ids"]) if best_dynamic is not None else "",
        "best_dynamic_loss": dynamic_loss,
        "best_any_dynamic_sensor_ids": str(best_any_dynamic["sensor_ids"]) if best_any_dynamic is not None else "",
        "best_any_dynamic_loss": any_dynamic_loss,
        "best_dynamic_event_loss": event_dynamic_loss,
        "best_dynamic_non_event_loss": _safe_float(best_dynamic["oracle_loss_non_event"])
        if best_dynamic is not None
        else float("nan"),
        "best_dynamic_has_laser": bool(best_dynamic["has_laser"]) if best_dynamic is not None else False,
        "best_dynamic_has_fc4": bool(best_dynamic["has_fc4"]) if best_dynamic is not None else False,
        "best_dynamic_has_spc": bool(best_dynamic["has_snow_particle_counter"]) if best_dynamic is not None else False,
        "best_dynamic_switches_per_step": _safe_float(best_dynamic.get("switches_per_step"))
        if best_dynamic is not None
        else float("nan"),
        "best_dynamic_mid_duty_sensor_count": int(_safe_float(best_dynamic.get("mid_duty_sensor_count"), -1))
        if best_dynamic is not None
        else -1,
        "best_dynamic_always_on_sensor_count": int(_safe_float(best_dynamic.get("always_on_sensor_count"), -1))
        if best_dynamic is not None
        else -1,
        "best_dynamic_always_off_sensor_count": int(_safe_float(best_dynamic.get("always_off_sensor_count"), -1))
        if best_dynamic is not None
        else -1,
        "best_dynamic_duty_entropy": _safe_float(best_dynamic.get("duty_entropy")) if best_dynamic is not None else float("nan"),
        "dynamic_margin": float(dynamic_margin),
        "raw_static_margin": float(raw_static_margin),
        "event_dynamic_margin": float(event_dynamic_margin),
        "laser_shortcut_broken": bool(laser_shortcut_broken),
        "dynamic_headroom": bool(dynamic_headroom),
        "raw_static_headroom": bool(raw_static_headroom),
        "dynamic_diversity_ok": bool(diversity_ok),
        "strict_static_gate_ok": bool(strict_static_gate_ok),
        "gate_pass": bool(gate_static_ok and dynamic_headroom and strict_static_gate_ok and diversity_ok),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search PD-PPO V3.1 static-break scenario settings.")
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v6_static_break.yaml")
    parser.add_argument("--out-dir", default="reports/v31_static_break_calibration")
    parser.add_argument("--profiles", nargs="+", choices=sorted(TARGET_PROFILES), default=["balanced_transport_v6", "transport_v6"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[1.10, 1.20, 1.30, 1.36])
    parser.add_argument("--startup-peak-budgets", nargs="+", type=float, default=[1.60])
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--required-sensors", nargs="*", default=[])
    parser.add_argument("--coverage-groups", action="store_true")
    parser.add_argument("--oracle-type", choices=["linear", "tcn"], default="linear")
    parser.add_argument("--truth-steps", type=int, default=30000)
    parser.add_argument("--freq-s", type=int, default=3600)
    parser.add_argument("--event-coverage", type=float, default=0.28)
    parser.add_argument("--min-duration", type=int, default=12)
    parser.add_argument("--max-duration", type=int, default=24)
    parser.add_argument("--min-gap", type=int, default=4)
    parser.add_argument("--lead-steps", type=int, default=6)
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
    parser.add_argument("--oracle-rollout-steps", type=int, default=1000)
    parser.add_argument("--oracle-rollouts-per-policy", type=int, default=3)
    parser.add_argument("--oracle-epochs", type=int, default=6)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--oracle-device", default="auto")
    parser.add_argument("--oracle-inference-device", default="cpu")
    parser.add_argument("--eval-steps", type=int, default=384)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--eval-event-fraction", type=float, default=0.75)
    parser.add_argument("--env-min-dwell-steps", type=int, default=1)
    parser.add_argument(
        "--eval-start-selection",
        choices=["event_fraction", "event_rich", "event_transport_rich"],
        default="event_fraction",
    )
    parser.add_argument("--eval-selection-stride", type=int, default=64)
    parser.add_argument("--schedule-lead-steps", type=int, default=4)
    parser.add_argument(
        "--schedule-family",
        choices=["v6_static_break", "auto_pairs", "diverse_auto", "subtype_static_break", "subtype_auto", "all"],
        default="auto_pairs",
    )
    parser.add_argument("--auto-schedule-top-k", type=int, default=4)
    parser.add_argument("--diverse-schedule-dwell-steps", type=int, default=16)
    parser.add_argument("--deployable-static-diagnostics", action="store_true")
    parser.add_argument("--deployable-static-top-k", type=int, default=6)
    parser.add_argument("--deployable-static-duty-low", type=float, default=0.12)
    parser.add_argument("--deployable-static-duty-high", type=float, default=0.75)
    parser.add_argument("--deployable-static-duty-score", type=float, default=12.0)
    parser.add_argument("--deployable-static-duty-feedback", type=float, default=2.5)
    parser.add_argument("--compare-deployable-static", action="store_true")
    parser.add_argument(
        "--require-raw-static-margin",
        action="store_true",
        help=(
            "When comparing against deployable static, also require the dynamic "
            "candidate to clear the raw best fixed-subset static margin."
        ),
    )
    parser.add_argument("--target-diagnostics", action="store_true")
    parser.add_argument("--energy-account", action="store_true")
    parser.add_argument("--energy-capacity", type=float, default=180.0)
    parser.add_argument("--initial-energy", type=float, default=180.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--reserve-energy", type=float, default=20.0)
    parser.add_argument("--min-dynamic-margin", type=float, default=0.01)
    parser.add_argument("--require-diverse-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-mid-duty-sensors", type=int, default=4)
    parser.add_argument("--max-always-on-sensors", type=int, default=2)
    parser.add_argument("--max-always-off-sensors", type=int, default=3)
    parser.add_argument("--min-switches-per-step", type=float, default=0.003)
    parser.add_argument("--max-switches-per-step", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-truth", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for profile in args.profiles:
        for budget in args.budgets:
            for peak_budget in args.startup_peak_budgets:
                combo = f"{profile}_b{_tag(float(budget))}_p{_tag(float(peak_budget))}"
                combo_dir = out_dir / combo
                if combo_dir.exists() and args.force:
                    shutil.rmtree(combo_dir)
                cmd = _build_oracle_lift_cmd(
                    args,
                    profile=str(profile),
                    budget=float(budget),
                    peak_budget=float(peak_budget),
                    combo_dir=combo_dir,
                )
                if args.dry_run:
                    print(" ".join(cmd))
                    continue
                log_path = out_dir / f"{combo}.log"
                print(f"[calibrate] {combo} -> {combo_dir}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    failures.append({"combo": combo, "returncode": int(result.returncode), "log_path": str(log_path)})
                    continue
                table_path = combo_dir / "oracle_lift_candidate_table.csv"
                if not table_path.exists():
                    failures.append({"combo": combo, "returncode": 0, "missing": str(table_path), "log_path": str(log_path)})
                    continue
                table = pd.read_csv(table_path)
                row = _summarize_combo(
                    table=table,
                    profile=str(profile),
                    budget=float(budget),
                    peak_budget=float(peak_budget),
                    required_sensors=args.required_sensors,
                    min_dynamic_margin=float(args.min_dynamic_margin),
                    min_mid_duty_sensors=int(args.min_mid_duty_sensors),
                    max_always_on_sensors=int(args.max_always_on_sensors),
                    max_always_off_sensors=int(args.max_always_off_sensors),
                    min_switches_per_step=float(args.min_switches_per_step),
                    max_switches_per_step=float(args.max_switches_per_step),
                    require_diverse_dynamic=bool(args.require_diverse_dynamic),
                    compare_deployable_static=bool(args.compare_deployable_static),
                    require_raw_static_margin=bool(args.require_raw_static_margin),
                )
                row["combo"] = combo
                row["out_dir"] = str(combo_dir)
                row["log_path"] = str(log_path)
                rows.append(row)
                pd.DataFrame(rows).sort_values(["gate_pass", "dynamic_margin"], ascending=[False, False]).to_csv(
                    out_dir / "calibration_summary.csv",
                    index=False,
                )
    summary = {
        "sensor_cfg": str(args.sensor_cfg),
            "oracle_type": str(args.oracle_type),
            "truth_event_design": {
                "event_coverage": float(args.event_coverage),
                "min_duration": int(args.min_duration),
                "max_duration": int(args.max_duration),
                "min_gap": int(args.min_gap),
                "lead_steps": int(args.lead_steps),
                "flux_wind_exponent": float(args.flux_wind_exponent),
                "event_microstructure_sigma": float(args.event_microstructure_sigma),
                "event_microstructure_alpha": float(args.event_microstructure_alpha),
                "event_microstructure_diameter_scale": float(args.event_microstructure_diameter_scale),
                "event_microstructure_velocity_scale": float(args.event_microstructure_velocity_scale),
                "event_particle_microstructure_correlation": float(args.event_particle_microstructure_correlation),
                "event_subtypes_enabled": bool(args.event_subtypes_enabled),
                "event_subtype_particle_prob": float(args.event_subtype_particle_prob),
                "event_subtype_flux_prob": float(args.event_subtype_flux_prob),
                "event_subtype_thermal_prob": float(args.event_subtype_thermal_prob),
                "event_subtype_particle_flux_multiplier": float(args.event_subtype_particle_flux_multiplier),
                "event_subtype_flux_multiplier": float(args.event_subtype_flux_multiplier),
                "event_subtype_thermal_flux_multiplier": float(args.event_subtype_thermal_flux_multiplier),
                "event_subtype_particle_diameter_shift_mm": float(args.event_subtype_particle_diameter_shift_mm),
                "event_subtype_particle_velocity_boost_ms": float(args.event_subtype_particle_velocity_boost_ms),
                "event_subtype_flux_diameter_shift_mm": float(args.event_subtype_flux_diameter_shift_mm),
                "event_subtype_flux_velocity_boost_ms": float(args.event_subtype_flux_velocity_boost_ms),
                "event_subtype_thermal_surface_drop_c": float(args.event_subtype_thermal_surface_drop_c),
            },
            "target_profiles": {name: list(TARGET_PROFILES[name]) for name in args.profiles},
            "diversity_gate": {
                "require_diverse_dynamic": bool(args.require_diverse_dynamic),
                "min_mid_duty_sensors": int(args.min_mid_duty_sensors),
                "max_always_on_sensors": int(args.max_always_on_sensors),
                "max_always_off_sensors": int(args.max_always_off_sensors),
                "min_switches_per_step": float(args.min_switches_per_step),
                "max_switches_per_step": float(args.max_switches_per_step),
                "deployable_static_diagnostics": bool(args.deployable_static_diagnostics),
                "compare_deployable_static": bool(args.compare_deployable_static),
                "require_raw_static_margin": bool(args.require_raw_static_margin),
            },
            "rows": rows,
        "failures": failures,
    }
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        frame = pd.DataFrame(rows).sort_values(["gate_pass", "dynamic_margin"], ascending=[False, False])
        frame.to_csv(out_dir / "calibration_summary.csv", index=False)
        cols = [
            "combo",
            "static_reference",
            "gate_pass",
            "dynamic_margin",
            "raw_static_margin",
            "event_dynamic_margin",
            "laser_shortcut_broken",
            "best_static_sensor_ids",
            "best_deployable_static_sensor_ids",
            "best_dynamic_sensor_ids",
            "best_dynamic_mid_duty_sensor_count",
            "best_dynamic_always_on_sensor_count",
            "best_dynamic_always_off_sensor_count",
            "dynamic_diversity_ok",
        ]
        print(frame[cols].to_string(index=False))
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
