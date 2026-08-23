#!/usr/bin/env python3
"""Aggregate frozen flexible-subset PD-PPO confirmation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ORDINARY = "oracle_loss_mean"
MACRO = "oracle_loss_macro_subtype_event_staticnorm"
STATIC_POLICIES = ("feasible_static_projected", "validation_selected_static")
DYNAMIC_POLICIES = ("aoi", "round_robin", "random")


def seed_from_dir(path: Path) -> int:
    return int(path.name.split("seed", 1)[1].split("_", 1)[0])


def bootstrap_mean_ci(values: np.ndarray, *, seed: int = 20260823) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def reference_row(context: pd.DataFrame, seed: int, policy: str) -> pd.Series | None:
    rows = context[(context["seed"] == seed) & (context["policy"] == policy)]
    return None if rows.empty else rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-glob", required=True)
    parser.add_argument("--context-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="Flexible-subset PD-PPO evaluation",
        help="Markdown report title without the leading hash.",
    )
    parser.add_argument(
        "--protocol-description",
        default="The protocol completed {seed_count} scene/policy seed pairs.",
        help="First report sentence; {seed_count} is replaced with the observed count.",
    )
    args = parser.parse_args()

    run_dirs = sorted({p.parent for p in Path().glob(args.policy_glob)})
    if not run_dirs:
        raise FileNotFoundError(f"no policy metrics matched {args.policy_glob}")
    context = pd.read_csv(args.context_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, float | int]] = []
    duty_rows: list[dict[str, float | int | str]] = []
    for run_dir in run_dirs:
        seed = seed_from_dir(run_dir)
        metrics = pd.read_csv(run_dir / "v2_custom_ppo_metrics.csv")
        custom = metrics.loc[metrics["policy"] == "custom_ppo"].iloc[0]
        static = metrics.loc[metrics["policy"].isin(STATIC_POLICIES)]
        dynamic = metrics.loc[metrics["policy"].isin(DYNAMIC_POLICIES)]
        full_open = metrics.loc[metrics["policy"] == "full_open_unconstrained"].iloc[0]

        rollout = np.load(run_dir / "rollout_custom_ppo.npz")
        selected = rollout["selected_masks"].astype(bool)
        event = rollout["event_flags"].astype(bool)
        geometry = json.loads((run_dir / "action_geometry.json").read_text())
        sensor_ids = [str(value) for value in geometry["sensor_ids"]]
        feasible = {
            tuple(sensor_id in set(row["sensor_ids"]) for sensor_id in sensor_ids)
            for row in geometry["masks"]
        }
        invalid_actions = sum(tuple(row) not in feasible for row in selected)
        power_violations = int(np.sum(rollout["powers"] > float(geometry["budget"]) + 1e-9))
        peak_violations = int(
            np.sum(rollout["peaks"] > float(geometry["startup_peak_budget"]) + 1e-9)
        )

        max_delta = 0.0
        for sensor_id, channel in zip(rollout["sensor_ids"].tolist(), selected.T):
            duty_event = float(channel[event].mean()) if event.any() else float("nan")
            duty_calm = float(channel[~event].mean()) if (~event).any() else float("nan")
            delta = duty_event - duty_calm
            max_delta = max(max_delta, abs(delta))
            duty_rows.append(
                {
                    "seed": seed,
                    "sensor_id": str(sensor_id),
                    "duty_all": float(channel.mean()),
                    "duty_event": duty_event,
                    "duty_non_event": duty_calm,
                    "event_minus_non_event": delta,
                }
            )

        context_row = reference_row(context, seed, "context_alert_bandit_t0p5")
        greedy_row = reference_row(context, seed, "forecast_greedy_one_step")
        exact_rows = context[(context["seed"] == seed) & context["policy"].str.startswith("event_label_reference")]
        exact_row = None if exact_rows.empty else exact_rows.iloc[0]

        row: dict[str, float | int] = {
            "seed": seed,
            "pdppo_ordinary": float(custom[ORDINARY]),
            "pdppo_macro": float(custom[MACRO]),
            "static_ordinary_margin": float(static[ORDINARY].min() - custom[ORDINARY]),
            "static_macro_margin": float(static[MACRO].min() - custom[MACRO]),
            "dynamic_ordinary_margin": float(dynamic[ORDINARY].min() - custom[ORDINARY]),
            "dynamic_macro_margin": float(dynamic[MACRO].min() - custom[MACRO]),
            "full_open_ordinary_margin": float(full_open[ORDINARY] - custom[ORDINARY]),
            "full_open_macro_margin": float(full_open[MACRO] - custom[MACRO]),
            "always_on": int(custom["always_on_sensor_count"]),
            "always_off": int(custom["always_off_sensor_count"]),
            "mid_duty": int(custom["mid_duty_sensor_count"]),
            "switches_per_step": float(custom["switches_per_step"]),
            "warmup_abort_count": int(custom["warmup_abort_count"]),
            "unique_action_count": int(len(np.unique(selected, axis=0))),
            "max_abs_event_calm_duty_delta": max_delta,
            "invalid_action_count": int(invalid_actions),
            "power_violation_count": power_violations,
            "startup_peak_violation_count": peak_violations,
        }
        for prefix, ref in (("context", context_row), ("forecast_greedy", greedy_row), ("exact_label", exact_row)):
            row[f"{prefix}_ordinary_margin"] = float("nan") if ref is None else float(ref[ORDINARY] - custom[ORDINARY])
            row[f"{prefix}_macro_margin"] = float("nan") if ref is None else float(ref[MACRO] - custom[MACRO])
        row["behavior_gate"] = int(
            row["always_on"] <= 1
            and row["always_off"] <= 1
            and row["switches_per_step"] > 0
            and row["warmup_abort_count"] == 0
            and row["unique_action_count"] > 1
            and row["max_abs_event_calm_duty_delta"] >= 0.05
            and row["invalid_action_count"] == 0
            and row["power_violation_count"] == 0
            and row["startup_peak_violation_count"] == 0
        )
        seed_rows.append(row)

    seeds = pd.DataFrame(seed_rows).sort_values("seed")
    duties = pd.DataFrame(duty_rows).sort_values(["seed", "sensor_id"])
    seeds.to_csv(args.out_dir / "seed_metrics.csv", index=False)
    duties.to_csv(args.out_dir / "channel_duty.csv", index=False)

    families = ("static", "dynamic", "context", "forecast_greedy", "exact_label", "full_open")
    summary_rows = []
    for family in families:
        ordinary = seeds[f"{family}_ordinary_margin"].dropna().to_numpy(float)
        macro = seeds[f"{family}_macro_margin"].dropna().to_numpy(float)
        if not len(ordinary):
            continue
        ordinary_ci = bootstrap_mean_ci(ordinary)
        macro_ci = bootstrap_mean_ci(macro)
        summary_rows.append(
            {
                "baseline_family": family,
                "seed_count": len(ordinary),
                "ordinary_wins": int(np.sum(ordinary > 0)),
                "macro_wins": int(np.sum(macro > 0)),
                "joint_wins": int(np.sum((ordinary > 0) & (macro > 0))),
                "ordinary_mean_margin": float(ordinary.mean()),
                "ordinary_ci_low": ordinary_ci[0],
                "ordinary_ci_high": ordinary_ci[1],
                "macro_mean_margin": float(macro.mean()),
                "macro_ci_low": macro_ci[0],
                "macro_ci_high": macro_ci[1],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "family_summary.csv", index=False)

    lines = [
        f"# {args.title}",
        "",
        args.protocol_description.format(seed_count=len(seeds)),
        "Positive margins indicate lower PD-PPO loss than the comparator.",
        "",
        "| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.baseline_family} | {row.ordinary_wins}/{row.seed_count} | "
            f"{row.macro_wins}/{row.seed_count} | {row.joint_wins}/{row.seed_count} | "
            f"{row.ordinary_mean_margin:+.6f} [{row.ordinary_ci_low:+.6f}, {row.ordinary_ci_high:+.6f}] | "
            f"{row.macro_mean_margin:+.6f} [{row.macro_ci_low:+.6f}, {row.macro_ci_high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Behavior and feasibility gate: {int(seeds['behavior_gate'].sum())}/{len(seeds)} seeds.",
            f"Invalid actions: {int(seeds['invalid_action_count'].sum())}; per-step power violations: "
            f"{int(seeds['power_violation_count'].sum())}; startup-peak violations: "
            f"{int(seeds['startup_peak_violation_count'].sum())}; warm-up aborts: "
            f"{int(seeds['warmup_abort_count'].sum())}.",
            "",
            "The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. "
            "The context row uses only supplied noisy warning scores and validation-calibrated actions.",
        ]
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
