#!/usr/bin/env python3
"""Build manuscript assets from the frozen clean PD-PPO evidence package.

The script deliberately requires every confirmatory/supplementary evidence block
to contain the same 24 seeds.  It is the only numerical source for the clean
result tables and figures introduced by the 2026-07-18 closure pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SEEDS = list(range(117, 141))
SENSOR_ORDER = [
    "shielded_thermo_hygro",
    "surface_temp_ir",
    "laser_disdrometer",
    "fc4_flux",
]
SENSOR_LABELS = ["Thermo-hygro", "Surface IR", "Laser", "FC4 flux"]
SUBTYPE_ORDER = ["calm", "particle", "flux", "thermal"]
SUBTYPE_LABELS = ["Calm", "Particle", "Flux", "Thermal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--main-dir", default="pdppo_clean_validation_frozen_24seed_20260718")
    parser.add_argument("--mechanism-dir", default="pdppo_clean_mechanism_24seed_20260718")
    parser.add_argument("--reference-dir", default="pdppo_framework_baselines_clean_24seed_20260718")
    parser.add_argument("--reward-dir", default="pdppo_clean_matched_reward_24seed_20260718")
    parser.add_argument("--dqn-dir", default="pdppo_matched_dqn_clean_24seed_20260718")
    parser.add_argument("--forecaster-dir", default="pdppo_secondary_forecaster_24seed_20260718")
    parser.add_argument("--full-final-dir", default="pdppo_full_final_partition_24seed_20260718")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def require_seed_set(df: pd.DataFrame, name: str, *, column: str = "seed") -> None:
    seeds = sorted({int(value) for value in df[column].tolist()})
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{name}: expected seeds {EXPECTED_SEEDS}, found {seeds}")


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("bootstrap input contains no finite values")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    chunk = 10_000
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        idx = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summary(values: pd.Series, samples: int, seed: int) -> dict[str, float | int | list[float]]:
    data = values.to_numpy(dtype=float)
    low, high = bootstrap_ci(data, samples, seed)
    return {
        "wins": int(np.sum(data > 0.0)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "minimum": float(np.min(data)),
        "ci95": [low, high],
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def fmt_ci(stats: dict[str, object], digits: int = 4) -> str:
    low, high = stats["ci95"]
    return f"{fmt(float(stats['mean']), digits)} [{fmt(float(low), digits)}, {fmt(float(high), digits)}]"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    aggregate = args.framework_root / "reports" / "aggregate"
    csv_paths = {
        "main": aggregate / args.main_dir / "validation_frozen_seed_metrics.csv",
        "behavior": aggregate / args.mechanism_dir / "clean_policy_behavior_seed_metrics.csv",
        "duty": aggregate / args.mechanism_dir / "clean_policy_subtype_sensor_duty_summary.csv",
        "references": aggregate / args.reference_dir / "framework_baseline_seed_metrics.csv",
        "reward": aggregate / args.reward_dir / "matched_reward_seed_metrics.csv",
        "dqn": aggregate / args.dqn_dir / "matched_dqn_seed_metrics.csv",
        "forecaster": aggregate / args.forecaster_dir / "secondary_forecaster_paired_metrics.csv",
        "full_final": aggregate / args.full_final_dir / "validation_frozen_seed_metrics.csv",
    }
    audit_paths = {
        "main_audit": aggregate / args.main_dir / "validation_frozen_claim_summary.json",
        "mechanism_audit": aggregate / args.mechanism_dir / "clean_policy_mechanism_summary.json",
        "reference_audit": aggregate / args.reference_dir / "framework_baseline_summary.json",
        "reward_audit": aggregate / args.reward_dir / "matched_reward_protocol_audit.json",
        "dqn_audit": aggregate / args.dqn_dir / "matched_dqn_protocol_audit.json",
        "forecaster_audit": aggregate / args.forecaster_dir / "secondary_forecaster_protocol.json",
        "full_final_audit": aggregate / args.full_final_dir / "validation_frozen_claim_summary.json",
    }
    frames = {name: read_csv(path) for name, path in csv_paths.items()}
    audits = {}
    for name, path in audit_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        audits[name] = json.loads(path.read_text(encoding="utf-8"))
    for name in ("main", "behavior", "references", "reward", "dqn", "forecaster", "full_final"):
        require_seed_set(frames[name], name)
    if len(frames["main"]) != 24 or len(frames["behavior"]) != 24:
        raise ValueError("main and behavior inputs must contain exactly one row per seed")
    if set(frames["reward"]["mode"]) != {"forecast", "aoi", "uncertainty"}:
        raise ValueError("reward input must contain forecast, aoi, and uncertainty modes")
    if frames["reward"].groupby("mode")["seed"].nunique().to_dict() != {
        "aoi": 24,
        "forecast": 24,
        "uncertainty": 24,
    }:
        raise ValueError("each reward mode must contain all 24 seeds")
    if frames["references"].groupby("policy")["seed"].nunique().min() != 24:
        raise ValueError("each strong reference must contain all 24 seeds")
    if int(audits["main_audit"].get("seed_count", -1)) != 24:
        raise ValueError("main evidence audit does not certify 24 seeds")
    if int(audits["full_final_audit"].get("seed_count", -1)) != 24:
        raise ValueError("full-final evidence audit does not certify 24 seeds")
    full_final = frames["full_final"]
    if set(pd.to_numeric(full_final["evaluation_steps"], errors="coerce")) != {5242}:
        raise ValueError("full-final evidence does not contain 5242 scoreable epochs per seed")
    if set(full_final["evaluation_scope_mode"].astype(str)) != {"full_scoreable_final_partition"}:
        raise ValueError("full-final evidence scope is not the complete scoreable final partition")
    if set(full_final["evaluation_oracle_device"].astype(str)) != {"cpu"}:
        raise ValueError("full-final evidence must use the common CPU evaluator")
    mechanism_audit = audits["mechanism_audit"]
    if int(mechanism_audit.get("behavior_gate_passes", -1)) != 24:
        raise ValueError("mechanism audit does not certify all behavior gates")
    if not bool(mechanism_audit.get("offline_subtype_labels_used_only_for_grouping", False)):
        raise ValueError("mechanism audit does not enforce offline-only subtype grouping")
    if int(audits["reference_audit"].get("complete_rows", -1)) != 72:
        raise ValueError("strong-reference audit is incomplete")
    for name in ("reward_audit", "dqn_audit", "forecaster_audit"):
        audit = audits[name]
        if str(audit.get("status")) != "passed":
            raise ValueError(f"{name} status is not passed")
        if sorted(int(value) for value in audit.get("seeds", ())) != EXPECTED_SEEDS:
            raise ValueError(f"{name} does not certify the frozen 24-seed set")
    return frames, {**csv_paths, **audit_paths}


def build_statistics(
    frames: dict[str, pd.DataFrame], samples: int, seed: int
) -> dict[str, object]:
    main = frames["main"].sort_values("seed")
    refs = frames["references"]
    reward = frames["reward"]
    dqn = frames["dqn"].sort_values("seed")
    forecaster = frames["forecaster"].sort_values("seed")
    full_final = frames["full_final"].sort_values("seed")

    stats: dict[str, object] = {"seed_count": 24, "seed_range": [117, 140]}
    stats["primary_means"] = {
        "pdppo_step_loss": float(main["custom_ppo_step_loss"].mean()),
        "fixed_step_loss": float(main["validation_selected_static_step_loss"].mean()),
        "step_relative_reduction": float(
            1.0
            - main["custom_ppo_step_loss"].mean()
            / main["validation_selected_static_step_loss"].mean()
        ),
        "pdppo_macro_score": float(
            main["custom_ppo_oracle_loss_macro_subtype_event_validationnorm"].mean()
        ),
        "fixed_macro_score": float(
            main["validation_selected_static_oracle_loss_macro_subtype_event_validationnorm"].mean()
        ),
        "macro_relative_reduction": float(
            1.0
            - main["custom_ppo_oracle_loss_macro_subtype_event_validationnorm"].mean()
            / main["validation_selected_static_oracle_loss_macro_subtype_event_validationnorm"].mean()
        ),
    }
    comparisons: dict[str, object] = {}
    for key, column in {
        "static_macro": "macro_margin_pdppo_vs_validation_selected_static",
        "static_step": "step_margin_pdppo_vs_validation_selected_static",
        "aoi_macro": "macro_margin_pdppo_vs_aoi",
        "aoi_step": "step_margin_pdppo_vs_aoi",
        "round_robin_macro": "macro_margin_pdppo_vs_round_robin",
        "round_robin_step": "step_margin_pdppo_vs_round_robin",
        "random_macro": "macro_margin_pdppo_vs_random",
        "random_step": "step_margin_pdppo_vs_random",
        "best_rule_macro": "macro_margin_pdppo_vs_posthoc_best_rule_dynamic",
    }.items():
        comparisons[key] = summary(main[column], samples, seed + len(comparisons))

    policy_map = {
        "forecast_greedy": "forecast_greedy_one_step",
        "context_alert": "context_alert_bandit_t0p5",
        "event_label": "event_label_reference_l16",
    }
    for short, policy in policy_map.items():
        rows = refs[refs["policy"] == policy].sort_values("seed")
        comparisons[f"{short}_step"] = summary(rows["margin_loss_vs_custom_ppo"], samples, seed + 20)
        comparisons[f"{short}_macro"] = summary(
            rows["margin_oracle_loss_macro_subtype_event_staticnorm_vs_custom_ppo"],
            samples,
            seed + 21,
        )
    comparisons["dqn_step"] = summary(dqn["step_margin_dqn_minus_ppo"], samples, seed + 30)
    comparisons["dqn_macro"] = summary(dqn["macro_margin_dqn_minus_ppo"], samples, seed + 31)
    stats["comparisons"] = comparisons
    full_step = summary(
        full_final["step_margin_pdppo_vs_validation_selected_static"],
        samples,
        seed + 70,
    )
    full_macro = summary(
        full_final["macro_margin_pdppo_vs_validation_selected_static"],
        samples,
        seed + 71,
    )
    stats["full_final_partition"] = {
        "scoreable_epochs_per_seed": int(full_final["evaluation_steps"].iloc[0]),
        "excluded_tail_epochs": 8,
        "pdppo_step_loss": float(full_final["custom_ppo_step_loss"].mean()),
        "fixed_step_loss": float(full_final["validation_selected_static_step_loss"].mean()),
        "pdppo_macro_score": float(
            full_final["custom_ppo_oracle_loss_macro_subtype_event_validationnorm"].mean()
        ),
        "fixed_macro_score": float(
            full_final["validation_selected_static_oracle_loss_macro_subtype_event_validationnorm"].mean()
        ),
        "step_margin": full_step,
        "macro_margin": full_macro,
    }
    stats["dqn_control"] = {
        "wins_vs_static": int(np.sum(dqn["dqn_macro_margin_vs_static"] > 0.0)),
        "behavior_valid": int(np.sum(dqn["dqn_behavior_gate_pass"].astype(bool))),
        "warmup_abort_total": int(np.nansum(dqn["dqn_warmup_abort_count"])),
    }

    regime_stats: dict[str, object] = {}
    for idx, subtype in enumerate(("particle", "flux", "thermal")):
        denominator = main[f"validation_normalizer_{subtype}"]
        pdppo = main[f"custom_ppo_oracle_loss_subtype_{subtype}"] / denominator
        fixed = main[f"validation_selected_static_oracle_loss_subtype_{subtype}"] / denominator
        regime_stats[subtype] = {
            "pdppo_mean": float(pdppo.mean()),
            "fixed_mean": float(fixed.mean()),
            "margin": summary(fixed - pdppo, samples, seed + 10 + idx),
        }
    regime_stats["macro"] = {
        "pdppo_mean": float(
            main["custom_ppo_oracle_loss_macro_subtype_event_validationnorm"].mean()
        ),
        "fixed_mean": float(
            main["validation_selected_static_oracle_loss_macro_subtype_event_validationnorm"].mean()
        ),
        "margin": comparisons["static_macro"],
    }
    stats["regime_decomposition"] = regime_stats

    reward_stats: dict[str, object] = {}
    forecast = reward[reward["mode"] == "forecast"].set_index("seed").sort_index()
    for mode in ("aoi", "uncertainty"):
        control = reward[reward["mode"] == mode].set_index("seed").sort_index()
        reward_stats[mode] = {
            "step": summary(
                control["oracle_loss_mean"] - forecast["oracle_loss_mean"],
                samples,
                seed + 38 + len(reward_stats),
            ),
            "macro": summary(
                control["macro_score"] - forecast["macro_score"],
                samples,
                seed + 40 + len(reward_stats),
            ),
            "forecast_wins_vs_static": int(np.sum(forecast["margin_vs_static"] > 0)),
            "control_wins_vs_static": int(np.sum(control["margin_vs_static"] > 0)),
            "control_behavior_valid": int(
                np.sum(
                    (control["warmup_abort_count"] == 0)
                    & (control["always_on_sensor_count"] == 1)
                    & (control["always_off_sensor_count"] == 1)
                    & control["mid_duty_sensor_count"].between(3, 4)
                )
            ),
        }
    stats["reward_controls"] = reward_stats

    forecaster_stats = {
        key: summary(forecaster[column], samples, seed + 50 + idx)
        for idx, (key, column) in enumerate(
            {
                "step_original_static": "step_margin_vs_original_static",
                "step_secondary_static": "step_margin_vs_secondary_static",
                "macro_original_static": "macro_margin_vs_original_static",
                "macro_secondary_static": "macro_margin_vs_secondary_static",
            }.items()
        )
    }
    stats["secondary_forecaster"] = forecaster_stats

    post_pilot_seeds = EXPECTED_SEEDS[2:]
    main_post = main[main["seed"].isin(post_pilot_seeds)]
    dqn_post = dqn[dqn["seed"].isin(post_pilot_seeds)]
    forecaster_post = forecaster[forecaster["seed"].isin(post_pilot_seeds)]
    reward_post = reward[reward["seed"].isin(post_pilot_seeds)]
    forecast_post = reward_post[reward_post["mode"] == "forecast"].set_index("seed").sort_index()
    stats["post_pilot_replication"] = {
        "seed_count": len(post_pilot_seeds),
        "seed_range": [post_pilot_seeds[0], post_pilot_seeds[-1]],
        "static_step": summary(
            main_post["step_margin_pdppo_vs_validation_selected_static"], samples, seed + 60
        ),
        "static_macro": summary(
            main_post["macro_margin_pdppo_vs_validation_selected_static"], samples, seed + 61
        ),
        "dqn_step": summary(dqn_post["step_margin_dqn_minus_ppo"], samples, seed + 62),
        "dqn_macro": summary(dqn_post["macro_margin_dqn_minus_ppo"], samples, seed + 63),
        "reward_controls": {
            mode: {
                "step": summary(
                    reward_post[reward_post["mode"] == mode]
                    .set_index("seed")
                    .sort_index()["oracle_loss_mean"]
                    - forecast_post["oracle_loss_mean"],
                    samples,
                    seed + 62 + idx,
                ),
                "macro": summary(
                    reward_post[reward_post["mode"] == mode]
                    .set_index("seed")
                    .sort_index()["macro_score"]
                    - forecast_post["macro_score"],
                    samples,
                    seed + 64 + idx,
                ),
            }
            for idx, mode in enumerate(("aoi", "uncertainty"))
        },
        "secondary_forecaster_macro": summary(
            forecaster_post["macro_margin_vs_secondary_static"], samples, seed + 66
        ),
    }
    return stats


def write_main_table(path: Path, stats: dict[str, object]) -> None:
    comp = stats["comparisons"]
    rows = [
        ("Validation-selected fixed schedule", "static"),
        ("AoI-priority schedule", "aoi"),
        ("Round robin", "round_robin"),
        ("Random feasible schedule", "random"),
        ("One-step forecast greedy", "forecast_greedy"),
        ("Warning-score rule", "context_alert"),
        ("Exact-label replay", "event_label"),
    ]
    body = []
    for label, key in rows:
        step = comp[f"{key}_step"]
        macro = comp[f"{key}_macro"]
        body.append(
            f"    {label} & {step['wins']}/24; {fmt_ci(step)} & "
            f"{macro['wins']}/24; {fmt_ci(macro)} \\\\" 
        )
    text = "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"  \centering",
            r"  \caption{Held-out schedule comparisons for PD-PPO. Positive margins mean that the reference has higher loss than PD-PPO.}",
            r"  \label{tab:clean_main_comparisons}",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{2.4pt}",
            r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.345\textwidth}>{\raggedright\arraybackslash}p{0.345\textwidth}@{}}",
            r"    \toprule",
            r"    Reference & Mean loss: wins; margin [95\% CI] & Macro: wins; margin [95\% CI] \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.98\textwidth}",
            r"  \vspace{0.4ex}\footnotesize The fixed, AoI, round-robin, and random schedules are deployable references under the common feasibility interface. Forecast greedy can inspect one-step future loss; exact-label replay receives held-out simulator labels; the warning-score rule receives supplied synthetic warning scores.",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_control_table(path: Path, stats: dict[str, object]) -> None:
    reward = stats["reward_controls"]
    comp = stats["comparisons"]
    rows = []
    for label, key in (("AoI reward", "aoi"), ("Diagonal-uncertainty reward", "uncertainty")):
        item = reward[key]
        rows.append(
            f"    {label} & {item['step']['wins']}/24; {fmt_ci(item['step'])} & "
            f"{item['macro']['wins']}/24; {fmt_ci(item['macro'])} & "
            f"{item['control_wins_vs_static']}/24 & {item['control_behavior_valid']}/24 \\\\"
        )
    dqn_step = comp["dqn_step"]
    dqn_macro = comp["dqn_macro"]
    dqn_control = stats["dqn_control"]
    rows.append(
        f"    Matched Double-DQN & {dqn_step['wins']}/24; {fmt_ci(dqn_step)} & "
        f"{dqn_macro['wins']}/24; {fmt_ci(dqn_macro)} & "
        f"{dqn_control['wins_vs_static']}/24 & {dqn_control['behavior_valid']}/24 \\\\"
    )
    text = "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"  \centering",
            r"  \caption{Matched objective and learned-policy controls. Positive differences indicate lower loss for forecast-reward PD-PPO.}",
            r"  \label{tab:clean_learning_controls}",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{2.2pt}",
            r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\centering\arraybackslash}p{0.10\textwidth}>{\centering\arraybackslash}p{0.10\textwidth}@{}}",
            r"    \toprule",
            r"    Control & Mean loss: wins; difference [95\% CI] & Macro: wins; difference [95\% CI] & Macro vs. fixed & Valid trace \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.98\textwidth}",
            r"  \vspace{0.4ex}\footnotesize Reward controls use the same PPO architecture, observation, training guide, auxiliary context task, and feasible masks; each difference is control loss minus forecast-reward loss. Double-DQN uses the same state, forecast reward, and masks, and its differences are DQN loss minus PD-PPO loss. The last column applies the reward-control execution check or the full DQN action-trace gate. All policies use the common frozen evaluator.",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_regime_table(path: Path, stats: dict[str, object]) -> None:
    regimes = stats["regime_decomposition"]
    rows = []
    for label, key in (
        ("Particle regime", "particle"),
        ("Flux regime", "flux"),
        ("Thermal regime", "thermal"),
        ("Equal-weight macro", "macro"),
    ):
        item = regimes[key]
        margin = item["margin"]
        rows.append(
            f"    {label} & {item['pdppo_mean']:.4f} & {item['fixed_mean']:.4f} & "
            f"{margin['wins']}/24 & {fmt_ci(margin)} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Validation-normalized loss by forecast regime for PD-PPO and the validation-selected fixed schedule. Positive margins favor PD-PPO.}",
            r"  \label{tab:clean_regime_decomposition}",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{3.2pt}",
            r"  \begin{tabular}{@{}lcccc@{}}",
            r"    \toprule",
            r"    Regime & PD-PPO & Fixed & Positive seeds & Margin [95\% CI] \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_forecaster_table(path: Path, stats: dict[str, object]) -> None:
    items = stats["secondary_forecaster"]
    rows = []
    for label, key in (
        ("Original validation-selected fixed schedule", "macro_original_static"),
        ("Ridge-validation-selected fixed schedule", "macro_secondary_static"),
    ):
        item = items[key]
        rows.append(f"    {label} & {item['wins']}/24 & {fmt_ci(item)} \\\\ ")
    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Frozen-trajectory sensitivity under an independently fitted ridge forecaster. Positive macro margins favor PD-PPO.}",
            r"  \label{tab:clean_secondary_forecaster}",
            r"  \footnotesize",
            r"  \begin{tabular}{@{}lcc@{}}",
            r"    \toprule",
            r"    Fixed-schedule reference & PD-PPO wins & Macro margin [95\% CI] \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_full_partition_table(path: Path, stats: dict[str, object]) -> None:
    primary = stats["primary_means"]
    primary_step = stats["comparisons"]["static_step"]
    primary_macro = stats["comparisons"]["static_macro"]
    full = stats["full_final_partition"]
    rows = [
        (
            "Prespecified subtype-balanced windows",
            4096,
            primary["pdppo_step_loss"],
            primary["fixed_step_loss"],
            primary_step,
            primary["pdppo_macro_score"],
            primary["fixed_macro_score"],
            primary_macro,
        ),
        (
            "Complete scoreable final partition",
            full["scoreable_epochs_per_seed"],
            full["pdppo_step_loss"],
            full["fixed_step_loss"],
            full["step_margin"],
            full["pdppo_macro_score"],
            full["fixed_macro_score"],
            full["macro_margin"],
        ),
    ]
    body = []
    for label, epochs, pd_step, fixed_step, step_margin, pd_macro, fixed_macro, macro_margin in rows:
        body.append(
            f"    {label} & {epochs:,} & {pd_step:.4f} / {fixed_step:.4f} & "
            f"{step_margin['wins']}/24; {fmt_ci(step_margin)} & "
            f"{pd_macro:.4f} / {fixed_macro:.4f} & "
            f"{macro_margin['wins']}/24; {fmt_ci(macro_margin)} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"  \centering",
            r"  \caption{Sensitivity to final-partition coverage. Positive margins mean that the validation-selected fixed schedule has higher loss than PD-PPO.}",
            r"  \label{tab:clean_full_partition_sensitivity}",
            r"  \scriptsize",
            r"  \renewcommand{\arraystretch}{1.05}",
            r"  \setlength{\tabcolsep}{2.2pt}",
            r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.16\textwidth}>{\centering\arraybackslash}p{0.06\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}@{}}",
            r"    \toprule",
            r"    Evaluation scope & Epochs per seed & Mean loss: PD-PPO / fixed & Mean-loss wins; margin [95\% CI] & Macro: PD-PPO / fixed & Macro wins; margin [95\% CI] \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.98\textwidth}",
            r"  \vspace{0.4ex}\footnotesize The continuous replay uses the same frozen checkpoints, validation-selected masks, and validation-fitted subtype normalizers as the primary evaluation. The last eight rows of each final partition are excluded because an eight-step future target is unavailable; no policy or reference is reselected.",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def setup_plot_style(paper_dir: Path) -> tuple[dict[str, str], object]:
    sys.path.insert(0, str(paper_dir / "figures"))
    from paper_plot_style import PALETTE, apply_paper_style, save_pdf_png

    apply_paper_style(base_size=8.2)
    return PALETTE, save_pdf_png


def build_main_figure(
    path: Path,
    frames: dict[str, pd.DataFrame],
    palette: dict[str, str],
    save_pdf_png: object,
) -> None:
    main = frames["main"].sort_values("macro_margin_pdppo_vs_validation_selected_static")
    refs = frames["references"]
    x = np.arange(24)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    ax1.axhline(0, color=palette["dark"], linewidth=0.7)
    ax1.plot(x, main["step_margin_pdppo_vs_validation_selected_static"], "o-", color=palette["blue"], markersize=3, linewidth=1.0, label="Mean forecast loss")
    ax1.plot(x, main["macro_margin_pdppo_vs_validation_selected_static"], "s-", color=palette["teal"], markersize=2.8, linewidth=1.0, label="Regime macro score")
    ax1.set_xticks(x[::3])
    ax1.set_xticklabels(main["seed"].iloc[::3].astype(int), rotation=35, ha="right")
    ax1.set_xlabel("Seed (sorted by macro margin)")
    ax1.set_ylabel("Reference loss minus PD-PPO loss")
    ax1.set_title("(a) PD-PPO versus validation-selected fixed schedule", loc="left")
    ax1.legend(loc="upper left")

    datasets = [
        main["macro_margin_pdppo_vs_validation_selected_static"].to_numpy(),
        main["macro_margin_pdppo_vs_aoi"].to_numpy(),
        main["macro_margin_pdppo_vs_round_robin"].to_numpy(),
        main["macro_margin_pdppo_vs_random"].to_numpy(),
    ]
    labels = ["Fixed", "AoI", "Round\nrobin", "Random"]
    greedy = refs[refs["policy"] == "forecast_greedy_one_step"].sort_values("seed")
    datasets.append(greedy["margin_oracle_loss_macro_subtype_event_staticnorm_vs_custom_ppo"].to_numpy())
    labels.append("One-step\ngreedy")
    box = ax2.boxplot(datasets, patch_artist=True, widths=0.58, showfliers=False)
    colors = [palette["teal"], palette["sky"], palette["blue"], palette["purple"], palette["vermillion"]]
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("white")
    rng = np.random.default_rng(7)
    for idx, values in enumerate(datasets, start=1):
        ax2.scatter(np.full(24, idx) + rng.normal(0, 0.035, 24), values, s=7, color=palette["dark"], alpha=0.35, linewidth=0)
    ax2.axhline(0, color=palette["dark"], linewidth=0.7)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Macro margin")
    ax2.set_title("(b) Matched reference families", loc="left")
    save_pdf_png(fig, path)
    plt.close(fig)


def build_mechanism_figure(
    path: Path,
    frames: dict[str, pd.DataFrame],
    palette: dict[str, str],
    save_pdf_png: object,
) -> None:
    duty = frames["duty"]
    behavior = frames["behavior"].sort_values("seed")
    heat = np.zeros((len(SUBTYPE_ORDER), len(SENSOR_ORDER)), dtype=float)
    for row_idx, subtype in enumerate(SUBTYPE_ORDER):
        for col_idx, sensor in enumerate(SENSOR_ORDER):
            row = duty[(duty["subtype"] == subtype) & (duty["sensor"] == sensor)]
            if len(row) != 1:
                raise ValueError(f"missing duty row for {subtype}/{sensor}")
            heat[row_idx, col_idx] = float(row.iloc[0]["selection_fraction_mean"])

    fig = plt.figure(figsize=(7.15, 2.75), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.75, 1.0])
    ax1 = fig.add_subplot(grid[0, 0])
    image = ax1.imshow(heat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(len(SENSOR_LABELS)), SENSOR_LABELS, rotation=28, ha="right")
    ax1.set_yticks(range(len(SUBTYPE_LABELS)), SUBTYPE_LABELS)
    ax1.set_title("(a) Specialist duty by regime", loc="left")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax1.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="white" if heat[i, j] > 0.55 else palette["dark"], fontsize=7.4)
    fig.colorbar(image, ax=ax1, fraction=0.046, pad=0.03, label="Selection fraction")

    ax2 = fig.add_subplot(grid[0, 1])
    counts = Counter(int(value) for value in frames["main"]["validation_selected_static_action_idx"])
    static_labels = ["Surface IR", "Laser", "FC4 flux"]
    static_counts = [counts[3], counts[4], counts[5]]
    ax2.barh(np.arange(3), static_counts, color=[palette["sky"], palette["orange"], palette["teal"]])
    ax2.set_yticks(np.arange(3), static_labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 24)
    ax2.set_xlabel("Seeds selected")
    ax2.set_title("(b) Fixed specialist", loc="left")
    for idx, value in enumerate(static_counts):
        ax2.text(value + 0.3, idx, str(value), va="center", fontsize=8)

    ax3 = fig.add_subplot(grid[0, 2])
    scatter = ax3.scatter(
        behavior["mask_entropy_bits"],
        behavior["subtype_mask_mi_bits"],
        c=1000 * behavior["switches_per_step"],
        cmap="viridis",
        s=24,
        edgecolor="white",
        linewidth=0.35,
    )
    ax3.set_xlabel("Mask entropy (bits)")
    ax3.set_ylabel("Regime-mask MI (bits)")
    ax3.set_title("(c) State-dependent traces", loc="left")
    fig.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04, label="Switches per 1000 steps")
    save_pdf_png(fig, path)
    plt.close(fig)


def build_control_figure(
    path: Path,
    frames: dict[str, pd.DataFrame],
    palette: dict[str, str],
    save_pdf_png: object,
) -> None:
    reward = frames["reward"]
    forecast = reward[reward["mode"] == "forecast"].set_index("seed").sort_index()
    aoi = reward[reward["mode"] == "aoi"].set_index("seed").sort_index()
    uncertainty = reward[reward["mode"] == "uncertainty"].set_index("seed").sort_index()
    dqn = frames["dqn"].sort_values("seed")
    forecaster = frames["forecaster"].sort_values("seed")
    reward_data = [
        (aoi["macro_score"] - forecast["macro_score"]).to_numpy(),
        (uncertainty["macro_score"] - forecast["macro_score"]).to_numpy(),
    ]
    learner_data = [dqn["macro_margin_dqn_minus_ppo"].to_numpy()]
    ridge_data = [
        forecaster["macro_margin_vs_original_static"].to_numpy(),
        forecaster["macro_margin_vs_secondary_static"].to_numpy(),
    ]
    reward_labels = ["AoI reward", "Uncertainty\nreward"]
    learner_labels = ["Double-DQN"]
    ridge_labels = ["Original\nfixed", "Ridge-selected\nfixed"]
    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.7),
        gridspec_kw={"width_ratios": [1.15, 0.75, 1.1]},
        constrained_layout=True,
    )
    reward_box = ax1.boxplot(reward_data, patch_artist=True, widths=0.55, showfliers=False)
    reward_colors = [palette["sky"], palette["purple"]]
    for patch, color in zip(reward_box["boxes"], reward_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("white")
    rng = np.random.default_rng(17)
    for idx, values in enumerate(reward_data, start=1):
        ax1.scatter(np.full(24, idx) + rng.normal(0, 0.035, 24), values, s=8, color=palette["dark"], alpha=0.4, linewidth=0)
    ax1.axhline(0, color=palette["dark"], linewidth=0.8)
    ax1.set_xticks(range(1, len(reward_labels) + 1), reward_labels)
    ax1.set_ylabel("Control macro loss minus PD-PPO macro loss")
    ax1.set_title("(a) Reward controls", loc="left")

    learner_box = ax2.boxplot(learner_data, patch_artist=True, widths=0.55, showfliers=False)
    learner_box["boxes"][0].set_facecolor(palette["orange"])
    learner_box["boxes"][0].set_alpha(0.72)
    learner_box["boxes"][0].set_edgecolor("white")
    ax2.scatter(
        np.full(24, 1) + rng.normal(0, 0.035, 24),
        learner_data[0],
        s=8,
        color=palette["dark"],
        alpha=0.4,
        linewidth=0,
    )
    ax2.axhline(0, color=palette["dark"], linewidth=0.8)
    ax2.set_xticks([1], learner_labels)
    ax2.set_ylabel("DQN macro loss minus PD-PPO macro loss")
    ax2.set_title("(b) Learner control", loc="left")

    ridge_box = ax3.boxplot(ridge_data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(ridge_box["boxes"], [palette["blue"], palette["teal"]], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("white")
    for idx, values in enumerate(ridge_data, start=1):
        ax3.scatter(np.full(24, idx) + rng.normal(0, 0.035, 24), values, s=8, color=palette["dark"], alpha=0.4, linewidth=0)
    ax3.axhline(0, color=palette["dark"], linewidth=0.8)
    ax3.set_xticks(range(1, len(ridge_labels) + 1), ridge_labels)
    ax3.set_ylabel("Fixed-schedule macro loss minus PD-PPO macro loss")
    ax3.set_title("(c) Ridge forecaster", loc="left")
    save_pdf_png(fig, path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.framework_root = args.framework_root.resolve()
    frames, input_paths = load_inputs(args)
    stats = build_statistics(frames, args.bootstrap_samples, args.bootstrap_seed)
    paper_dir = args.framework_root / "paper"
    table_dir = paper_dir / "tables"
    figure_dir = paper_dir / "figures"
    report_dir = args.framework_root / "reports" / "aggregate" / "pdppo_clean_paper_assets_20260718"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    write_main_table(table_dir / "clean_main_comparisons.tex", stats)
    write_regime_table(table_dir / "clean_regime_decomposition.tex", stats)
    write_control_table(table_dir / "clean_learning_controls.tex", stats)
    write_forecaster_table(table_dir / "clean_secondary_forecaster.tex", stats)
    write_full_partition_table(table_dir / "clean_full_partition_sensitivity.tex", stats)
    palette, save_pdf_png = setup_plot_style(paper_dir)
    build_main_figure(figure_dir / "figure_clean_main_evidence", frames, palette, save_pdf_png)
    build_mechanism_figure(figure_dir / "figure_clean_mechanism", frames, palette, save_pdf_png)
    build_control_figure(figure_dir / "figure_clean_controls", frames, palette, save_pdf_png)

    output_paths = [
        table_dir / "clean_main_comparisons.tex",
        table_dir / "clean_regime_decomposition.tex",
        table_dir / "clean_learning_controls.tex",
        table_dir / "clean_secondary_forecaster.tex",
        table_dir / "clean_full_partition_sensitivity.tex",
        figure_dir / "figure_clean_main_evidence.pdf",
        figure_dir / "figure_clean_main_evidence.png",
        figure_dir / "figure_clean_mechanism.pdf",
        figure_dir / "figure_clean_mechanism.png",
        figure_dir / "figure_clean_controls.pdf",
        figure_dir / "figure_clean_controls.png",
    ]
    manifest = {
        "inputs": {
            name: {"path": str(path.relative_to(args.framework_root)), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "statistics": stats,
        "outputs": {
            str(path.relative_to(args.framework_root)): {"sha256": sha256(path)}
            for path in output_paths
        },
    }
    (report_dir / "paper_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"paper_assets_complete report={report_dir}")


if __name__ == "__main__":
    main()
