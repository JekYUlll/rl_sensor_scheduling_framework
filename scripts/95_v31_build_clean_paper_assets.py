#!/usr/bin/env python3
"""Build manuscript assets from the frozen clean PD-PPO evidence package.

The script requires every evidence block to contain the same 24 frozen runs.
Main-text statistics and plots use the 22 post-selection evaluation seeds
(119--140); the two pilot/model-selection seeds remain in a separately labelled
24-seed descriptive aggregate for the supplementary material.
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
PRIMARY_SEEDS = list(range(119, 141))
SENSOR_ORDER = [
    "shielded_thermo_hygro",
    "surface_temp_ir",
    "laser_disdrometer",
    "fc4_flux",
]
SENSOR_LABELS = [
    "Shielded temperature–\nhumidity channel",
    "Infrared snow-surface-\ntemperature channel",
    "Laser disdrometer\nchannel",
    "FC4 snow mass-flux\nchannel",
]
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
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional manifest path; defaults to the frozen aggregate report directory.",
    )
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
        "duty": aggregate / args.mechanism_dir / "clean_policy_subtype_sensor_duty.csv",
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


def select_analysis_seeds(
    frames: dict[str, pd.DataFrame], seeds: list[int]
) -> dict[str, pd.DataFrame]:
    selected: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        if "seed" not in frame.columns:
            selected[name] = frame.copy()
            continue
        subset = frame[frame["seed"].astype(int).isin(seeds)].copy()
        found = sorted({int(value) for value in subset["seed"].tolist()})
        if found != sorted(seeds):
            raise ValueError(f"{name}: expected analysis seeds {seeds}, found {found}")
        selected[name] = subset
    return selected


def build_statistics(
    frames: dict[str, pd.DataFrame], samples: int, seed: int
) -> dict[str, object]:
    main = frames["main"].sort_values("seed")
    refs = frames["references"]
    reward = frames["reward"]
    dqn = frames["dqn"].sort_values("seed")
    forecaster = frames["forecaster"].sort_values("seed")
    full_final = frames["full_final"].sort_values("seed")
    analysis_seeds = sorted({int(value) for value in main["seed"].tolist()})
    seed_count = len(analysis_seeds)

    stats: dict[str, object] = {
        "seed_count": seed_count,
        "seeds": analysis_seeds,
        "seed_range": [analysis_seeds[0], analysis_seeds[-1]],
        "one_sided_all_positive_sign_test_p": float(0.5**seed_count),
    }
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
        "behavior_valid": int(
            np.sum(
                (dqn["dqn_warmup_abort_count"] == 0)
                & (dqn["dqn_always_on_sensor_count"] == 1)
                & (dqn["dqn_always_off_sensor_count"] == 1)
                & dqn["dqn_mid_duty_sensor_count"].between(3, 4)
            )
        ),
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
    seed_count = int(stats["seed_count"])
    rows = [
        ("Fixed-schedule baseline", "static"),
        ("AoI-priority baseline", "aoi"),
        ("Round-robin baseline", "round_robin"),
        ("Random-feasible baseline", "random"),
        ("Privileged one-step look-ahead reference using next-step test targets", "forecast_greedy"),
        ("Warning-rule baseline", "context_alert"),
        ("Privileged true-label baseline", "event_label"),
    ]
    body = []
    for label, key in rows:
        step = comp[f"{key}_step"]
        macro = comp[f"{key}_macro"]
        body.append(
            f"    {label} & {step['wins']} of {seed_count}; {fmt_ci(step)} & "
            f"{macro['wins']} of {seed_count}; {fmt_ci(macro)} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"  \centering",
            rf"  \caption{{Baseline comparisons with the complete PD-PPO training configuration on the event-balanced evaluation windows over {seed_count} post-selection evaluation seeds. A positive paired difference means that the baseline has higher loss than PD-PPO.}}",
            r"  \label{tab:clean_main_comparisons}",
            r"  \footnotesize",
            r"  \setlength{\tabcolsep}{2.4pt}",
            r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.36\textwidth}>{\raggedright\arraybackslash}p{0.36\textwidth}@{}}",
            r"    \toprule",
            r"    Baseline & \shortstack[l]{Mean forecast loss\\Seeds with lower loss;\\mean paired difference\\{[95\% bootstrap CI]}} & \shortstack[l]{Macro-averaged loss\\Seeds with lower loss;\\mean paired difference\\{[95\% bootstrap CI]}} \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.98\textwidth}",
            r"  \vspace{0.4ex}\footnotesize Each interval is a 95\% bootstrap confidence interval for the paired difference. The validation-selected fixed-schedule baseline is the structural primary comparator; the warning-rule baseline is the practical adaptive comparator. The one-step look-ahead reference uses next-step test targets, and the true-label baseline uses the exact sequence $c_t,\ldots,c_{t+16}$; both receive information unavailable during deployment, and neither is an upper bound. The warning-rule baseline uses the same simulator-generated warning signals available to PD-PPO and a mapping selected on the calibration/validation partition.",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_control_table(path: Path, stats: dict[str, object]) -> None:
    reward = stats["reward_controls"]
    comp = stats["comparisons"]
    seed_count = int(stats["seed_count"])
    rows = []
    for label, key in (("AoI-objective PPO configuration", "aoi"), ("Uncertainty-objective PPO configuration", "uncertainty")):
        item = reward[key]
        rows.append(
            f"    {label} & {item['step']['wins']} of {seed_count}; {fmt_ci(item['step'])} & "
            f"{item['macro']['wins']} of {seed_count}; {fmt_ci(item['macro'])} & "
            f"{item['control_wins_vs_static']} of {seed_count} & "
            f"{item['control_behavior_valid']} of {seed_count} \\\\"
        )
    dqn_step = comp["dqn_step"]
    dqn_macro = comp["dqn_macro"]
    dqn_control = stats["dqn_control"]
    rows.append(
        f"    Double DQN training configuration & {dqn_step['wins']} of {seed_count}; {fmt_ci(dqn_step)} & "
        f"{dqn_macro['wins']} of {seed_count}; {fmt_ci(dqn_macro)} & "
        f"{dqn_control['wins_vs_static']} of {seed_count} & "
        f"{dqn_control['behavior_valid']} of {seed_count} \\\\"
    )
    text = "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"  \centering",
            r"  \caption{Scalar-objective PPO and Double DQN training-configuration comparisons. A positive paired difference indicates lower loss for PD-PPO.}",
            r"  \label{tab:clean_learning_controls}",
            r"  \scriptsize",
            r"  \setlength{\tabcolsep}{2.1pt}",
            r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.14\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.14\textwidth}@{}}",
            r"    \toprule",
            r"    Comparison & Mean forecast loss\newline Seeds with lower loss;\newline mean paired difference\newline [95\% bootstrap CI] & Macro-averaged loss\newline Seeds with lower loss;\newline mean paired difference\newline [95\% bootstrap CI] & Lower macro-averaged\newline loss than fixed baseline & Channel-use acceptance\newline criterion \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.98\textwidth}",
            r"  \vspace{0.4ex}\footnotesize Each interval is a 95\% bootstrap confidence interval. The objective rows compare matched PPO training configurations with different scalar objectives. Each objective changes the PPO surrogate, value target, advantage, and the weights in the advantage-weighted behavior cloning term; the rows are therefore not isolated scalar-reward ablations. The configurations otherwise share the scheduler input, training-only supervised targets, auxiliary future-condition classifier, candidate action set, and constraints. The Double DQN row is a training-configuration comparison: it shares the state representation, reward based on forecast loss, action set, and constraints, but omits behavior cloning pretraining, the continuing behavior cloning term, and the auxiliary classifier. All configurations have zero warm-up interruption structurally because $D_{\min}=6$ exceeds the maximum two-step warm-up; the final column is therefore determined by the channel-use acceptance criterion. All test metrics use the same frozen primary forecaster.",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_regime_table(path: Path, stats: dict[str, object]) -> None:
    regimes = stats["regime_decomposition"]
    seed_count = int(stats["seed_count"])
    rows = []
    for label, key in (
        ("Particle event type", "particle"),
        ("Flux event type", "flux"),
        ("Thermal event type", "thermal"),
        ("Macro-averaged loss", "macro"),
    ):
        item = regimes[key]
        margin = item["margin"]
        rows.append(
            f"    {label} & {item['pdppo_mean']:.4f} & {item['fixed_mean']:.4f} & "
            f"{margin['wins']} of {seed_count} & {fmt_ci(margin)} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Event-type-specific forecast losses and the macro-averaged normalized forecast loss. Lower values are better; a positive paired difference favors PD-PPO over the fixed-schedule baseline.}",
            r"  \label{tab:clean_regime_decomposition}",
            r"  \scriptsize",
            r"  \setlength{\tabcolsep}{2.6pt}",
            r"  \begin{tabular}{@{}lcccc@{}}",
            r"    \toprule",
            r"    Event type & PD-PPO & Fixed baseline & \shortstack{Seeds with\\lower loss} & \shortstack{Mean paired difference\\{[95\% bootstrap CI]}} \\",
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
    seed_count = int(stats["seed_count"])
    rows = []
    for label, key in (
        ("Primary-forecaster fixed-schedule baseline", "macro_original_static"),
        ("Ridge-forecaster fixed-schedule baseline", "macro_secondary_static"),
    ):
        item = items[key]
        rows.append(f"    {label} & {item['wins']} of {seed_count} & {fmt_ci(item)} \\\\")
    text = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Alternative-forecaster analysis on unchanged observation sequences. Positive macro-averaged loss differences favor PD-PPO.}",
            r"  \label{tab:clean_secondary_forecaster}",
            r"  \footnotesize",
            r"  \begin{tabular}{@{}lcc@{}}",
            r"    \toprule",
            r"    Fixed-schedule baseline & Seeds with lower loss & Mean paired difference [95\% bootstrap CI] \\",
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
    seed_count = int(stats["seed_count"])
    primary = stats["primary_means"]
    primary_step = stats["comparisons"]["static_step"]
    primary_macro = stats["comparisons"]["static_macro"]
    full = stats["full_final_partition"]
    rows = [
        (
            "Event-balanced evaluation windows",
            4096,
            primary["pdppo_step_loss"],
            primary["fixed_step_loss"],
            primary_step,
            primary["pdppo_macro_score"],
            primary["fixed_macro_score"],
            primary_macro,
        ),
        (
            "Continuous full-partition evaluation",
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
            f"{step_margin['wins']} of {seed_count}; {fmt_ci(step_margin)} & "
            f"{pd_macro:.4f} / {fixed_macro:.4f} & "
            f"{macro_margin['wins']} of {seed_count}; {fmt_ci(macro_margin)} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"  \centering",
            r"  \caption{Sensitivity to evaluation coverage in the test partition. A positive paired difference means that the fixed-schedule baseline has higher loss than PD-PPO.}",
            r"  \label{tab:clean_full_partition_sensitivity}",
            r"  \scriptsize",
            r"  \renewcommand{\arraystretch}{1.05}",
            r"  \setlength{\tabcolsep}{2.2pt}",
            r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.16\textwidth}>{\centering\arraybackslash}p{0.06\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\centering\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}@{}}",
            r"    \toprule",
            r"    Evaluation scope & Time steps per seed & Mean forecast loss\newline (PD-PPO / fixed baseline) & Seeds with lower loss;\newline mean paired difference\newline [95\% bootstrap CI] & Macro-averaged loss\newline (PD-PPO / fixed baseline) & Seeds with lower loss;\newline mean paired difference\newline [95\% bootstrap CI] \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.98\textwidth}",
            r"  \vspace{0.4ex}\footnotesize Each interval is a 95\% bootstrap confidence interval for the paired difference. The event-balanced evaluation uses eight separately reset 512-step rollouts. Continuous full-partition evaluation uses the same final policy checkpoint and the fixed channel subset selected on the calibration/validation partition once over $[64750,69992)$. The last eight time steps are excluded because an eight-step target is unavailable; no policy, subset, or normalization factor is reselected.",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_reported_descriptive_table(path: Path, stats: dict[str, object]) -> None:
    """Write the 24-seed aggregate without assigning confirmatory status."""
    seed_count = int(stats["seed_count"])
    comparisons = stats["comparisons"]
    full = stats["full_final_partition"]
    reward = stats["reward_controls"]
    forecaster = stats["secondary_forecaster"]
    rows = [
        ("Event-balanced evaluation windows: fixed-schedule baseline", "Mean forecast loss", comparisons["static_step"]),
        ("Event-balanced evaluation windows: fixed-schedule baseline", "Macro-averaged loss", comparisons["static_macro"]),
        ("Continuous full partition: fixed-schedule baseline", "Mean forecast loss", full["step_margin"]),
        ("Continuous full partition: fixed-schedule baseline", "Macro-averaged loss", full["macro_margin"]),
        ("Warning-rule baseline", "Macro-averaged loss", comparisons["context_alert_macro"]),
        ("Privileged true-label baseline", "Macro-averaged loss", comparisons["event_label_macro"]),
        ("Double DQN training configuration", "Macro-averaged loss", comparisons["dqn_macro"]),
        ("AoI-objective PPO configuration", "Macro-averaged loss", reward["aoi"]["macro"]),
        ("Uncertainty-objective PPO configuration", "Macro-averaged loss", reward["uncertainty"]["macro"]),
        ("Ridge-forecaster fixed-schedule baseline", "Macro-averaged loss", forecaster["macro_secondary_static"]),
    ]
    body = [
        f"    {label} & {endpoint} & {item['wins']} of {seed_count} & {fmt_ci(item)} \\\\"
        for label, endpoint, item in rows
    ]
    text = "\n".join(
        [
            r"\begin{table*}[htbp]",
            r"  \centering",
            r"  \caption{Descriptive aggregate over all 24 reported seeds, including the two pilot/model-selection seeds. Positive differences favor the complete PD-PPO training configuration. These values are retained for transparency and are not the post-selection inferential analysis.}",
            r"  \label{tab:reported_24seed_descriptive}",
            r"  \scriptsize",
            r"  \setlength{\tabcolsep}{2.5pt}",
            r"  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}p{0.19\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}@{}}",
            r"    \toprule",
            r"    Comparison and scope & Endpoint & Seeds favoring PD-PPO & Mean paired difference [95\% bootstrap CI] \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{0.92\textwidth}",
            r"  \vspace{0.4ex}\footnotesize The objective rows are matched PPO training configurations with different scalar objectives, not isolated scalar-reward ablations. The Double DQN row compares complete training configurations. The warning-rule and privileged true-label confidence intervals include zero.",
            r"  \end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def setup_plot_style(paper_dir: Path) -> tuple[dict[str, str], object]:
    sys.path.insert(0, str(paper_dir / "figures"))
    from paper_plot_style import PALETTE, apply_paper_style, save_pdf_png

    apply_paper_style(base_size=8.5)
    return PALETTE, save_pdf_png


def draw_box_swarm(
    ax: plt.Axes,
    datasets: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    win_labels: list[str],
    *,
    seed: int,
) -> None:
    """Draw a consistent box-and-seed distribution panel."""
    box = ax.boxplot(
        datasets,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.1},
        whiskerprops={"color": "#606060", "linewidth": 0.8},
        capprops={"color": "#606060", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    rng = np.random.default_rng(seed)
    for idx, (values, color) in enumerate(zip(datasets, colors, strict=True), start=1):
        ax.scatter(
            np.full(len(values), idx) + rng.normal(0, 0.038, len(values)),
            values,
            s=9,
            color=color,
            alpha=0.68,
            linewidth=0,
            zorder=3,
        )
    values = np.concatenate(datasets)
    span = max(float(np.ptp(values)), 0.02)
    low = min(float(values.min()), 0.0) - 0.08 * span
    high = float(values.max()) + 0.20 * span
    ax.set_ylim(low, high)
    label_y = float(values.max()) + 0.10 * span
    for idx, label in enumerate(win_labels, start=1):
        ax.text(idx, label_y, label, ha="center", va="bottom", fontsize=8.0, weight="bold")
    ax.axhline(0, color="#303030", linewidth=0.8)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.grid(axis="x", visible=False)


def build_main_figure(
    path: Path,
    frames: dict[str, pd.DataFrame],
    stats: dict[str, object],
    palette: dict[str, str],
    save_pdf_png: object,
) -> None:
    main = frames["main"].sort_values("macro_margin_pdppo_vs_validation_selected_static")
    refs = frames["references"]
    dqn = frames["dqn"].sort_values("seed")
    seed_count = int(stats["seed_count"])
    rank = np.arange(1, seed_count + 1)
    fig = plt.figure(figsize=(7.15, 3.85), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.45, 0.88], hspace=0.10)
    ax1 = fig.add_subplot(grid[0, 0])
    ax1b = fig.add_subplot(grid[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(grid[:, 1])
    ax3 = fig.add_subplot(grid[:, 2])

    step_values = main["step_margin_pdppo_vs_validation_selected_static"].to_numpy()
    macro_values = main["macro_margin_pdppo_vs_validation_selected_static"].to_numpy()
    for axis, values, color, marker, label in (
        (ax1, step_values, palette["blue"], "o", "Mean forecast loss"),
        (ax1b, macro_values, palette["teal"], "s", "Macro-averaged loss"),
    ):
        axis.scatter(rank, values, s=16, marker=marker, color=color, edgecolor="white", linewidth=0.35, zorder=3)
        axis.axhline(0, color=palette["dark"], linewidth=0.8)
        axis.set_ylabel(f"{label}\n(fixed baseline - PD-PPO)")
        axis.grid(axis="x", visible=False)
    ax1.set_title("A  Fixed-schedule baseline", loc="left")
    primary = stats["primary_means"]
    ax1.text(
        0.03,
        0.92,
        f"{stats['comparisons']['static_step']['wins']}/{seed_count}; "
        f"{100 * primary['step_relative_reduction']:.1f}% lower",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        weight="bold",
    )
    ax1b.text(
        0.03,
        0.92,
        f"{stats['comparisons']['static_macro']['wins']}/{seed_count}; "
        f"{100 * primary['macro_relative_reduction']:.1f}% lower",
        transform=ax1b.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        weight="bold",
    )
    tick_idx = np.unique(np.linspace(0, seed_count - 1, 7, dtype=int))
    ax1b.set_xticks(rank[tick_idx], main["seed"].iloc[tick_idx].astype(int), rotation=30, ha="right")
    ax1b.set_xlabel("Seeds ordered by macro difference")
    ax1.tick_params(labelbottom=False)

    datasets = [
        main["macro_margin_pdppo_vs_validation_selected_static"].to_numpy(),
        main["macro_margin_pdppo_vs_aoi"].to_numpy(),
        main["macro_margin_pdppo_vs_round_robin"].to_numpy(),
        main["macro_margin_pdppo_vs_random"].to_numpy(),
    ]
    labels = ["Fixed", "AoI", "Round\nrobin", "Random"]
    greedy = refs[refs["policy"] == "forecast_greedy_one_step"].sort_values("seed")
    datasets.append(greedy["margin_oracle_loss_macro_subtype_event_staticnorm_vs_custom_ppo"].to_numpy())
    labels.append("Next-step test\ntargets\n(privileged)")
    colors = ["#4D4D4D", palette["teal"], palette["vermillion"], palette["purple"], palette["orange"]]
    draw_box_swarm(
        ax2,
        datasets,
        labels,
        colors,
        [f"{int(np.sum(values > 0))}/{seed_count}" for values in datasets],
        seed=7,
    )
    ax2.tick_params(axis="x", labelsize=8.0, pad=3)
    plt.setp(ax2.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    ax2.set_ylabel("Macro-averaged loss difference\n(baseline - PD-PPO)")
    ax2.set_title("B  Baselines", loc="left")

    warning = refs[refs["policy"] == "context_alert_bandit_t0p5"].sort_values("seed")
    exact = refs[refs["policy"] == "event_label_reference_l16"].sort_values("seed")
    strong_data = [
        warning["margin_oracle_loss_macro_subtype_event_staticnorm_vs_custom_ppo"].to_numpy(),
        exact["margin_oracle_loss_macro_subtype_event_staticnorm_vs_custom_ppo"].to_numpy(),
        dqn["macro_margin_dqn_minus_ppo"].to_numpy(),
    ]
    draw_box_swarm(
        ax3,
        strong_data,
        [
            "Warning\nrule",
            "True label\n(privileged)",
            "Double DQN\nconfig.",
        ],
        [palette["sky"], palette["gray"], "#7B3294"],
        [f"{int(np.sum(values > 0))}/{seed_count}" for values in strong_data],
        seed=11,
    )
    ax3.tick_params(axis="x", labelsize=8.0, pad=3)
    plt.setp(ax3.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    ax3.set_ylabel("Macro-averaged loss difference\n(baseline - PD-PPO)")
    ax3.set_title("C  Additional comparisons", loc="left")
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
    seed_count = int(frames["main"]["seed"].nunique())
    heat = np.zeros((len(SUBTYPE_ORDER), len(SENSOR_ORDER)), dtype=float)
    for row_idx, subtype in enumerate(SUBTYPE_ORDER):
        for col_idx, sensor in enumerate(SENSOR_ORDER):
            row = duty[(duty["subtype"] == subtype) & (duty["sensor"] == sensor)]
            if len(row) != seed_count:
                raise ValueError(f"missing duty row for {subtype}/{sensor}")
            heat[row_idx, col_idx] = float(row["selection_fraction"].mean())

    fig = plt.figure(figsize=(7.15, 3.15), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.78, 1.08])
    ax1 = fig.add_subplot(grid[0, 0])
    image = ax1.imshow(heat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    mechanism_sensor_labels = [
        "T–RH",
        "Snow\nsurface\nIR",
        "Laser",
        "FC4\nmass flux",
    ]
    ax1.set_xticks(
        range(len(mechanism_sensor_labels)),
        mechanism_sensor_labels,
        rotation=0,
        ha="center",
        rotation_mode="anchor",
    )
    ax1.tick_params(axis="x", labelsize=7.0, pad=3)
    ax1.set_yticks(range(len(SUBTYPE_LABELS)), SUBTYPE_LABELS)
    ax1.set_title("A  Channel activation by operating condition", loc="left", fontsize=9.0, pad=4)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax1.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="white" if heat[i, j] > 0.55 else palette["dark"], fontsize=8.5)
    cbar1 = fig.colorbar(image, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label("Channel activation frequency", fontsize=7.0)
    cbar1.ax.tick_params(labelsize=7.0)

    ax2 = fig.add_subplot(grid[0, 1])
    counts = Counter(int(value) for value in frames["main"]["validation_selected_static_action_idx"])
    static_labels = [
        "Infrared snow-surface-\ntemperature channel",
        "Laser disdrometer\nchannel",
        "FC4 snow mass-flux\nchannel",
    ]
    static_counts = [counts[3], counts[4], counts[5]]
    ax2.barh(np.arange(3), static_counts, color=[palette["sky"], palette["orange"], palette["teal"]])
    ax2.set_yticks(np.arange(3), static_labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, seed_count)
    ax2.set_xlabel("Number of seeds")
    ax2.set_title("B  Cal./val.-selected\nfixed channels", loc="left", fontsize=8.5, pad=4)
    ax2.text(0.98, 0.98, f"n={seed_count}", transform=ax2.transAxes, ha="right", va="top", fontsize=7.5, weight="bold")
    for idx, value in enumerate(static_counts):
        ax2.text(value + 0.3, idx, str(value), va="center", fontsize=8.5)

    ax3 = fig.add_subplot(grid[0, 2])
    switches = behavior["switches_per_step"].to_numpy()
    switch_span = max(float(np.ptp(switches)), 1e-9)
    marker_sizes = 22 + 58 * (switches - float(switches.min())) / switch_span
    scatter = ax3.scatter(
        behavior["mask_entropy_bits"],
        behavior["subtype_mask_mi_bits"],
        c=switches,
        cmap="viridis",
        s=marker_sizes,
        edgecolor="white",
        linewidth=0.35,
    )
    ax3.set_xlabel("Executed-action\nentropy (bits)")
    ax3.set_ylabel("Condition–executed-action\nmutual information (bits)")
    ax3.set_title(
        "C  Executed-action\ndiagnostics", loc="left", fontsize=8.5, pad=4, x=0.05
    )
    cbar3 = fig.colorbar(
        scatter,
        ax=ax3,
        fraction=0.046,
        pad=0.04,
    )
    cbar3.set_label("Normalized Hamming distance\nper transition", fontsize=7.0)
    cbar3.ax.tick_params(labelsize=7.0)
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
    forecaster = frames["forecaster"].sort_values("seed")
    reward_data = [
        (aoi["macro_score"] - forecast["macro_score"]).to_numpy(),
        (uncertainty["macro_score"] - forecast["macro_score"]).to_numpy(),
    ]
    ridge_data = [
        forecaster["macro_margin_vs_original_static"].to_numpy(),
        forecaster["macro_margin_vs_secondary_static"].to_numpy(),
    ]
    reward_labels = ["AoI-objective\nPPO", "Uncertainty-objective\nPPO"]
    ridge_labels = ["Primary-forecaster\nfixed schedule", "Ridge-forecaster\nfixed schedule"]
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.75),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=6 / 72, h_pad=3 / 72, wspace=0.08, hspace=0.04)
    draw_box_swarm(
        ax1,
        reward_data,
        reward_labels,
        [palette["teal"], palette["purple"]],
        [
            f"Lower in\n{int(np.sum(values > 0))} of {len(values)} seeds"
            for values in reward_data
        ],
        seed=17,
    )
    ax1.set_ylabel("Macro-averaged loss difference\n(variant - PD-PPO)")
    ax1.set_title("A  Matched PPO configurations", loc="left")

    draw_box_swarm(
        ax2,
        ridge_data,
        ridge_labels,
        ["#4D4D4D", palette["blue"]],
        [
            f"Lower in\n{int(np.sum(values > 0))} of {len(values)} seeds"
            for values in ridge_data
        ],
        seed=23,
    )
    ax2.set_ylabel("Macro-averaged loss difference\n(baseline - PD-PPO)")
    ax2.set_title("B  Alternative-forecaster analysis", loc="left")
    save_pdf_png(fig, path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.framework_root = args.framework_root.resolve()
    frozen_frames, input_paths = load_inputs(args)
    frames = select_analysis_seeds(frozen_frames, PRIMARY_SEEDS)
    reported_frames = select_analysis_seeds(frozen_frames, EXPECTED_SEEDS)
    stats = build_statistics(frames, args.bootstrap_samples, args.bootstrap_seed)
    reported_stats = build_statistics(
        reported_frames, args.bootstrap_samples, args.bootstrap_seed
    )
    paper_dir = args.framework_root / "paper"
    table_dir = paper_dir / "tables"
    figure_dir = paper_dir / "figures"
    report_dir = args.framework_root / "reports" / "aggregate" / "pdppo_claim_evidence_assets_20260729"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest_output is None:
        manifest_output = report_dir / "paper_asset_manifest.json"
    else:
        manifest_output = args.manifest_output.resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    write_main_table(table_dir / "clean_main_comparisons.tex", stats)
    write_regime_table(table_dir / "clean_regime_decomposition.tex", stats)
    write_control_table(table_dir / "clean_learning_controls.tex", stats)
    write_forecaster_table(table_dir / "clean_secondary_forecaster.tex", stats)
    write_full_partition_table(table_dir / "clean_full_partition_sensitivity.tex", stats)
    write_reported_descriptive_table(
        table_dir / "clean_reported_24seed_descriptive.tex", reported_stats
    )
    palette, save_pdf_png = setup_plot_style(paper_dir)
    build_main_figure(
        figure_dir / "figure_clean_main_evidence", frames, stats, palette, save_pdf_png
    )
    build_mechanism_figure(figure_dir / "figure_clean_mechanism", frames, palette, save_pdf_png)
    build_control_figure(figure_dir / "figure_clean_controls", frames, palette, save_pdf_png)

    output_paths = [
        table_dir / "clean_main_comparisons.tex",
        table_dir / "clean_regime_decomposition.tex",
        table_dir / "clean_learning_controls.tex",
        table_dir / "clean_secondary_forecaster.tex",
        table_dir / "clean_full_partition_sensitivity.tex",
        table_dir / "clean_reported_24seed_descriptive.tex",
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
        "statistics": {
            "post_selection_22_primary": stats,
            "reported_24_descriptive": reported_stats,
        },
        "outputs": {
            str(path.relative_to(args.framework_root)): {"sha256": sha256(path)}
            for path in output_paths
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"paper_assets_complete manifest={manifest_output}")


if __name__ == "__main__":
    main()
