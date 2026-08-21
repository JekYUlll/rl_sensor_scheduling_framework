#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "energy_account_convergence_20260524"
PAPER_TABLE = ROOT / "paper" / "tables" / "energy_account_curriculum_results.tex"
MEMO = ROOT / "docs" / "05-24-results-convergence.md"

POLICY_ORDER = [
    "full_open_unconstrained",
    "feasible_static_projected",
    "custom_ppo",
    "aoi",
    "round_robin",
    "random",
]

POLICY_LABEL = {
    "full_open_unconstrained": "Full obs.",
    "feasible_static_projected": "Static proj.",
    "custom_ppo": "PD-PPO",
    "aoi": "AoI",
    "round_robin": "Round-robin",
    "random": "Random",
}

STORM_RUNS = {
    41: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed41",
    42: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed42",
    43: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed43",
    44: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed_seed44",
    45: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed_seed45",
}

FULL_RUNS = {
    41: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_fast" / "seed41" / "all" / "budget1p20_seed41",
    42: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_fast" / "seed42" / "all" / "budget1p20_seed42",
    43: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_fast" / "seed43" / "all" / "budget1p20_seed43",
    44: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_seed44_45_fast" / "all" / "budget1p20_seed44",
    45: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_seed44_45_fast" / "all" / "budget1p20_seed45",
}

PROBES = {
    "100k baseline": {
        "storm": STORM_RUNS[41],
        "full": FULL_RUNS[41],
    },
    "300k baseline": {
        "storm": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_300k_seed41",
        "full": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_300k_full_eval_fast" / "all" / "budget1p20_seed41",
    },
    "event-gated actor": {
        "storm": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_eventgated_200k_seed41",
        "full": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_eventgated_200k_full_eval_fast" / "all" / "budget1p20_seed41",
    },
    "SOC auxiliary": {
        "storm": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_socaux_h16_c01_200k_seed41",
        "full": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_socaux_h16_c01_200k_full_eval_fast" / "all" / "budget1p20_seed41",
    },
    "SOC soft penalty": {
        "storm": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_soc001_seed41",
    },
    "event reward x1.5": {
        "storm": REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_evt15_seed41",
    },
}


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_?seed(\d+)|seed(\d+)", str(path))
    if not match:
        return -1
    return int(next(group for group in match.groups() if group))


def load_storm_metrics(seed: int, run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "v2_custom_ppo_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for item in df.to_dict("records"):
        rows.append(
            {
                "scenario": "storm",
                "seed": int(seed),
                "policy": str(item["policy"]),
                "oracle_loss_mean": float(item["oracle_loss_mean"]),
                "reward_mean": float(item.get("reward_mean", np.nan)),
                "power_mean": float(item.get("power_mean", np.nan)),
                "warmup_abort_count": float(item.get("warmup_abort_count", np.nan)),
                "event_rate": float(item.get("event_rate", np.nan)),
                "source": str(run_dir),
            }
        )
    return rows


def load_rollout_metrics(seed: int, run_dir: Path, scenario: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("rollout_*.npz")):
        z = np.load(path, allow_pickle=True)
        policy = str(np.asarray(z["policy"]).reshape(-1)[0])
        oracle_losses = np.asarray(z["oracle_losses"], dtype=float)
        rewards = np.asarray(z["rewards"], dtype=float)
        powers = np.asarray(z["powers"], dtype=float)
        events = np.asarray(z["event_flags"], dtype=float)
        abort = float(np.asarray(z["warmup_abort_count"]).reshape(-1)[0]) if "warmup_abort_count" in z.files else np.nan
        rows.append(
            {
                "scenario": str(scenario),
                "seed": int(seed),
                "policy": policy,
                "oracle_loss_mean": float(np.nanmean(oracle_losses)),
                "reward_mean": float(np.nanmean(rewards)),
                "power_mean": float(np.nanmean(powers)),
                "warmup_abort_count": abort,
                "event_rate": float(np.nanmean(events)),
                "source": str(run_dir),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no rollout_*.npz files under {run_dir}")
    return rows


def collect_main_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, run_dir in STORM_RUNS.items():
        rows.extend(load_storm_metrics(seed, run_dir))
    for seed, run_dir in FULL_RUNS.items():
        rows.extend(load_rollout_metrics(seed, run_dir, "full"))
    df = pd.DataFrame(rows)
    df = df[df["policy"].isin(POLICY_ORDER)].copy()
    df["policy"] = pd.Categorical(df["policy"], categories=POLICY_ORDER, ordered=True)
    return df.sort_values(["scenario", "seed", "policy"]).reset_index(drop=True)


def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, scenario_df in df.groupby("scenario", observed=True):
        pivot = scenario_df.pivot_table(index="seed", columns="policy", values="oracle_loss_mean", observed=True)
        ppo = pivot["custom_ppo"]
        for policy in POLICY_ORDER:
            sub = scenario_df[scenario_df["policy"].astype(str) == policy]
            wins = ""
            if policy != "custom_ppo" and policy in pivot.columns:
                wins = f"{int((ppo < pivot[policy]).sum())}/{int(pivot[[policy, 'custom_ppo']].dropna().shape[0])}"
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "policy_label": POLICY_LABEL[policy],
                    "oracle_loss_mean": float(sub["oracle_loss_mean"].mean()),
                    "oracle_loss_std": float(sub["oracle_loss_mean"].std(ddof=1)),
                    "power_mean": float(sub["power_mean"].mean()),
                    "abort_mean": float(sub["warmup_abort_count"].mean()),
                    "event_rate_mean": float(sub["event_rate"].mean()),
                    "ppo_wins_vs_policy": wins,
                }
            )
    return pd.DataFrame(rows)


def collect_probe_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for probe_name, payload in PROBES.items():
        for scenario, run_dir in payload.items():
            if not run_dir.exists():
                continue
            if scenario == "storm":
                metrics = load_storm_metrics(41, run_dir)
            else:
                metrics = load_rollout_metrics(41, run_dir, "full")
            values = {row["policy"]: row for row in metrics}
            ppo = values.get("custom_ppo", {})
            aoi = values.get("aoi", {})
            static = values.get("feasible_static_projected", {})
            rows.append(
                {
                    "probe": probe_name,
                    "scenario": scenario,
                    "ppo_oracle_loss": float(ppo.get("oracle_loss_mean", np.nan)),
                    "aoi_oracle_loss": float(aoi.get("oracle_loss_mean", np.nan)),
                    "static_oracle_loss": float(static.get("oracle_loss_mean", np.nan)),
                    "ppo_minus_aoi": float(ppo.get("oracle_loss_mean", np.nan)) - float(aoi.get("oracle_loss_mean", np.nan)),
                    "ppo_abort": float(ppo.get("warmup_abort_count", np.nan)),
                    "ppo_power": float(ppo.get("power_mean", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def write_latex_table(summary: pd.DataFrame) -> None:
    storm = summary[summary["scenario"] == "storm"].set_index("policy")
    full = summary[summary["scenario"] == "full"].set_index("policy")
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Retrospective energy-account curriculum diagnostic from archived, non-independent windows. Lower oracle loss is better. Entries are mean $\\pm$ standard deviation over five seeds; win counts describe the stored procedure only and must not be interpreted as held-out learned-policy evidence.}",
        "  \\label{tab:energy_account_curriculum}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{@{}lcccccc@{}}",
        "    \\toprule",
        "    Policy & Storm loss & Storm wins & Full loss & Full wins & Storm aborts & Full aborts \\\\",
        "    \\midrule",
    ]
    for policy in POLICY_ORDER:
        label = POLICY_LABEL[policy]
        storm_loss = fmt_mean_std(float(storm.loc[policy, "oracle_loss_mean"]), float(storm.loc[policy, "oracle_loss_std"]))
        full_loss = fmt_mean_std(float(full.loc[policy, "oracle_loss_mean"]), float(full.loc[policy, "oracle_loss_std"]))
        storm_wins = "--" if policy == "custom_ppo" else str(storm.loc[policy, "ppo_wins_vs_policy"])
        full_wins = "--" if policy == "custom_ppo" else str(full.loc[policy, "ppo_wins_vs_policy"])
        storm_abort = f"{float(storm.loc[policy, 'abort_mean']):.1f}"
        full_abort = f"{float(full.loc[policy, 'abort_mean']):.1f}"
        lines.append(
            f"    {label} & {storm_loss} & {storm_wins} & {full_loss} & {full_wins} & {storm_abort} & {full_abort} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}%",
            "  }",
            "\\end{table}",
            "",
        ]
    )
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE.write_text("\n".join(lines), encoding="utf-8")


def write_memo(summary: pd.DataFrame, probes: pd.DataFrame) -> None:
    storm = summary[summary["scenario"] == "storm"].set_index("policy")
    full = summary[summary["scenario"] == "full"].set_index("policy")
    text = f"""# Results Convergence Memo - 2026-05-24

## Superseded Diagnostic Table

Do not use the 100k storm-window curriculum PD-PPO runs (`seed=41--45`) as a
main learned-policy result. The 2026-05-26 protocol audit found identical
storm training/evaluation starts in all five seeds, overlap in the later
full-distribution replay, reconstructed oracle overlap, and no declared
training-only normalisation. See
`reports/energy_account_protocol_audit_20260526/energy_account_protocol_audit_summary.json`.

Recorded descriptive values from the archived procedure:

- Storm-window: PD-PPO `{storm.loc['custom_ppo', 'oracle_loss_mean']:.4f} +/- {storm.loc['custom_ppo', 'oracle_loss_std']:.4f}`.
- Storm-window AoI: `{storm.loc['aoi', 'oracle_loss_mean']:.4f} +/- {storm.loc['aoi', 'oracle_loss_std']:.4f}`.
- Storm-window static projection: `{storm.loc['feasible_static_projected', 'oracle_loss_mean']:.4f} +/- {storm.loc['feasible_static_projected', 'oracle_loss_std']:.4f}`.
- Full-distribution: PD-PPO `{full.loc['custom_ppo', 'oracle_loss_mean']:.4f} +/- {full.loc['custom_ppo', 'oracle_loss_std']:.4f}`.
- Full-distribution AoI: `{full.loc['aoi', 'oracle_loss_mean']:.4f} +/- {full.loc['aoi', 'oracle_loss_std']:.4f}`.
- Full-distribution static projection: `{full.loc['feasible_static_projected', 'oracle_loss_mean']:.4f} +/- {full.loc['feasible_static_projected', 'oracle_loss_std']:.4f}`.

## Claim Boundary After Protocol Audit

Permitted as retrospective mechanism diagnostics only:

- Under its stored non-independent procedure, PD-PPO records lower storm-window
  loss than feasible static projection, round-robin, and random scheduling.
- These values motivate a corrected chronological split experiment; they do not
  measure held-out learned-policy performance.

Not supported:

- Submission-level learned-policy comparison in the energy-account setting.
- Held-out or full-distribution generalization from these archived runs.
- PD-PPO robustly dominates AoI.
- PD-PPO reliably learns clean event-triggered laser gating.
- Fixed-budget V3.1 results alone prove dynamic scheduling value; those results are
  compatible with strong static projection.

## Diagnostic Probe Summary

{dataframe_to_markdown(probes)}
"""
    MEMO.write_text(text, encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for item in df.to_dict("records"):
        values = []
        for col in cols:
            value = item[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_df = collect_main_rows()
    summary = summarize_main(main_df)
    probes = collect_probe_rows()
    main_df.to_csv(OUT_DIR / "energy_account_main_long.csv", index=False)
    summary.to_csv(OUT_DIR / "energy_account_main_summary.csv", index=False)
    probes.to_csv(OUT_DIR / "energy_account_probe_summary.csv", index=False)
    write_latex_table(summary)
    write_memo(summary, probes)
    print(OUT_DIR / "energy_account_main_summary.csv")
    print(PAPER_TABLE)
    print(MEMO)


if __name__ == "__main__":
    main()
