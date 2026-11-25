#!/usr/bin/env python
"""Summarise mechanism-ablation seed-level margins with bootstrap intervals."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_LABELS = {
    "full_reference": "Full PD-PPO",
    "no_imitation_guide": "No imitation guide",
    "no_regime_aux_path": "No event context auxiliary signal",
    "no_staticnorm_train": "No balanced training loss",
}


def seed_label(seeds: list[int]) -> str:
    return "_".join(str(seed) for seed in seeds)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def finite_array(values: pd.Series | list[float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values: np.ndarray, *, statistic: str, draws: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True)
    if statistic == "median":
        stats = np.median(samples, axis=1)
    else:
        stats = np.mean(samples, axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def count_true(df: pd.DataFrame, column: str) -> int:
    if column not in df:
        return 0
    return int(as_bool(df[column]).sum())


def load_variant(root: Path, variant: str, seeds: list[int], date_tag: str) -> pd.DataFrame:
    label = seed_label(seeds)
    macro_path = root / f"mechanism_ablation_{variant}_{label}_macro_{date_tag}" / "metpair_seed_summary.csv"
    old_path = root / f"mechanism_ablation_{variant}_{label}_oldclaim_{date_tag}" / "oldclaim_seed_summary.csv"
    if not macro_path.exists():
        raise FileNotFoundError(f"Missing macro seed summary for {variant}: {macro_path}")
    if not old_path.exists():
        raise FileNotFoundError(f"Missing old-claim seed summary for {variant}: {old_path}")

    macro = pd.read_csv(macro_path)
    old = pd.read_csv(old_path)
    keep_macro = [
        "seed",
        "complete",
        "learned_macro_margin_abs_vs_macro_static_reference",
        "replay_macro_margin_abs_vs_static_reference",
        "macro_seed_positive_pass",
        "macro_seed_gate_pass",
        "behavior_gate_pass",
    ]
    keep_old = [
        "seed",
        "custom_ppo_loss",
        "selected_static_loss",
        "step_margin_vs_best_operational_baseline",
        "step_margin_vs_replay_static_reference",
        "learned_true_static_step_gate_pass",
        "learned_true_static_macro_gate_pass",
        "old_claim_step_gate_pass",
        "old_claim_macro_gate_pass",
    ]
    rows = macro[[col for col in keep_macro if col in macro.columns]].merge(
        old[[col for col in keep_old if col in old.columns]],
        on="seed",
        how="outer",
        suffixes=("_macro", "_old"),
    )
    rows.insert(0, "variant", variant)
    rows.insert(1, "variant_label", VARIANT_LABELS.get(variant, variant))
    return rows


def summarise_variant(rows: pd.DataFrame, reference_rows: pd.DataFrame | None, *, draws: int) -> dict[str, object]:
    variant = str(rows["variant"].iloc[0])
    macro = finite_array(rows["learned_macro_margin_abs_vs_macro_static_reference"])
    replay_macro = finite_array(rows.get("replay_macro_margin_abs_vs_static_reference", []))
    step = finite_array(rows.get("step_margin_vs_best_operational_baseline", []))
    strict_step = finite_array(rows.get("step_margin_vs_replay_static_reference", []))

    mean_lo, mean_hi = bootstrap_ci(macro, statistic="mean", draws=draws, seed=1729)
    median_lo, median_hi = bootstrap_ci(macro, statistic="median", draws=draws, seed=1730)

    paired_delta = np.array([], dtype=float)
    if reference_rows is not None and variant != "full_reference":
        left = rows[["seed", "learned_macro_margin_abs_vs_macro_static_reference"]].rename(
            columns={"learned_macro_margin_abs_vs_macro_static_reference": "variant_macro_margin"}
        )
        right = reference_rows[["seed", "learned_macro_margin_abs_vs_macro_static_reference"]].rename(
            columns={"learned_macro_margin_abs_vs_macro_static_reference": "reference_macro_margin"}
        )
        merged = left.merge(right, on="seed", how="inner")
        paired_delta = finite_array(merged["variant_macro_margin"] - merged["reference_macro_margin"])
    delta_lo, delta_hi = bootstrap_ci(paired_delta, statistic="mean", draws=draws, seed=1731)

    return {
        "variant": variant,
        "variant_label": str(rows["variant_label"].iloc[0]),
        "complete_seeds": int(len(rows)),
        "macro_gate_count": count_true(rows, "macro_seed_gate_pass"),
        "macro_positive_count": count_true(rows, "macro_seed_positive_pass"),
        "strict_true_static_step_count": count_true(rows, "learned_true_static_step_gate_pass"),
        "true_static_macro_count": count_true(rows, "learned_true_static_macro_gate_pass"),
        "behavior_gate_count": count_true(rows, "behavior_gate_pass"),
        "macro_margin_mean": float(np.mean(macro)) if macro.size else float("nan"),
        "macro_margin_median": float(np.median(macro)) if macro.size else float("nan"),
        "macro_margin_min": float(np.min(macro)) if macro.size else float("nan"),
        "macro_margin_mean_ci_low": mean_lo,
        "macro_margin_mean_ci_high": mean_hi,
        "macro_margin_median_ci_low": median_lo,
        "macro_margin_median_ci_high": median_hi,
        "paired_delta_macro_margin_mean_vs_full": float(np.mean(paired_delta)) if paired_delta.size else float("nan"),
        "paired_delta_macro_margin_ci_low_vs_full": delta_lo,
        "paired_delta_macro_margin_ci_high_vs_full": delta_hi,
        "step_margin_mean": float(np.mean(step)) if step.size else float("nan"),
        "strict_step_margin_mean": float(np.mean(strict_step)) if strict_step.size else float("nan"),
        "replay_macro_margin_mean": float(np.mean(replay_macro)) if replay_macro.size else float("nan"),
    }


def write_report(summary: pd.DataFrame, seed_rows: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Mechanism Ablation Continuous-Margin Summary",
        "",
        "This report is generated from seed-level mechanism ablation artifacts. Macro margins use the event type macro score normalised by static references.",
        "",
        "## Variant Summary",
        "",
        "| Variant | Complete | Macro gate | Strict step | Behaviour | Mean macro margin [95% CI] | Median macro margin [95% CI] | Delta vs full [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        n = int(row["complete_seeds"])
        delta = "--"
        if math.isfinite(float(row["paired_delta_macro_margin_mean_vs_full"])):
            delta = (
                f"{row['paired_delta_macro_margin_mean_vs_full']:.4f} "
                f"[{row['paired_delta_macro_margin_ci_low_vs_full']:.4f}, "
                f"{row['paired_delta_macro_margin_ci_high_vs_full']:.4f}]"
            )
        lines.append(
            "| {label} | {n} | {macro}/{n} | {strict}/{n} | {beh}/{n} | {mean:.4f} [{lo:.4f}, {hi:.4f}] | {med:.4f} [{mlo:.4f}, {mhi:.4f}] | {delta} |".format(
                label=row["variant_label"],
                n=n,
                macro=int(row["macro_gate_count"]),
                strict=int(row["strict_true_static_step_count"]),
                beh=int(row["behavior_gate_count"]),
                mean=float(row["macro_margin_mean"]),
                lo=float(row["macro_margin_mean_ci_low"]),
                hi=float(row["macro_margin_mean_ci_high"]),
                med=float(row["macro_margin_median"]),
                mlo=float(row["macro_margin_median_ci_low"]),
                mhi=float(row["macro_margin_median_ci_high"]),
                delta=delta,
            )
        )
    lines.extend(
        [
            "",
            "## Seed Rows",
            "",
            f"Seed-level rows: `{out_path.parent / 'mechanism_ablation_seed_margins.csv'}`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", default="reports/aggregate")
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full_reference", "no_imitation_guide", "no_regime_aux_path", "no_staticnorm_train"],
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    args = parser.parse_args()

    root = Path(args.aggregate_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = {
        variant: load_variant(root, variant, args.seeds, args.date_tag)
        for variant in args.variants
    }
    reference_rows = variant_rows.get("full_reference")
    seed_rows = pd.concat(variant_rows.values(), ignore_index=True)
    summary_rows = [
        summarise_variant(rows, reference_rows, draws=args.bootstrap_draws)
        for rows in variant_rows.values()
    ]
    summary = pd.DataFrame(summary_rows)

    seed_csv = out_dir / "mechanism_ablation_seed_margins.csv"
    summary_csv = out_dir / "mechanism_ablation_continuous_margins.csv"
    report_md = out_dir / "mechanism_ablation_continuous_margins.md"

    seed_rows.to_csv(seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_report(summary, seed_rows, report_md)

    print(f"wrote {summary_csv}")
    print(f"wrote {seed_csv}")
    print(f"wrote {report_md}")


if __name__ == "__main__":
    main()
