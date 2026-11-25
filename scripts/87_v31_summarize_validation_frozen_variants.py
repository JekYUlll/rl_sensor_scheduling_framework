#!/usr/bin/env python
"""Summarise validation-frozen macro margins for the full policy and ablations."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_LABELS = {
    "full_reference": "Full PD-PPO",
    "no_imitation": "No imitation guide",
    "no_regime_aux": "No event context auxiliary",
    "no_staticnorm": "No balanced training loss",
}


def parse_entry(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=CSV, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=CSV, got {value!r}")
    return name, Path(path)


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def bootstrap(values: np.ndarray, *, statistic: str, seed: int, draws: int) -> tuple[float, float]:
    values = finite(values)
    if not values.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))]
    if statistic == "median":
        samples = np.median(samples, axis=1)
    else:
        samples = samples.mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summary_row(name: str, rows: pd.DataFrame, reference: pd.DataFrame | None, *, draws: int) -> dict[str, object]:
    macro = finite(rows["macro_margin_pdppo_vs_validation_selected_static"])
    step = finite(rows["step_margin_pdppo_vs_validation_selected_static"])
    lo, hi = bootstrap(macro, statistic="mean", seed=20260711, draws=draws)
    median_lo, median_hi = bootstrap(macro, statistic="median", seed=20260712, draws=draws)
    result: dict[str, object] = {
        "variant": name,
        "variant_label": VARIANT_LABELS.get(name, name),
        "complete_seeds": int(len(rows)),
        "macro_win_count": int(np.sum(macro > 0.0)),
        "step_win_count": int(np.sum(step > 0.0)),
        "zero_abort_count": int(np.sum(pd.to_numeric(rows["warmup_abort_count"], errors="coerce") == 0.0)),
        "macro_margin_mean": float(np.mean(macro)),
        "macro_margin_median": float(np.median(macro)),
        "macro_margin_min": float(np.min(macro)),
        "macro_margin_mean_ci_low": lo,
        "macro_margin_mean_ci_high": hi,
        "macro_margin_median_ci_low": median_lo,
        "macro_margin_median_ci_high": median_hi,
    }
    if reference is not None and name != "full_reference":
        paired = rows[["seed", "macro_margin_pdppo_vs_validation_selected_static"]].merge(
            reference[["seed", "macro_margin_pdppo_vs_validation_selected_static"]],
            on="seed",
            suffixes=("_variant", "_full"),
        )
        delta = finite(
            paired["macro_margin_pdppo_vs_validation_selected_static_variant"]
            - paired["macro_margin_pdppo_vs_validation_selected_static_full"]
        )
        delta_lo, delta_hi = bootstrap(delta, statistic="mean", seed=20260713, draws=draws)
        result.update(
            {
                "paired_delta_macro_margin_mean_vs_full": float(np.mean(delta)),
                "paired_delta_macro_margin_ci_low_vs_full": delta_lo,
                "paired_delta_macro_margin_ci_high_vs_full": delta_hi,
            }
        )
    else:
        result.update(
            {
                "paired_delta_macro_margin_mean_vs_full": float("nan"),
                "paired_delta_macro_margin_ci_low_vs_full": float("nan"),
                "paired_delta_macro_margin_ci_high_vs_full": float("nan"),
            }
        )
    return result


def write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Validation-Frozen Mechanism Ablation",
        "",
        "All macro margins use validation-partition static-candidate denominators. "
        "Positive margins are validation-selected static loss minus PD-PPO loss.",
        "",
        "| Variant | Macro wins | Step wins | Zero-abort | Mean macro margin [95% CI] | Paired delta vs full [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        delta = "--"
        if math.isfinite(float(row["paired_delta_macro_margin_mean_vs_full"])):
            delta = (
                f"{row['paired_delta_macro_margin_mean_vs_full']:.4f} "
                f"[{row['paired_delta_macro_margin_ci_low_vs_full']:.4f}, "
                f"{row['paired_delta_macro_margin_ci_high_vs_full']:.4f}]"
            )
        lines.append(
            f"| {row['variant_label']} | {int(row['macro_win_count'])}/{int(row['complete_seeds'])} | "
            f"{int(row['step_win_count'])}/{int(row['complete_seeds'])} | "
            f"{int(row['zero_abort_count'])}/{int(row['complete_seeds'])} | "
            f"{row['macro_margin_mean']:.4f} [{row['macro_margin_mean_ci_low']:.4f}, "
            f"{row['macro_margin_mean_ci_high']:.4f}] | {delta} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", nargs="+", required=True, type=parse_entry)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    variants = {name: pd.read_csv(path).sort_values("seed") for name, path in args.entries}
    if "full_reference" not in variants:
        raise SystemExit("Entries must include full_reference=...")
    reference = variants["full_reference"]
    seed_rows = pd.concat(
        [table.assign(variant=name, variant_label=VARIANT_LABELS.get(name, name)) for name, table in variants.items()],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        [summary_row(name, table, reference, draws=args.bootstrap_draws) for name, table in variants.items()]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows.to_csv(args.out_dir / "validation_frozen_variant_seed_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "validation_frozen_variant_summary.csv", index=False)
    write_markdown(summary, args.out_dir / "validation_frozen_variant_summary.md")
    print(args.out_dir / "validation_frozen_variant_summary.csv")


if __name__ == "__main__":
    main()
