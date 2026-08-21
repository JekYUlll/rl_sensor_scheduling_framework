#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DYNAMIC = ("round_robin", "aoi", "random")
STATIC_FAMILY = ("validation_selected_static", "feasible_static_projected", "oracle_static_projected")
DEPLOYABLE_STATIC = (
    "duty_constrained_validation_selected_static",
    "duty_constrained_feasible_static_projected",
    "duty_constrained_oracle_static_projected",
)
REFERENCE_POLICIES = ("full_open_unconstrained",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect operational PD-PPO split-protocol metrics from per-seed CSVs."
    )
    parser.add_argument(
        "--base-dir",
        default="reports/v31_split_protocol_no_warmup_hguard_envdwell12_reduced",
        help="Experiment directory containing raw/budget*_seed*/v2_custom_ppo_metrics.csv.",
    )
    parser.add_argument("--budget-label", default="budget1p70")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument(
        "--out-name",
        default="operational_pdppo_summary.csv",
        help="Per-seed summary CSV name written under base-dir.",
    )
    return parser.parse_args()


def resolve_base(path: str) -> Path:
    base = Path(path)
    if not base.is_absolute():
        base = ROOT / base
    return base


def best_row(by_policy: dict[str, pd.Series], names: tuple[str, ...]) -> pd.Series | None:
    rows = [by_policy[name] for name in names if name in by_policy]
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["oracle_loss_mean"]))


def first_row(by_policy: dict[str, pd.Series], names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in by_policy:
            return by_policy[name]
    return None


def best_prefixed_row(
    by_policy: dict[str, pd.Series],
    *,
    prefix: str,
    exclude: set[str] | None = None,
) -> pd.Series | None:
    exclude = exclude or set()
    rows = [
        row
        for name, row in by_policy.items()
        if name.startswith(prefix) and name not in exclude
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["oracle_loss_mean"]))


def loss(row: pd.Series | None) -> float:
    return float(row["oracle_loss_mean"]) if row is not None else float("nan")


def policy(row: pd.Series | None) -> str:
    return str(row["policy"]) if row is not None else ""


def beats(a: pd.Series, b: pd.Series | None) -> bool:
    return bool(b is not None and float(a["oracle_loss_mean"]) < float(b["oracle_loss_mean"]))


def collect_seed(base_dir: Path, budget_label: str, seed: int) -> dict[str, object]:
    metrics_path = base_dir / "raw" / f"{budget_label}_seed{seed}" / "v2_custom_ppo_metrics.csv"
    row: dict[str, object] = {
        "seed": int(seed),
        "metrics_path": str(metrics_path),
        "complete": bool(metrics_path.exists()),
    }
    if not metrics_path.exists():
        return row

    df = pd.read_csv(metrics_path)
    by_policy = {str(item["policy"]): item for _, item in df.iterrows()}
    pdppo = by_policy.get("custom_ppo")
    if pdppo is None:
        row["complete"] = False
        row["missing_custom_ppo"] = True
        return row

    best_static = best_row(by_policy, STATIC_FAMILY)
    best_original_dynamic = best_row(by_policy, ORIGINAL_DYNAMIC)
    selected_static = first_row(by_policy, ("validation_selected_static", "oracle_static_projected"))
    deployable_selected_static = by_policy.get("duty_constrained_validation_selected_static")
    best_deployable_static = best_row(by_policy, DEPLOYABLE_STATIC)
    duty_exclude = {"duty_constrained_validation_selected_static"}
    best_duty_non_pdppo = best_prefixed_row(
        by_policy,
        prefix="duty_constrained_",
        exclude=duty_exclude,
    )
    full_open = best_row(by_policy, REFERENCE_POLICIES)

    row.update(
        {
            "pdppo_oracle_loss": loss(pdppo),
            "pdppo_mid": int(pdppo.get("mid_duty_sensor_count", -1)),
            "pdppo_always_on": int(pdppo.get("always_on_sensor_count", -1)),
            "pdppo_always_off": int(pdppo.get("always_off_sensor_count", -1)),
            "pdppo_switches_per_step": float(pdppo.get("switches_per_step", float("nan"))),
            "pdppo_warmup_abort_count": int(pdppo.get("warmup_abort_count", -1)),
            "pdppo_duty_min": float(pdppo.get("duty_min", float("nan"))),
            "pdppo_duty_max": float(pdppo.get("duty_max", float("nan"))),
            "full_open_policy": policy(full_open),
            "full_open_oracle_loss": loss(full_open),
            "best_static_policy": policy(best_static),
            "best_static_oracle_loss": loss(best_static),
            "selected_static_policy": policy(selected_static),
            "selected_static_oracle_loss": loss(selected_static),
            "deployable_selected_static_policy": policy(deployable_selected_static),
            "deployable_selected_static_oracle_loss": loss(deployable_selected_static),
            "best_deployable_static_policy": policy(best_deployable_static),
            "best_deployable_static_oracle_loss": loss(best_deployable_static),
            "best_original_dynamic_policy": policy(best_original_dynamic),
            "best_original_dynamic_oracle_loss": loss(best_original_dynamic),
            "best_duty_non_pdppo_policy": policy(best_duty_non_pdppo),
            "best_duty_non_pdppo_oracle_loss": loss(best_duty_non_pdppo),
            "pdppo_beats_full_open": beats(pdppo, full_open),
            "pdppo_beats_best_static": beats(pdppo, best_static),
            "pdppo_beats_selected_static": beats(pdppo, selected_static),
            "pdppo_beats_deployable_selected_static": beats(pdppo, deployable_selected_static),
            "pdppo_beats_best_deployable_static": beats(pdppo, best_deployable_static),
            "pdppo_beats_best_original_dynamic": beats(pdppo, best_original_dynamic),
            "pdppo_beats_best_duty_non_pdppo": beats(pdppo, best_duty_non_pdppo),
            "pdppo_valid_behavior": bool(
                int(pdppo.get("mid_duty_sensor_count", -1)) == 8
                and int(pdppo.get("always_on_sensor_count", -1)) == 0
                and int(pdppo.get("always_off_sensor_count", -1)) == 0
                and int(pdppo.get("warmup_abort_count", -1)) == 0
            ),
        }
    )
    return row


def comparison_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for flag, loss_col in (
        ("pdppo_beats_full_open", "full_open_oracle_loss"),
        ("pdppo_beats_best_static", "best_static_oracle_loss"),
        ("pdppo_beats_selected_static", "selected_static_oracle_loss"),
        ("pdppo_beats_deployable_selected_static", "deployable_selected_static_oracle_loss"),
        ("pdppo_beats_best_deployable_static", "best_deployable_static_oracle_loss"),
        ("pdppo_beats_best_original_dynamic", "best_original_dynamic_oracle_loss"),
        ("pdppo_beats_best_duty_non_pdppo", "best_duty_non_pdppo_oracle_loss"),
    ):
        if flag not in summary.columns or loss_col not in summary.columns:
            continue
        subset = summary[summary[loss_col].notna()]
        if subset.empty:
            continue
        delta = subset[loss_col].astype(float) - subset["pdppo_oracle_loss"].astype(float)
        rows.append(
            {
                "comparison": flag.removeprefix("pdppo_beats_"),
                "n": int(len(subset)),
                "pdppo_win_count": int(subset[flag].astype(bool).sum()),
                "mean_delta_baseline_minus_pdppo": float(delta.mean()),
                "min_delta_baseline_minus_pdppo": float(delta.min()),
                "max_delta_baseline_minus_pdppo": float(delta.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_dir = resolve_base(args.base_dir)
    rows = [collect_seed(base_dir, str(args.budget_label), int(seed)) for seed in args.seeds]
    summary = pd.DataFrame(rows)
    out_path = base_dir / str(args.out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    comparison = comparison_summary(summary[summary.get("complete", False) == True])  # noqa: E712
    comparison_path = out_path.with_name(out_path.stem + "_comparisons.csv")
    comparison.to_csv(comparison_path, index=False)
    complete = int(summary["complete"].sum()) if "complete" in summary else 0
    print(f"complete_seeds={complete}/{len(summary)}")
    print(out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path)
    print(comparison_path.relative_to(ROOT) if comparison_path.is_relative_to(ROOT) else comparison_path)
    if not comparison.empty:
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
