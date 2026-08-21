#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect split-protocol energy-account seed results."
    )
    parser.add_argument(
        "--base-dir",
        default="reports/energy_account_split_protocol_gate_semimarkov",
        help="Directory containing budget1p20_seed*/ outputs.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--budget-label", default="budget1p20")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <base-dir>/aggregate.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_seed(seed_dir: Path, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    metrics_path = seed_dir / "v2_custom_ppo_metrics.csv"
    eval_path = seed_dir / "evaluation" / "v2_eval_overall.csv"
    manifest_path = seed_dir / "split_protocol_manifest.json"
    status = {
        "seed": int(seed),
        "seed_dir": str(seed_dir),
        "complete": False,
        "missing": [],
    }
    for path in (metrics_path, eval_path, manifest_path):
        if not path.exists():
            status["missing"].append(str(path))
    if status["missing"]:
        return [], status

    metrics = pd.read_csv(metrics_path)
    eval_overall = pd.read_csv(eval_path)
    manifest = read_json(manifest_path)
    final_test = manifest.get("final_test", {})
    ppo_controls = manifest.get("ppo_controls", {})
    energy_account = manifest.get("energy_account", {})
    rows: list[dict[str, object]] = []
    for _, item in metrics.iterrows():
        policy = str(item["policy"])
        eval_row = eval_overall.loc[eval_overall["policy"].astype(str) == policy]
        extra = eval_row.iloc[0].to_dict() if not eval_row.empty else {}
        row = {
            "seed": int(seed),
            "policy": policy,
            "oracle_loss_mean": float(item["oracle_loss_mean"]),
            "reward_mean": float(item["reward_mean"]),
            "power_mean": float(item["power_mean"]),
            "peak_power_max": float(item["peak_power_max"]),
            "warmup_abort_count": int(item["warmup_abort_count"]),
            "event_rate": float(item["event_rate"]),
            "final_selected_event_rate_mean": float(
                final_test.get("selected_event_rate_mean", float("nan"))
            ),
            "protocol": str(manifest.get("protocol", "")),
            "evidence_role": str(manifest.get("evidence_role", "")),
            "lambda_warmup_abort": float(ppo_controls.get("lambda_warmup_abort", float("nan"))),
            "soc_aux_horizon": int(ppo_controls.get("soc_aux_horizon", 0) or 0),
            "soc_aux_coef": float(ppo_controls.get("soc_aux_coef", float("nan"))),
            "soc_soft_penalty_buffer": float(
                energy_account.get("soc_soft_penalty_buffer", float("nan"))
            ),
            "lambda_soc_soft_penalty": float(
                energy_account.get("lambda_soc_soft_penalty", float("nan"))
            ),
        }
        for key in (
            "weighted_normalized_mae",
            "forecast_weighted_mae_overall",
            "forecast_weighted_mae_event",
            "forecast_weighted_mae_non_event",
            "warmup_abort_rate",
            "steady_violation_rate",
            "peak_violation_rate",
        ):
            if key in extra:
                row[key] = float(extra[key])
        rows.append(row)
    status["complete"] = True
    return rows, status


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "oracle_loss_mean",
        "reward_mean",
        "power_mean",
        "peak_power_max",
        "warmup_abort_count",
        "event_rate",
        "final_selected_event_rate_mean",
        "weighted_normalized_mae",
    ]
    present = [col for col in numeric if col in rows.columns]
    summary = (
        rows.groupby("policy", as_index=False)[present]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        for col in summary.columns.to_flat_index()
    ]
    return summary.sort_values("oracle_loss_mean_mean")


def compare(rows: pd.DataFrame) -> pd.DataFrame:
    pivot = rows.pivot(index="seed", columns="policy", values="oracle_loss_mean")
    if "custom_ppo" not in pivot.columns:
        return pd.DataFrame()
    out = []
    for comparator in (
        "validation_selected_static",
        "round_robin",
        "aoi",
        "feasible_static_projected",
        "random",
    ):
        if comparator not in pivot.columns:
            continue
        pair = pivot[["custom_ppo", comparator]].dropna()
        if pair.empty:
            continue
        delta = pair["custom_ppo"] - pair[comparator]
        out.append(
            {
                "comparator": comparator,
                "n": int(len(delta)),
                "custom_better_count": int((delta < 0).sum()),
                "custom_worse_count": int((delta > 0).sum()),
                "custom_tie_count": int((delta == 0).sum()),
                "mean_delta_custom_minus_comparator": float(delta.mean()),
                "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "mean_percent_delta": float(((pair["custom_ppo"] / pair[comparator]) - 1.0).mean() * 100.0),
            }
        )
    return pd.DataFrame(out).sort_values("comparator")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT / base_dir
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "aggregate"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    statuses = []
    for seed in args.seeds:
        seed_dir = base_dir / f"{args.budget_label}_seed{seed}"
        seed_rows, status = collect_seed(seed_dir, int(seed))
        rows.extend(seed_rows)
        statuses.append(status)

    rows_df = pd.DataFrame(rows)
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out_dir / "energy_split_seed_status.csv", index=False)
    (out_dir / "energy_split_seed_status.json").write_text(
        json.dumps(statuses, indent=2), encoding="utf-8"
    )
    if rows_df.empty:
        print(f"No complete seeds found under {base_dir}")
        return

    rows_df.to_csv(out_dir / "energy_split_long.csv", index=False)
    summarize(rows_df).to_csv(out_dir / "energy_split_policy_summary.csv", index=False)
    compare(rows_df).to_csv(out_dir / "energy_split_custom_comparisons.csv", index=False)
    complete = int(status_df["complete"].sum())
    print(f"complete_seeds={complete}/{len(status_df)}")
    print((out_dir / "energy_split_policy_summary.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
