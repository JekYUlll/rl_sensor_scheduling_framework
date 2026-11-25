#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_SCORE = "oracle_loss_macro_subtype_event_staticnorm"
BEHAVIOR_COLUMNS = (
    "switches_per_step",
    "warmup_abort_count",
    "always_on_sensor_count",
    "always_off_sensor_count",
    "mid_duty_sensor_count",
    "duty_entropy",
    "duty_min",
    "duty_max",
)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows)) if rows else len(columns[idx])
        for idx in range(len(columns))
    ]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    header = render(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([header, separator, *(render(row) for row in rows)])


def bootstrap_mean_ci(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.mean(rng.choice(data, size=(int(draws), int(data.size)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def find_policy(metrics: pd.DataFrame, policy: str) -> pd.Series:
    rows = metrics.loc[metrics["policy"].astype(str) == str(policy)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {policy!r} row, found {len(rows)}")
    return rows.iloc[0]


def normalized_protocol_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Remove only fields expected to differ between matched reward runs."""

    normalized = deepcopy(metadata)
    normalized.pop("model_path", None)
    normalized.pop("oracle_path", None)
    custom_ppo = dict(normalized.get("custom_ppo", {}))
    custom_ppo.pop("history_path", None)
    normalized["custom_ppo"] = custom_ppo
    reward = dict(normalized.get("reward_shaping", {}))
    reward.pop("reward_proxy_mode", None)
    reward.pop("reward_staticnorm_candidates_path", None)
    normalized["reward_shaping"] = reward
    return normalized


def protocol_sha256(metadata: dict[str, object]) -> str:
    payload = json.dumps(
        normalized_protocol_metadata(metadata),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect strict matched PPO reward controls.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--date-tag", default="20260718pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[117, 118])
    parser.add_argument("--modes", nargs="+", default=["forecast", "aoi", "uncertainty"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    protocol_by_seed: dict[int, dict[str, object]] = {}
    metadata_fingerprint_by_seed: dict[int, str] = {}
    for mode in args.modes:
        for seed in args.seeds:
            run_dir = args.reports_root / (
                f"v31_scenebal2_matched_reward_{mode}_noexactevent_seed{seed}_"
                f"h075{mode}ctrl_{args.date_tag}"
            )
            metrics_path = run_dir / "v2_custom_ppo_metrics.csv"
            metadata_path = run_dir / "v2_ppo_metadata.json"
            if not metrics_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Incomplete matched control: {run_dir}")
            metrics = pd.read_csv(metrics_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            control = dict(metadata.get("control_source", {}))
            alert = dict(metadata.get("agent_alert_context", {}))
            policy_cfg = dict(metadata.get("custom_ppo", {}))
            if not bool(control.get("enabled", False)):
                raise ValueError(f"{run_dir} did not record frozen source reuse")
            if str(metadata.get("reward_shaping", {}).get("reward_proxy_mode")) != str(mode):
                raise ValueError(f"{run_dir} reward mode does not match its label")
            if bool(alert.get("include_event_flag_in_state", True)):
                raise ValueError(f"{run_dir} exposes the exact event flag online")
            if bool(alert.get("truth_event_labels_used_online", True)):
                raise ValueError(f"{run_dir} records final-policy access to truth event labels")
            if bool(policy_cfg.get("subtype_router_enabled", False)):
                raise ValueError(f"{run_dir} uses the hard subtype-to-action router")
            if str(control.get("source_oracle_sha256")) != str(control.get("copied_oracle_sha256")):
                raise ValueError(f"{run_dir} frozen evaluator checksum mismatch")
            metadata_fingerprint = protocol_sha256(metadata)
            previous_fingerprint = metadata_fingerprint_by_seed.get(int(seed))
            if previous_fingerprint is not None and previous_fingerprint != metadata_fingerprint:
                raise ValueError(
                    f"Matched-control metadata differs beyond reward mode and output paths for seed {seed}"
                )
            metadata_fingerprint_by_seed[int(seed)] = metadata_fingerprint

            protocol = {
                "truth_sha256": str(control.get("truth_sha256")),
                "oracle_sha256": str(control.get("source_oracle_sha256")),
                "eval_start_indices": tuple(int(value) for value in metadata.get("eval_start_indices", ())),
                "static_selection_start_indices": tuple(
                    int(value)
                    for value in dict(metadata.get("partition_protocol", {})).get("static_selection_start_indices", ())
                ),
                "candidate_count": int(dict(metadata.get("custom_ppo", {})).get("candidate_count", -1)),
                "context_encoder_enabled": bool(policy_cfg.get("context_encoder_enabled", False)),
                "context_feature_dim": int(policy_cfg.get("context_feature_dim", 0)),
                "normalized_metadata_sha256": metadata_fingerprint,
            }
            if int(seed) in protocol_by_seed and protocol_by_seed[int(seed)] != protocol:
                raise ValueError(f"Matched-control protocol differs across reward modes for seed {seed}")
            protocol_by_seed[int(seed)] = protocol

            learned = find_policy(metrics, "custom_ppo")
            static = find_policy(metrics, "validation_selected_static")
            row: dict[str, object] = {
                "mode": str(mode),
                "seed": int(seed),
                "run_dir": str(run_dir),
                "oracle_loss_mean": float(learned["oracle_loss_mean"]),
                "macro_score": float(learned[PRIMARY_SCORE]),
                "static_macro_score": float(static[PRIMARY_SCORE]),
                "margin_vs_static": float(static[PRIMARY_SCORE] - learned[PRIMARY_SCORE]),
                "truth_sha256": protocol["truth_sha256"],
                "oracle_sha256": protocol["oracle_sha256"],
            }
            for column in BEHAVIOR_COLUMNS:
                row[column] = float(learned[column]) if column in learned.index else float("nan")
            rows.append(row)

    seed_metrics = pd.DataFrame(rows).sort_values(["seed", "mode"]).reset_index(drop=True)
    forecast = seed_metrics.loc[
        seed_metrics["mode"] == "forecast",
        ["seed", "oracle_loss_mean", "macro_score"],
    ].rename(
        columns={
            "oracle_loss_mean": "forecast_oracle_loss_mean",
            "macro_score": "forecast_macro_score",
        }
    )
    seed_metrics = seed_metrics.merge(forecast, on="seed", how="left", validate="many_to_one")
    seed_metrics["forecast_step_margin_vs_mode"] = (
        seed_metrics["oracle_loss_mean"] - seed_metrics["forecast_oracle_loss_mean"]
    )
    seed_metrics["forecast_margin_vs_mode"] = seed_metrics["macro_score"] - seed_metrics["forecast_macro_score"]

    summary_rows = []
    for mode, group in seed_metrics.groupby("mode", sort=False):
        step_margins = group["forecast_step_margin_vs_mode"].to_numpy(dtype=float)
        margins = group["forecast_margin_vs_mode"].to_numpy(dtype=float)
        static_margins = group["margin_vs_static"].to_numpy(dtype=float)
        step_ci_low, step_ci_high = bootstrap_mean_ci(
            step_margins,
            draws=int(args.bootstrap_draws),
            seed=71_800,
        )
        ci_low, ci_high = bootstrap_mean_ci(margins, draws=int(args.bootstrap_draws), seed=71_801)
        static_ci_low, static_ci_high = bootstrap_mean_ci(
            static_margins,
            draws=int(args.bootstrap_draws),
            seed=71_802,
        )
        summary_rows.append(
            {
                "mode": str(mode),
                "n_seeds": int(len(group)),
                "oracle_loss_mean": float(group["oracle_loss_mean"].mean()),
                "forecast_step_margin_vs_mode_mean": float(np.mean(step_margins)),
                "forecast_step_margin_vs_mode_ci95_low": step_ci_low,
                "forecast_step_margin_vs_mode_ci95_high": step_ci_high,
                "forecast_step_wins": int(np.sum(step_margins > 0.0)),
                "macro_score_mean": float(group["macro_score"].mean()),
                "forecast_margin_vs_mode_mean": float(np.mean(margins)),
                "forecast_margin_vs_mode_ci95_low": ci_low,
                "forecast_margin_vs_mode_ci95_high": ci_high,
                "forecast_wins": int(np.sum(margins > 0.0)),
                "ties": int(np.sum(np.isclose(margins, 0.0))),
                "margin_vs_static_mean": float(np.mean(static_margins)),
                "margin_vs_static_ci95_low": static_ci_low,
                "margin_vs_static_ci95_high": static_ci_high,
                "wins_vs_static": int(np.sum(static_margins > 0.0)),
                "warmup_abort_total": int(np.nansum(group["warmup_abort_count"])),
                "switches_per_step_mean": float(group["switches_per_step"].mean()),
                "always_on_sensor_count_mean": float(group["always_on_sensor_count"].mean()),
                "always_off_sensor_count_mean": float(group["always_off_sensor_count"].mean()),
                "mid_duty_sensor_count_mean": float(group["mid_duty_sensor_count"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(out_dir / "matched_reward_seed_metrics.csv", index=False)
    summary.to_csv(out_dir / "matched_reward_summary.csv", index=False)
    audit = {
        "status": "passed",
        "primary_score": PRIMARY_SCORE,
        "date_tag": str(args.date_tag),
        "seeds": [int(value) for value in args.seeds],
        "modes": [str(value) for value in args.modes],
            "protocol_by_seed": {
            str(seed): {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in payload.items()
            }
            for seed, payload in protocol_by_seed.items()
        },
    }
    (out_dir / "matched_reward_protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# Matched reward-control summary",
        "",
        "Positive `forecast_step_margin_vs_mode` or `forecast_margin_vs_mode` means forecast-reward PPO has lower final frozen-forecaster loss.",
        "All variants reuse the same seed-specific truth, frozen evaluator, candidate masks, validation selection, and final windows.",
        "The final policy does not receive exact simulator event labels.",
        "",
        dataframe_to_markdown(summary),
    ]
    (out_dir / "matched_reward_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "matched_reward_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
