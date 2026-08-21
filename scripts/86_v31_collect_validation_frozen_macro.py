#!/usr/bin/env python
"""Recompute final-test macro evidence with validation-frozen normalizers.

This collector is intentionally separate from the historical V31 collectors.
Those collectors derived subtype normalizers from a final-test static replay
table.  The manuscript protocol instead specifies that the fixed-schedule
candidate set from the validation stage supplies those denominators.  This
script reads the saved rollouts only: it does not retrain a policy, refit an
oracle, or select a final-test policy as a primary reference.

The primary comparison is PD-PPO versus ``validation_selected_static``.  The
rule-based policies are reported both individually and as a clearly labelled
post-hoc strongest-per-seed diagnostic.  Replay and event-label results are
written separately because their action selection can use held-out information
or privileged labels.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("PD_PPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
SUBTYPES: dict[int, str] = {1: "particle", 2: "flux", 3: "thermal"}
LOSS_COLUMNS = tuple(f"oracle_loss_subtype_{label}" for label in SUBTYPES.values())
PRIMARY_POLICIES = ("validation_selected_static", "aoi", "round_robin", "random")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_truth_path(run_dir: Path, metadata: dict[str, Any]) -> Path:
    control_source = dict(metadata.get("control_source", {}))
    source_value = str(
        control_source.get("source_run_dir") or control_source.get("run_dir") or ""
    ).strip()
    source_dir = Path(source_value) if source_value else None
    truth_value = str(metadata.get("truth_csv", "")).strip()
    candidates = [run_dir / "truth_v31_split.csv"]
    if truth_value:
        truth_path = Path(truth_value)
        candidates.extend([truth_path, ROOT / truth_path])
    if source_dir is not None:
        candidates.extend([source_dir / "truth_v31_split.csv", ROOT / source_dir / "truth_v31_split.csv"])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        expected_hash = str(control_source.get("truth_sha256", "")).strip()
        if expected_hash and sha256_file(candidate) != expected_hash:
            raise ValueError(f"Control-source truth checksum mismatch: {candidate}")
        return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve source truth for {run_dir}; checked: {checked}")


def validation_normalizers(path: Path) -> dict[str, float]:
    table = pd.read_csv(path)
    result: dict[str, float] = {}
    for column in LOSS_COLUMNS:
        if column not in table:
            continue
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            result[column] = float(np.median(values))
    if len(result) != len(LOSS_COLUMNS):
        missing = sorted(set(LOSS_COLUMNS).difference(result))
        raise ValueError(f"Validation normalizers missing {missing} in {path}")
    return result


def macro_from_rollout(
    rollout_path: Path,
    truth_subtypes: np.ndarray,
    normalizers: dict[str, float],
) -> dict[str, float]:
    if not rollout_path.exists():
        return {}
    with np.load(rollout_path, allow_pickle=False) as data:
        losses = np.asarray(data["oracle_losses"], dtype=float).reshape(-1)
        steps = np.asarray(data["step_indices"], dtype=int).reshape(-1)
    if losses.size != steps.size:
        raise ValueError(f"Mismatched rollout arrays in {rollout_path}")
    valid = (steps >= 0) & (steps < truth_subtypes.size) & np.isfinite(losses)
    result: dict[str, float] = {}
    normalized: list[float] = []
    raw: list[float] = []
    for subtype_id, label in SUBTYPES.items():
        mask = valid & (truth_subtypes[steps.clip(0, truth_subtypes.size - 1)] == subtype_id)
        loss = float(losses[mask].mean()) if np.any(mask) else float("nan")
        column = f"oracle_loss_subtype_{label}"
        result[column] = loss
        result[f"steps_subtype_{label}"] = int(np.sum(mask))
        if np.isfinite(loss):
            raw.append(loss)
            normalized.append(loss / normalizers[column])
    result["oracle_loss_macro_subtype_event"] = finite_mean(raw)
    result["oracle_loss_macro_subtype_event_validationnorm"] = finite_mean(normalized)
    result["macro_subtype_event_count"] = int(len(normalized))
    return result


def macro_from_table_row(row: pd.Series, normalizers: dict[str, float]) -> dict[str, float]:
    raw: list[float] = []
    normalized: list[float] = []
    result: dict[str, float] = {}
    for column in LOSS_COLUMNS:
        value = as_float(row.get(column))
        result[column] = value
        if np.isfinite(value):
            raw.append(value)
            normalized.append(value / normalizers[column])
    result["oracle_loss_macro_subtype_event"] = finite_mean(raw)
    result["oracle_loss_macro_subtype_event_validationnorm"] = finite_mean(normalized)
    result["macro_subtype_event_count"] = int(len(normalized))
    return result


def metric_value(table: pd.DataFrame, policy: str, column: str = "oracle_loss_mean") -> float:
    matches = table[table["policy"].astype(str) == str(policy)]
    return as_float(matches.iloc[0].get(column)) if not matches.empty else float("nan")


def behavior_values(table: pd.DataFrame) -> dict[str, float]:
    matches = table[table["policy"].astype(str) == "custom_ppo"]
    if matches.empty:
        return {}
    row = matches.iloc[0]
    return {
        "warmup_abort_count": as_float(row.get("warmup_abort_count")),
        "switches_per_step": as_float(row.get("switches_per_step")),
        "always_on_sensor_count": as_float(row.get("always_on_sensor_count")),
        "always_off_sensor_count": as_float(row.get("always_off_sensor_count")),
        "mid_duty_sensor_count": as_float(row.get("mid_duty_sensor_count")),
    }


def bootstrap_interval(values: np.ndarray, *, seed: int, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def sign_test_p_all_positive(wins: int, total: int) -> float:
    if total <= 0 or wins < 0 or wins > total:
        return float("nan")
    return float(sum(math.comb(total, k) for k in range(wins, total + 1)) / (2**total))


def margin_summary(values: pd.Series, *, bootstrap_seed: int, bootstrap_samples: int) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    wins = int(np.sum(finite > 0.0))
    lo, hi = bootstrap_interval(finite, seed=bootstrap_seed, samples=bootstrap_samples)
    return {
        "available_seeds": int(finite.size),
        "wins": wins,
        "mean_margin": float(np.mean(finite)) if finite.size else float("nan"),
        "median_margin": float(np.median(finite)) if finite.size else float("nan"),
        "minimum_margin": float(np.min(finite)) if finite.size else float("nan"),
        "bootstrap_95_ci": [lo, hi],
        "one_sided_sign_test_p": sign_test_p_all_positive(wins, int(finite.size)),
    }


def column_or_nan(table: pd.DataFrame, column: str) -> pd.Series:
    if column in table.columns:
        return table[column]
    return pd.Series(np.nan, index=table.index, dtype=float)


def collect_run(run_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = parse_seed(run_dir)
    router_dir = run_dir / args.router_eval_dir
    replay_dir = run_dir / args.replay_dir
    metrics_path = router_dir / "v2_custom_ppo_metrics.csv"
    normalizer_path = run_dir / "validation_static_candidates.csv"
    metadata = read_json(run_dir / "v2_ppo_metadata.json")
    evaluation_metadata = read_json(router_dir / "v2_ppo_metadata.json")
    policy_cfg = dict(metadata.get("custom_ppo", {}))
    alert_cfg = dict(metadata.get("agent_alert_context", {}))
    if bool(policy_cfg.get("subtype_router_enabled", False)):
        raise ValueError(f"Primary evidence cannot use the hard subtype router: {run_dir}")
    if bool(alert_cfg.get("include_event_flag_in_state", True)):
        raise ValueError(f"Primary evidence exposes the exact event flag online: {run_dir}")
    if bool(alert_cfg.get("truth_event_labels_used_online", True)):
        raise ValueError(f"Primary evidence records online truth-label use: {run_dir}")
    truth_path = resolve_truth_path(run_dir, metadata)
    if not (metrics_path.exists() and normalizer_path.exists()):
        raise FileNotFoundError(f"Incomplete primary evidence for {run_dir}")

    metrics = pd.read_csv(metrics_path)
    truth = pd.read_csv(truth_path, usecols=["event_subtype_id"])
    truth_subtypes = truth["event_subtype_id"].to_numpy(dtype=int)
    normalizers = validation_normalizers(normalizer_path)
    selected_static = metadata.get("selected_static_reference", {})
    selected_action = int(selected_static.get("action_idx", -1))

    row: dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "router_eval_dir": args.router_eval_dir,
        "normalizer_source": "validation_static_candidates.csv median subtype losses",
        "validation_selected_static_action_idx": selected_action,
        "evaluation_steps": evaluation_metadata.get("eval_steps"),
        "evaluation_start_indices": json.dumps(evaluation_metadata.get("eval_start_indices", [])),
        "evaluation_scope_mode": dict(evaluation_metadata.get("evaluation_scope", {})).get("mode"),
        "evaluation_oracle_device": evaluation_metadata.get("oracle_inference_device"),
    }
    row.update({f"validation_normalizer_{column.removeprefix('oracle_loss_subtype_')}": value for column, value in normalizers.items()})
    row.update(behavior_values(metrics))

    macro_by_policy: dict[str, dict[str, float]] = {}
    for policy in (*PRIMARY_POLICIES, "custom_ppo"):
        result = macro_from_rollout(router_dir / f"rollout_{policy}.npz", truth_subtypes, normalizers)
        if result:
            macro_by_policy[policy] = result
            row[f"{policy}_step_loss"] = metric_value(metrics, policy)
            row.update({f"{policy}_{key}": value for key, value in result.items()})

    pdppo_macro = as_float(macro_by_policy.get("custom_ppo", {}).get("oracle_loss_macro_subtype_event_validationnorm"))
    pdppo_step = as_float(row.get("custom_ppo_step_loss"))
    for policy in PRIMARY_POLICIES:
        baseline_macro = as_float(macro_by_policy.get(policy, {}).get("oracle_loss_macro_subtype_event_validationnorm"))
        baseline_step = as_float(row.get(f"{policy}_step_loss"))
        row[f"macro_margin_pdppo_vs_{policy}"] = baseline_macro - pdppo_macro
        row[f"step_margin_pdppo_vs_{policy}"] = baseline_step - pdppo_step

    rule_rows = [
        (policy, as_float(macro_by_policy.get(policy, {}).get("oracle_loss_macro_subtype_event_validationnorm")))
        for policy in ("aoi", "round_robin", "random")
    ]
    rule_rows = [(policy, value) for policy, value in rule_rows if np.isfinite(value)]
    if rule_rows:
        best_policy, best_macro = min(rule_rows, key=lambda item: item[1])
        row["posthoc_best_rule_dynamic_policy"] = best_policy
        row["posthoc_best_rule_dynamic_macro"] = best_macro
        row["macro_margin_pdppo_vs_posthoc_best_rule_dynamic"] = best_macro - pdppo_macro

    diagnostics: dict[str, Any] = {
        "seed": seed,
        "run_dir": row["run_dir"],
        "diagnostic_scope": "held-out diagnostic; not a validation-selected primary reference",
    }
    replay_table_path = replay_dir / "split_static_candidate_event_table.csv"
    replay_summary = read_json(replay_dir / "split_replay_gate_summary.json")
    if replay_table_path.exists():
        replay_table = pd.read_csv(replay_table_path)
        scored = replay_table.copy()
        scored_values = [macro_from_table_row(item, normalizers) for _, item in scored.iterrows()]
        scored["validationnorm_macro"] = [item["oracle_loss_macro_subtype_event_validationnorm"] for item in scored_values]
        validation_action = scored[pd.to_numeric(scored["action_idx"], errors="coerce") == selected_action]
        if not validation_action.empty:
            score = as_float(validation_action.iloc[0]["validationnorm_macro"])
            diagnostics["validation_selected_static_replay_macro"] = score
            diagnostics["macro_margin_pdppo_vs_validation_selected_static_replay"] = score - pdppo_macro
        best_static = scored.loc[pd.to_numeric(scored["validationnorm_macro"], errors="coerce").idxmin()]
        best_static_score = as_float(best_static["validationnorm_macro"])
        diagnostics["posthoc_best_static_action_idx"] = int(best_static["action_idx"])
        diagnostics["posthoc_best_static_macro"] = best_static_score
        diagnostics["macro_margin_pdppo_vs_posthoc_best_static"] = best_static_score - pdppo_macro
    event_policy = str(replay_summary.get("best_replay_macro_subtype_policy", ""))
    if event_policy:
        event_rollout = replay_dir / f"rollout_{event_policy}.npz"
        event_macro = macro_from_rollout(event_rollout, truth_subtypes, normalizers)
        if event_macro:
            value = as_float(event_macro.get("oracle_loss_macro_subtype_event_validationnorm"))
            diagnostics["privileged_event_label_policy"] = event_policy
            diagnostics["privileged_event_label_macro"] = value
            diagnostics["macro_margin_privileged_event_label_vs_pdppo"] = pdppo_macro - value
    return row, diagnostics


def expand_runs(args: argparse.Namespace) -> list[Path]:
    runs: list[Path] = []
    for value in args.runs or []:
        runs.append(resolve_path(value))
    for pattern in args.run_glob or []:
        runs.extend(Path(value) for value in glob.glob(str(resolve_path(pattern))))
    unique = {str(path.resolve()): path.resolve() for path in runs}
    if args.seeds:
        allowed = set(args.seeds)
        unique = {key: path for key, path in unique.items() if parse_seed(path) in allowed}
    return sorted(unique.values(), key=lambda path: (parse_seed(path) or 10**9, path.name))


def write_markdown(path: Path, summary: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    lines = [
        "# Validation-Frozen Macro Evidence",
        "",
        "This report recomputes every macro score from final-test rollout losses using subtype denominators fixed by the validation static-candidate table. No model, forecaster, or reference action is fitted or selected from final-test losses for the primary comparison.",
        "",
        "## Primary Comparisons",
        "",
        "| Comparison | Wins | Mean margin | 95% bootstrap CI | Minimum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, title in (
        ("pdppo_vs_validation_selected_static", "PD-PPO vs validation-selected static"),
        ("pdppo_vs_aoi", "PD-PPO vs AoI"),
        ("pdppo_vs_round_robin", "PD-PPO vs round robin"),
        ("pdppo_vs_random", "PD-PPO vs random"),
        ("pdppo_vs_posthoc_best_rule_dynamic", "PD-PPO vs post-hoc strongest rule dynamic"),
    ):
        item = summary.get(key, {})
        ci = item.get("bootstrap_95_ci", [float("nan"), float("nan")])
        lines.append(
            f"| {title} | {item.get('wins', 0)}/{item.get('available_seeds', 0)} | "
            f"{item.get('mean_margin', float('nan')):.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | "
            f"{item.get('minimum_margin', float('nan')):.6f} |"
        )
    lines += [
        "",
        "## Diagnostic Boundaries",
        "",
        "The fixed static replay is labelled post-hoc because it ranks constant actions on held-out loss. The event-label diagnostic is privileged because it has access to simulator event labels. Neither diagnostic defines the primary result.",
        "",
        f"- Primary seed count: {summary.get('seed_count', 0)}",
        f"- Action-trace operational rows available: {summary.get('behavior_available_seeds', 0)}",
        f"- Privileged event-label rows available: {diagnostics.get('privileged_event_label_available_seeds', 0)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--run-glob", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--router-eval-dir")
    parser.add_argument("--replay-dir", default="replay_gate_explicit_static_noguard")
    parser.add_argument(
        "--seed-metrics-csv",
        help="Use seed-level rows extracted from a read-only remote collector instead of reading run directories.",
    )
    parser.add_argument(
        "--diagnostic-csv",
        help="Optional diagnostic rows paired with --seed-metrics-csv.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed_metrics_csv:
        seed_table = pd.read_csv(resolve_path(args.seed_metrics_csv)).sort_values("seed")
        diagnostic_table = (
            pd.read_csv(resolve_path(args.diagnostic_csv)).sort_values("seed")
            if args.diagnostic_csv
            else pd.DataFrame({"seed": seed_table["seed"]})
        )
        if args.seeds:
            allowed = set(args.seeds)
            seed_table = seed_table[seed_table["seed"].isin(allowed)].copy()
            diagnostic_table = diagnostic_table[diagnostic_table["seed"].isin(allowed)].copy()
    else:
        if not args.router_eval_dir:
            raise SystemExit("--router-eval-dir is required when collecting directly from run directories.")
        runs = expand_runs(args)
        if not runs:
            raise SystemExit("No run directories matched.")
        seed_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        for run_dir in runs:
            row, diagnostic = collect_run(run_dir, args)
            seed_rows.append(row)
            diagnostic_rows.append(diagnostic)
        seed_table = pd.DataFrame(seed_rows).sort_values("seed")
        diagnostic_table = pd.DataFrame(diagnostic_rows).sort_values("seed")
    summary = {
        "metric": "equal-weight event-subtype macro loss normalized by validation static-candidate medians",
        "normalizer_source": "validation_static_candidates.csv",
        "primary_reference": "validation_selected_static",
        "seed_count": int(len(seed_table)),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "pdppo_vs_validation_selected_static": margin_summary(column_or_nan(seed_table, "macro_margin_pdppo_vs_validation_selected_static"), bootstrap_seed=args.bootstrap_seed + 1, bootstrap_samples=args.bootstrap_samples),
        "pdppo_vs_aoi": margin_summary(column_or_nan(seed_table, "macro_margin_pdppo_vs_aoi"), bootstrap_seed=args.bootstrap_seed + 2, bootstrap_samples=args.bootstrap_samples),
        "pdppo_vs_round_robin": margin_summary(column_or_nan(seed_table, "macro_margin_pdppo_vs_round_robin"), bootstrap_seed=args.bootstrap_seed + 3, bootstrap_samples=args.bootstrap_samples),
        "pdppo_vs_random": margin_summary(column_or_nan(seed_table, "macro_margin_pdppo_vs_random"), bootstrap_seed=args.bootstrap_seed + 4, bootstrap_samples=args.bootstrap_samples),
        "pdppo_vs_posthoc_best_rule_dynamic": margin_summary(column_or_nan(seed_table, "macro_margin_pdppo_vs_posthoc_best_rule_dynamic"), bootstrap_seed=args.bootstrap_seed + 5, bootstrap_samples=args.bootstrap_samples),
        "step_pdppo_vs_validation_selected_static": margin_summary(column_or_nan(seed_table, "step_margin_pdppo_vs_validation_selected_static"), bootstrap_seed=args.bootstrap_seed + 9, bootstrap_samples=args.bootstrap_samples),
        "behavior_available_seeds": int(column_or_nan(seed_table, "mid_duty_sensor_count").notna().sum()),
        "behavior_all_zero_abort": bool((column_or_nan(seed_table, "warmup_abort_count") == 0.0).all()),
        "behavior_mid_duty_sensor_range": [
            float(column_or_nan(seed_table, "mid_duty_sensor_count").min()),
            float(column_or_nan(seed_table, "mid_duty_sensor_count").max()),
        ],
        "behavior_always_on_sensor_range": [
            float(column_or_nan(seed_table, "always_on_sensor_count").min()),
            float(column_or_nan(seed_table, "always_on_sensor_count").max()),
        ],
        "behavior_always_off_sensor_range": [
            float(column_or_nan(seed_table, "always_off_sensor_count").min()),
            float(column_or_nan(seed_table, "always_off_sensor_count").max()),
        ],
        "behavior_switches_per_step_range": [
            float(column_or_nan(seed_table, "switches_per_step").min()),
            float(column_or_nan(seed_table, "switches_per_step").max()),
        ],
    }
    privileged_vs_static = (
        column_or_nan(diagnostic_table, "validation_selected_static_replay_macro")
        - column_or_nan(diagnostic_table, "privileged_event_label_macro")
    )
    diagnostic_summary = {
        "scope": "diagnostic only; final-test action selection or privileged event labels prevent primary use",
        "posthoc_best_static": margin_summary(column_or_nan(diagnostic_table, "macro_margin_pdppo_vs_posthoc_best_static"), bootstrap_seed=args.bootstrap_seed + 6, bootstrap_samples=args.bootstrap_samples),
        "validation_selected_static_replay": margin_summary(column_or_nan(diagnostic_table, "macro_margin_pdppo_vs_validation_selected_static_replay"), bootstrap_seed=args.bootstrap_seed + 7, bootstrap_samples=args.bootstrap_samples),
        "privileged_event_label_available_seeds": int(column_or_nan(diagnostic_table, "privileged_event_label_macro").notna().sum()),
        "privileged_event_label_vs_pdppo": margin_summary(column_or_nan(diagnostic_table, "macro_margin_privileged_event_label_vs_pdppo"), bootstrap_seed=args.bootstrap_seed + 8, bootstrap_samples=args.bootstrap_samples),
        "privileged_event_label_vs_validation_selected_static_replay": margin_summary(privileged_vs_static, bootstrap_seed=args.bootstrap_seed + 10, bootstrap_samples=args.bootstrap_samples),
    }
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_table.to_csv(out_dir / "validation_frozen_seed_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    diagnostic_table.to_csv(out_dir / "validation_frozen_privileged_diagnostics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    (out_dir / "validation_frozen_claim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "validation_frozen_diagnostic_summary.json").write_text(json.dumps(diagnostic_summary, indent=2), encoding="utf-8")
    write_markdown(out_dir / "validation_frozen_claim_summary.md", summary, diagnostic_summary)
    print(out_dir)
    print(json.dumps(summary, indent=2))
    print(json.dumps(diagnostic_summary, indent=2))


if __name__ == "__main__":
    main()
