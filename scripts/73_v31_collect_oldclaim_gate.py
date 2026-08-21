#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STATIC_POLICIES = (
    "validation_selected_static",
    "feasible_static_projected",
    "oracle_static_projected",
)
RULE_DYNAMIC_POLICIES = ("round_robin", "aoi", "random")
RULE_DYNAMIC_PREFIXES = ("dwell", "duty_dwell")
STEP_SCORE = "oracle_loss_mean"
MACRO_SCORE = "oracle_loss_macro_subtype_event_staticnorm"
RAW_MACRO_SCORE = "oracle_loss_macro_subtype_event"


_METPAIR_HELPERS: Any | None = None


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def metpair_helpers() -> Any | None:
    global _METPAIR_HELPERS
    if _METPAIR_HELPERS is not None:
        return _METPAIR_HELPERS
    helper_path = ROOT / "scripts" / "72_v31_collect_metpair_strongclaim.py"
    if not helper_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("v31_metpair_strongclaim_helpers", helper_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _METPAIR_HELPERS = module
    return module


def rollout_macro_replay_scale(
    run_dir: Path,
    *,
    replay_dir: str,
    metrics_eval_dir: str,
    policy: str,
    score_col: str,
) -> float:
    helper = metpair_helpers()
    if helper is None:
        return float("nan")
    try:
        static_candidate_metrics = helper.add_macro_subtype_column(
            helper.read_csv(run_dir / replay_dir / "split_static_candidate_event_table.csv")
        )
        normalizers = helper.subtype_static_normalizers(static_candidate_metrics)
        truth = helper.truth_for_run(run_dir)
        return float(
            helper.rollout_macro_for_policy(
                run_dir,
                router_eval_dir=str(metrics_eval_dir),
                policy=str(policy),
                truth=truth,
                score_col=str(score_col),
                normalizers=normalizers,
            )
        )
    except Exception:
        return float("nan")


def ensure_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if RAW_MACRO_SCORE not in result.columns:
        cols = ["oracle_loss_subtype_particle", "oracle_loss_subtype_flux", "oracle_loss_subtype_thermal"]
        if all(col in result.columns for col in cols):
            result[RAW_MACRO_SCORE] = result[cols].apply(lambda row: finite_mean([as_float(v) for v in row]), axis=1)
    if MACRO_SCORE not in result.columns:
        cols = [
            "oracle_loss_subtype_particle_staticnorm",
            "oracle_loss_subtype_flux_staticnorm",
            "oracle_loss_subtype_thermal_staticnorm",
        ]
        if all(col in result.columns for col in cols):
            result[MACRO_SCORE] = result[cols].apply(lambda row: finite_mean([as_float(v) for v in row]), axis=1)
    return result


def row_for_policy(df: pd.DataFrame, policy: str) -> pd.Series | None:
    if "policy" not in df.columns:
        return None
    subset = df[df["policy"].astype(str) == policy]
    return None if subset.empty else subset.iloc[0]


def best_row(df: pd.DataFrame, policies: tuple[str, ...], score_col: str) -> pd.Series | None:
    if "policy" not in df.columns or score_col not in df.columns:
        return None
    subset = df[df["policy"].astype(str).isin(policies)].copy()
    if subset.empty:
        return None
    subset[score_col] = pd.to_numeric(subset[score_col], errors="coerce")
    subset = subset[np.isfinite(subset[score_col])]
    return None if subset.empty else subset.sort_values(score_col).iloc[0]


def is_rule_dynamic_policy(name: str) -> bool:
    value = str(name)
    if value in RULE_DYNAMIC_POLICIES:
        return True
    if value.startswith(RULE_DYNAMIC_PREFIXES):
        return any(value.endswith(f"_{policy}") for policy in RULE_DYNAMIC_POLICIES)
    if value.startswith("duty_constrained_"):
        base = value.removeprefix("duty_constrained_")
        return base in RULE_DYNAMIC_POLICIES
    return False


def best_rule_dynamic_row(df: pd.DataFrame, score_col: str) -> pd.Series | None:
    if "policy" not in df.columns or score_col not in df.columns:
        return None
    subset = df[df["policy"].astype(str).map(is_rule_dynamic_policy)].copy()
    if subset.empty:
        return None
    subset[score_col] = pd.to_numeric(subset[score_col], errors="coerce")
    subset = subset[np.isfinite(subset[score_col])]
    return None if subset.empty else subset.sort_values(score_col).iloc[0]


def first_existing_row(df: pd.DataFrame, policies: tuple[str, ...]) -> pd.Series | None:
    for policy in policies:
        row = row_for_policy(df, policy)
        if row is not None:
            return row
    return None


def policy_name(row: pd.Series | None) -> str:
    return str(row["policy"]) if row is not None and "policy" in row.index else ""


def score(row: pd.Series | None, score_col: str) -> float:
    return as_float(row[score_col]) if row is not None and score_col in row.index else float("nan")


def margin_gate(margin: float, baseline_loss: float, min_abs: float, min_rel: float) -> bool:
    if not (np.isfinite(margin) and np.isfinite(baseline_loss)):
        return False
    required = max(float(min_abs), float(min_rel) * float(baseline_loss))
    return bool(margin >= required)


def load_behavior(run_dir: Path, behavior_dir: str, behavior_eval_dir: str = ".") -> dict[str, Any]:
    path = run_dir / behavior_dir / "behavior_complexity_summary.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    normalized = str(behavior_eval_dir).strip().strip("/")
    if normalized and normalized != ".":
        fragment = f"/{normalized}/"
        for row in rows:
            row_path = str(row.get("path", ""))
            if fragment in row_path and row_path.endswith("rollout_custom_ppo.npz"):
                return dict(row)
    else:
        for row in rows:
            row_path = str(row.get("path", ""))
            if row_path.endswith("/rollout_custom_ppo.npz") and "/eval_" not in row_path:
                return dict(row)
    for row in rows:
        row_path = str(row.get("path", ""))
        if "eval_router_conf08" in row_path and row_path.endswith("rollout_custom_ppo.npz"):
            return dict(row)
    for row in rows:
        row_path = str(row.get("path", ""))
        if row_path.endswith("rollout_custom_ppo.npz"):
            return dict(row)
    return {}


def load_replay(run_dir: Path, replay_dir: str) -> dict[str, Any]:
    path = run_dir / replay_dir / "split_replay_gate_summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def metrics_path_for(run_dir: Path, metrics_eval_dir: str) -> Path:
    normalized = str(metrics_eval_dir).strip().strip("/")
    if not normalized or normalized == ".":
        return run_dir / "v2_custom_ppo_metrics.csv"
    return run_dir / normalized / "v2_custom_ppo_metrics.csv"


def backfill_macro_scores_from_rollouts(
    metrics: pd.DataFrame,
    *,
    run_dir: Path,
    metrics_eval_dir: str,
    replay_dir: str,
    macro_score_column: str,
) -> pd.DataFrame:
    """Fill macro scores for eval-only metrics tables that only contain step loss."""
    if "policy" not in metrics.columns or not str(macro_score_column):
        return metrics
    result = metrics.copy()
    if macro_score_column not in result.columns:
        result[macro_score_column] = np.nan
    values = pd.to_numeric(result[macro_score_column], errors="coerce")
    missing = ~np.isfinite(values)
    if not bool(missing.any()):
        return result
    for idx, policy in result.loc[missing, "policy"].astype(str).items():
        value = rollout_macro_replay_scale(
            run_dir,
            replay_dir=str(replay_dir),
            metrics_eval_dir=str(metrics_eval_dir),
            policy=str(policy),
            score_col=str(macro_score_column),
        )
        if np.isfinite(value):
            result.at[idx, macro_score_column] = float(value)
    return result


def collect_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = resolve_path(run_dir)
    metrics_path = metrics_path_for(run_dir, str(args.metrics_eval_dir))
    row: dict[str, Any] = {
        "seed": parse_seed(run_dir),
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "metrics_eval_dir": str(args.metrics_eval_dir),
        "has_metrics": metrics_path.exists(),
    }
    if not metrics_path.exists():
        row["complete"] = False
        return row

    metrics = ensure_macro_columns(pd.read_csv(metrics_path))
    metrics = backfill_macro_scores_from_rollouts(
        metrics,
        run_dir=run_dir,
        metrics_eval_dir=str(args.metrics_eval_dir),
        replay_dir=str(args.replay_dir),
        macro_score_column=str(args.macro_score_column),
    )
    ppo = row_for_policy(metrics, "custom_ppo")
    selected_static = first_existing_row(metrics, ("validation_selected_static",))
    best_static = best_row(metrics, STATIC_POLICIES, STEP_SCORE)
    best_rule = best_rule_dynamic_row(metrics, STEP_SCORE)
    best_static_macro = best_row(metrics, STATIC_POLICIES, args.macro_score_column)
    best_rule_macro = best_rule_dynamic_row(metrics, args.macro_score_column)

    ppo_step = score(ppo, STEP_SCORE)
    selected_static_step = score(selected_static, STEP_SCORE)
    best_static_step = score(best_static, STEP_SCORE)
    best_rule_step = score(best_rule, STEP_SCORE)
    best_baseline_step = finite_mean([])  # initialized as NaN
    best_baseline_policy = ""
    baseline_candidates = [
        (policy_name(best_static), best_static_step),
        (policy_name(best_rule), best_rule_step),
    ]
    finite_candidates = [(name, value) for name, value in baseline_candidates if name and np.isfinite(value)]
    if finite_candidates:
        best_baseline_policy, best_baseline_step = min(finite_candidates, key=lambda item: item[1])

    ppo_macro = score(ppo, args.macro_score_column)
    best_static_macro_loss = score(best_static_macro, args.macro_score_column)
    best_rule_macro_loss = score(best_rule_macro, args.macro_score_column)
    best_baseline_macro = finite_mean([])
    best_baseline_macro_policy = ""
    macro_candidates = [
        (policy_name(best_static_macro), best_static_macro_loss),
        (policy_name(best_rule_macro), best_rule_macro_loss),
    ]
    finite_macro_candidates = [(name, value) for name, value in macro_candidates if name and np.isfinite(value)]
    if finite_macro_candidates:
        best_baseline_macro_policy, best_baseline_macro = min(finite_macro_candidates, key=lambda item: item[1])

    step_margin_static = best_static_step - ppo_step
    step_margin_selected_static = selected_static_step - ppo_step
    step_margin_rule = best_rule_step - ppo_step
    step_margin_all = best_baseline_step - ppo_step
    macro_margin_static = best_static_macro_loss - ppo_macro
    macro_margin_rule = best_rule_macro_loss - ppo_macro
    macro_margin_all = best_baseline_macro - ppo_macro

    replay = load_replay(run_dir, str(args.replay_dir))
    behavior = load_behavior(run_dir, str(args.behavior_dir), str(args.behavior_eval_dir))
    behavior_gate = as_bool(behavior.get("behavior_complexity_gate_pass", False))
    replay_gate = as_bool(replay.get("gate_pass", False))
    replay_macro_gate = as_bool(replay.get("static_macro_reference_gate_pass", False))
    replay_static_reference_policy = str(replay.get("static_reference_policy", ""))
    replay_static_reference_loss = as_float(replay.get("static_reference_oracle_loss_mean"))
    replay_macro_static_reference_policy = str(replay.get("static_macro_reference_policy", ""))
    replay_macro_static_reference_loss = as_float(
        replay.get(
            "static_macro_reference_oracle_loss_macro_subtype_event",
            replay.get("static_macro_reference_score", float("nan")),
        )
    )
    if str(args.macro_score_column) == MACRO_SCORE:
        replay_macro_static_reference_loss = as_float(
            replay.get(
                "static_macro_reference_score",
                replay.get("static_macro_reference_oracle_loss_macro_subtype_event", float("nan")),
            )
        )
    ppo_macro_replay_scale = rollout_macro_replay_scale(
        run_dir,
        replay_dir=str(args.replay_dir),
        metrics_eval_dir=str(args.metrics_eval_dir),
        policy="custom_ppo",
        score_col=str(args.macro_score_column),
    )
    ppo_macro_for_replay_static = ppo_macro_replay_scale if np.isfinite(ppo_macro_replay_scale) else ppo_macro

    step_static_gate = margin_gate(step_margin_static, best_static_step, args.min_margin_abs, args.min_margin_rel)
    step_rule_gate = margin_gate(step_margin_rule, best_rule_step, args.min_margin_abs, args.min_margin_rel)
    step_all_gate = margin_gate(step_margin_all, best_baseline_step, args.min_margin_abs, args.min_margin_rel)
    macro_static_gate = margin_gate(macro_margin_static, best_static_macro_loss, args.min_margin_abs, args.min_margin_rel)
    macro_rule_gate = margin_gate(macro_margin_rule, best_rule_macro_loss, args.min_margin_abs, args.min_margin_rel)
    macro_all_gate = margin_gate(macro_margin_all, best_baseline_macro, args.min_margin_abs, args.min_margin_rel)
    learned_replay_static_margin = replay_static_reference_loss - ppo_step
    learned_replay_macro_static_margin = replay_macro_static_reference_loss - ppo_macro_for_replay_static
    learned_replay_static_gate = margin_gate(
        learned_replay_static_margin,
        replay_static_reference_loss,
        args.min_margin_abs,
        args.min_margin_rel,
    )
    learned_replay_macro_static_gate = margin_gate(
        learned_replay_macro_static_margin,
        replay_macro_static_reference_loss,
        args.min_margin_abs,
        args.min_margin_rel,
    )

    row.update(
        {
            "custom_ppo_loss": ppo_step,
            "custom_ppo_macro_loss": ppo_macro,
            "custom_ppo_macro_loss_replay_scale": ppo_macro_replay_scale,
            "custom_ppo_macro_loss_for_replay_static": ppo_macro_for_replay_static,
            "selected_static_policy": policy_name(selected_static),
            "selected_static_loss": selected_static_step,
            "best_static_policy": policy_name(best_static),
            "best_static_loss": best_static_step,
            "best_rule_dynamic_policy": policy_name(best_rule),
            "best_rule_dynamic_loss": best_rule_step,
            "best_static_macro_policy": policy_name(best_static_macro),
            "best_static_macro_loss": best_static_macro_loss,
            "best_rule_dynamic_macro_policy": policy_name(best_rule_macro),
            "best_rule_dynamic_macro_loss": best_rule_macro_loss,
            "best_operational_baseline_policy": best_baseline_policy,
            "best_operational_baseline_loss": best_baseline_step,
            "best_operational_baseline_macro_policy": best_baseline_macro_policy,
            "best_operational_baseline_macro_loss": best_baseline_macro,
            "step_margin_vs_selected_static": step_margin_selected_static,
            "step_margin_vs_best_static": step_margin_static,
            "step_margin_vs_best_rule_dynamic": step_margin_rule,
            "step_margin_vs_best_operational_baseline": step_margin_all,
            "macro_margin_vs_best_static": macro_margin_static,
            "macro_margin_vs_best_rule_dynamic": macro_margin_rule,
            "macro_margin_vs_best_operational_baseline": macro_margin_all,
            "step_static_gate_pass": step_static_gate,
            "step_rule_dynamic_gate_pass": step_rule_gate,
            "step_operational_gate_pass": step_all_gate,
            "macro_static_gate_pass": macro_static_gate,
            "macro_rule_dynamic_gate_pass": macro_rule_gate,
            "macro_operational_gate_pass": macro_all_gate,
            "replay_gate_pass": replay_gate,
            "replay_macro_gate_pass": replay_macro_gate,
            "behavior_eval_dir": str(args.behavior_eval_dir),
            "replay_static_reference_policy": replay_static_reference_policy,
            "replay_static_reference_loss": replay_static_reference_loss,
            "replay_macro_static_reference_policy": replay_macro_static_reference_policy,
            "replay_macro_static_reference_loss": replay_macro_static_reference_loss,
            "step_margin_vs_replay_static_reference": learned_replay_static_margin,
            "macro_margin_vs_replay_static_reference": learned_replay_macro_static_margin,
            "learned_replay_static_gate_pass": learned_replay_static_gate,
            "learned_replay_macro_static_gate_pass": learned_replay_macro_static_gate,
            "behavior_gate_pass": behavior_gate,
            "behavior_state_dependent": as_bool(behavior.get("state_dependent", False)),
            "behavior_fixed_like": as_bool(behavior.get("fixed_like", True)),
            "behavior_simple_cycle_like": as_bool(behavior.get("simple_cycle_like", True)),
            "behavior_unique_mask_count": int(behavior.get("unique_mask_count", -1)),
            "complete": bool(metrics_path.exists() and replay and behavior),
        }
    )
    row["old_claim_step_gate_pass"] = bool(
        row["complete"] and step_static_gate and step_rule_gate and replay_gate and behavior_gate
    )
    row["old_claim_macro_gate_pass"] = bool(
        row["complete"] and macro_static_gate and macro_rule_gate and replay_macro_gate and behavior_gate
    )
    row["learned_true_static_step_gate_pass"] = bool(
        row["complete"] and step_static_gate and step_rule_gate and learned_replay_static_gate and behavior_gate
    )
    row["learned_true_static_macro_gate_pass"] = bool(
        row["complete"]
        and macro_static_gate
        and macro_rule_gate
        and learned_replay_macro_static_gate
        and behavior_gate
    )
    return row


def binomial_one_sided_p(win_count: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, k) * (0.5**n) for k in range(win_count, n + 1)))


def aggregate(summary: pd.DataFrame) -> dict[str, Any]:
    complete = summary[summary["complete"].astype(bool)].copy() if "complete" in summary else pd.DataFrame()
    n = int(len(complete))
    if n == 0:
        return {"complete_seeds": 0, "claim_strength": "not_supported", "reason": "no complete seeds"}

    def count(flag: str) -> int:
        return int(complete.get(flag, pd.Series(False, index=complete.index)).astype(bool).sum())

    step_count = count("old_claim_step_gate_pass")
    macro_count = count("old_claim_macro_gate_pass")
    required_80 = int(math.ceil(0.8 * n))
    result: dict[str, Any] = {
        "complete_seeds": n,
        "old_claim_step_gate_count": step_count,
        "old_claim_macro_gate_count": macro_count,
        "behavior_eval_dirs": sorted({str(value) for value in complete.get("behavior_eval_dir", [])}),
        "step_static_gate_count": count("step_static_gate_pass"),
        "step_rule_dynamic_gate_count": count("step_rule_dynamic_gate_pass"),
        "step_operational_gate_count": count("step_operational_gate_pass"),
        "macro_static_gate_count": count("macro_static_gate_pass"),
        "macro_rule_dynamic_gate_count": count("macro_rule_dynamic_gate_pass"),
        "macro_operational_gate_count": count("macro_operational_gate_pass"),
        "replay_gate_count": count("replay_gate_pass"),
        "replay_macro_gate_count": count("replay_macro_gate_pass"),
        "learned_replay_static_gate_count": count("learned_replay_static_gate_pass"),
        "learned_replay_macro_static_gate_count": count("learned_replay_macro_static_gate_pass"),
        "learned_true_static_step_gate_count": count("learned_true_static_step_gate_pass"),
        "learned_true_static_macro_gate_count": count("learned_true_static_macro_gate_pass"),
        "behavior_gate_count": count("behavior_gate_pass"),
        "mean_step_margin_vs_best_operational_baseline": float(
            complete["step_margin_vs_best_operational_baseline"].astype(float).mean()
        ),
        "median_step_margin_vs_best_operational_baseline": float(
            complete["step_margin_vs_best_operational_baseline"].astype(float).median()
        ),
        "mean_macro_margin_vs_best_operational_baseline": float(
            complete["macro_margin_vs_best_operational_baseline"].astype(float).mean()
        ),
        "median_macro_margin_vs_best_operational_baseline": float(
            complete["macro_margin_vs_best_operational_baseline"].astype(float).median()
        ),
        "one_sided_sign_test_p_old_claim_step_gate": binomial_one_sided_p(step_count, n),
        "one_sided_sign_test_p_old_claim_macro_gate": binomial_one_sided_p(macro_count, n),
        "one_sided_sign_test_p_learned_true_static_step_gate": binomial_one_sided_p(
            count("learned_true_static_step_gate_pass"), n
        ),
        "one_sided_sign_test_p_learned_true_static_macro_gate": binomial_one_sided_p(
            count("learned_true_static_macro_gate_pass"), n
        ),
    }
    if n >= 10 and step_count >= required_80:
        result["claim_strength"] = "strong_old_claim_step"
        result["reason"] = ">=10 complete seeds with at least 80% old-claim step gates"
    elif n >= 5 and step_count >= required_80:
        result["claim_strength"] = "moderate_old_claim_step"
        result["reason"] = ">=5 complete seeds with at least 80% old-claim step gates"
    elif n >= 2 and step_count == n:
        result["claim_strength"] = "replicated_old_claim_step_pilot"
        result["reason"] = "all completed pilot seeds pass old-claim step gates, but n is below 5"
    else:
        result["claim_strength"] = "not_supported"
        result["reason"] = "old-claim step gates are not consistently positive"

    if n >= 10 and macro_count >= required_80:
        result["macro_claim_strength"] = "strong_old_claim_macro"
        result["macro_reason"] = ">=10 complete seeds with at least 80% old-claim macro gates"
    elif n >= 5 and macro_count >= required_80:
        result["macro_claim_strength"] = "moderate_old_claim_macro"
        result["macro_reason"] = ">=5 complete seeds with at least 80% old-claim macro gates"
    elif n >= 2 and macro_count == n:
        result["macro_claim_strength"] = "replicated_old_claim_macro_pilot"
        result["macro_reason"] = "all completed pilot seeds pass old-claim macro gates, but n is below 5"
    else:
        result["macro_claim_strength"] = "macro_not_supported"
        result["macro_reason"] = "old-claim macro gates are not consistently positive"
    return result


def expand_runs(args: argparse.Namespace) -> list[Path]:
    runs: list[Path] = []
    if args.runs:
        runs.extend(resolve_path(value) for value in args.runs)
    for pattern in args.run_glob or []:
        runs.extend(Path(path) for path in glob.glob(str(resolve_path(pattern))))
    unique = {str(path.resolve()): path.resolve() for path in runs}
    values = list(unique.values())
    if args.seeds:
        allowed = {int(seed) for seed in args.seeds}
        values = [path for path in values if parse_seed(path) in allowed]
    return sorted(values, key=lambda path: (parse_seed(path) if parse_seed(path) is not None else 10**9, path.name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect strict old-claim gates for V31 metpair runs.")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--run-glob", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--replay-dir", default="replay_gate_explicit_static_noguard")
    parser.add_argument("--behavior-dir", default="behavior_audit_v2")
    parser.add_argument(
        "--behavior-eval-dir",
        default=".",
        help="Evaluation subdirectory whose custom_ppo rollout should be used for behavior audit; '.' uses the raw run.",
    )
    parser.add_argument(
        "--metrics-eval-dir",
        default=".",
        help="Evaluation subdirectory whose v2_custom_ppo_metrics.csv should be used; '.' uses the raw run.",
    )
    parser.add_argument("--macro-score-column", default=MACRO_SCORE)
    parser.add_argument("--min-margin-abs", type=float, default=0.001)
    parser.add_argument("--min-margin-rel", type=float, default=0.002)
    parser.add_argument("--out-dir", default="reports/aggregate/metpair_oldclaim_gate_20260620")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = expand_runs(args)
    if not runs:
        raise SystemExit("No runs matched.")
    rows = [collect_run(path, args) for path in runs]
    summary = pd.DataFrame(rows)
    claim = aggregate(summary)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "oldclaim_seed_summary.csv"
    claim_path = out_dir / "oldclaim_summary.json"
    summary.to_csv(summary_path, index=False)
    claim_path.write_text(json.dumps(claim, indent=2), encoding="utf-8")

    print(summary_path.relative_to(ROOT) if summary_path.is_relative_to(ROOT) else summary_path)
    print(claim_path.relative_to(ROOT) if claim_path.is_relative_to(ROOT) else claim_path)
    print(json.dumps(claim, indent=2))
    cols = [
        "seed",
        "complete",
        "custom_ppo_loss",
        "best_static_loss",
        "best_rule_dynamic_policy",
        "best_rule_dynamic_loss",
        "step_margin_vs_best_operational_baseline",
        "step_margin_vs_replay_static_reference",
        "macro_margin_vs_best_operational_baseline",
        "macro_margin_vs_replay_static_reference",
        "old_claim_step_gate_pass",
        "old_claim_macro_gate_pass",
        "learned_true_static_step_gate_pass",
        "learned_true_static_macro_gate_pass",
    ]
    print(summary[[col for col in cols if col in summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
