#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBTYPE_LABELS = {
    1: "particle",
    2: "flux",
    3: "thermal",
}
SUBTYPE_LOSS_COLUMNS = tuple(f"oracle_loss_subtype_{label}" for label in SUBTYPE_LABELS.values())
MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event"
STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN = "oracle_loss_macro_subtype_event_staticnorm"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    return int(match.group(1)) if match else None


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def metric_row(df: pd.DataFrame | None, policy: str) -> pd.Series | None:
    if df is None or "policy" not in df.columns:
        return None
    matches = df[df["policy"].astype(str) == policy]
    if matches.empty:
        return None
    return matches.iloc[0]


def best_policy(df: pd.DataFrame | None, policies: tuple[str, ...]) -> pd.Series | None:
    if df is None or "policy" not in df.columns:
        return None
    subset = df[df["policy"].astype(str).isin(policies)].copy()
    if subset.empty or "oracle_loss_mean" not in subset.columns:
        return None
    return subset.sort_values("oracle_loss_mean").iloc[0]


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
    try:
        return bool(value)
    except Exception:
        return False


def resolve_run_path(value: str | Path, *, run_dir: Path) -> Path:
    path = Path(value)
    candidates = [path, run_dir / path.name, ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve {value!r} from {run_dir}")


def finite_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def add_macro_subtype_column(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    result = df.copy()
    if MACRO_SUBTYPE_LOSS_COLUMN not in result.columns and all(col in result.columns for col in SUBTYPE_LOSS_COLUMNS):
        result[MACRO_SUBTYPE_LOSS_COLUMN] = result[list(SUBTYPE_LOSS_COLUMNS)].apply(
            lambda row: finite_mean([as_float(value) for value in row.to_list()]),
            axis=1,
        )
    return result


def subtype_static_normalizers(table: pd.DataFrame | None) -> dict[str, float]:
    normalizers: dict[str, float] = {}
    if table is None or table.empty:
        return normalizers
    for col in SUBTYPE_LOSS_COLUMNS:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            normalizers[col] = float(np.median(values))
    return normalizers


def add_staticnorm_macro_column(df: pd.DataFrame | None, normalizers: dict[str, float]) -> pd.DataFrame | None:
    if df is None or df.empty or not normalizers:
        return df
    result = df.copy()
    norm_cols: list[str] = []
    for col in SUBTYPE_LOSS_COLUMNS:
        denom = float(normalizers.get(col, float("nan")))
        if col not in result.columns or not np.isfinite(denom) or denom <= 0.0:
            continue
        norm_col = f"{col}_staticnorm"
        result[norm_col] = pd.to_numeric(result[col], errors="coerce") / denom
        norm_cols.append(norm_col)
    if norm_cols:
        result[STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN] = result[norm_cols].apply(
            lambda row: finite_mean([as_float(value) for value in row.to_list()]),
            axis=1,
        )
    return result


def best_by_column(df: pd.DataFrame | None, score_col: str) -> pd.Series | None:
    if df is None or score_col not in df.columns:
        return None
    values = pd.to_numeric(df[score_col], errors="coerce")
    subset = df[np.isfinite(values)].copy()
    if subset.empty:
        return None
    subset[score_col] = pd.to_numeric(subset[score_col], errors="coerce")
    return subset.sort_values(score_col).iloc[0]


def truth_for_run(run_dir: Path) -> pd.DataFrame | None:
    metadata_path = run_dir / "v2_ppo_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        truth_path = resolve_run_path(str(metadata["truth_csv"]), run_dir=run_dir)
        truth = pd.read_csv(truth_path)
    except Exception:
        return None
    if "event_subtype_id" not in truth.columns:
        return None
    return truth


def rollout_macro_subtype_loss(run_dir: Path, rollout_path: Path, *, truth: pd.DataFrame | None = None) -> dict[str, Any]:
    if not rollout_path.exists():
        return {}
    if truth is None:
        truth = truth_for_run(run_dir)
    if truth is None or "event_subtype_id" not in truth.columns:
        return {}
    try:
        data = np.load(rollout_path, allow_pickle=False)
        losses = np.asarray(data["oracle_losses"], dtype=float).reshape(-1)
        step_indices = np.asarray(data["step_indices"], dtype=int).reshape(-1)
    except Exception:
        return {}
    if losses.size != step_indices.size:
        return {}

    valid = (step_indices >= 0) & (step_indices < len(truth))
    subtype_values = np.zeros_like(step_indices, dtype=int)
    subtype_values[valid] = truth["event_subtype_id"].to_numpy(dtype=int)[step_indices[valid]]
    finite = np.isfinite(losses)

    result: dict[str, Any] = {}
    subtype_losses: list[float] = []
    for subtype_id, label in SUBTYPE_LABELS.items():
        subtype_mask = (subtype_values == int(subtype_id)) & finite
        subtype_loss = float(np.mean(losses[subtype_mask])) if np.any(subtype_mask) else float("nan")
        result[f"oracle_loss_subtype_{label}"] = subtype_loss
        result[f"steps_subtype_{label}"] = int(np.sum(subtype_values == int(subtype_id)))
        if np.isfinite(subtype_loss):
            subtype_losses.append(subtype_loss)
    result[MACRO_SUBTYPE_LOSS_COLUMN] = finite_mean(subtype_losses)
    result["macro_subtype_event_count"] = int(len(subtype_losses))
    return result


def rollout_macro_for_policy(
    run_dir: Path,
    *,
    router_eval_dir: str,
    policy: str,
    truth: pd.DataFrame | None = None,
    score_col: str = MACRO_SUBTYPE_LOSS_COLUMN,
    normalizers: dict[str, float] | None = None,
) -> float:
    values = rollout_macro_subtype_loss(
        run_dir,
        run_dir / router_eval_dir / f"rollout_{policy}.npz",
        truth=truth,
    )
    if not values:
        return float("nan")
    if str(score_col) == STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN and normalizers:
        normalized: list[float] = []
        for col in SUBTYPE_LOSS_COLUMNS:
            denom = as_float(normalizers.get(col, float("nan")))
            value = as_float(values.get(col))
            if np.isfinite(value) and np.isfinite(denom) and denom > 0.0:
                normalized.append(value / denom)
        return finite_mean(normalized)
    return as_float(values.get(MACRO_SUBTYPE_LOSS_COLUMN))


def binomial_one_sided_p(win_count: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    tail = 0.0
    for k in range(int(win_count), int(n) + 1):
        tail += math.comb(int(n), k) * (0.5 ** int(n))
    return float(tail)


def load_behavior_row(path: Path, *, preferred_eval_dir: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return None
    normalized = str(preferred_eval_dir).strip().strip("/")
    if normalized and normalized != ".":
        fragment = f"/{normalized}/"
        for row in rows:
            row_path = str(row.get("path", ""))
            if fragment in row_path and row_path.endswith("rollout_custom_ppo.npz"):
                return dict(row)
    elif normalized in {"", "."}:
        for row in rows:
            row_path = str(row.get("path", ""))
            if row_path.endswith("/rollout_custom_ppo.npz") and "/eval_" not in row_path:
                return dict(row)
    for row in rows:
        row_path = str(row.get("path", ""))
        if "eval_router_conf08" in row_path and row_path.endswith("rollout_custom_ppo.npz"):
            return dict(row)
    for row in rows:
        if str(row.get("policy", "")) == "custom_ppo":
            return dict(row)
    return dict(rows[0]) if rows else None


def collect_run(
    run_dir: Path,
    *,
    router_eval_dir: str,
    replay_dir: str,
    behavior_dir: str,
    min_learned_margin_abs: float,
    min_learned_margin_rel: float,
    macro_score_column: str,
) -> dict[str, Any]:
    run_dir = resolve_path(run_dir)
    seed = parse_seed(run_dir)
    row: dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
    }

    source_metrics = read_csv(run_dir / "v2_custom_ppo_metrics.csv")
    router_metrics = add_macro_subtype_column(read_csv(run_dir / router_eval_dir / "v2_custom_ppo_metrics.csv"))
    replay_summary_path = run_dir / replay_dir / "split_replay_gate_summary.json"
    replay_metrics = add_macro_subtype_column(read_csv(run_dir / replay_dir / "split_replay_gate_metrics.csv"))
    static_candidate_metrics = add_macro_subtype_column(
        read_csv(run_dir / replay_dir / "split_static_candidate_event_table.csv")
    )
    behavior_path = run_dir / behavior_dir / "behavior_complexity_summary.json"
    if replay_summary_path.exists():
        replay = json.loads(replay_summary_path.read_text(encoding="utf-8"))
    else:
        replay = {}
    effective_macro_score_column = str(macro_score_column)
    if effective_macro_score_column == "auto":
        effective_macro_score_column = str(replay.get("macro_score_column") or MACRO_SUBTYPE_LOSS_COLUMN)
    if effective_macro_score_column not in {MACRO_SUBTYPE_LOSS_COLUMN, STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN}:
        effective_macro_score_column = MACRO_SUBTYPE_LOSS_COLUMN
    normalizers = subtype_static_normalizers(static_candidate_metrics)
    router_metrics = add_staticnorm_macro_column(router_metrics, normalizers)
    replay_metrics = add_staticnorm_macro_column(replay_metrics, normalizers)
    static_candidate_metrics = add_staticnorm_macro_column(static_candidate_metrics, normalizers)

    row.update(
        {
            "has_source_metrics": source_metrics is not None,
            "has_router_metrics": router_metrics is not None,
            "has_replay_summary": replay_summary_path.exists(),
            "has_behavior_summary": behavior_path.exists(),
        }
    )

    router_ppo = metric_row(router_metrics, "custom_ppo")
    selected_static = metric_row(router_metrics, "validation_selected_static")
    if selected_static is None:
        selected_static = best_policy(
            router_metrics,
            ("validation_selected_static", "feasible_static_projected", "oracle_static_projected"),
        )
    full_open = metric_row(router_metrics, "full_open_unconstrained")
    source_ppo = metric_row(source_metrics, "custom_ppo")

    router_loss = as_float(router_ppo["oracle_loss_mean"]) if router_ppo is not None else float("nan")
    selected_loss = as_float(selected_static["oracle_loss_mean"]) if selected_static is not None else float("nan")
    full_open_loss = as_float(full_open["oracle_loss_mean"]) if full_open is not None else float("nan")
    source_loss = as_float(source_ppo["oracle_loss_mean"]) if source_ppo is not None else float("nan")
    run_truth = truth_for_run(run_dir)
    router_macro_loss = (
        as_float(router_ppo.get(effective_macro_score_column))
        if router_ppo is not None and effective_macro_score_column in router_ppo.index
        else float("nan")
    )
    if not np.isfinite(router_macro_loss):
        router_macro_loss = rollout_macro_for_policy(
            run_dir,
            router_eval_dir=router_eval_dir,
            policy="custom_ppo",
            truth=run_truth,
            score_col=effective_macro_score_column,
            normalizers=normalizers,
        )
    selected_policy_name = str(selected_static["policy"]) if selected_static is not None else ""
    selected_macro_loss = (
        as_float(selected_static.get(effective_macro_score_column))
        if selected_static is not None and effective_macro_score_column in selected_static.index
        else float("nan")
    )
    if selected_policy_name and not np.isfinite(selected_macro_loss):
        selected_macro_loss = rollout_macro_for_policy(
            run_dir,
            router_eval_dir=router_eval_dir,
            policy=selected_policy_name,
            truth=run_truth,
            score_col=effective_macro_score_column,
            normalizers=normalizers,
        )
    learned_margin_abs = selected_loss - router_loss
    learned_margin_rel = learned_margin_abs / selected_loss if np.isfinite(selected_loss) and selected_loss != 0 else float("nan")
    learned_macro_margin_abs = selected_macro_loss - router_macro_loss
    learned_macro_margin_rel = (
        learned_macro_margin_abs / selected_macro_loss
        if np.isfinite(selected_macro_loss) and selected_macro_loss != 0
        else float("nan")
    )
    learned_required_abs = max(float(min_learned_margin_abs), float(min_learned_margin_rel) * selected_loss)
    learned_gate = bool(np.isfinite(learned_margin_abs) and learned_margin_abs >= learned_required_abs)

    row.update(
        {
            "source_custom_ppo_loss": source_loss,
            "router_custom_ppo_loss": router_loss,
            "selected_static_policy": str(selected_static["policy"]) if selected_static is not None else "",
            "selected_static_loss": selected_loss,
            "full_open_loss": full_open_loss,
            "router_custom_ppo_macro_subtype_loss": router_macro_loss,
            "selected_static_macro_subtype_loss": selected_macro_loss,
            "macro_score_column": effective_macro_score_column,
            "learned_margin_abs_vs_selected_static": learned_margin_abs,
            "learned_margin_rel_vs_selected_static": learned_margin_rel,
            "learned_macro_margin_abs_vs_selected_static": learned_macro_margin_abs,
            "learned_macro_margin_rel_vs_selected_static": learned_macro_margin_rel,
            "learned_macro_positive_pass": bool(np.isfinite(learned_macro_margin_abs) and learned_macro_margin_abs > 0.0),
            "learned_required_margin_abs": learned_required_abs,
            "learned_gate_pass": learned_gate,
            "learned_beats_full_open": bool(np.isfinite(full_open_loss) and router_loss < full_open_loss),
        }
    )

    replay_macro_row = best_by_column(replay_metrics, effective_macro_score_column)
    static_macro_row = best_by_column(static_candidate_metrics, effective_macro_score_column)
    replay_macro_loss = as_float(replay.get("best_replay_oracle_loss_macro_subtype_event"))
    if effective_macro_score_column == STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN:
        replay_macro_loss = as_float(replay.get("best_replay_macro_score"))
    if not np.isfinite(replay_macro_loss) and replay_macro_row is not None:
        replay_macro_loss = as_float(replay_macro_row[effective_macro_score_column])
    replay_macro_policy = str(replay.get("best_replay_macro_subtype_policy", ""))
    if not replay_macro_policy and replay_macro_row is not None:
        replay_macro_policy = str(replay_macro_row.get("policy", ""))
    static_macro_loss = as_float(replay.get("static_macro_reference_oracle_loss_macro_subtype_event"))
    if effective_macro_score_column == STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN:
        static_macro_loss = as_float(replay.get("static_macro_reference_score"))
    if not np.isfinite(static_macro_loss) and static_macro_row is not None:
        static_macro_loss = as_float(static_macro_row[effective_macro_score_column])
    static_macro_policy = str(replay.get("static_macro_reference_policy", ""))
    if not static_macro_policy and static_macro_row is not None:
        static_macro_policy = str(static_macro_row.get("policy", ""))
    replay_macro_margin_abs = as_float(replay.get("margin_abs_vs_static_macro_reference"))
    if not np.isfinite(replay_macro_margin_abs):
        replay_macro_margin_abs = static_macro_loss - replay_macro_loss
    replay_macro_margin_rel = as_float(replay.get("margin_rel_vs_static_macro_reference"))
    if not np.isfinite(replay_macro_margin_rel):
        replay_macro_margin_rel = (
            replay_macro_margin_abs / static_macro_loss
            if np.isfinite(static_macro_loss) and static_macro_loss != 0
            else float("nan")
        )
    macro_required_abs = as_float(replay.get("static_macro_required_margin_abs"))
    if not np.isfinite(macro_required_abs):
        macro_required_abs = (
            max(float(min_learned_margin_abs), float(min_learned_margin_rel) * static_macro_loss)
            if np.isfinite(static_macro_loss)
            else float("nan")
        )
    learned_macro_margin_abs_vs_macro_static = static_macro_loss - router_macro_loss
    learned_macro_margin_rel_vs_macro_static = (
        learned_macro_margin_abs_vs_macro_static / static_macro_loss
        if np.isfinite(static_macro_loss) and static_macro_loss != 0
        else float("nan")
    )
    learned_macro_gate = bool(
        np.isfinite(learned_macro_margin_abs_vs_macro_static)
        and np.isfinite(macro_required_abs)
        and learned_macro_margin_abs_vs_macro_static >= macro_required_abs
    )
    row.update(
        {
            "replay_best_policy": str(replay.get("best_replay_policy", "")),
            "replay_best_loss": as_float(replay.get("best_replay_oracle_loss_mean")),
            "replay_reference_policy": str(replay.get("reference_policy", "")),
            "replay_reference_loss": as_float(replay.get("reference_oracle_loss_mean")),
            "replay_static_reference_policy": str(replay.get("static_reference_policy", "")),
            "replay_static_reference_loss": as_float(replay.get("static_reference_oracle_loss_mean")),
            "replay_margin_abs_vs_static_reference": as_float(replay.get("margin_abs_vs_static_reference")),
            "replay_margin_rel_vs_static_reference": as_float(replay.get("margin_rel_vs_static_reference")),
            "replay_gate_pass": as_bool(replay.get("gate_pass", False)),
            "replay_best_macro_subtype_policy": replay_macro_policy,
            "replay_best_macro_subtype_loss": replay_macro_loss,
            "replay_static_macro_reference_policy": static_macro_policy,
            "replay_static_macro_reference_loss": static_macro_loss,
            "macro_required_margin_abs": macro_required_abs,
            "learned_macro_margin_abs_vs_macro_static_reference": learned_macro_margin_abs_vs_macro_static,
            "learned_macro_margin_rel_vs_macro_static_reference": learned_macro_margin_rel_vs_macro_static,
            "learned_macro_gate_pass": learned_macro_gate,
            "replay_macro_margin_abs_vs_static_reference": replay_macro_margin_abs,
            "replay_macro_margin_rel_vs_static_reference": replay_macro_margin_rel,
            "replay_macro_positive_pass": bool(np.isfinite(replay_macro_margin_abs) and replay_macro_margin_abs > 0.0),
            "replay_macro_gate_pass": as_bool(replay.get("static_macro_reference_gate_pass", False)),
        }
    )

    behavior = load_behavior_row(behavior_path, preferred_eval_dir=router_eval_dir) or {}
    row.update(
        {
            "behavior_unique_mask_count": int(behavior.get("unique_mask_count", -1)),
            "behavior_top1_mask_fraction": as_float(behavior.get("top1_mask_fraction")),
            "behavior_top3_mask_fraction": as_float(behavior.get("top3_mask_fraction")),
            "behavior_mask_entropy_bits": as_float(behavior.get("mask_entropy_bits")),
            "behavior_transition_entropy_bits": as_float(behavior.get("transition_entropy_bits")),
            "behavior_event_sensor_l1": as_float(behavior.get("event_sensor_l1")),
            "behavior_event_mask_mi_bits": as_float(behavior.get("event_mask_mi_bits")),
            "behavior_state_dependent": as_bool(behavior.get("state_dependent", False)),
            "behavior_fixed_like": as_bool(behavior.get("fixed_like", True)),
            "behavior_simple_cycle_like": as_bool(behavior.get("simple_cycle_like", True)),
            "behavior_gate_pass": as_bool(behavior.get("behavior_complexity_gate_pass", False)),
        }
    )
    row["complete"] = bool(
        row["has_source_metrics"]
        and row["has_router_metrics"]
        and row["has_replay_summary"]
        and row["has_behavior_summary"]
    )
    row["seed_gate_pass"] = bool(
        row["complete"]
        and row["learned_gate_pass"]
        and row["replay_gate_pass"]
        and row["behavior_gate_pass"]
    )
    row["macro_seed_positive_pass"] = bool(
        row["complete"]
        and row["learned_macro_positive_pass"]
        and row["replay_macro_positive_pass"]
        and row["behavior_gate_pass"]
    )
    row["macro_seed_gate_pass"] = bool(
        row["complete"]
        and row["learned_macro_gate_pass"]
        and row["replay_macro_gate_pass"]
        and row["behavior_gate_pass"]
    )
    return row


def claim_strength(summary: pd.DataFrame) -> dict[str, Any]:
    complete = summary[summary["complete"].astype(bool)].copy() if "complete" in summary else pd.DataFrame()
    n = int(len(complete))
    if n == 0:
        return {
            "complete_seeds": 0,
            "seed_gate_pass_count": 0,
            "claim_strength": "not_supported",
            "reason": "no complete seeds",
        }
    pass_count = int(complete["seed_gate_pass"].astype(bool).sum())
    learned_win_count = int(complete["learned_gate_pass"].astype(bool).sum())
    replay_pass_count = int(complete["replay_gate_pass"].astype(bool).sum())
    behavior_pass_count = int(complete["behavior_gate_pass"].astype(bool).sum())
    learned_macro_positive_count = int(complete["learned_macro_positive_pass"].astype(bool).sum())
    replay_macro_positive_count = int(complete["replay_macro_positive_pass"].astype(bool).sum())
    macro_seed_positive_count = int(complete["macro_seed_positive_pass"].astype(bool).sum())
    learned_macro_gate_count = int(complete.get("learned_macro_gate_pass", pd.Series(False, index=complete.index)).astype(bool).sum())
    replay_macro_gate_count = int(complete.get("replay_macro_gate_pass", pd.Series(False, index=complete.index)).astype(bool).sum())
    macro_seed_gate_count = int(complete.get("macro_seed_gate_pass", pd.Series(False, index=complete.index)).astype(bool).sum())
    mean_learned_margin = float(complete["learned_margin_abs_vs_selected_static"].astype(float).mean())
    median_learned_margin = float(complete["learned_margin_abs_vs_selected_static"].astype(float).median())
    mean_replay_margin = float(complete["replay_margin_abs_vs_static_reference"].astype(float).mean())
    median_replay_margin = float(complete["replay_margin_abs_vs_static_reference"].astype(float).median())
    mean_learned_macro_margin = float(complete["learned_macro_margin_abs_vs_selected_static"].astype(float).mean())
    median_learned_macro_margin = float(complete["learned_macro_margin_abs_vs_selected_static"].astype(float).median())
    mean_learned_macro_margin_vs_macro_static = float(
        complete["learned_macro_margin_abs_vs_macro_static_reference"].astype(float).mean()
    )
    median_learned_macro_margin_vs_macro_static = float(
        complete["learned_macro_margin_abs_vs_macro_static_reference"].astype(float).median()
    )
    mean_replay_macro_margin = float(complete["replay_macro_margin_abs_vs_static_reference"].astype(float).mean())
    median_replay_macro_margin = float(complete["replay_macro_margin_abs_vs_static_reference"].astype(float).median())
    p_value = binomial_one_sided_p(pass_count, n)
    macro_p_value = binomial_one_sided_p(macro_seed_positive_count, n)
    macro_gate_p_value = binomial_one_sided_p(macro_seed_gate_count, n)

    required_80 = int(math.ceil(0.8 * n))
    if n >= 10 and pass_count >= required_80 and mean_learned_margin > 0 and mean_replay_margin > 0:
        strength = "strong_multiseed"
        reason = ">=10 complete seeds with at least 80% full gate passes"
    elif n >= 5 and pass_count >= required_80 and mean_learned_margin > 0 and mean_replay_margin > 0:
        strength = "moderate_multiseed"
        reason = ">=5 complete seeds with at least 80% full gate passes"
    elif n >= 2 and pass_count == n:
        strength = "replicated_pilot"
        reason = "all completed pilot seeds pass, but n is below 5"
    elif n == 1 and pass_count == 1:
        strength = "single_seed_only"
        reason = "one complete passing seed"
    else:
        strength = "not_supported"
        reason = "seed-level gates are not consistently positive"

    if (
        n >= 10
        and macro_seed_gate_count >= required_80
        and mean_learned_macro_margin_vs_macro_static > 0
        and mean_replay_macro_margin > 0
    ):
        macro_strength = "strong_macro_multiseed"
        macro_reason = ">=10 complete seeds with at least 80% macro gate passes"
    elif (
        n >= 5
        and macro_seed_gate_count >= required_80
        and mean_learned_macro_margin_vs_macro_static > 0
        and mean_replay_macro_margin > 0
    ):
        macro_strength = "moderate_macro_multiseed"
        macro_reason = ">=5 complete seeds with at least 80% macro gate passes"
    elif n >= 2 and macro_seed_gate_count == n:
        macro_strength = "replicated_macro_pilot"
        macro_reason = "all completed pilot seeds pass the macro gate, but n is below 5"
    elif n == 1 and macro_seed_gate_count == 1:
        macro_strength = "single_seed_macro_only"
        macro_reason = "one complete seed passes the macro gate"
    else:
        macro_strength = "macro_not_supported"
        macro_reason = "macro seed-level gates are not consistently positive"

    return {
        "complete_seeds": n,
        "seed_gate_pass_count": pass_count,
        "learned_gate_pass_count": learned_win_count,
        "replay_gate_pass_count": replay_pass_count,
        "behavior_gate_pass_count": behavior_pass_count,
        "learned_macro_positive_count": learned_macro_positive_count,
        "replay_macro_positive_count": replay_macro_positive_count,
        "macro_seed_positive_count": macro_seed_positive_count,
        "learned_macro_gate_count": learned_macro_gate_count,
        "replay_macro_gate_count": replay_macro_gate_count,
        "macro_seed_gate_count": macro_seed_gate_count,
        "mean_learned_margin_abs": mean_learned_margin,
        "median_learned_margin_abs": median_learned_margin,
        "mean_replay_margin_abs_vs_static_reference": mean_replay_margin,
        "median_replay_margin_abs_vs_static_reference": median_replay_margin,
        "mean_learned_macro_margin_abs": mean_learned_macro_margin,
        "median_learned_macro_margin_abs": median_learned_macro_margin,
        "mean_learned_macro_margin_abs_vs_macro_static_reference": mean_learned_macro_margin_vs_macro_static,
        "median_learned_macro_margin_abs_vs_macro_static_reference": median_learned_macro_margin_vs_macro_static,
        "mean_replay_macro_margin_abs_vs_static_reference": mean_replay_macro_margin,
        "median_replay_macro_margin_abs_vs_static_reference": median_replay_macro_margin,
        "one_sided_sign_test_p_seed_gate": p_value,
        "one_sided_sign_test_p_macro_seed_positive": macro_p_value,
        "one_sided_sign_test_p_macro_seed_gate": macro_gate_p_value,
        "claim_strength": strength,
        "reason": reason,
        "macro_claim_strength": macro_strength,
        "macro_reason": macro_reason,
    }


def expand_runs(args: argparse.Namespace) -> list[Path]:
    runs: list[Path] = []
    if args.runs:
        runs.extend(resolve_path(value) for value in args.runs)
    default_globs = [
        "reports/v31_metpair_stronglatent_seed*_h075_20260620",
        "reports/v31_metpair_strongclaim_seed*_h075_20260620",
        "reports/v31_metpair_backbone_context_seed*_h075ctx_20260620",
    ]
    patterns = args.run_glob
    if patterns is None:
        patterns = [] if args.runs else default_globs
    for pattern in patterns:
        absolute_pattern = str(resolve_path(pattern))
        runs.extend(Path(value) for value in glob.glob(absolute_pattern))
    unique: dict[str, Path] = {}
    for path in runs:
        unique[str(path.resolve())] = path.resolve()
    values = list(unique.values())
    if args.seeds:
        allowed = {int(seed) for seed in args.seeds}
        values = [path for path in values if parse_seed(path) in allowed]
    return sorted(values, key=lambda path: (parse_seed(path) if parse_seed(path) is not None else 10**9, path.name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V31 metpair strong-claim seed evidence.")
    parser.add_argument("--runs", nargs="*", default=None, help="Explicit run directories.")
    parser.add_argument(
        "--run-glob",
        nargs="*",
        default=None,
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--router-eval-dir", default="eval_router_conf08")
    parser.add_argument("--replay-dir", default="replay_gate_explicit_static_noguard")
    parser.add_argument("--behavior-dir", default="behavior_audit_v2")
    parser.add_argument("--min-learned-margin-abs", type=float, default=0.001)
    parser.add_argument("--min-learned-margin-rel", type=float, default=0.002)
    parser.add_argument(
        "--macro-score-column",
        choices=["auto", MACRO_SUBTYPE_LOSS_COLUMN, STATICNORM_MACRO_SUBTYPE_LOSS_COLUMN],
        default="auto",
    )
    parser.add_argument("--out-dir", default="reports/aggregate/metpair_strongclaim_20260620")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = expand_runs(args)
    if not runs:
        raise SystemExit("No runs matched.")

    rows = [
        collect_run(
            path,
            router_eval_dir=str(args.router_eval_dir),
            replay_dir=str(args.replay_dir),
            behavior_dir=str(args.behavior_dir),
            min_learned_margin_abs=float(args.min_learned_margin_abs),
            min_learned_margin_rel=float(args.min_learned_margin_rel),
            macro_score_column=str(args.macro_score_column),
        )
        for path in runs
    ]
    summary = pd.DataFrame(rows)
    aggregate = claim_strength(summary)

    out_dir = resolve_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "metpair_seed_summary.csv"
    aggregate_path = out_dir / "metpair_claim_summary.json"
    summary.to_csv(summary_path, index=False)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(summary_path.relative_to(ROOT) if summary_path.is_relative_to(ROOT) else summary_path)
    print(aggregate_path.relative_to(ROOT) if aggregate_path.is_relative_to(ROOT) else aggregate_path)
    print(json.dumps(aggregate, indent=2))
    display_cols = [
        "seed",
        "complete",
        "router_custom_ppo_loss",
        "selected_static_loss",
        "macro_score_column",
        "learned_margin_abs_vs_selected_static",
        "replay_margin_abs_vs_static_reference",
        "learned_macro_margin_abs_vs_selected_static",
        "learned_macro_margin_abs_vs_macro_static_reference",
        "replay_macro_margin_abs_vs_static_reference",
        "macro_seed_positive_pass",
        "macro_seed_gate_pass",
        "behavior_gate_pass",
        "seed_gate_pass",
    ]
    print(summary[[col for col in display_cols if col in summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
