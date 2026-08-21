#!/usr/bin/env python
"""Failure analysis for CA-PD-PPO against the context-alert bandit.

This script is diagnostic only. It does not train, tune, or select policies.
It reads completed CA-PD-PPO development runs plus the replayed
``context_alert_bandit_t0p5`` rollouts and writes seed-level, context-bin,
alert-lag, and PPO-training-stability summaries.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTEXT_POLICY = "context_alert_bandit_t0p5"
SUBTYPES = {
    1: "particle",
    2: "flux",
    3: "thermal",
}
ALERT_COLUMNS = {
    "particle": "agent_context_particle_alert",
    "flux": "agent_context_flux_alert",
    "thermal": "agent_context_thermal_alert",
}
CONF_BINS = [
    (0.00, 0.40),
    (0.40, 0.55),
    (0.55, 0.70),
    (0.70, 0.85),
    (0.85, 1.0000001),
]


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def entropy_from_labels(labels: list[str]) -> float:
    if not labels:
        return float("nan")
    values, counts = np.unique(np.asarray(labels, dtype=str), return_counts=True)
    _ = values
    probs = counts.astype(float) / float(counts.sum())
    probs = probs[probs > 0.0]
    return float(-(probs * np.log(probs)).sum())


def dict_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def selected_distribution(selected_masks: np.ndarray, sensor_ids: list[str]) -> dict[str, float]:
    masks = np.asarray(selected_masks, dtype=float)
    if masks.ndim != 2 or masks.shape[1] != len(sensor_ids):
        return {}
    return {sensor_ids[i]: float(masks[:, i].mean()) for i in range(len(sensor_ids))}


def active_specialist_labels(selected_masks: np.ndarray, sensor_ids: list[str]) -> list[str]:
    masks = np.asarray(selected_masks, dtype=bool)
    if masks.ndim != 2:
        return []
    labels: list[str] = []
    for row in masks:
        active = [sensor_ids[i] for i, is_on in enumerate(row) if is_on and sensor_ids[i] != "met_station_core"]
        labels.append("|".join(active) if active else "none")
    return labels


def switches_per_step(selected_masks: np.ndarray) -> float:
    masks = np.asarray(selected_masks, dtype=float)
    if masks.ndim != 2 or masks.shape[0] <= 1:
        return float("nan")
    return float(np.mean(np.abs(np.diff(masks, axis=0)).sum(axis=1)))


def event_mix(labels: np.ndarray) -> dict[str, float]:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    if arr.size == 0:
        return {}
    out: dict[str, float] = {}
    denom = float(arr.size)
    for subtype_id, name in {0: "calm", **SUBTYPES}.items():
        out[name] = float(np.sum(arr == int(subtype_id)) / denom)
    return out


def top_two_gap(scores: np.ndarray, sensor_ids: list[str]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2:
        return np.full(0, np.nan)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        unique = set(np.unique(np.round(finite, decimals=8)).tolist())
        if unique.issubset({-1.0, 1.0}):
            # These are saturated execution scores, not actor logits or action
            # probabilities. Reporting a constant gap would be misleading.
            return np.full(arr.shape[0], np.nan)
    specialist_idx = [i for i, sid in enumerate(sensor_ids) if sid != "met_station_core"]
    if len(specialist_idx) < 2:
        return np.full(arr.shape[0], np.nan)
    sub = arr[:, specialist_idx]
    ordered = np.sort(sub, axis=1)
    return ordered[:, -1] - ordered[:, -2]


@dataclass
class Rollout:
    path: Path
    sensor_ids: list[str]
    selected_masks: np.ndarray
    scores: np.ndarray
    oracle_losses: np.ndarray
    step_indices: np.ndarray
    warmup_abort_count: float


def load_rollout(path: Path) -> Rollout:
    data = np.load(path, allow_pickle=True)
    sensor_ids = [str(x) for x in data["sensor_ids"].tolist()]
    return Rollout(
        path=path,
        sensor_ids=sensor_ids,
        selected_masks=np.asarray(data["selected_masks"]),
        scores=np.asarray(data["scores"], dtype=float),
        oracle_losses=np.asarray(data["oracle_losses"], dtype=float).reshape(-1),
        step_indices=np.asarray(data["step_indices"], dtype=int).reshape(-1),
        warmup_abort_count=finite_float(np.asarray(data["warmup_abort_count"]).reshape(-1)[0]),
    )


def read_truth_subset(path: Path) -> pd.DataFrame:
    usecols = ["event_subtype_id", *ALERT_COLUMNS.values()]
    return pd.read_csv(path, usecols=usecols)


def confidence_frame(truth: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=truth.index)
    for label, col in ALERT_COLUMNS.items():
        out[label] = pd.to_numeric(truth[col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    labels = list(ALERT_COLUMNS)
    values = out[labels].to_numpy(dtype=float)
    argmax = np.argmax(values, axis=1)
    max_conf = np.max(values, axis=1)
    out["max_confidence"] = max_conf
    out["alert_argmax"] = [labels[int(i)] for i in argmax]
    out["event_subtype_id"] = pd.to_numeric(truth["event_subtype_id"], errors="coerce").fillna(0).astype(int)
    return out


def add_conf_bin(value: float) -> str:
    for lo, hi in CONF_BINS:
        if float(lo) <= float(value) < float(hi):
            return f"[{lo:.2f},{hi if hi <= 1 else 1.0:.2f})" if hi <= 1 else "[0.85,1.00]"
    return "out_of_range"


def alert_lag_bins(conf: pd.DataFrame, *, threshold: float, lag_window_steps: int) -> np.ndarray:
    alert = conf["max_confidence"].to_numpy(dtype=float) >= float(threshold)
    n = alert.size
    labels = np.full(n, "outside_alert", dtype=object)
    if n == 0:
        return labels
    starts: list[int] = []
    ends: list[int] = []
    i = 0
    while i < n:
        if not alert[i]:
            i += 1
            continue
        start = i
        while i < n and alert[i]:
            i += 1
        end = i - 1
        starts.append(start)
        ends.append(end)
        labels[start : end + 1] = "mid_event"
        early_end = min(end, start + int(lag_window_steps))
        late_start = max(start, end - int(lag_window_steps))
        labels[start : early_end + 1] = "early_event"
        labels[late_start : end + 1] = "late_event"
        if end - start > 2 * int(lag_window_steps):
            labels[start + int(lag_window_steps) + 1 : end - int(lag_window_steps)] = "mid_event"
        pre_start = max(0, start - int(lag_window_steps))
        labels[pre_start:start] = "pre_onset"
        post_end = min(n - 1, end + int(lag_window_steps))
        labels[end + 1 : post_end + 1] = "post_offset"
    return labels


def summarise_slice(
    *,
    seed: int,
    label: str,
    mask: np.ndarray,
    pdppo: Rollout,
    bandit: Rollout,
    conf_rows: pd.DataFrame,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != pdppo.oracle_losses.size:
        raise ValueError("Slice mask size does not match rollout length")
    p_loss = pdppo.oracle_losses[mask]
    b_loss = bandit.oracle_losses[mask]
    p_labels = active_specialist_labels(pdppo.selected_masks[mask], pdppo.sensor_ids)
    b_labels = active_specialist_labels(bandit.selected_masks[mask], bandit.sensor_ids)
    p_gap = top_two_gap(pdppo.scores[mask], pdppo.sensor_ids)
    return {
        "seed": int(seed),
        "bin": label,
        "num_windows": int(mask.sum()),
        "pdppo_loss": finite_mean(p_loss),
        "bandit_loss": finite_mean(b_loss),
        "margin": finite_mean(b_loss) - finite_mean(p_loss),
        "pdppo_specialist_entropy": entropy_from_labels(p_labels),
        "pdppo_action_confidence": finite_mean(p_gap),
        "bandit_action_entropy": entropy_from_labels(b_labels),
        "event_type_mix": dict_json(event_mix(conf_rows["event_subtype_id"].to_numpy(dtype=int))),
    }


def load_training_tail(path: Path, *, tail_n: int = 10) -> dict[str, Any]:
    if not path.exists():
        return {"training_log_available": False}
    table = pd.read_csv(path)
    tail = table.tail(int(tail_n))
    return {
        "training_log_available": True,
        "value_prediction_error_proxy_tail_value_loss": finite_mean(tail.get("value_loss", [])),
        "advantage_mean_tail": finite_mean(tail.get("advantage_mean", [])),
        "advantage_std_tail": finite_mean(tail.get("advantage_std", [])),
        "policy_entropy_tail": finite_mean(tail.get("entropy_mean", [])),
        "subtype_aux_accuracy_tail": finite_mean(tail.get("subtype_aux_accuracy", [])),
        "final_entropy": finite_float(table.iloc[-1].get("entropy_mean")) if not table.empty else float("nan"),
        "final_value_loss": finite_float(table.iloc[-1].get("value_loss")) if not table.empty else float("nan"),
    }


def analyse(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_metrics = pd.read_csv(args.seed_metrics)
    ca_seed_metrics = seed_metrics[seed_metrics["variant"].astype(str) == "ca_pdppo"].copy()
    baseline_metrics = pd.read_csv(args.ca_out_root / "framework_baseline_seed_metrics.csv")

    losing_rows: list[dict[str, Any]] = []
    conf_rows_out: list[dict[str, Any]] = []
    lag_rows_out: list[dict[str, Any]] = []
    instability_rows: list[dict[str, Any]] = []
    limitations: set[str] = set()

    for _, seed_row in ca_seed_metrics.sort_values("seed").iterrows():
        seed = int(seed_row["seed"])
        run_dir = Path(str(seed_row["run_dir"]))
        if not run_dir.is_absolute():
            run_dir = args.project_root / run_dir
        seed_dir = args.ca_out_root / f"seed{seed}"
        pdppo_path = run_dir / "rollout_custom_ppo.npz"
        bandit_path = seed_dir / "rollout_context_alert_bandit_t0p5.npz"
        truth_path = run_dir / "truth_v31_split.csv"
        context_group = baseline_metrics[
            (baseline_metrics["seed"].astype(int) == seed)
            & (baseline_metrics["policy"].astype(str) == CONTEXT_POLICY)
        ]
        if context_group.empty:
            limitations.add(f"seed {seed}: missing context baseline row")
            continue
        context_row = context_group.iloc[0]
        if not pdppo_path.exists() or not bandit_path.exists():
            limitations.add(f"seed {seed}: missing rollout npz")
            continue

        pdppo = load_rollout(pdppo_path)
        bandit = load_rollout(bandit_path)
        if pdppo.step_indices.size != bandit.step_indices.size:
            limitations.add(f"seed {seed}: rollout lengths differ")
            continue

        event_margins = {
            label: finite_float(context_row.get(f"oracle_loss_subtype_{label}"))
            - finite_float(context_row.get(f"custom_ppo_oracle_loss_subtype_{label}"))
            for label in SUBTYPES.values()
        }
        dominant = min(event_margins, key=lambda key: event_margins[key])
        if event_margins[dominant] >= 0.0:
            dominant = "none"

        losing_rows.append(
            {
                "seed": seed,
                "macro_margin_vs_bandit": finite_float(seed_row.get("macro_margin_vs_context_bandit")),
                "step_margin_vs_bandit": finite_float(seed_row.get("step_margin_vs_context_bandit")),
                "event_particle_margin": event_margins["particle"],
                "event_flux_margin": event_margins["flux"],
                "event_thermal_margin": event_margins["thermal"],
                "dominant_losing_event_type": dominant,
                "pdppo_selected_specialist_distribution": dict_json(
                    selected_distribution(pdppo.selected_masks, pdppo.sensor_ids)
                ),
                "bandit_selected_specialist_distribution": dict_json(
                    selected_distribution(bandit.selected_masks, bandit.sensor_ids)
                ),
                "switches_per_step": switches_per_step(pdppo.selected_masks),
                "warmup_abort_count": pdppo.warmup_abort_count,
                "min_on_blocked_action_rate": float("nan"),
            }
        )

        score_gap = top_two_gap(pdppo.scores, pdppo.sensor_ids)
        bandit_selected = np.asarray(bandit.selected_masks, dtype=bool)
        if bandit_selected.shape == pdppo.scores.shape:
            preferred_scores = np.where(bandit_selected, pdppo.scores, np.nan)
            nonpreferred_scores = np.where(~bandit_selected, pdppo.scores, np.nan)
            bandit_score_gap_proxy = np.nanmean(preferred_scores, axis=1) - np.nanmean(nonpreferred_scores, axis=1)
        else:
            bandit_score_gap_proxy = np.full(pdppo.scores.shape[0], np.nan)
        exact_agreement = (
            np.all(np.asarray(pdppo.selected_masks, dtype=bool) == np.asarray(bandit.selected_masks, dtype=bool), axis=1)
            if pdppo.selected_masks.shape == bandit.selected_masks.shape
            else np.asarray([], dtype=bool)
        )
        inst = {
            "seed": seed,
            "macro_margin_vs_bandit": finite_float(seed_row.get("macro_margin_vs_context_bandit")),
            "step_margin_vs_bandit": finite_float(seed_row.get("step_margin_vs_context_bandit")),
            "masked_action_probability_for_bandit_preferred_action": float("nan"),
            "masked_action_probability_note": "not_recorded_in_rollout",
            "bandit_preferred_score_gap_proxy": finite_mean(bandit_score_gap_proxy),
            "top_two_specialist_score_gap": finite_mean(score_gap),
            "exact_mask_agreement_with_bandit": float(np.mean(exact_agreement)) if exact_agreement.size else float("nan"),
            **load_training_tail(run_dir / "custom_ppo_training_log.csv"),
        }
        instability_rows.append(inst)

        if not truth_path.exists():
            limitations.add(f"seed {seed}: missing truth_v31_split.csv; skipped confidence/lag bins")
            continue
        truth = read_truth_subset(truth_path)
        conf = confidence_frame(truth)
        valid = (pdppo.step_indices >= 0) & (pdppo.step_indices < len(conf))
        if not np.all(valid):
            limitations.add(f"seed {seed}: invalid step indices dropped for bin analysis")
        step_idx = pdppo.step_indices[valid]
        conf_eval = conf.iloc[step_idx].reset_index(drop=True)
        pdppo_eval = Rollout(
            path=pdppo.path,
            sensor_ids=pdppo.sensor_ids,
            selected_masks=pdppo.selected_masks[valid],
            scores=pdppo.scores[valid],
            oracle_losses=pdppo.oracle_losses[valid],
            step_indices=step_idx,
            warmup_abort_count=pdppo.warmup_abort_count,
        )
        bandit_eval = Rollout(
            path=bandit.path,
            sensor_ids=bandit.sensor_ids,
            selected_masks=bandit.selected_masks[valid],
            scores=bandit.scores[valid],
            oracle_losses=bandit.oracle_losses[valid],
            step_indices=step_idx,
            warmup_abort_count=bandit.warmup_abort_count,
        )

        bin_labels = conf_eval["max_confidence"].map(add_conf_bin).to_numpy(dtype=object)
        for label in sorted(set(bin_labels)):
            mask = bin_labels == label
            conf_rows_out.append(
                summarise_slice(
                    seed=seed,
                    label=str(label),
                    mask=mask,
                    pdppo=pdppo_eval,
                    bandit=bandit_eval,
                    conf_rows=conf_eval.loc[mask],
                )
            )

        lag_all = alert_lag_bins(conf, threshold=float(args.alert_threshold), lag_window_steps=int(args.lag_window_steps))
        lag_labels = lag_all[step_idx]
        for label in ["pre_onset", "early_event", "mid_event", "late_event", "post_offset", "outside_alert"]:
            mask = lag_labels == label
            if not np.any(mask):
                continue
            row = summarise_slice(
                seed=seed,
                label=label,
                mask=mask,
                pdppo=pdppo_eval,
                bandit=bandit_eval,
                conf_rows=conf_eval.loc[mask],
            )
            row["lag_window_steps"] = int(args.lag_window_steps)
            lag_rows_out.append(row)

    losing = pd.DataFrame(losing_rows).sort_values("macro_margin_vs_bandit")
    losing_only = losing[losing["macro_margin_vs_bandit"] < 0.0].copy()
    conf_bins = pd.DataFrame(conf_rows_out)
    lag_bins = pd.DataFrame(lag_rows_out)
    instability = pd.DataFrame(instability_rows).sort_values("macro_margin_vs_bandit")

    losing.to_csv(out_dir / "all_seed_failure_metrics.csv", index=False)
    losing_only.to_csv(out_dir / "losing_seed_list.csv", index=False)
    conf_bins.to_csv(out_dir / "context_confidence_bins.csv", index=False)
    lag_bins.to_csv(out_dir / "alert_lag_analysis.csv", index=False)
    instability.to_csv(out_dir / "ppo_instability_audit.csv", index=False)

    write_summary(
        out_dir=out_dir,
        losing=losing,
        losing_only=losing_only,
        conf_bins=conf_bins,
        lag_bins=lag_bins,
        instability=instability,
        limitations=sorted(limitations),
        alert_threshold=float(args.alert_threshold),
        lag_window_steps=int(args.lag_window_steps),
    )


def aggregate_by_bin(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("bin", sort=False):
        rows.append(
            {
                "bin": label,
                "num_windows": int(group["num_windows"].sum()),
                "pdppo_loss": finite_mean(group["pdppo_loss"]),
                "bandit_loss": finite_mean(group["bandit_loss"]),
                "margin": finite_mean(group["margin"]),
                "pdppo_specialist_entropy": finite_mean(group["pdppo_specialist_entropy"]),
                "pdppo_action_confidence": finite_mean(group["pdppo_action_confidence"]),
                "bandit_action_entropy": finite_mean(group["bandit_action_entropy"]),
            }
        )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, *, digits: int = 6) -> str:
    if frame.empty:
        return ""
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)
    headers = [str(c) for c in frame.columns]
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines += ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join(lines)


def write_summary(
    *,
    out_dir: Path,
    losing: pd.DataFrame,
    losing_only: pd.DataFrame,
    conf_bins: pd.DataFrame,
    lag_bins: pd.DataFrame,
    instability: pd.DataFrame,
    limitations: list[str],
    alert_threshold: float,
    lag_window_steps: int,
) -> None:
    losing_event_counts = (
        losing_only["dominant_losing_event_type"].value_counts().rename_axis("dominant_losing_event_type").reset_index(name="count")
        if not losing_only.empty
        else pd.DataFrame(columns=["dominant_losing_event_type", "count"])
    )
    conf_agg = aggregate_by_bin(conf_bins)
    lag_agg = aggregate_by_bin(lag_bins)
    instability_display_cols = [
        "seed",
        "macro_margin_vs_bandit",
        "value_prediction_error_proxy_tail_value_loss",
        "advantage_mean_tail",
        "advantage_std_tail",
        "policy_entropy_tail",
        "bandit_preferred_score_gap_proxy",
        "top_two_specialist_score_gap",
        "exact_mask_agreement_with_bandit",
    ]
    lines = [
        "# CA-PD-PPO Failure Analysis",
        "",
        "This analysis uses completed development seeds 201--224. Margins are `bandit - PD-PPO`; positive values mean CA-PD-PPO is better.",
        "",
        "## Losing Seeds",
        "",
        f"- Losing macro seeds: {len(losing_only)}/{len(losing)}.",
        f"- Worst macro seed: `{int(losing.iloc[0]['seed'])}` with margin `{float(losing.iloc[0]['macro_margin_vs_bandit']):.6f}`." if not losing.empty else "- No seed rows available.",
        "",
        md_table(
            losing_only[
                [
                    "seed",
                    "macro_margin_vs_bandit",
                    "step_margin_vs_bandit",
                    "event_particle_margin",
                    "event_flux_margin",
                    "event_thermal_margin",
                    "dominant_losing_event_type",
                    "switches_per_step",
                    "warmup_abort_count",
                    "min_on_blocked_action_rate",
                ]
            ],
            digits=6,
        ),
        "",
        "Dominant event type among losing seeds:",
        "",
        md_table(losing_event_counts, digits=0),
        "",
        "## Context Confidence Bins",
        "",
        md_table(conf_agg, digits=6),
        "",
        "Interpretation: negative margins in high-confidence bins point to weak context-to-action mapping; negative margins in low-confidence bins point to uncertain-context handling.",
        "",
        "## Alert Onset / Offset Lag",
        "",
        f"Alert threshold: `{alert_threshold}`. Lag bins use ±`{lag_window_steps}` simulation steps because the current benchmark does not expose wall-clock timestamps in these rollouts.",
        "",
        md_table(lag_agg, digits=6),
        "",
        "## PPO Stability Proxies",
        "",
        "Exact masked action probabilities and critic prediction errors are not stored in the rollout artifacts. The table reports available proxies from the training log and rollout scores.",
        "Stored rollout scores are saturated execution scores in this run, so score-gap confidence is intentionally left blank.",
        "",
        md_table(instability[[c for c in instability_display_cols if c in instability.columns]].head(24), digits=6),
        "",
        "## Artifact Limits",
        "",
    ]
    if limitations:
        lines += [f"- {item}" for item in limitations]
    else:
        lines += ["- `min_on_blocked_action_rate` is not directly recorded and is intentionally left as NaN rather than inferred from ambiguous score/projection mismatches."]
        lines += ["- `masked_action_probability_for_bandit_preferred_action` is not recorded; `bandit_preferred_score_gap_proxy` is provided as a non-equivalent diagnostic proxy."]
        lines += ["- Stored rollout scores are saturated execution scores rather than actor logits; score-gap confidence is therefore not interpreted."]
    lines += [
        "",
        "## Decision",
        "",
        "Do not launch fresh final seeds from this analysis alone. The next clean step is a bounded development wave only if the failure structure supports one of the method-consistent variants: stronger context encoder, gated fusion, entropy decay, longer rollout, or lower learning rate.",
        "",
    ]
    (out_dir / "failure_summary.md").write_text("\n".join(lines), encoding="utf-8")

    conf_lines = [
        "# Context Confidence Bin Analysis",
        "",
        "Margins are `bandit - PD-PPO`; positive means CA-PD-PPO is better.",
        "",
        md_table(conf_agg, digits=6),
        "",
        f"Per-seed details: `{out_dir / 'context_confidence_bins.csv'}`",
        "",
    ]
    (out_dir / "context_confidence_bins.md").write_text("\n".join(conf_lines), encoding="utf-8")

    lag_lines = [
        "# Alert Lag Analysis",
        "",
        "Margins are `bandit - PD-PPO`; positive means CA-PD-PPO is better.",
        f"Lag bins use ±`{lag_window_steps}` simulation steps around thresholded alert onsets/offsets.",
        "",
        md_table(lag_agg, digits=6),
        "",
        f"Per-seed details: `{out_dir / 'alert_lag_analysis.csv'}`",
        "",
    ]
    (out_dir / "alert_lag_analysis.md").write_text("\n".join(lag_lines), encoding="utf-8")

    instability_lines = [
        "# PPO Instability Audit",
        "",
        "Exact masked action probabilities and value prediction errors are not stored. This audit reports the available training-log and score-based proxies.",
        "Stored rollout scores are saturated execution scores in this run, so top-two score-gap confidence is not interpreted.",
        "",
        md_table(instability[[c for c in instability_display_cols if c in instability.columns]], digits=6),
        "",
        "Missing exact diagnostics to add in any next run:",
        "",
        "- masked action probability assigned to the context-bandit-preferred candidate mask;",
        "- per-step critic value prediction error;",
        "- per-step chosen-candidate logit gap after feasibility masking;",
        "- explicit minimum-on blocked-action indicator.",
        "",
    ]
    (out_dir / "ppo_instability_audit.md").write_text("\n".join(instability_lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="rl_sensor_scheduling_framework root",
    )
    parser.add_argument(
        "--seed-metrics",
        type=Path,
        default=Path("reports/aggregate/contextaware_pdppo_dev_20260703capdppo/contextaware_pdppo_dev_seed_metrics.csv"),
    )
    parser.add_argument(
        "--ca-out-root",
        type=Path,
        default=Path("reports/aggregate/contextaware_pdppo_ca_pdppo_dev_20260703capdppo"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/analysis/ca_pdppo_failure_20260703"),
    )
    parser.add_argument("--alert-threshold", type=float, default=0.5)
    parser.add_argument("--lag-window-steps", type=int, default=12)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.seed_metrics.is_absolute():
        args.seed_metrics = args.project_root / args.seed_metrics
    if not args.ca_out_root.is_absolute():
        args.ca_out_root = args.project_root / args.ca_out_root
    if not args.output_dir.is_absolute():
        args.output_dir = args.project_root / args.output_dir
    analyse(args)


if __name__ == "__main__":
    main()
