#!/usr/bin/env python
"""Quantify simulated warning information from frozen truth series.

This script is analysis-only. It verifies each truth CSV against the SHA-256
recorded in the frozen 2026-07-18 evidence metadata and never loads a policy,
checkpoint, or training routine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SUBTYPES = {1: "particle", 2: "flux", 3: "thermal"}
ALERT_COLUMNS = {
    1: "agent_context_particle_alert",
    2: "agent_context_flux_alert",
    3: "agent_context_thermal_alert",
}
PRIMARY_SEEDS = tuple(range(119, 141))
REPORTED_SEEDS = tuple(range(117, 141))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(
            "reproducibility/pdppo_eswa_evidence_20260718/"
            "full_partition_source_rows"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/aggregate/pdppo_warning_information_20260729"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lead-window", type=int, default=16)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def load_seed(
    *, root: Path, metadata_root: Path, seed: int, horizon: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_path = metadata_root / f"seed{seed}" / "v2_ppo_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    truth_path = resolve(root, Path(str(metadata["truth_csv"])))
    actual_hash = file_sha256(truth_path)
    expected_hash = str(metadata["control_source"]["truth_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"seed {seed}: truth SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )

    final_start, final_end = (
        int(value)
        for value in metadata["control_source"]["source_manifest"]["partitions"][
            "final_test"
        ]
    )
    scoreable_end = final_end - int(horizon)
    if scoreable_end <= final_start:
        raise ValueError(f"seed {seed}: invalid final-test interval")
    usecols = [
        "event_subtype_id",
        "agent_context_particle_alert",
        "agent_context_flux_alert",
        "agent_context_thermal_alert",
        "agent_context_event_alert",
    ]
    truth = pd.read_csv(truth_path, usecols=usecols)
    if len(truth) < final_end:
        raise ValueError(f"seed {seed}: truth series shorter than final partition")
    frame = truth.iloc[final_start:scoreable_end].copy().reset_index(drop=True)
    frame["absolute_index"] = np.arange(final_start, scoreable_end, dtype=int)
    audit = {
        "seed": int(seed),
        "truth_path": str(truth_path.relative_to(root)),
        "truth_sha256": actual_hash,
        "final_partition": [final_start, final_end],
        "scoreable_interval": [final_start, scoreable_end],
        "scoreable_steps": int(scoreable_end - final_start),
    }
    return frame, audit


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= float(threshold)
    true = y_true.astype(bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    tn = int(np.sum(~pred & ~true))
    fn = int(np.sum(~pred & true))
    return {
        "roc_auc": float(roc_auc_score(true, score)),
        "pr_auc": float(average_precision_score(true, score)),
        "precision_at_threshold": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall_at_threshold": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "positive_prevalence": float(np.mean(true)),
        "alert_positive_fraction": float(np.mean(pred)),
    }


def event_lead_rows(
    frame: pd.DataFrame, *, seed: int, threshold: float, lead_window: int
) -> list[dict[str, Any]]:
    labels = frame["event_subtype_id"].to_numpy(dtype=int)
    absolute = frame["absolute_index"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for subtype_id, subtype_name in SUBTYPES.items():
        score = frame[ALERT_COLUMNS[subtype_id]].to_numpy(dtype=float)
        onsets = np.flatnonzero(
            (labels == subtype_id)
            & np.concatenate(([True], labels[:-1] != subtype_id))
        )
        for onset in onsets:
            start = max(0, int(onset) - int(lead_window))
            candidates = np.flatnonzero(score[start : int(onset) + 1] >= threshold)
            first = int(start + candidates[0]) if candidates.size else None
            rows.append(
                {
                    "seed": int(seed),
                    "subtype": subtype_name,
                    "event_onset_absolute_index": int(absolute[onset]),
                    "threshold_crossing_absolute_index": (
                        int(absolute[first]) if first is not None else np.nan
                    ),
                    "lead_steps": int(onset - first) if first is not None else np.nan,
                    "detected_within_lead_window": bool(first is not None),
                }
            )
    return rows


def summarize_analysis_set(
    frames: dict[int, pd.DataFrame], *, seeds: tuple[int, ...], threshold: float
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [frames[seed].assign(seed=seed) for seed in seeds], ignore_index=True
    )
    labels = combined["event_subtype_id"].to_numpy(dtype=int)
    alert_matrix = combined[[ALERT_COLUMNS[i] for i in SUBTYPES]].to_numpy(dtype=float)
    max_score = alert_matrix.max(axis=1)
    predicted = np.where(max_score >= threshold, alert_matrix.argmax(axis=1) + 1, 0)

    class_rows = []
    class_summary: dict[str, Any] = {}
    for subtype_id, subtype_name in SUBTYPES.items():
        metrics = binary_metrics(
            labels == subtype_id,
            alert_matrix[:, subtype_id - 1],
            threshold,
        )
        class_summary[subtype_name] = {
            key: safe_float(value) for key, value in metrics.items()
        }
        class_rows.append({"subtype": subtype_name, **metrics})

    cm = confusion_matrix(labels, predicted, labels=[0, 1, 2, 3])
    confusion_rows = []
    names = {0: "calm", **SUBTYPES}
    for true_id in range(4):
        total = int(cm[true_id].sum())
        for pred_id in range(4):
            confusion_rows.append(
                {
                    "true_label": names[true_id],
                    "predicted_label": names[pred_id],
                    "count": int(cm[true_id, pred_id]),
                    "row_fraction": (
                        float(cm[true_id, pred_id] / total) if total else np.nan
                    ),
                }
            )

    label_entropy = mutual_info_score(labels, labels) / np.log(2.0)
    mi_bits = mutual_info_score(labels, predicted) / np.log(2.0)
    aggregate = combined["agent_context_event_alert"].to_numpy(dtype=float)
    summary = {
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "analysis_support": "continuous scoreable final-test partition",
        "steps_per_seed": sorted(combined.groupby("seed").size().unique().tolist()),
        "threshold": float(threshold),
        "subtype_one_vs_rest": class_summary,
        "thresholded_four_class": {
            "accuracy": float(np.mean(labels == predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
            "macro_precision": float(
                precision_score(labels, predicted, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(labels, predicted, average="macro", zero_division=0)
            ),
            "mutual_information_bits": float(mi_bits),
            "label_entropy_bits": float(label_entropy),
            "normalized_mutual_information_fraction": float(mi_bits / label_entropy),
        },
        "aggregate_warning": {
            "mean": float(np.mean(aggregate)),
            "std": float(np.std(aggregate)),
            "q05": float(np.quantile(aggregate, 0.05)),
            "median": float(np.median(aggregate)),
            "q95": float(np.quantile(aggregate, 0.95)),
            "fraction_at_or_above_threshold": float(np.mean(aggregate >= threshold)),
        },
    }
    return summary, pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def main() -> None:
    args = parse_args()
    root = args.framework_root.resolve()
    metadata_root = resolve(root, args.metadata_root).resolve()
    output_dir = resolve(root, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[int, pd.DataFrame] = {}
    hash_audits = []
    lead_rows = []
    seed_subtype_rows = []
    for seed in REPORTED_SEEDS:
        frame, audit = load_seed(
            root=root,
            metadata_root=metadata_root,
            seed=seed,
            horizon=8,
        )
        frames[seed] = frame
        hash_audits.append(audit)
        labels = frame["event_subtype_id"].to_numpy(dtype=int)
        for subtype_id, subtype_name in SUBTYPES.items():
            metrics = binary_metrics(
                labels == subtype_id,
                frame[ALERT_COLUMNS[subtype_id]].to_numpy(dtype=float),
                float(args.threshold),
            )
            seed_subtype_rows.append(
                {"seed": seed, "subtype": subtype_name, **metrics}
            )
        lead_rows.extend(
            event_lead_rows(
                frame,
                seed=seed,
                threshold=float(args.threshold),
                lead_window=int(args.lead_window),
            )
        )

    lead_frame = pd.DataFrame(lead_rows)
    seed_subtype = pd.DataFrame(seed_subtype_rows)
    seed_subtype.to_csv(output_dir / "warning_quality_seed_subtype.csv", index=False)
    lead_frame.to_csv(output_dir / "warning_lead_times.csv", index=False)
    pd.DataFrame(hash_audits).to_csv(output_dir / "truth_hash_audit.csv", index=False)

    outputs: dict[str, Any] = {
        "status": "passed",
        "analysis_only": True,
        "policy_or_checkpoint_loaded": False,
        "training_or_policy_replay_performed": False,
        "truth_hashes_verified": True,
        "lead_window_steps": int(args.lead_window),
        "warning_rule_definition": (
            "argmax of the three subtype warning scores when their maximum is "
            f">={float(args.threshold):g}; calm otherwise"
        ),
    }
    for name, seeds in (
        ("post_selection_22", PRIMARY_SEEDS),
        ("reported_24_descriptive", REPORTED_SEEDS),
    ):
        summary, class_frame, confusion_frame = summarize_analysis_set(
            frames, seeds=seeds, threshold=float(args.threshold)
        )
        leads = lead_frame[lead_frame["seed"].isin(seeds)]
        valid_leads = pd.to_numeric(leads["lead_steps"], errors="coerce").dropna()
        summary["lead_time"] = {
            "event_runs": int(len(leads)),
            "detected_within_16_steps": int(leads["detected_within_lead_window"].sum()),
            "detection_fraction": float(leads["detected_within_lead_window"].mean()),
            "median_steps": float(valid_leads.median()),
            "q05_steps": float(valid_leads.quantile(0.05)),
            "q95_steps": float(valid_leads.quantile(0.95)),
        }
        outputs[name] = summary
        class_frame.to_csv(output_dir / f"warning_quality_{name}.csv", index=False)
        confusion_frame.to_csv(
            output_dir / f"warning_confusion_{name}.csv", index=False
        )

    (output_dir / "warning_information_summary.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
