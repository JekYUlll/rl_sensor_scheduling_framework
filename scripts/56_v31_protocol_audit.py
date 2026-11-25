#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def reconstructed_start_indices(
    truth: pd.DataFrame,
    *,
    steps: int,
    horizon: int,
    n_rollouts: int,
    event_fraction: float,
    seed: int,
    event_column: str = "event_flag",
) -> tuple[int, ...]:
    """Mirror the V3.1 S2 start sampler for metadata reconstruction."""
    max_start = max(0, len(truth) - int(steps) - int(horizon) - 1)
    if max_start <= 0 or int(n_rollouts) <= 1:
        return (0,)
    rng = np.random.default_rng(int(seed))
    starts: list[int] = []
    n_event = int(round(float(np.clip(event_fraction, 0.0, 1.0)) * int(n_rollouts)))
    event_flags = (
        truth[event_column].astype(bool).to_numpy()
        if event_column in truth.columns
        else np.zeros(len(truth), dtype=bool)
    )
    event_indices = np.flatnonzero(event_flags[: max_start + int(steps)])
    for _ in range(min(n_event, int(n_rollouts))):
        if event_indices.size == 0:
            break
        event_idx = int(rng.choice(event_indices))
        starts.append(int(np.clip(event_idx - int(steps) // 3, 0, max_start)))
    while len(starts) < int(n_rollouts):
        starts.append(int(rng.integers(0, max_start + 1)))
    return tuple(starts)


def interval_overlap_pairs(
    left: tuple[int, ...],
    left_steps: int,
    right: tuple[int, ...],
    right_steps: int,
    *,
    exclude_identity: bool = False,
) -> list[tuple[int, int]]:
    pairs = []
    for left_idx, left_start in enumerate(left):
        for right_idx, right_start in enumerate(right):
            if exclude_identity and left_idx == right_idx:
                continue
            if int(left_start) < int(right_start) + int(right_steps) and int(right_start) < int(left_start) + int(left_steps):
                pairs.append((left_idx, right_idx))
    return pairs


def audit_run(run_dir: Path) -> dict[str, object]:
    metadata = json.loads((run_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    prior_cfg = dict(metadata.get("candidate_prior", {}))
    truth = pd.read_csv(run_dir / "truth_v31.csv", usecols=["event_flag"])
    prior_steps = int(prior_cfg["steps"])
    eval_steps = int(metadata.get("eval_steps", 1024))
    prior_starts = reconstructed_start_indices(
        truth,
        steps=prior_steps,
        horizon=int(metadata["horizon"]),
        n_rollouts=int(prior_cfg["rollouts"]),
        event_fraction=float(metadata["eval_event_fraction"]),
        seed=int(metadata["seed"]) + 811,
    )
    eval_starts = tuple(int(x) for x in metadata["eval_start_indices"])
    prior_eval_pairs = interval_overlap_pairs(prior_starts, prior_steps, eval_starts, eval_steps)
    eval_internal_pairs = interval_overlap_pairs(
        eval_starts,
        eval_steps,
        eval_starts,
        eval_steps,
        exclude_identity=True,
    )
    prior_internal_pairs = interval_overlap_pairs(
        prior_starts,
        prior_steps,
        prior_starts,
        prior_steps,
        exclude_identity=True,
    )
    prior_table = pd.read_csv(run_dir / "custom_ppo_candidate_prior.csv")
    best = prior_table.iloc[0]
    return {
        "run_tag": run_dir.name,
        "budget": float(metadata["constraints"]["per_step_budget"]),
        "seed": int(metadata["seed"]),
        "prior_starts": json.dumps(list(prior_starts)),
        "eval_starts": json.dumps(list(eval_starts)),
        "prior_eval_overlap_pair_count": int(len(prior_eval_pairs)),
        "has_prior_eval_overlap": bool(prior_eval_pairs),
        "eval_internal_overlap_pair_count": int(len(eval_internal_pairs) // 2),
        "has_eval_internal_overlap": bool(eval_internal_pairs),
        "prior_internal_overlap_pair_count": int(len(prior_internal_pairs) // 2),
        "has_prior_internal_overlap": bool(prior_internal_pairs),
        "prior_best_action_idx": int(best["action_idx"]),
        "prior_best_sensor_ids": str(best["sensor_ids"]),
        "prior_best_oracle_loss_mean": float(best["oracle_loss_mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V3.1 S2 prior/evaluation window independence.")
    parser.add_argument("--source-dir", default="reports/v31_s2_main")
    parser.add_argument("--out-dir", default="reports/v31_s2_protocol_audit")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    run_dirs = sorted((source_dir / "raw").glob("budget*_seed*"))
    if not run_dirs:
        raise FileNotFoundError(f"No S2 run directories found under {source_dir / 'raw'}")
    rows = [audit_run(run_dir) for run_dir in run_dirs]
    table = pd.DataFrame(rows).sort_values(["budget", "seed"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "v31_s2_protocol_audit_runs.csv", index=False)
    summary = {
        "source_dir": str(source_dir),
        "run_count": int(len(table)),
        "runs_with_prior_eval_overlap": int(table["has_prior_eval_overlap"].sum()),
        "runs_with_eval_internal_overlap": int(table["has_eval_internal_overlap"].sum()),
        "runs_with_prior_internal_overlap": int(table["has_prior_internal_overlap"].sum()),
        "by_budget": [
            {
                "budget": float(budget),
                "run_count": int(len(group)),
                "runs_with_prior_eval_overlap": int(group["has_prior_eval_overlap"].sum()),
                "runs_with_eval_internal_overlap": int(group["has_eval_internal_overlap"].sum()),
                "runs_with_prior_internal_overlap": int(group["has_prior_internal_overlap"].sum()),
            }
            for budget, group in table.groupby("budget", sort=True)
        ],
        "interpretation": (
            "The stored S2 result does not implement independent/non-overlapping "
            "prior selection and policy evaluation windows. Any added replay on "
            "the existing windows is diagnostic only."
        ),
    }
    (out_dir / "v31_s2_protocol_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(out_dir / "v31_s2_protocol_audit_runs.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
