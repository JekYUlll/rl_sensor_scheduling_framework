#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
REPORTS = ROOT / "reports"

STORM_RUNS = {
    41: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed41",
    42: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed42",
    43: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed43",
    44: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed_seed44",
    45: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed_seed45",
}

FULL_RUNS = {
    41: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_fast" / "seed41" / "all" / "budget1p20_seed41",
    42: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_fast" / "seed42" / "all" / "budget1p20_seed42",
    43: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_fast" / "seed43" / "all" / "budget1p20_seed43",
    44: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_seed44_45_fast" / "all" / "budget1p20_seed44",
    45: REPORTS / "physical_event_v4_energy_ppo_h092_cap180_stormcurr_full_eval_seed44_45_fast" / "all" / "budget1p20_seed45",
}


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, PROJECT_ROOT / path, ROOT / path):
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def intervals_from_rollout(path: Path) -> list[tuple[int, int]]:
    with np.load(path, allow_pickle=True) as saved:
        indices = np.asarray(saved["step_indices"], dtype=int).reshape(-1)
    if indices.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            intervals.append((start, previous + 1))
            start = value
        previous = value
    intervals.append((start, previous + 1))
    return intervals


def overlap_steps(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    return int(
        sum(max(0, min(left_end, right_end) - max(left_start, right_start))
            for left_start, left_end in left
            for right_start, right_end in right)
    )


def internal_overlap_steps(intervals: list[tuple[int, int]]) -> int:
    diagonal = sum(end - start for start, end in intervals)
    return int(overlap_steps(intervals, intervals) - diagonal)


def select_eval_start_indices(
    event_flags: np.ndarray,
    *,
    steps: int,
    horizon: int,
    n_rollouts: int,
    event_fraction: float,
    seed: int,
) -> tuple[int, ...]:
    """Reproduce scripts/23_v2_train_ppo.py::select_eval_start_indices."""
    max_start = max(0, int(event_flags.size) - int(steps) - int(horizon) - 1)
    if max_start <= 0 or int(n_rollouts) <= 1:
        return (0,)
    rng = np.random.default_rng(int(seed))
    starts: list[int] = []
    n_event = int(round(float(np.clip(event_fraction, 0.0, 1.0)) * int(n_rollouts)))
    event_indices = np.flatnonzero(event_flags[: max_start + int(steps)])
    for _ in range(min(n_event, int(n_rollouts))):
        if event_indices.size == 0:
            break
        event_idx = int(rng.choice(event_indices))
        starts.append(int(np.clip(event_idx - int(steps) // 3, 0, max_start)))
    while len(starts) < int(n_rollouts):
        starts.append(int(rng.integers(0, max_start + 1)))
    return tuple(starts)


def reconstruct_oracle_intervals(metadata: dict[str, object], truth_path: Path) -> list[tuple[int, int]]:
    truth = pd.read_csv(truth_path)
    event_flags = truth["event_flag"].astype(bool).to_numpy()
    per_rollout_steps = max(
        int(metadata["lookback"]) + int(metadata["horizon"]) + 2,
        int(np.ceil(float(metadata["oracle_rollout_steps"]) / float(metadata["oracle_rollouts_per_policy"]))),
    )
    summary = metadata["oracle_pretrain_rollout_summary"]
    total_specs = int(summary["total_specs"])
    intervals: list[tuple[int, int]] = []
    for offset in range(total_specs):
        starts = select_eval_start_indices(
            event_flags,
            steps=per_rollout_steps,
            horizon=int(metadata["horizon"]),
            n_rollouts=int(metadata["oracle_rollouts_per_policy"]),
            event_fraction=float(metadata["oracle_event_fraction"]),
            seed=int(metadata["seed"]) + 10_000 + offset,
        )
        intervals.extend((int(start), int(start) + per_rollout_steps) for start in starts)
    return intervals


def audit_seed(seed: int, *, train_episode_steps: int) -> dict[str, object]:
    storm_dir = STORM_RUNS[seed]
    full_dir = FULL_RUNS[seed]
    storm_meta = json.loads((storm_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    full_meta = json.loads((full_dir / "v2_ppo_metadata.json").read_text(encoding="utf-8"))
    storm_intervals = intervals_from_rollout(storm_dir / "rollout_custom_ppo.npz")
    full_intervals = intervals_from_rollout(full_dir / "rollout_custom_ppo.npz")
    train_starts = [int(value) for value in storm_meta.get("train_start_indices", [])]
    storm_starts = [int(value) for value in storm_meta.get("eval_start_indices", [])]
    train_intervals = [(start, start + int(train_episode_steps)) for start in train_starts]
    truth_path = resolve_recorded_path(str(storm_meta["truth_csv"]))
    oracle_intervals = reconstruct_oracle_intervals(storm_meta, truth_path)
    shared_starts = sorted(set(train_starts).intersection(storm_starts))
    return {
        "seed": int(seed),
        "truth_csv": str(truth_path),
        "storm_train_starts_equal": bool(train_starts == storm_starts),
        "storm_train_shared_start_count": int(len(shared_starts)),
        "storm_training_overlap_certified": bool(shared_starts),
        "storm_vs_train_overlap_steps_assuming_episode_len": overlap_steps(storm_intervals, train_intervals),
        "full_vs_train_overlap_steps_assuming_episode_len": overlap_steps(full_intervals, train_intervals),
        "full_vs_storm_overlap_steps": overlap_steps(full_intervals, storm_intervals),
        "oracle_vs_storm_overlap_steps_reconstructed": overlap_steps(oracle_intervals, storm_intervals),
        "oracle_vs_full_overlap_steps_reconstructed": overlap_steps(oracle_intervals, full_intervals),
        "storm_internal_overlap_steps": internal_overlap_steps(storm_intervals),
        "full_internal_overlap_steps": internal_overlap_steps(full_intervals),
        "oracle_internal_overlap_steps_reconstructed": internal_overlap_steps(oracle_intervals),
        "normalization_partition_declared": bool(
            storm_meta.get("protocol_partitions")
            or storm_meta.get("normalization_partition")
            or storm_meta.get("normalization_start_idx") is not None
        ),
        "storm_eval_starts": storm_starts,
        "full_eval_starts": [int(value) for value in full_meta.get("eval_start_indices", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit energy-account curriculum evaluation independence.")
    parser.add_argument("--out-dir", default=str(REPORTS / "energy_account_protocol_audit_20260526"))
    parser.add_argument(
        "--train-episode-steps",
        type=int,
        default=512,
        help="Episode length used only to quantify overlap steps; equal train/eval starts are independently fatal.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [audit_seed(seed, train_episode_steps=int(args.train_episode_steps)) for seed in sorted(STORM_RUNS)]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "energy_account_protocol_audit_by_seed.csv", index=False)

    summary = {
        "status": "diagnostic_only_requires_split_protocol_rerun",
        "audited_seeds": [int(seed) for seed in sorted(STORM_RUNS)],
        "train_episode_steps_for_quantified_overlap": int(args.train_episode_steps),
        "storm_training_overlap_certified_count": int(table["storm_training_overlap_certified"].sum()),
        "storm_train_eval_identical_starts_count": int(table["storm_train_starts_equal"].sum()),
        "full_eval_overlaps_training_count_assuming_episode_len": int(
            (table["full_vs_train_overlap_steps_assuming_episode_len"] > 0).sum()
        ),
        "full_eval_overlaps_storm_eval_count": int((table["full_vs_storm_overlap_steps"] > 0).sum()),
        "oracle_overlaps_storm_eval_count_reconstructed": int(
            (table["oracle_vs_storm_overlap_steps_reconstructed"] > 0).sum()
        ),
        "oracle_overlaps_full_eval_count_reconstructed": int(
            (table["oracle_vs_full_overlap_steps_reconstructed"] > 0).sum()
        ),
        "full_eval_has_internal_overlap_count": int((table["full_internal_overlap_steps"] > 0).sum()),
        "normalization_partition_declared_count": int(table["normalization_partition_declared"].sum()),
        "fatal_reasons": [
            "Storm-window curriculum evaluation uses the same recorded starts as PPO training in every audited seed.",
            "No audited curriculum metadata declares training-only normalization statistics.",
        ],
        "additional_reconstructed_risks": [
            "Under the deterministic oracle sampler in scripts/23_v2_train_ppo.py, oracle-pretraining windows overlap both storm and full-distribution evaluation in every seed.",
            "Full-distribution no-retrain evaluation overlaps storm evaluation in every seed and overlaps the default-length training windows in every seed.",
        ],
        "interpretation": (
            "Existing energy-account curriculum summaries may be retained as same-protocol "
            "mechanism diagnostics only. Learned-policy submission evidence requires a "
            "chronological split-protocol retraining and final-test evaluation."
        ),
    }
    (out_dir / "energy_account_protocol_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(out_dir / "energy_account_protocol_audit_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
