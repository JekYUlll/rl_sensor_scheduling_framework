from __future__ import annotations

from pathlib import Path

import numpy as np

from v2.evaluation import (
    action_score_metrics,
    event_group_metrics,
    load_rollout_npz,
    overall_metrics,
    sensor_usage_metrics,
    subset_rollout_columns,
    variable_metrics,
)


def test_evaluation_suite_reads_rollout_and_computes_tables(tmp_path: Path) -> None:
    path = tmp_path / "rollout_demo.npz"
    truth = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    observations = truth + np.asarray([[0.0, 0.5], [0.2, 0.0], [0.0, -0.5], [0.1, 0.0]])
    np.savez(
        path,
        observations=observations,
        masks=np.ones_like(observations),
        truth=truth,
        rewards=-np.ones(4),
        scores=np.asarray([[0.9, -0.2], [0.8, 0.7], [-0.1, 0.6], [0.5, -0.4]]),
        powers=np.asarray([1.0, 1.2, 1.1, 1.3]),
        peaks=np.asarray([1.0, 1.4, 1.1, 1.6]),
        selected_masks=np.asarray([[1, 0], [1, 1], [0, 1], [1, 0]]),
        mode_ids=np.asarray([[2, 0], [2, 1], [0, 2], [2, 0]]),
        event_flags=np.asarray([0, 1, 1, 0]),
        oracle_losses=np.asarray([0.2, 0.3, 0.4, 0.5]),
        step_indices=np.asarray([10, 11, 12, 13]),
        warmup_abort_count=np.asarray([2]),
        sensor_ids=np.asarray(["a", "b"]),
        state_columns=np.asarray(["x", "y"]),
        policy=np.asarray(["demo"]),
    )

    rollout = load_rollout_npz(path)
    rollout = subset_rollout_columns(rollout, ["x", "y"])
    overall = overall_metrics(
        rollout,
        per_step_budget=1.2,
        startup_peak_budget=1.5,
        dtw_window=2,
        target_weights=[1.0, 2.0],
        target_scales=[1.0, 2.0],
    )
    by_variable = variable_metrics(rollout, dtw_window=2, target_weights=[1.0, 2.0], target_scales=[1.0, 2.0])
    by_event = event_group_metrics(rollout, dtw_window=2, target_weights=[1.0, 2.0], target_scales=[1.0, 2.0])
    usage = sensor_usage_metrics(rollout)
    action_scores = action_score_metrics(rollout)

    assert overall["policy"] == "demo"
    assert overall["steps"] == 4
    assert overall["weighted_normalized_mae"] > 0.0
    assert overall["steady_violation_rate"] == 0.25
    assert overall["peak_violation_rate"] == 0.25
    assert len(by_variable) == 2
    assert by_variable[1]["weight"] == 2.0
    assert by_variable[1]["scale"] == 2.0
    assert {row["group"] for row in by_event} == {"all", "event", "non_event"}
    assert len(usage) == 2
    assert len(action_scores) == 2
    assert action_scores[0]["score_mean"] > 0.0
