#!/usr/bin/env python3
"""Verify seed roles and the manuscript's aggregate result directions."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AGG = ROOT / "aggregates"
PILOT_SEEDS = {117, 118}
POST_SELECTION_SEEDS = set(range(119, 141))
REPORTED_SEEDS = PILOT_SEEDS | POST_SELECTION_SEEDS


def rows(relative: str) -> list[dict[str, str]]:
    with (AGG / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_seeds(
    data: list[dict[str, str]], expected: set[int], label: str
) -> None:
    seeds = {int(row["seed"]) for row in data}
    if seeds != expected:
        raise AssertionError(
            f"{label}: expected {sorted(expected)}, found {sorted(seeds)}"
        )


def post_selection(data: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in data if int(row["seed"]) in POST_SELECTION_SEEDS]
    require_seeds(selected, POST_SELECTION_SEEDS, "post-selection subset")
    return selected


primary = rows(
    "pdppo_clean_validation_frozen_24seed_20260718/"
    "validation_frozen_seed_metrics.csv"
)
require_seeds(primary, REPORTED_SEEDS, "reported event-balanced-window aggregate")
primary_post = post_selection(primary)
assert all(
    float(row["macro_margin_pdppo_vs_validation_selected_static"]) > 0
    for row in primary_post
)
assert all(
    float(row["step_margin_pdppo_vs_validation_selected_static"]) > 0
    for row in primary_post
)

continuous = rows(
    "pdppo_full_final_partition_24seed_20260718/"
    "validation_frozen_seed_metrics.csv"
)
require_seeds(continuous, REPORTED_SEEDS, "reported continuous evaluation")
assert all(int(row["evaluation_steps"]) == 5242 for row in continuous)
continuous_post = post_selection(continuous)
assert all(
    float(row["macro_margin_pdppo_vs_validation_selected_static"]) > 0
    for row in continuous_post
)
assert all(
    float(row["step_margin_pdppo_vs_validation_selected_static"]) > 0
    for row in continuous_post
)

dqn = rows(
    "pdppo_matched_dqn_clean_24seed_20260718/"
    "matched_dqn_seed_metrics.csv"
)
require_seeds(dqn, REPORTED_SEEDS, "Double DQN comparison")
dqn_post = post_selection(dqn)
assert sum(float(row["macro_margin_dqn_minus_ppo"]) > 0 for row in dqn_post) == 22
assert sum(float(row["step_margin_dqn_minus_ppo"]) > 0 for row in dqn_post) == 21

ridge = rows(
    "pdppo_secondary_forecaster_24seed_20260718/"
    "secondary_forecaster_paired_metrics.csv"
)
require_seeds(ridge, REPORTED_SEEDS, "ridge")
ridge_post = post_selection(ridge)
assert sum(
    float(row["macro_margin_vs_secondary_static"]) > 0 for row in ridge_post
) == 21

print("PASS: pilot/model-selection seeds are 117--118")
print("PASS: post-selection evaluation seeds are 119--140")
print("PASS: 24 reported seeds remain available as a descriptive aggregate")
print("PASS: fixed-schedule margins are positive for both endpoints in 22/22 post-selection seeds")
print("PASS: continuous evaluation covers 5,242 time steps with positive fixed-schedule margins in 22/22 post-selection seeds")
print("PASS: the Double DQN training-configuration margins are positive in 22/22 macro and 21/22 mean comparisons")
print("PASS: the ridge-forecaster fixed-schedule margin is positive in 21/22 post-selection seeds")
