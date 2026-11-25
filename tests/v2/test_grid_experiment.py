from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_grid_script_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "26_v2_grid_experiment.py"),
            "--dry-run",
            "--budget",
            "1.70",
            "--seed",
            "41",
            "--total-timesteps",
            "16",
            "--force",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "25_v2_train_custom_ppo.py" in result.stdout
    assert "budget1p70_seed41" in result.stdout


def test_grid_script_dqn_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "26_v2_grid_experiment.py"),
            "--dry-run",
            "--policies",
            "dqn",
            "--budget",
            "1.70",
            "--seed",
            "41",
            "--total-timesteps",
            "16",
            "--force",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "29_v2_train_dqn.py" in result.stdout
    assert "budget1p70_seed41" in result.stdout


def test_aggregate_script_with_mock_data(tmp_path: Path) -> None:
    policies = [
        "random",
        "round_robin",
        "aoi",
        "custom_ppo",
        "full_open_unconstrained",
        "feasible_static_projected",
    ]
    for budget in ("1p65", "1p70", "1p75"):
        for seed in (41, 42, 43):
            run_dir = tmp_path / f"budget{budget}_seed{seed}"
            eval_dir = run_dir / "evaluation"
            eval_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "policy": policies,
                    "forecast_weighted_mae_overall": [0.48, 0.45, 0.47, 0.44, 0.41, 0.439],
                    "power_mean": [1.6, 1.55, 1.62, 1.61, 4.62, 1.46],
                }
            ).to_csv(eval_dir / "v2_eval_overall.csv", index=False)
            pd.DataFrame(
                {
                    "policy": ["custom_ppo", "round_robin"],
                    "variable": ["air_temperature_c", "air_temperature_c"],
                    "forecast_mae": [2.1, 2.4],
                }
            ).to_csv(eval_dir / "v2_eval_by_variable.csv", index=False)
            pd.DataFrame(
                {
                    "policy": ["custom_ppo", "round_robin"],
                    "condition": ["all", "all"],
                    "forecast_weighted_mae": [0.44, 0.45],
                }
            ).to_csv(eval_dir / "v2_eval_by_condition.csv", index=False)
            pd.DataFrame({"step": [0, 16], "loss": [1.0, 0.5]}).to_csv(
                run_dir / "custom_ppo_training_log.csv",
                index=False,
            )

    out_dir = tmp_path / "paper_tables"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "27_v2_aggregate_results.py"),
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "table2_main_results.csv").exists()
    assert (out_dir / "table3_by_variable.csv").exists()
    assert (out_dir / "table3_by_condition.csv").exists()
    assert (out_dir / "figure4_learning_curves" / "seed41_training_log.csv").exists()


def test_aggregate_script_merges_dqn_grid(tmp_path: Path) -> None:
    ppo_dir = tmp_path / "ppo_grid"
    dqn_dir = tmp_path / "dqn_grid"
    for grid_dir, policies in (
        (ppo_dir, ["custom_ppo", "round_robin"]),
        (dqn_dir, ["dqn", "round_robin"]),
    ):
        for budget in ("1p65", "1p70", "1p75"):
            for seed in (41, 42, 43):
                run_dir = grid_dir / f"budget{budget}_seed{seed}"
                eval_dir = run_dir / "evaluation"
                eval_dir.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "policy": policies,
                        "forecast_weighted_mae_overall": [0.40 + 0.01 * idx for idx in range(len(policies))],
                        "power_mean": [1.6 for _ in policies],
                    }
                ).to_csv(eval_dir / "v2_eval_overall.csv", index=False)
                pd.DataFrame({"step": [0, 16], "loss": [1.0, 0.5]}).to_csv(
                    run_dir / "custom_ppo_training_log.csv",
                    index=False,
                )

    out_dir = tmp_path / "merged_tables"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "27_v2_aggregate_results.py"),
            "--input-dirs",
            str(ppo_dir),
            str(dqn_dir),
            "--output-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    table = pd.read_csv(out_dir / "table2_main_results.csv", index_col=0)
    assert "custom_ppo" in table.index
    assert "dqn" in table.index


def test_table2_format_if_generated() -> None:
    table2_path = ROOT / "reports" / "v2_paper_tables" / "table2_main_results.csv"
    if not table2_path.exists():
        pytest.skip("Table 2 has not been generated yet")
    df = pd.read_csv(table2_path, index_col=0)
    for policy in ("custom_ppo", "round_robin", "aoi", "random"):
        assert policy in df.index
    columns = {str(col) for col in df.columns}
    assert "1.65" in columns
    assert ("1.70" in columns) or ("1.7" in columns)
    assert "1.75" in columns
