#!/usr/bin/env python3
"""Run the exact-geometry receding diagnostic for one frozen V32 run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def append_sensor_quality_args(command: list[str], metadata: dict[str, object]) -> None:
    quality = dict(metadata.get("sensor_quality") or {})
    quality_columns = list(quality.get("columns") or [])
    if not quality_columns:
        return
    command.extend(["--sensor-quality-columns", *map(str, quality_columns)])
    command.extend([
        "--sensor-quality-max-noise-multiplier",
        str(quality.get("max_noise_multiplier", 1.0)),
        "--sensor-quality-availability-floor",
        str(quality.get("availability_floor", 1.0)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-subdir", default="receding_oracle_l8_scene_gate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--partition",
        choices=("rl_train", "validation", "final_test"),
        default="final_test",
    )
    args = parser.parse_args()

    run = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((run / "v2_ppo_metadata.json").read_text())
    manifest = json.loads((run / "split_protocol_manifest.json").read_text())
    geometry = json.loads((run / "action_geometry.json").read_text())
    local_truth_path = run / "truth_v31_split.csv"
    metadata_truth_path = Path(str(metadata.get("truth_csv") or ""))
    truth_path = local_truth_path if local_truth_path.is_file() else metadata_truth_path
    if not truth_path.is_file():
        raise FileNotFoundError(
            "Expected a local truth_v31_split.csv or a valid truth_csv path in run metadata"
        )
    if args.partition == "rl_train":
        start_indices = manifest["rl_train"]["candidate_prior_starts"]
        eval_steps = manifest["rl_train"]["candidate_prior_steps"]
    elif args.partition == "validation":
        start_indices = manifest["validation"]["static_selection_starts"]
        eval_steps = manifest["validation"]["static_selection_steps"]
    else:
        start_indices = manifest["final_test"]["eval_starts"]
        eval_steps = manifest["final_test"]["eval_steps"]
    command = [
        sys.executable,
        str(root / "scripts/49_v31_physical_event_oracle_lift.py"),
        "--truth-csv", str(truth_path),
        "--out-dir", str(run / args.output_subdir),
        "--sensor-cfg", str(geometry["sensor_cfg"]),
        "--budget", str(geometry["budget"]),
        "--startup-peak-budget", str(geometry["startup_peak_budget"]),
        "--max-active", str(len(geometry["sensor_ids"])),
        "--required-sensors",
        "--target-weights", *map(str, metadata["target_weights"]),
        "--target-scales", *map(str, metadata["target_scales"]),
        "--lookback", str(metadata["lookback"]),
        "--horizon", str(metadata["horizon"]),
        "--oracle-type", "tcn",
        "--oracle-path", str(run / "v2_tcn_oracle.pt"),
        "--oracle-inference-device", args.device,
        "--oracle-loss-clip", str(manifest["ppo_controls"]["oracle_loss_clip"]),
        "--freq-s", str(metadata["freq_s"]),
        "--eval-steps", str(eval_steps),
        "--eval-rollouts", str(metadata["eval_rollouts"]),
        "--eval-start-indices", *map(str, start_indices),
        "--env-min-dwell-steps", str(metadata["reward_shaping"]["min_dwell_steps"]),
        "--schedule-diagnostics",
        "--schedule-family", "receding_oracle",
        "--receding-oracle-lookahead-steps", str(metadata["horizon"]),
        "--seed", str(manifest["seed"]),
    ]
    if metadata.get("state_columns"):
        command.extend(["--state-columns", *map(str, metadata["state_columns"])])
    append_sensor_quality_args(command, metadata)
    shaping = dict(metadata.get("reward_shaping") or {})
    command.append(
        "--common-random-numbers"
        if bool(shaping.get("common_random_numbers", False))
        else "--no-common-random-numbers"
    )
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
