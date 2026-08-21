from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_v2_pipeline_script_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    truth = root / "data" / "generated" / "public_weather_truth_smoke.csv"
    if not truth.exists():
        return
    out_dir = tmp_path / "v2_pipeline"
    cmd = [
        sys.executable,
        str(root / "scripts" / "22_v2_run_pipeline.py"),
        "--truth-csv",
        str(truth),
        "--out-dir",
        str(out_dir),
        "--oracle-rollout-steps",
        "80",
        "--eval-steps",
        "40",
        "--lookback",
        "6",
        "--horizon",
        "2",
    ]
    subprocess.run(cmd, cwd=root, check=True)

    assert (out_dir / "v2_metrics.csv").exists()
    assert (out_dir / "v2_policy_summary.png").exists()
