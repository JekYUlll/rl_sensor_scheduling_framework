#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.common import evaluate_scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", required=True)
    parser.add_argument("--sensor_cfg", required=True)
    parser.add_argument("--estimator_cfg", required=True)
    parser.add_argument("--scheduler_cfg", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    out = evaluate_scheduler(
        env_cfg_path=args.env_cfg,
        sensor_cfg_path=args.sensor_cfg,
        estimator_cfg_path=args.estimator_cfg,
        scheduler_cfg_path=args.scheduler_cfg,
        run_id=args.run_id,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
