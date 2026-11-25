from __future__ import annotations

from core.config import load_yaml
from pipelines.common import run_scheduler_training


def main() -> None:
    sweep = load_yaml("configs/sweeps/phase1_debug.yaml")
    env_cfg = sweep["env_cfg"]
    sensor_cfg = sweep["sensor_cfg"]
    est_cfg = sweep["estimator_cfg"]
    for sched_cfg in sweep["schedulers"]:
        run_id = f"phase1_{load_yaml(sched_cfg).get('scheduler_name', 'unknown')}"
        run_scheduler_training(env_cfg, sensor_cfg, est_cfg, sched_cfg, run_id=run_id)


if __name__ == "__main__":
    main()
