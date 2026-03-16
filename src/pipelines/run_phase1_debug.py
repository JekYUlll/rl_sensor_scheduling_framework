from __future__ import annotations

from pipelines.common import run_scheduler_training


def main() -> None:
    run_scheduler_training(
        env_cfg_path="configs/env/linear_gaussian_case.yaml",
        sensor_cfg_path="configs/sensors/linear_gaussian_sensors.yaml",
        estimator_cfg_path="configs/estimator/kalman.yaml",
        scheduler_cfg_path="configs/scheduler/dqn.yaml",
        run_id="phase1_debug",
    )


if __name__ == "__main__":
    main()
