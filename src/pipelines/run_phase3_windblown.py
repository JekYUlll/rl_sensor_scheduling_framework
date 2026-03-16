from __future__ import annotations

from business_cases.windblown_case.generator_adapter import generate_windblown_csv


def main() -> None:
    out = generate_windblown_csv(
        env_cfg_path="configs/env/windblown_case.yaml",
        sensor_cfg_path="configs/sensors/windblown_sensors.yaml",
        n_steps=2000,
        out_csv="data/generated/windblown_phase3.csv",
    )
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
