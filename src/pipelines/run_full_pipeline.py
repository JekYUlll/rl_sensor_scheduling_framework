from __future__ import annotations

from pipelines.run_phase1_training import main as run_phase1
from pipelines.run_phase3_windblown import main as run_phase3
from pipelines.run_phase4_forecast_eval import main as run_phase4


def main() -> None:
    run_phase1()
    run_phase3()
    run_phase4()


if __name__ == "__main__":
    main()
