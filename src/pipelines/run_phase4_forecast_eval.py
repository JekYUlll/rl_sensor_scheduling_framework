from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_yaml
from evaluation.forecast_metrics import compute_forecast_metrics
from forecasting.dataset_builder import build_window_dataset, split_dataset
from forecasting.factory import build_predictor


def main() -> None:
    csv_path = Path("data/generated/windblown_phase3.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Run phase3 first to generate data")

    df = pd.read_csv(csv_path)
    cols = [c for c in ["wind_speed_ms", "air_temperature_c", "relative_humidity", "snow_mass_flux_kg_m2_s"] if c in df.columns]
    series = df[cols].to_numpy(dtype=float)
    ds = build_window_dataset(series, lookback=30, horizon=3)
    train, val, test = split_dataset(ds)

    out_dir = Path("reports/aggregate/phase4")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg_name in ["naive", "mlp", "lstm", "transformer", "informer", "tcn"]:
        cfg = load_yaml(f"configs/predictor/{cfg_name}.yaml")
        model = build_predictor(cfg)
        model.fit(train, val)
        pred = model.predict(test)
        m = compute_forecast_metrics(test.Y, pred)
        rows.append({"model": cfg_name, **m})

    pd.DataFrame(rows).to_csv(out_dir / "metrics_forecast.csv", index=False)
    print(out_dir / "metrics_forecast.csv")


if __name__ == "__main__":
    main()
