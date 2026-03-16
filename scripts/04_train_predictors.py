#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import load_yaml
from forecasting.dataset_builder import build_window_dataset, split_dataset
from forecasting.factory import build_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_npz", default="data/processed/forecast_dataset.npz")
    parser.add_argument("--predictor_cfg", required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    data = np.load(args.series_npz)
    series = data["series"]
    ds = build_window_dataset(series, lookback=20, horizon=3)
    train, val, test = split_dataset(ds)

    cfg = load_yaml(args.predictor_cfg)
    predictor = build_predictor(cfg)
    predictor.fit(train, val)
    metrics = predictor.evaluate(test)

    out_dir = ROOT / "reports" / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": cfg.get("predictor_name", "unknown"), **metrics}]).to_csv(out_dir / "metrics_forecast.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
