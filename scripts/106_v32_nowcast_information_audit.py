#!/usr/bin/env python3
"""Audit a label-free exogenous meteorological nowcast proxy.

The proxy exposes noisy forecasts of future wind, humidity, and air temperature.
It never uses event labels, target losses, candidate costs, or a policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score


FEATURES = ("wind_speed_ms", "relative_humidity", "air_temperature_c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lead-steps", type=int, default=4)
    parser.add_argument("--wind-noise-std", type=float, default=1.0)
    parser.add_argument("--humidity-noise-std", type=float, default=3.0)
    parser.add_argument("--temperature-noise-std", type=float, default=0.7)
    return parser.parse_args()


def final_frame(run: Path, *, lead_steps: int, noise: np.ndarray) -> pd.DataFrame:
    manifest = json.loads((run / "split_protocol_manifest.json").read_text())
    starts = manifest["final_test"]["eval_starts"]
    steps = int(manifest["final_test"]["eval_steps"])
    truth = pd.read_csv(run / "truth_v31_split.csv", usecols=[*FEATURES, "event_subtype_id"])
    rows: list[pd.DataFrame] = []
    for start in starts:
        idx = np.arange(int(start), int(start) + steps, dtype=int)
        valid = idx + int(lead_steps) < len(truth)
        frame = truth.iloc[idx[valid]].copy().reset_index(drop=True)
        future = truth.iloc[(idx[valid] + int(lead_steps))][list(FEATURES)].to_numpy(dtype=float)
        frame.loc[:, list(FEATURES)] = future + noise
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    noise_scale = np.asarray(
        [args.wind_noise_std, args.humidity_noise_std, args.temperature_noise_std], dtype=float
    )
    rows: list[dict[str, float | int]] = []
    for run in args.run_dirs:
        seed = int(run.name.split("seed", 1)[1].split("_", 1)[0])
        rng = np.random.default_rng(seed + 106_000)
        manifest = json.loads((run / "split_protocol_manifest.json").read_text())
        n_rows = len(manifest["final_test"]["eval_starts"]) * int(manifest["final_test"]["eval_steps"])
        noise = rng.normal(0.0, noise_scale, size=(n_rows, len(FEATURES)))
        frame = final_frame(run, lead_steps=args.lead_steps, noise=noise)
        noise = noise[: len(frame)]
        # Use alternating final windows as a development-only split within the
        # final-frame diagnostic; this never modifies a policy or scene.
        split = np.arange(len(frame)) % 2 == 0
        model = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=8, max_features=1.0, n_jobs=-1, random_state=seed
        ).fit(frame.loc[split, list(FEATURES)], frame.loc[split, "event_subtype_id"])
        prediction = model.predict(frame.loc[~split, list(FEATURES)])
        y_true = frame.loc[~split, "event_subtype_id"].to_numpy(dtype=int)
        rows.append({
            "seed": seed,
            "rows": int(len(frame)),
            "four_class_accuracy": float(np.mean(prediction == y_true)),
            "four_class_macro_f1": float(f1_score(y_true, prediction, average="macro")),
            "event_recall": float(np.mean((prediction > 0)[y_true > 0])),
        })
    output = pd.DataFrame(rows).sort_values("seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    summary = output.mean(numeric_only=True).to_frame("mean").T
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(output.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
