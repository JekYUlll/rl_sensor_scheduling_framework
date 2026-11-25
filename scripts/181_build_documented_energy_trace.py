"""Build a reproducible PV/physical-load trace for the V699 screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v2.entity_energy import CHANNEL_DEVICE, ENTITY_POWER, system_power_w
from v2.entity_resource_envelope import FIXED_AUXILIARY_POWER_W


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-truth", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--pv-derating", type=float, default=0.80)
    args = parser.parse_args()

    truth = pd.read_csv(args.input_truth)
    irradiance = np.clip(truth["solar_radiation_wm2"].to_numpy(dtype=float), 0.0, None)
    pv_w = np.minimum(600.0, 600.0 * irradiance / 1000.0 * float(args.pv_derating))
    frame = pd.DataFrame({"time_idx": truth["time_idx"].to_numpy(dtype=int)})
    for channel, device_id in CHANNEL_DEVICE.items():
        frame[f"resource_power_w_{channel}"] = float(ENTITY_POWER[device_id].steady_power_w)
    frame["energy_harvest_wh"] = pv_w
    frame["generation_pv_w"] = pv_w
    fixed_load = float(FIXED_AUXILIARY_POWER_W + system_power_w([])["steady_power_w"])
    baseline_soc = np.empty(len(frame), dtype=float)
    soc = 8640.0
    for index, harvest in enumerate(pv_w):
        soc = float(np.clip(soc + float(harvest) - fixed_load, 0.0, 8640.0))
        baseline_soc[index] = soc
    frame["baseline_soc_wh"] = baseline_soc
    frame.to_csv(args.output_trace, index=False)
    print({"rows": int(len(frame)), "mean_pv_w": float(np.mean(pv_w)), "zero_fraction": float(np.mean(pv_w <= 1e-12))})


if __name__ == "__main__":
    main()
