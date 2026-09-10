#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
OUT_ROOT="${V666_OUT_ROOT:-reports/v666_extra_train_transfer_20260910}"
mkdir -p "$OUT_ROOT"
for seed in 7231 7232 7233 7234; do
  "$PYTHON_BIN" - "$seed" "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
seed, out_root = sys.argv[1:]
train_root = Path("reports/v661_observable_specialist_mode_geometry_mode_labels_20260910")
extra_root = Path("reports/v665_additional_train_start_geometry_20260910")
test_root = Path("reports/v662_observable_specialist_mode_geometry_testgrid_20260910")
obs_root = Path("reports/v663_observable_specialist_mode_observations_20260910")
extra_obs_root = Path("reports/v665_additional_train_start_observations_20260910")
out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
train_paths = sorted((train_root / f"seed{seed}" / "train").glob("subset_condition_losses_seed*.csv"))
train_paths += sorted((extra_root / f"seed{seed}").glob("subset_condition_losses_seed*.csv"))
test_paths = sorted((test_root / f"seed{seed}").glob("test_*/subset_condition_losses_seed*.csv"))
pd.concat([pd.read_csv(p) for p in train_paths], ignore_index=True).to_csv(out / f"train_losses_seed{seed}.csv", index=False)
pd.concat([pd.read_csv(p) for p in test_paths], ignore_index=True).to_csv(out / f"test_losses_seed{seed}.csv", index=False)
arrays = [np.load(obs_root / "train" / f"seed{seed}" / "online_observation_probe.npz"), np.load(extra_obs_root / f"seed{seed}" / "online_observation_probe.npz")]
np.savez_compressed(out / f"train_observations_seed{seed}.npz", time_idx=np.concatenate([a["time_idx"] for a in arrays]), agent_observations=np.concatenate([a["agent_observations"] for a in arrays]), selected_masks=np.concatenate([a["selected_masks"] for a in arrays]))
PY
  "$PYTHON_BIN" scripts/169_audit_full_observation_transfer.py \
    --train-losses "$OUT_ROOT/train_losses_seed${seed}.csv" \
    --test-losses "$OUT_ROOT/test_losses_seed${seed}.csv" \
    --train-observations "$OUT_ROOT/train_observations_seed${seed}.npz" \
    --test-observations "reports/v663_observable_specialist_mode_observations_20260910/test/seed${seed}/online_observation_probe.npz" \
    --train-trace "reports/v658_observable_specialist_mode_truth_20260910/resource/resource_seed${seed}.csv" \
    --test-trace "reports/v658_observable_specialist_mode_truth_20260910/resource/resource_seed${seed}.csv" \
    --effective-budget 5.0 --fixed-power 0.4104 --seed "$seed" \
    --output "$OUT_ROOT/seed${seed}/transfer" > "$OUT_ROOT/seed${seed}.log" 2>&1
done
