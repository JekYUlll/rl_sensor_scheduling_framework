#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate darts
fi

SRC_ROOT="reports/v31_split_protocol_no_warmup_hguard_envdwell12_reduced/raw"
OUT_ROOT="reports/v31_split_protocol_no_warmup_hguard_envdwell12_selected_static_duty_replay"
mkdir -p "$OUT_ROOT/raw" "$OUT_ROOT/logs" "$OUT_ROOT/done"

for seed in 41 42 43; do
  run_tag="budget1p70_seed${seed}"
  echo "[selected-static-duty] replay ${run_tag}"
  python scripts/64_v31_eval_saved_run_operational_baselines.py \
    --source-run-dir "${SRC_ROOT}/${run_tag}" \
    --out-dir "${OUT_ROOT}/raw/${run_tag}" \
    --device cpu \
    --oracle-device cpu \
    --eval-duty-constrained-baselines \
    --baseline-duty-hard-low 0.12 \
    --baseline-duty-hard-high 0.85 \
    --baseline-duty-hard-score 12 \
    --baseline-duty-score-feedback 2.5 \
    --env-min-dwell-steps 12 \
    > "${OUT_ROOT}/logs/${run_tag}.log" 2>&1
  touch "${OUT_ROOT}/done/${run_tag}.done"
done

python - <<'PY'
from pathlib import Path

import pandas as pd

out = Path("reports/v31_split_protocol_no_warmup_hguard_envdwell12_selected_static_duty_replay")
rows = []
for seed in (41, 42, 43):
    path = out / "raw" / f"budget1p70_seed{seed}" / "v2_custom_ppo_metrics.csv"
    df = pd.read_csv(path)
    by_policy = {str(row.policy): row for row in df.itertuples(index=False)}
    pdppo = by_policy["custom_ppo"]
    original_static = by_policy.get("validation_selected_static") or by_policy.get("oracle_static_projected")
    constrained_static = (
        by_policy.get("duty_constrained_validation_selected_static")
        or by_policy.get("duty_constrained_oracle_static_projected")
    )
    duty_candidates = [
        row for name, row in by_policy.items()
        if name.startswith("duty_constrained_") and name != "duty_constrained_validation_selected_static"
    ]
    best_duty = min(duty_candidates, key=lambda row: float(row.oracle_loss_mean))
    rows.append(
        {
            "seed": seed,
            "pdppo_oracle_loss": float(pdppo.oracle_loss_mean),
            "original_static_policy": str(original_static.policy) if original_static is not None else "",
            "original_static_oracle_loss": float(original_static.oracle_loss_mean) if original_static is not None else float("nan"),
            "deployable_selected_static_policy": str(constrained_static.policy) if constrained_static is not None else "",
            "deployable_selected_static_oracle_loss": float(constrained_static.oracle_loss_mean) if constrained_static is not None else float("nan"),
            "best_duty_non_pdppo_policy": str(best_duty.policy),
            "best_duty_non_pdppo_oracle_loss": float(best_duty.oracle_loss_mean),
            "pdppo_beats_original_static": bool(original_static is not None and float(pdppo.oracle_loss_mean) < float(original_static.oracle_loss_mean)),
            "pdppo_beats_deployable_selected_static": bool(constrained_static is not None and float(pdppo.oracle_loss_mean) < float(constrained_static.oracle_loss_mean)),
            "pdppo_beats_best_duty_non_pdppo": bool(float(pdppo.oracle_loss_mean) < float(best_duty.oracle_loss_mean)),
            "pdppo_mid": int(pdppo.mid_duty_sensor_count),
            "pdppo_always_on": int(pdppo.always_on_sensor_count),
            "pdppo_always_off": int(pdppo.always_off_sensor_count),
            "pdppo_switches_per_step": float(pdppo.switches_per_step),
        }
    )
summary = pd.DataFrame(rows)
summary.to_csv(out / "selected_static_duty_summary.csv", index=False)
print(summary.to_string(index=False))
PY
