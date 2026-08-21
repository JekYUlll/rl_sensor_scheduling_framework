#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
ORACLE_DEVICE="${ORACLE_DEVICE:-cpu}"
RUN_GLOB="${RUN_GLOB:-reports/v31_metpair_backbone_context_ortholinear_strongteacher_seed*_h075ctxolst_20260620}"
REPLAY_DIR_NAME="${REPLAY_DIR_NAME:-replay_gate_explicit_staticnorm_noguard}"
OUT_ROUTER="${OUT_ROUTER:-reports/aggregate/metpair_ortholinear_strongteacher_14seed_staticnorm_replay_20260620}"
OUT_RAW="${OUT_RAW:-reports/aggregate/metpair_ortholinear_strongteacher_14seed_raw_staticnorm_replay_20260620}"
LOG="${LOG:-logs/metpair_staticnorm_replay_14seed_20260620.log}"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
echo "[staticnorm14] start $(date)" | tee -a "$LOG"

runs=()
while IFS= read -r d; do
  runs+=("$d")
done < <(find reports -maxdepth 1 -type d -name "$(basename "$RUN_GLOB")" | sort)

if [[ "${#runs[@]}" -eq 0 ]]; then
  echo "[staticnorm14] no runs matched: $RUN_GLOB" | tee -a "$LOG"
  exit 2
fi

for d in "${runs[@]}"; do
  out="$d/$REPLAY_DIR_NAME"
  if [[ -f "$out/split_replay_gate_summary.json" ]]; then
    echo "[staticnorm14] skip existing $d" | tee -a "$LOG"
    continue
  fi
  echo "[staticnorm14] replay $d" | tee -a "$LOG"
  "$PY" scripts/70_v31_split_replay_gate.py \
    --source-run-dir "$d" \
    --out-dir "$out" \
    --oracle-device "$ORACLE_DEVICE" \
    --replay-family subtype_explicit \
    --explicit-policy-name split_metpair_subtype_explicit \
    --explicit-calm-sensors met_station_core shielded_thermo_hygro \
    --explicit-particle-sensors met_station_core laser_disdrometer \
    --explicit-flux-sensors met_station_core fc4_flux \
    --explicit-thermal-sensors met_station_core surface_temp_ir \
    --lead-steps 0 2 4 8 10 \
    --dwell-steps 6 12 24 \
    --subtype-top-size-cap 2 \
    --static-reference-duty-guard off \
    --enforce-static-candidate-reference \
    --min-margin-abs 0.005 \
    --min-margin-rel 0.01 \
    --macro-score-column oracle_loss_macro_subtype_event_staticnorm \
    2>&1 | tee -a "$LOG"
done

"$PY" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${runs[@]}" \
  --replay-dir "$REPLAY_DIR_NAME" \
  --out-dir "$OUT_ROUTER" \
  2>&1 | tee -a "$LOG"

"$PY" scripts/72_v31_collect_metpair_strongclaim.py \
  --runs "${runs[@]}" \
  --router-eval-dir . \
  --replay-dir "$REPLAY_DIR_NAME" \
  --out-dir "$OUT_RAW" \
  2>&1 | tee -a "$LOG"

echo "[staticnorm14] done $(date)" | tee -a "$LOG"
