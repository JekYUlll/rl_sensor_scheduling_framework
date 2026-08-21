#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/_code/microclimate_demo/rl_sensor_scheduling_framework}"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
fi
conda activate darts

SOURCE_RUN="reports/v31_static_break_v23_met_laser_exchange_dual_flux_dwell12_ppo_seed45_b1p10_h082_prior0p5_balanced_eventfraction_20260613/raw/budget1p10_seed45"
OUT_DIR="reports/v31_static_break_v23_met_laser_exchange_dual_flux_dwell12_diverse_replay_seed45_h082_20260613"
mkdir -p "$OUT_DIR"

CALM_TOP2="met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter;met_station_core,radiometer_basic,snow_particle_counter"
EVENT_TOP2="met_station_core,radiometer_basic,laser_disdrometer;met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter"
CALM_TOP3="${CALM_TOP2};met_station_core,radiometer_basic,laser_disdrometer"
EVENT_TOP3="${EVENT_TOP2};met_station_core,radiometer_basic,snow_particle_counter"
CALM_TOP5="${CALM_TOP3};radiometer_basic,surface_temp_ir,shielded_thermo_hygro,laser_disdrometer;radiometer_basic,shielded_thermo_hygro,snow_particle_counter"
EVENT_TOP5="${EVENT_TOP3};surface_temp_ir,ultrasonic_anemometer_hd,shielded_thermo_hygro,snow_particle_counter;surface_temp_ir,shielded_thermo_hygro,fc4_flux"

python scripts/69_v31_eval_event_pair_policy.py \
  --source-run-dir "$SOURCE_RUN" \
  --out-dir "$OUT_DIR" \
  --oracle-device cpu \
  --env-min-dwell-steps 12 \
  --cyclic-policy-spec "v23_dual_diverse_top2_l0_dwell12=${CALM_TOP2}|${EVENT_TOP2}|0|12" \
  --cyclic-policy-spec "v23_dual_diverse_top2_l6_dwell12=${CALM_TOP2}|${EVENT_TOP2}|6|12" \
  --cyclic-policy-spec "v23_dual_diverse_top3_l6_dwell12=${CALM_TOP3}|${EVENT_TOP3}|6|12" \
  --cyclic-policy-spec "v23_dual_diverse_top5_l0_dwell12=${CALM_TOP5}|${EVENT_TOP5}|0|12" \
  --cyclic-policy-spec "v23_dual_diverse_top5_l6_dwell12=${CALM_TOP5}|${EVENT_TOP5}|6|12" \
  2>&1 | tee "$OUT_DIR/diverse_replay.log"
