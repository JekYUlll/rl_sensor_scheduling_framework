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
OUT_DIR="reports/v31_static_break_v23_met_laser_exchange_dual_flux_dwell12_eventpair_replay_seed45_h082_20260613"
mkdir -p "$OUT_DIR"

python scripts/69_v31_eval_event_pair_policy.py \
  --source-run-dir "$SOURCE_RUN" \
  --out-dir "$OUT_DIR" \
  --oracle-device cpu \
  --env-min-dwell-steps 12 \
  --policy-spec "v23_dual_auto_non10_event11_l0=surface_temp_ir,ultrasonic_anemometer_hd,shielded_thermo_hygro,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v23_dual_auto_non10_event11_l6=surface_temp_ir,ultrasonic_anemometer_hd,shielded_thermo_hygro,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|6" \
  --policy-spec "v23_dual_auto_non6_event11_l0=met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v23_dual_auto_non6_event11_l6=met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|6" \
  --policy-spec "v23_dual_auto_non0_event11_l0=met_station_core,radiometer_basic,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v23_dual_auto_non3_event11_l0=surface_temp_ir,ultrasonic_anemometer_hd,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v23_dual_auto_non7_event11_l0=surface_temp_ir,shielded_thermo_hygro,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v23_dual_auto_non10_event21_l0=surface_temp_ir,ultrasonic_anemometer_hd,shielded_thermo_hygro,snow_particle_counter|radiometer_basic,shielded_thermo_hygro,fc4_flux|0" \
  --policy-spec "v23_dual_auto_non6_event21_l0=met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter|radiometer_basic,shielded_thermo_hygro,fc4_flux|0" \
  2>&1 | tee "$OUT_DIR/event_pair_replay.log"
