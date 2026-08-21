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

SOURCE_RUN="reports/v31_static_break_v22_fc4_boundary_event_flux_dwell12_ppo_seed45_b1p10_h082_prior0p5_balanced_eventfraction_20260613/raw/budget1p10_seed45"
OUT_DIR="reports/v31_static_break_v22_fc4_boundary_event_flux_dwell12_eventpair_replay_seed45_h082_20260613"
mkdir -p "$OUT_DIR"

python scripts/69_v31_eval_event_pair_policy.py \
  --source-run-dir "$SOURCE_RUN" \
  --out-dir "$OUT_DIR" \
  --oracle-device cpu \
  --env-min-dwell-steps 12 \
  --policy-spec "v22_eventflux_auto_non7_event15_l0=radiometer_basic,surface_temp_ir,ultrasonic_anemometer_hd,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v22_eventflux_auto_non7_event15_l6=radiometer_basic,surface_temp_ir,ultrasonic_anemometer_hd,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|6" \
  --policy-spec "v22_eventflux_auto_non2_event15_l0=met_station_core,radiometer_basic,surface_temp_ir,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v22_eventflux_auto_non2_event15_l6=met_station_core,radiometer_basic,surface_temp_ir,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|6" \
  --policy-spec "v22_eventflux_auto_non4_event15_l0=met_station_core,radiometer_basic,ultrasonic_anemometer_hd,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v22_eventflux_auto_non9_event15_l0=met_station_core,radiometer_basic,shielded_thermo_hygro,snow_particle_counter|met_station_core,radiometer_basic,laser_disdrometer|0" \
  --policy-spec "v22_eventflux_auto_non2_event21_l0=met_station_core,radiometer_basic,surface_temp_ir,snow_particle_counter|surface_temp_ir,shielded_thermo_hygro,fc4_flux|0" \
  --policy-spec "v22_eventflux_auto_non7_event21_l0=radiometer_basic,surface_temp_ir,ultrasonic_anemometer_hd,snow_particle_counter|surface_temp_ir,shielded_thermo_hygro,fc4_flux|0" \
  2>&1 | tee "$OUT_DIR/event_pair_replay.log"
