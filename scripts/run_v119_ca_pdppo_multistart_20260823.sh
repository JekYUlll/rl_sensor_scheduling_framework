#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for offset in 10000 11000; do
  POLICY_SEED_OFFSET="$offset" \
  RUN_PREFIX_OVERRIDE="v119_ca_pdppo_jointselect_init${offset}" \
  bash scripts/run_v118_ca_pdppo_jointselect_20260823.sh
done

python scripts/101_v32_select_policy_initialization.py \
  --run-glob 'reports/v118_ca_pdppo_jointselect_seed*_b1p75_20260822' \
  --run-glob 'reports/v119_ca_pdppo_jointselect_init*_seed*_b1p75_20260822' \
  --out-root reports/aggregate/v119_ca_pdppo_multistart_20260823
