#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ALIGNED_QUALITY_ACTION_SCORE=1 \
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-1601}" \
GPU_IDS="${GPU_IDS:-0}" \
RUN_PREFIX_OVERRIDE=v166_aligned_quality_actor_pilot \
LOG_DIR_OVERRIDE=logs/v166_aligned_quality_actor_pilot \
bash scripts/run_v163_crn_quality_pdppo_20260826.sh
