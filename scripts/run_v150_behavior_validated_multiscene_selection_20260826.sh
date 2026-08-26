#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_PREFIX_OVERRIDE=v150_behavior_validated_train \
HOLDOUT_PREFIX_OVERRIDE=v150_behavior_validated_holdout \
CHECKPOINT_SELECTION_INTERVAL_UPDATES_OVERRIDE=5 \
CHECKPOINT_REQUIRE_VALID_BEHAVIOR_OVERRIDE=1 \
bash scripts/run_v147_interleaved_multiscene_holdout1505_20260826.sh
