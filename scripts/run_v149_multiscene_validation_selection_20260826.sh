#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_PREFIX_OVERRIDE=v149_multiscene_validation_train \
HOLDOUT_PREFIX_OVERRIDE=v149_multiscene_validation_holdout \
CHECKPOINT_SELECTION_INTERVAL_UPDATES_OVERRIDE=5 \
bash scripts/run_v147_interleaved_multiscene_holdout1505_20260826.sh
