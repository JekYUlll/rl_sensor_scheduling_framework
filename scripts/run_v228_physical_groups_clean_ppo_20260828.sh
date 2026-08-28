#!/usr/bin/env bash
set -euo pipefail

# Clean PD-PPO development screen after the V227 physical-scene gate. This
# launcher changes no method component: it only raises the policy-training
# budget and locks fresh development seeds.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2471 2472 2473 2474 2475}"
export RUN_PREFIX_OVERRIDE="${RUN_PREFIX_OVERRIDE:-v228_physical_groups_clean_ppo_dev}"
export TOTAL_TIMESTEPS_OVERRIDE="${TOTAL_TIMESTEPS_OVERRIDE:-50000}"

exec bash scripts/run_v227_physical_groups_scene_gate_20260828.sh scene
