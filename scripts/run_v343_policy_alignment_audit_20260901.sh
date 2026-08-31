#!/usr/bin/env bash
set -euo pipefail

# V343 is an offline policy-alignment diagnostic.  It loads the completed
# V342 checkpoint and evaluates all feasible candidate costs from the same
# environment state as the frozen policy rollout.  Candidate costs are never
# exposed to the policy or used for training.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SEEDS=(6871 6872)
POLICY_SEEDS=(7351 7352)

run_one() {
  local idx="$1" seed="${SEEDS[$1]}" policy_seed="${POLICY_SEEDS[$1]}"
  local checkpoint="reports/v342_decision_only_bc_diag_seed${seed}_b1p75_20260822/custom_ppo.pt"
  local starts
  if [[ "$seed" == "6871" ]]; then
    starts="33300 33684 34452 34836 35220 35609"
  else
    starts="33300 33684 34116 34500 34884 35609"
  fi
  (
    export CUDA_VISIBLE_DEVICES="$idx"
    export RUN_PREFIX_OVERRIDE="v343_policy_alignment_audit"
    export LOG_PREFIX_OVERRIDE="v343_policy_alignment_audit"
    export SEEDS_OVERRIDE="$seed" POLICY_SEEDS_OVERRIDE="$policy_seed"
    export CONTROL_SOURCE_RUN_DIR_OVERRIDE="reports/v342_decision_only_bc_diag_seed${seed}_b1p75_20260822"
    export POLICY_CHECKPOINT_SOURCE_OVERRIDE="$checkpoint"
    export POLICY_ALIGNMENT_AUDIT_OUTPUT_OVERRIDE="reports/v343_policy_alignment_audit_seed${seed}_b1p75_20260822/policy_alignment_audit.csv"
    export EVAL_START_INDICES="$starts" EVAL_STEPS=384 EVAL_ROLLOUTS=6
    export TOTAL_TIMESTEPS_OVERRIDE=0
    export BC_PRETRAIN_DECISION_ONLY_OVERRIDE=1
    bash scripts/run_v341_recalibrated_scene_bc_only_pdppo_diag_20260901.sh
  ) >"logs/v343_policy_alignment_audit_seed${seed}.log" 2>&1
}

mkdir -p logs
run_one 0 & p1=$!
run_one 1 & p2=$!
wait "$p1" "$p2"
