# PROJECT_BRIEF: PD-PPO Strong-Claim Autoresearch

## Goal
Find and validate a PD-PPO/RL sensor-scheduling framework that is forecast-optimal
under the tested protocol and produces genuinely state-dependent scheduling:
not fixed sensors, not a simple round-robin or small periodic rotation.

## Current Target Journal
Expert Systems with Applications (ESWA).

## Active Remote
Use only SSH alias `remote-gpu`.

Remote project path:
`~/_code/microclimate_demo/rl_sensor_scheduling_framework`

Conda environment:
`darts`

## Current Experiment Track
BO-1: `v31_metpair_backbone_context_ortholinear_balancedobjective`.

The active 24h runner is remote tmux:
`bo24_autonomy_20260621`.

It runs 12-seed waves, aggregates after every wave, and writes reports under:
`reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_41_<end_seed>_20260621_oldclaim/`.

## Current Evidence Boundary
Through seeds 41--66:
- macro old-claim gate: 26/26
- learned operational macro: 26/26
- behaviour complexity: 26/26
- strict explicit replay step gate: 20/26

The ESWA claim is currently strong for regime-balanced event-subtype scheduling,
but not yet an unqualified step-weighted strict replay claim.

## Next Hypothesis
`subtype_static_auto` may close step-gate failures by using static-candidate
calm/subtype losses to choose the AWBC teacher instead of hand-written
calm/particle/flux/thermal masks.

Seed55 step diagnostic already supports this direction:
`subtype_auto` replay with no static duty guard passes the strict step gate.

## Autonomy Rules
- Keep the 24h BO runner alive unless it fails.
- Do not kill existing remote experiments.
- Do not use old IPs, UniVPN, aTrust, or hardcoded host addresses.
- After every wave, sync aggregate outputs locally and update findings/progress.
- If 10 complete hypothesis rounds do not produce a stronger step claim, pivot to
  deeper teacher/framework/simulator changes.
