# PD-PPO Return-to-Basics Plan (2026-09-12)

## Goal

Reorganize the flexible-subset work into two separable evidence tracks. The
algorithm track must establish that execution-consistent masked PD-PPO scales
from the original exclusive-specialist task to an exactly enumerated
arbitrary-subset action space. The physical track must not claim dynamic
energy-aware deployment value without measured system-level power/SOC inputs.

The final target remains a defensible PD-PPO result for the entity six-channel
system. A policy result is admissible only after the relevant policy-free
geometry, observability, and executable-opportunity gates pass.

## Scope boundaries

- Primary policy remains forecast-loss PD-PPO with categorical hard-feasible
  mask actions.
- Five selectable channels plus the mandatory CR1000Xe backbone produce at
  most 32 selectable masks; no combinatorial approximation is needed yet.
- No bandit prior, residual action, counterfactual label, or bandit-margin
  reward enters the primary method.
- No PPO is trained on V699/V700 or any closed heater-quality route.
- The paper is not modified during this planning and evidence phase.
- All long jobs run on `remote-gpu` under `tmux`; local execution is limited
  to syntax, unit tests, and aggregation.

## Acceptance gates

### Algorithm ladder

1. `q=1` regression reproduces the known exclusive-specialist capability.
2. 32-mask feasibility enumeration is exact and has no runtime violations.
3. Locked steps use a singleton action mask containing the currently executed
   mask, or are represented as decision epochs; stored PPO action and executed
   action must agree.
4. A controlled arbitrary-subset scene with predeclared subset-level forecast
   opportunity passes the policy-free geometry gate.
5. A low-capacity online observation probe captures meaningful opportunity on
   held-out chronological windows (`eta_transfer > 0.3` target).
6. Only then run a four-seed PPO development probe; fresh confirmation requires
   at least 12 locked seeds with no post-selection tuning.

### Physical deployment track

The entity route may advance only after a new external input provides at least
one of system power, battery current, SOC, charging/generation power, or a
documented deployment resource trace. The new input must be frozen before truth
generation. The same resource occupancy, feasible-frontier, subset-loss,
executability, and observability gates then apply.

## Phases

### Phase A: evidence and implementation audit

- Inventory the original `q=1` assets, current arbitrary-mask helpers, dwell
  execution, and rollout buffer semantics.
- Write a minimal invariant test for forced-step action identity.
- Decide whether the singleton-mask fix belongs in the environment, policy
  collector, or both; do not silently change historical experiment defaults.

### Phase B: execution-consistent PD-PPO

- Add an opt-in execution-consistent mode.
- At a dwell-locked step, expose only the currently executed mask to the actor
  and store that same mask/log-probability in the rollout.
- Preserve rewards and critic/GAE updates; do not invent a bandit-derived
  objective.
- Log forced-step count, override count, executed/proposed equality, and
  action-mask cardinality.
- Verify old default behavior remains available for historical reproduction.

### Phase C: minimal benchmark ladder

- Run the known `q=1` regression with the corrected semantics.
- Build a controlled 32-mask synthetic benchmark in which subset-level
  forecast values, not merely individual channel scores, have a predeclared
  state-dependent crossover.
- Keep target dynamics, observation access, and resource constraints fixed
  while changing one mechanism at a time.
- Evaluate fixed, myopic forecast-greedy, contextual probe, original PD-PPO,
  and execution-consistent PD-PPO before any fresh physical claim.

### Phase D: physical re-entry gate

- Do not reuse V699/V700 as positive policy evidence.
- When new telemetry exists, generate a frozen hardware/resource manifest and
  a trace provenance record.
- Re-run resource occupancy, feasible-set Jaccard, complete-subset losses,
  near-optimal intersection, and executable oracle screens.
- Proceed to online transfer/PPO only if all four fresh entity seeds pass.

### Phase E: confirmation and packaging

- Freeze code/configuration and seed list before confirmation.
- Run paired seed-level comparisons and behavior/constraint audits.
- Aggregate ordinary and macro margins, win counts, bootstrap intervals,
  static-regret coverage, switching, warm-up, and action identity metrics.
- Update CHANGELOG and evidence manifests; only then consider manuscript use.

## Explicit stop rules

- If complete-subset opportunity is below `0.01`, stop before PPO.
- If held-out online transfer is below the declared target, stop before PPO.
- If singleton-lock semantics cannot be verified, stop policy comparisons.
- If a proposed change only adjusts a threshold, quality floor, target weight,
  or budget after seeing geometry, reject it as post-hoc scene fitting.

## Deliverables

- `algorithm_ladder_manifest.json`
- execution-consistency tests and opt-in implementation
- policy-free subset geometry reports
- chronological observability/transfer report
- seed-locked PPO development and confirmation summaries
- physical-input provenance manifest when the deployment track resumes
