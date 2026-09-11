# Active Research Goal

## Status

`active`

The previously scoped arbitrary-subset PD-PPO goal is retained as historical
evidence and is not falsely marked complete. Its current physical deployment
track is blocked because no system-level power/SOC trace is available and the
existing policy-free geometry routes did not pass their all-seed gates.

## Abstract objective

Develop and verify a reproducible, prediction-driven constrained sensor
scheduling system whose algorithmic claims, physical assumptions, executable
constraints, and empirical evidence remain aligned. The work must distinguish
algorithm capability, task-level adaptive opportunity, online observability,
and deployment-specific physical validity. No result is promoted to a policy
claim until the preceding evidence gates pass.

## Authority

This file is the active goal specification for subsequent work. The current
operational plan is:

`.planning/2026-09-12-pdppo-return-to-basics/`

The following files provide evidence but do not override this goal:

- `docs/09-12-01.md`
- `reports/arbitrary_subset_exploration_status_20260911.md`
- V699/V700 geometry reports
- historical planning logs and manuscript drafts

## Work tracks

### Algorithm track

Establish a clean PD-PPO capability ladder:

1. Reproduce the original exclusive-specialist (`q=1`) capability.
2. Verify exact arbitrary-subset mask enumeration.
3. Verify execution-consistent dwell semantics and decision-only PPO updates.
4. Construct a controlled subset-level forecast-value benchmark.
5. Verify chronological online transfer from deployable observations.
6. Run development PPO only after geometry and transfer gates pass.
7. Run fresh confirmation seeds only after the method and seed list are frozen.

### Physical deployment track

Maintain the entity six-channel hardware mapping, power manifest, and dynamic
resource implementation. Resume physical scenario claims only after obtaining
system-level power/SOC/controller evidence or a separately documented
deployment resource trace. Until then, preserve V699/V700 and heater-quality
routes as diagnostic evidence and do not retune them.

## Evidence gates

- Exact feasible action masks and zero runtime constraint violations.
- Complete-subset operating opportunity gap greater than `0.01` on every
  development gate seed.
- Empty intersection of `0.01` near-optimal static subsets.
- Held-out online-transfer capture ratio greater than `0.3`.
- Zero warm-up aborts and no unexplained forced-action mismatch.
- At least four development seeds before any confirmation wave.
- At least twelve fresh confirmation seeds with no post-selection tuning.

## Prohibited shortcuts

- Do not use bandit-dependent priors, residual actions, margin rewards, or
  counterfactual bandit labels in the primary PD-PPO method.
- Do not hide losing seeds or select a scene after observing final-test policy
  performance.
- Do not convert device manuals into synthetic SOC trajectories without
  measured or documented system-level support.
- Do not mark this goal complete merely because code, reports, or a paper
  compile successfully.

## Required reporting

Every phase must update `task_plan.md`, `findings.md`, and `progress.md` in the
active planning directory. Completed experiments must record commands, seed
lists, artifacts, gates, and whether results are admissible as policy
evidence. Repository changes must be selectively committed and pushed without
including unrelated historical worktree files.
