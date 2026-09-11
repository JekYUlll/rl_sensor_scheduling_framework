# Progress: PD-PPO Return-to-Basics

## 2026-09-12

- Read `docs/09-12-01.md` and reconciled it with V699/V700, V673/V685/V686/V688,
  and the real supply-log audit.
- Reframed the work into an executable algorithm track and a physically gated
  deployment track.
- Created `task_plan.md`, `findings.md`, and this progress log.
- No paper changes and no remote training jobs were started.

- Audited `src/v2/custom_ppo.py` and `src/v2/env.py`: forced dwell rows are
  already represented by a singleton feasible action mask, and the trainer
  supports decision-only policy updates and decision-block credit. Existing
  tests in `tests/v2/test_custom_ppo.py` cover these primitives.

## Next action

Add an explicit ladder manifest and run a CPU smoke check with the
execution-consistent flags before launching any remote experiment.

## Goal transition (2026-09-12)

The prior goal was not marked complete because its physical and policy-evidence
requirements remain unmet. A new abstract active goal is defined in
`.planning/GOAL.md`; subsequent planning and execution should use that file
as the authority while retaining the prior blocked route as evidence.
