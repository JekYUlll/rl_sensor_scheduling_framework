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

- Completed the relevant local execution-consistency tests: `72 passed`.
- Created an isolated remote snapshot at
  `/data/zhangzhuyu/pdppo_return_to_basics/framework_0c2eef0` so the dirty
  historical remote checkout is not modified.
- Started the q=1 regression wave in tmux session `pdppo-q1-regression`.
  Seeds are 41--44, using the historical exclusive-specialist sensor geometry,
  `B=0.75`, `max_active=2`, a mandatory core channel, minimum dwell 6, and
  `decision_only_policy_updates + decision_block_credit(sum)`. Frozen truth
  files are copied from the existing seed-specific no-warmup artifacts; this
  is an algorithm/semantics regression and is not admissible as new physical
  evidence.
- The first launch attempt terminated before training because the isolated
  snapshot lacked `scripts/23_v2_train_ppo.py`; the missing tracked helper was
  added and the wave was restarted. No result was taken from the failed launch.
- The corrected preflight audit, with `cr1000xe_backbone` explicitly required,
  reports 16 feasible masks at `B=2.15` and startup budget `2.60`. The mask
  cardinalities are 1:1, 2:5, and 3:10; the remaining 16 projected masks are
  infeasible under the declared fixed-cost geometry. This is the correct
  entity action surface for the current budget, not an assumed full 32-mask
  surface.

## Next action

Monitor the q=1 wave to completion, aggregate its seed-level metrics and
action-identity diagnostics, then proceed to exact 32-mask enumeration and a
policy-free controlled subset geometry screen.

## Goal transition (2026-09-12)

The prior goal was not marked complete because its physical and policy-evidence
requirements remain unmet. A new abstract active goal is defined in
`.planning/GOAL.md`; subsequent planning and execution should use that file
as the authority while retaining the prior blocked route as evidence.
