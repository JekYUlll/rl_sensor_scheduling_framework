# Findings: PD-PPO Return-to-Basics

## 2026-09-12 starting evidence

- The original `q=1` task made specialist competition explicit by allowing at
  most one optional channel. Its success does not establish arbitrary-subset
  generalization.
- The current entity system has five selectable channels and one mandatory
  logger backbone, so exact enumeration contains at most 32 masks.
- V699/V700 established state-dependent resource feasibility but not stable
  downstream forecast opportunity across seeds. They remain diagnostic only.
- V673/V685/V686/V688 already tested heater-state/quality/resource coupling and
  did not pass the all-seed geometry gate.
- Real room/freezer logs contain voltage and rare Parsivel heater telemetry,
  but not system current, charging power, SOC, or an Antarctic deployment
  distribution. They cannot define a new SOC scene.
- The known algorithmic risk is a dwell-lock mismatch: the policy can propose
  an action and store its probability while the environment executes the prior
  action. Flexible masks increase the number of such uncontrolled proposals.
- The current implementation already has an opt-in correction path:
  `feasible_candidate_mask` exposes only the held mask during a valid dwell
  lock, `decision_only_policy_updates` excludes forced rows from actor loss,
  and `decision_block_credit` aggregates rewards to decision epochs. Existing
  tests cover the action-mask singleton and decision-block calculations. The
  next task is therefore a configuration/coverage audit, not an immediate
  rewrite of PPO internals.

## Scientific separation

The required causal hierarchy is:

`physical/resource variation -> complete-subset forecast geometry -> online
observability -> executable opportunity -> RL optimization`.

A failure before the last stage is not evidence that PPO is intrinsically
ineffective. Conversely, a PPO result cannot repair a zero or negligible
policy-free opportunity gap.

## Current decision

Continue through the algorithm ladder and execution-consistency audit. Keep the
entity physical route paused until new system-level telemetry or a documented
deployment resource trace is available.
