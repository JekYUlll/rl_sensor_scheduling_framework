# PD-PPO Method and Baseline Reproducibility Audit

Date: 2026-07-18

## Scope and status

This note records the executable method contract for the SCENEBAL-2 PD-PPO
experiments requested by `docs/07-18-01.md`. It also records two defects found
while reconstructing that contract:

1. the archived SCENEBAL-2 policy supplied the exact simulator `event_flag` to
   the actor and event-aware critic; and
2. PPO and DQN episode environments were reconstructed field by field, omitting
   subtype reward multipliers, static normalizers, and other configuration
   fields.

The implementation has been corrected, but the correction changes the policy
training contract. The archived 24-seed aggregate therefore remains historical
evidence, not evidence for the corrected online-observability claim. A matched
pilot is retraining forecast-, AoI-, and uncertainty-reward PPO policies on
seeds 117 and 118 before any corrected 24-seed expansion.

## Authoritative source assets

The corrective runs reuse, for each seed, the archived SCENEBAL-2 truth,
frozen TCN evaluator, six candidate masks, validation-start indices,
validation-frozen static normalizers, selected fixed schedule, and final-start
indices. The strict preflight records are:

- `reports/v31_scenebal2_matched_reward_forecast_noexactevent_seed117_h075forecastctrl_20260718preflight/control_source_preflight.json`
- `reports/v31_scenebal2_matched_reward_forecast_noexactevent_seed118_h075forecastctrl_20260718preflight/control_source_preflight.json`

For seed 117, the source truth SHA-256 begins `e3a6`, the frozen evaluator
SHA-256 begins `e773`, the validation starts are 60128, 61300, 62453, and
63603, and the final starts are 64750, 65262, 66158, 66670, 67182, 67694,
68206, and 68718. For seed 118, the corresponding hash prefixes are `dc511`
and `dac94`, the validation starts are 60167, 61272, 62408, and 63555, and the
final starts are 64750, 65262, 65774, 66542, 67054, 67566, 68078, and 69294.
The full hashes are stored in the preflight JSON files.

## Partial-observation update

`WarmupSchedulingEnv` implements a normalized sample-and-hold observation
buffer. It is not a Kalman filter.

At epoch t, every selected sensor advances through its off, warming, or ready
runtime state. A ready sensor produces observations for its assigned variables
subject to the configured availability probability and Gaussian noise. If
several ready sensors observe the same scalar variable, their measurements are
combined using inverse-noise-variance weights. Wind direction is fused through
weighted circular means. Variables without a valid measurement retain their
last value. The newest value vector and binary observation mask are appended to
lookback histories of length 20.

For the uncertainty-reward control only, a diagonal variance state is updated
as

```
P_minus[t, j] = min(P[t-1, j] + Q[j], P_max)
P[t, j]       = (1 / P_minus[t, j] + 1 / R[t, j])^(-1), if j is observed
P[t, j]       = P_minus[t, j],                              otherwise
```

where `Q` is estimated from normalized first differences on the RL
normalization partition and `R` is obtained from the active sensor noise model.
This variance is used to define the matched uncertainty reward. It is not an
input to the current policy, and the manuscript must not claim otherwise.

## Policy observation

The corrected PPO and matched DQN receive the same online state vector:

1. normalized sample-and-hold value history;
2. binary observation-mask history;
3. per-sensor runtime mode and normalized warm-up time remaining;
4. per-sensor freshness, defined from the last ready acquisition attempt;
5. the previous executed sensor mask;
6. duty estimates when the duty guard is enabled;
7. current power ratio and diurnal sine/cosine;
8. partial-observation-derived regime summary features;
9. four supplied context signals (`particle`, `flux`, `thermal`, and their
   maximum event-alert score).

The exact simulator event flag is disabled at policy execution. The actor and
event-aware critic now use `agent_context_event_alert` rather than
`event_flag`. Exact event subtype labels remain available only on the RL
training partition for the existing auxiliary labels and event-conditioned
episode sampling.

The supplied context signals are synthetic alert proxies. The simulator forms
each proxy from the corresponding subtype interval, adds a 16-step leading
ramp and Gaussian noise (standard deviation 0.01), and clips it to [0, 1].
They model an upstream event-warning input; they are not outputs of a validated
field detector. This construction must be stated in the simulator appendix.

## Action set and online feasibility

The sensor configuration contains one mandatory meteorological core and five
optional channels. At budget 0.75, startup-peak budget 0.95, and maximum two
active sensors, exhaustive projection produces six unique candidate masks:

| Action | Executed subset |
|---:|---|
| 0 | meteorological core |
| 1 | meteorological core + radiometer |
| 2 | meteorological core + shielded temperature/humidity sensor |
| 3 | meteorological core + surface-temperature sensor |
| 4 | meteorological core + laser disdrometer |
| 5 | meteorological core + snow-flux sensor |

At each decision epoch, the policy produces one logit or Q value per candidate.
Candidates that are infeasible under the current startup and runtime state are
masked. The selected candidate then passes through the same power projector and
environment guard. A global minimum-duration rule of six epochs holds the last
executed mask for five further epochs after a change. Thus warm-up, steady
power, startup-peak power, maximum active count, and minimum duration are
enforced by the environment rather than learned through a soft penalty alone.

## Training objectives

The three matched PPO controls share the policy architecture, candidate masks,
observation, hard constraints, training windows, optimizer settings, and final
frozen evaluator. They differ only in the immediate objective used for PPO
training:

- **Forecast:** subtype-normalized multi-step loss from the fixed TCN evaluator.
- **AoI:** mean normalized age since each forecast target was last validly
  observed within the lookback buffer.
- **Uncertainty:** mean bounded diagonal target variance, `P / (1 + P)`.

All three are evaluated by the same frozen multi-step forecast loss. Training
return magnitudes are not compared across the three reward scales.

The matched Double-DQN comparator uses the forecast reward, the same state and
six candidate masks, online feasibility masking, the same RL and final
partitions, and no imitation, bandit prior, or final-test label. It is intended
to isolate learning-backbone effects rather than provide a broad RL benchmark.

## Executable baseline definitions

| Policy | Online input and decision rule | Selection or tie break | Constraint handling | Privilege |
|---|---|---|---|---|
| Validation-selected fixed schedule | Replays one constant candidate mask. | Evaluate all six masks on the validation starts; minimize the validation-frozen subtype-normalized macro forecast loss, then ordinary evaluator loss. | Same projector, warm-up model, and minimum-duration rule. | Deployable fixed design; no final labels. |
| Fixed priority | Assign linearly decreasing priority by sensor index at every epoch. | Deterministic projector order. | Same projector and runtime constraints. | None. |
| Round robin | Give highest scores to a cyclic group of `n_sensors // 3` sensors. | Sensor index and epoch determine the cycle. | Same projector; the reported constrained form uses the same minimum-duration rule. | None. |
| AoI rule | Score each sensor by time since its last ready acquisition attempt. | Project the score ranking. | Same projector; the reported constrained form uses the same minimum-duration rule. | None. |
| Random rule | Draw fixed-seed independent Gaussian sensor scores each epoch. | Project the sampled ranking. | Same projector; the reported constrained form uses the same minimum-duration rule. | Pseudorandom only. |
| Context-alert bandit | Read the three supplied synthetic alert scores; choose their argmax above threshold 0.5, otherwise calm. Replay a validation-selected mask for that context. | For calm/particle/flux/thermal, independently minimize the corresponding validation loss, then ordinary evaluator loss. | Same projector and minimum-duration rule. | Online synthetic alert proxy; no final labels. |
| One-step forecast greedy | Snapshot the environment, simulate every candidate for one step, and choose the smallest next-step frozen-evaluator loss. | Lower switching fraction, then lower action index. | Each simulated and executed step uses the same projector and guards. | Privileged final-future-loss diagnostic; not deployable. |
| Event-label reference | Use the exact current or look-ahead event subtype and replay its prescribed specialist mask. | First active subtype in the configured look-ahead window. | Same projector and guards. | Privileged simulator-label diagnostic; not deployable. |
| Full observation | Request all sensors at every epoch under an empty constraint object. | Not applicable. | Deliberately unconstrained. | Upper/reference condition, not a fair constrained baseline. |

The fixed-priority, round-robin, AoI, and random implementations are in
`src/v2/policies.py`. Validation selection and standard evaluation are in
`scripts/25_v2_train_custom_ppo.py`. Context-alert and one-step greedy are in
`scripts/81_v31_framework_baseline_supplements.py`. Privileged event-label
replays are in `scripts/70_v31_split_replay_gate.py`.

## Current evidence gate

The bounded pilot consists of corrected forecast, AoI, and uncertainty PPO on
seeds 117 and 118, each trained for 200,000 steps, plus a matched Double-DQN
pilot. The forecast policy must preserve positive held-out margins against the
validation-selected fixed schedule and conventional rules, while satisfying
the existing behavior checks, before a corrected multi-seed expansion is
launched. Reward alignment will then be judged by paired final forecast loss,
not by training returns. A second robustness experiment is deferred until
these mandatory controls are frozen.

## Manuscript consequences

Before the new evidence is inserted, the paper requires four global repairs:

1. replace Kalman-like `estimator state and uncertainty` wording with the
   actual sample-and-hold partial-observation update;
2. describe the online event input as a supplied synthetic warning signal and
   reserve exact event labels for training-only auxiliaries or privileged
   diagnostics;
3. define conventional rules and context-aware/privileged diagnostics as
   separate comparator families; and
4. replace `24 independent held-out seeds` with `24 evaluation seeds,
   including a 22-seed post-pilot replication` wherever that archived result is
   discussed.

The paper remains a prediction-driven constrained scheduling method paper.
These changes make its implementation and evidence contract reconstructable;
they do not reframe it as a benchmark catalogue.
