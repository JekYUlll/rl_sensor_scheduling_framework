# 2026-05-23 Curriculum PPO Mechanism Diagnosis

## Protocol Status Update - 2026-05-26

This document now records mechanism observations from a non-independent curriculum
procedure, not submission-level learned-policy evidence. The saved storm evaluation
starts equal the PPO training starts for all five completed seeds, and the later
full-distribution replay intersects the training/oracle evaluation support. See
`reports/energy_account_protocol_audit_20260526/energy_account_protocol_audit_summary.json`.

## Question

The storm-window curriculum PPO result is positive at the aggregate level:

- PPO mean oracle loss over seeds 41/42/43: `0.4136 +/- 0.0061`
- AoI mean oracle loss over seeds 41/42/43: `0.4194 +/- 0.0141`

However, the mean laser event/non-event selected ratio is only `1.03x`. This diagnosis checks whether PPO is winning through the intended event-triggered laser mechanism or through a different allocation strategy.

## Existing Rollout Limitation

The saved rollout files contain total `warmup_abort_count`, but not per-step abort markers. Per-step aborts can still be reconstructed exactly from `mode_ids`, `selected_masks`, and `step_indices`:

- if sensor mode at step `t-1` is `WARMING`;
- and the same sensor is not selected at step `t`;
- and `step_indices[t] == step_indices[t-1] + 1`;
- then `SensorRuntime.begin_step()` increments the abort count at step `t`.

This reconstruction matches the stored total abort counts for PPO, AoI, round-robin, and random in all three seeds.

## Abort Diagnosis

For PPO across seeds 41/42/43:

| sensor | total aborts | event aborts | non-event aborts |
|---|---:|---:|---:|
| laser_disdrometer | 41 | 10 | 31 |
| snow_particle_counter | 16 | 11 | 5 |

This does not support the hypothesis that laser event triggering is mainly being suppressed by event-time SOC aborts. Most laser aborts occur during non-event steps.

## Event/Non-Event Allocation

Mean PPO event/non-event selected ratios over seeds:

| sensor | ratio |
|---|---:|
| radiometer_basic | `1.63x` |
| snow_particle_counter | `1.13x` |
| fc4_flux | `1.04x` |
| laser_disdrometer | `1.03x` |
| surface_temp_ir | `0.64x` |

PPO is not abandoning laser; it selects laser at a high rate in both event and non-event portions. But it is not using laser as a strongly event-conditioned instrument.

## Where PPO Beats AoI

Mean event/non-event oracle losses over seeds:

| policy | event loss | non-event loss |
|---|---:|---:|
| PPO | `0.3274` | `0.5256` |
| AoI | `0.3243` | `0.5431` |

AoI is slightly better on event steps. PPO wins overall mainly by reducing non-event loss while maintaining comparable event loss.

## Interpretation

The current curriculum result supports learnability, but not the intended event-triggered laser mechanism.

The best-supported explanation is:

> PPO learns a conservative storm-window allocation and energy-management strategy. It preserves enough event performance while reducing non-event degradation, rather than learning a clean laser-on-event / laser-off-calm policy.

This is distinct from the original target mechanism.

## Implications for Next Optimization

1. SOC soft penalty may reduce aborts, but the current data do not indicate that event-time laser aborts are the main bottleneck.
2. Increasing event-step reward importance is more directly aligned with the observed gap, because PPO currently loses slightly to AoI on event steps and wins in non-event steps.
3. A laser-specific claim remains unsafe unless a follow-up run shows a large increase in laser event/non-event ratio without destroying SOC feasibility.

Recommended next test:

- keep the same storm-window curriculum setup;
- add a default-off event-loss multiplier;
- run one seed with event multiplier `1.5`;
- evaluate whether event loss improves and whether laser event/non-event ratio increases.

Acceptance criteria for mechanism improvement:

- PPO still beats or matches AoI on overall oracle loss;
- laser event/non-event selected ratio increases materially above `1.03x`;
- warmup aborts do not exceed the current three-seed range by a large margin;
- event-step oracle loss moves closer to or below AoI.

## Event Multiplier Probe Result

A single seed probe with `event_reward_multiplier = 1.5` was run using the same storm-window curriculum setup as seed 41.

| run | PPO oracle | AoI oracle | PPO event loss | PPO non-event loss | PPO aborts |
|---|---:|---:|---:|---:|---:|
| baseline seed 41 | `0.4106` | `0.4130` | `0.3307` | `0.5144` | `5` |
| event x1.5 seed 41 | `0.4088` | `0.4118` | `0.3247` | `0.5180` | `7` |

The event multiplier improved event loss and overall oracle loss without materially increasing aborts. However, it still did not produce event-triggered laser gating:

| run | laser event selected | laser non-event selected | ratio |
|---|---:|---:|---:|
| baseline seed 41 | `0.519` | `0.744` | `0.70x` |
| event x1.5 seed 41 | `0.740` | `0.768` | `0.96x` |

The multiplier makes laser a higher-duty storm-window instrument, not an event-step-gated instrument.

## Additional Mechanism Check

Non-event gaps inside the selected storm windows are usually short, but the high non-event laser rate is not explained only by bridging short calm gaps:

- non-event gap median: `5` steps;
- non-event gap 75th percentile: `6` steps;
- maximum non-event gap: `419` steps.

For baseline seed 41, laser selected rate remains high even for non-event steps far from the nearest event:

- distance 33+ steps from event: `0.867`;
- event steps: `0.519`.

For the event x1.5 run:

- distance 33+ steps from event: `0.798`;
- event steps: `0.740`.

Thus the low event/non-event laser ratio is not merely a warmup-bridge artifact. The current policy treats laser as a storm-context sensor rather than as an event-triggered sensor.

## Revised Interpretation

The event multiplier is useful for improving event loss, but it does not solve the mechanism mismatch. The current defensible claim remains:

> PPO can learn a storm-window adaptive allocation that beats AoI on oracle loss, but the learned mechanism is not robust event-triggered laser gating.

If the paper requires laser-gating evidence, the next change should alter the scenario or objective so that laser's value is uniquely event-local. Generic event-loss weighting is insufficient.

## Full-Distribution Generalization Check

The three curriculum PPO models were also evaluated without retraining on random full-distribution windows. This used six 1024-step windows per seed, with mean event rate about `0.296`, instead of the storm-window event rate about `0.565`.

| seed | event rate | PPO oracle | AoI oracle | static projected oracle |
|---|---:|---:|---:|---:|
| 41 | `0.295` | `0.3106` | `0.3116` | `0.3358` |
| 42 | `0.273` | `0.3311` | `0.3349` | `0.3220` |
| 43 | `0.320` | `0.2973` | `0.3016` | `0.3349` |

Mean over seeds:

| policy | oracle loss |
|---|---:|
| PPO | `0.3130 +/- 0.0171` |
| AoI | `0.3160 +/- 0.0171` |
| feasible static projected | `0.3309 +/- 0.0077` |
| round-robin | `0.3422 +/- 0.0241` |
| random | `0.3471 +/- 0.0236` |

This result is useful but should be framed carefully:

- PPO remains slightly better than AoI in all three seeds.
- PPO beats the static projected baseline on average, but not in every seed; seed 42 static projected is better.
- The laser event/non-event selected ratio remains near-neutral (`1.04x`), while `radiometer_basic` remains the clearest event-conditioned sensor (`2.04x`).

Thus the current evidence supports generalization of the learned allocation advantage beyond the hand-picked storm windows, but it still does not establish robust dominance over every static baseline in every seed.

## Five-Seed Curriculum Extension

Seeds 44 and 45 were added with the same storm-window curriculum configuration. This changed the interpretation.

Storm-window evaluation:

| policy | mean oracle loss |
|---|---:|
| PPO | `0.4153 +/- 0.0051` |
| AoI | `0.4176 +/- 0.0105` |
| feasible static projected | `0.4742 +/- 0.0236` |
| round-robin | `0.4451 +/- 0.0167` |
| random | `0.4565 +/- 0.0140` |

Per-seed wins:

| comparison | PPO wins |
|---|---:|
| vs AoI | `3/5` |
| vs feasible static projected | `5/5` |
| vs round-robin | `5/5` |
| vs random | `5/5` |

The new seeds weaken the claim that PPO reliably beats AoI. Seeds 44 and 45 lose narrowly to AoI. However, the storm-window claim against static projected, round-robin, and random is stronger after expansion.

Full-distribution no-retrain evaluation was also extended to seeds 44 and 45:

| policy | mean oracle loss |
|---|---:|
| PPO | `0.3155 +/- 0.0133` |
| AoI | `0.3168 +/- 0.0135` |
| feasible static projected | `0.3318 +/- 0.0062` |
| round-robin | `0.3375 +/- 0.0195` |
| random | `0.3431 +/- 0.0188` |

Per-seed full-distribution wins:

| comparison | PPO wins |
|---|---:|
| vs AoI | `4/5` |
| vs feasible static projected | `4/5` |
| vs round-robin | `5/5` |
| vs random | `5/5` |

Revised claim boundary:

> Curriculum PPO consistently beats static projected, round-robin, and random in storm-window evaluation, and shows a small average full-distribution advantage over AoI. It does not yet support a robust per-seed claim over AoI or a robust per-seed claim over static projected in full-distribution evaluation.

## SOC Soft-Penalty Probe

A seed-41 probe added a soft SOC penalty with `soc_soft_penalty_buffer=20` and `lambda_soc_soft_penalty=0.01`.

| run | PPO oracle | PPO event loss | PPO non-event loss | aborts | mean power |
|---|---:|---:|---:|---:|---:|
| baseline seed 41 | `0.4106` | `0.3307` | `0.5144` | `5` | `1.0128` |
| SOC penalty seed 41 | `0.4069` | `0.3234` | `0.5155` | `9` | `0.9960` |

The SOC penalty improves overall and event loss while reducing mean power, but it does not reduce warmup aborts in this seed. The rollout also records `31` energy-guard-dropped selections and SOC reaches the reserve boundary.

This probe should not be treated as an abort-management solution yet. It is better interpreted as a promising event-performance modification that needs a multi-seed check before becoming a default setting.
