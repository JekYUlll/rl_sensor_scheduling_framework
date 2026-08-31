# V320 decision-only hard forecast-value pretraining

Date: 2026-08-31

## Protocol

- Scene seeds: 6811 and 6812; policy seeds: 7001 and 7002.
- V320 keeps V319's dwell-aligned hard forecast-value teacher
  (`greedy-lookahead=6`), 16,384 pretraining steps, and 50,000 decision-only
  PPO timesteps.
- The controlled change is to retain only states at which a new executable
  action can be selected during hard pretraining. Forced dwell rows are
  excluded from the pretraining loss.
- No bandit-dependent signal, test-time event label, residual action, or
  trainable action prior is used.

## Results

Baseline-minus-PD-PPO margins are positive when PD-PPO has lower loss.

| Reference | Ordinary mean | Ordinary wins | Macro mean | Macro wins |
|---|---:|---:|---:|---:|
| validation-selected static | -0.06207826419822651 | 0/2 | -0.11337990022836941 | 0/2 |
| feasible static | -0.037941887226821736 | 0/2 | -0.027463407524587424 | 1/2 |
| full-open unconstrained | -0.05171214322544007 | 0/2 | -0.1036908813907721 | 1/2 |
| AoI | -0.03901113478749979 | 0/2 | -0.07943499314444458 | 1/2 |
| random | -0.021524284534836985 | 0/2 | -0.005113637750391198 | 1/2 |
| round-robin | -0.01618008664953688 | 0/2 | -0.03095876183507129 | 1/2 |

## Behavior

- Warm-up aborts: 0/2 seeds.
- Always-on channels: 0/2 seeds.
- Always-off channels: 0/2 seeds for seed 6811 and 1 for seed 6812.
- Mid-duty channels: 6 and 3.
- Switching rates: 0.03907946157186279 and 0.005355333622810826 per step.

## Decision

The decision-only pretraining filter causes a clear predictive regression:
validation-selected static ordinary and macro wins are both `0/2`, and the
mean margins are strongly negative. It is rejected as a primary repair. The
result indicates that removing forced rows without compensating the teacher
state distribution loses useful temporal context; no final-seed expansion is
justified.
