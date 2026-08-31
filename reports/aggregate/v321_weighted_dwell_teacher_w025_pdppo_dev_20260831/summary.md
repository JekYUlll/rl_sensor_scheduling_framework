# V321 weighted forced-dwell supervision

Date: 2026-08-31

## Protocol

- Scene seeds: 6811 and 6812; policy seeds 7011 and 7012.
- The run retained all teacher states and used hard forecast-value pretraining
  with lookahead 6, 16,384 pretraining steps, and 50,000 PPO timesteps.
- The action loss on forced dwell rows was weighted by 0.25. Auxiliary
  state/context supervision still used the complete state batch.
- No bandit-dependent signal, test-time event label, residual action, or
  trainable action prior was used.

## Results

Baseline-minus-PD-PPO margins are positive when PD-PPO has lower loss.

| Reference | Ordinary mean | Ordinary wins | Macro mean | Macro wins |
|---|---:|---:|---:|---:|
| validation-selected static | -0.043538 | 0/2 | +0.000477 | 1/2 |
| feasible static | -0.019402 | 0/2 | +0.086393 | 2/2 |
| full-open unconstrained | -0.033172 | 0/2 | +0.010166 | 1/2 |
| AoI | -0.020471 | 0/2 | +0.034422 | 2/2 |
| random | -0.002984 | 0/2 | +0.108743 | 2/2 |
| round-robin | +0.002360 | 1/2 | +0.082898 | 2/2 |

## Behavior

- Warm-up aborts: 0/2 seeds.
- Always-on channels: 0/2 seeds.
- Always-off channels: 0 for seed 6811 and 1 for seed 6812.
- Mid-duty channels: five and three.
- Switching rates: 0.025691 and 0.003040 per step.

## Decision

The weight-0.25 intervention does not repair predictive transfer. Ordinary
validation-static wins remain `0/2`, and the mean ordinary margin is strongly
negative. The small positive macro mean is driven by only one seed and does
not satisfy the primary endpoint gate. The variant is rejected for expansion.
