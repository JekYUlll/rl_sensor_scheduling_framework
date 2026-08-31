# V321 weighted forced-dwell supervision, weight 0.10

Date: 2026-08-31

## Protocol

- Scene seeds: 6811 and 6812; policy seeds 7011 and 7012.
- All teacher states were retained. Hard forecast-value pretraining used
  lookahead 6, 16,384 pretraining steps, followed by 50,000 PPO timesteps.
- Forced-dwell rows received action-loss weight 0.10. State/context supervision
  still used the complete state batch.
- No bandit-dependent signal, test-time event label, residual action, or
  trainable action prior was used.

## Results

Baseline-minus-PD-PPO margins are positive when PD-PPO has lower loss.

| Reference | Ordinary mean | Ordinary wins | Macro mean | Macro wins |
|---|---:|---:|---:|---:|
| validation-selected static | -0.043407 | 0/2 | -0.042695 | 1/2 |
| feasible static | -0.019270 | 1/2 | +0.043222 | 2/2 |
| full-open unconstrained | -0.033040 | 0/2 | -0.033006 | 1/2 |
| AoI | -0.020339 | 0/2 | -0.008750 | 1/2 |
| random | -0.002853 | 1/2 | +0.065571 | 2/2 |
| round-robin | +0.002492 | 1/2 | +0.039726 | 2/2 |

## Behavior

- Warm-up aborts: 0/2 seeds.
- Always-on channels: 0/2 seeds.
- Always-off channels: 1 for each seed.
- Mid-duty channels: five and three.
- Switching rates: 0.029816 and 0.015559 per step.

## Decision

The weight-0.10 intervention does not repair predictive transfer or behavior.
Validation-selected static ordinary-loss wins remain `0/2`, and both seeds
contain an always-off channel. The forced-row action-weight family is closed;
no final-seed expansion is justified.
