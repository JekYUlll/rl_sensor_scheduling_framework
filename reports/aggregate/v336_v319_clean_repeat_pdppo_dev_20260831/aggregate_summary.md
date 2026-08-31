# V336 clean V319 repeat

## Protocol

- Corrected scenes 6811/6812 with the corresponding V294 control assets.
- Fresh policy seeds: 7251/7252.
- Same configuration as V319: static-normalized forecast reward, hard
  forecast-value pretraining, decision-only PPO updates, minimum dwell 6, and
  no forecast-value head/auxiliary or candidate-interaction head.
- Selected updates: 30 for seed 6811 and 20 for seed 6812, both from valid
  non-empty validation scenes.

## Predictive result

The value is `baseline - custom_ppo`; positive values indicate a PD-PPO win.

| Comparator | Ordinary-loss wins | Mean ordinary margin | Macro wins | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 0/2 | -0.080137 | 0/2 | -0.091588 |
| feasible static projected | 0/2 | -0.105102 | 1/2 | -0.008459 |
| AoI | 0/2 | -0.056580 | 0/2 | -0.042761 |
| round-robin | 0/2 | -0.084238 | 0/2 | -0.001047 |
| random | 0/2 | -0.099583 | 0/2 | -0.004951 |
| full-open unconstrained | 0/2 | -0.112271 | 0/2 | -0.026603 |

## Operational result

- Warm-up aborts: 0/2.
- Constant-on channels: 0/2.
- Constant-off channels: 1/2.
- Mid-duty channel counts: 3 and 4.
- Switching rates: 0.012520 and 0.029527 per step.

## Decision

V336 does not reproduce V319's transfer result. The negative result is treated
as evidence of substantial policy-seed sensitivity in the current training
pipeline; no final evaluation or primary-claim promotion is made.
