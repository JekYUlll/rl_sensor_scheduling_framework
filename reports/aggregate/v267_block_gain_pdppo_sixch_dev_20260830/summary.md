# V267 action-sensitive dwell-block reward

The protocol completed 2 fresh physical six-channel scene/policy seed pairs.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 0/2 | 0/2 | -0.040334 [-0.042456, -0.038213] | -0.096793 [-0.126093, -0.067494] |
| dynamic | 0/2 | 0/2 | 0/2 | -0.007487 [-0.011167, -0.003806] | -0.005586 [-0.005888, -0.005283] |
| full_open | 0/2 | 0/2 | 0/2 | -0.036566 [-0.044189, -0.028942] | -0.054503 [-0.055230, -0.053775] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
