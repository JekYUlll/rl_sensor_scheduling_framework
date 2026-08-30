# V268 relative action-sensitive dwell-block reward

The protocol completed 2 fresh physical six-channel scene/policy seed pairs.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 0/2 | 0/2 | -0.096345 [-0.141467, -0.051223] | -0.124932 [-0.180367, -0.069498] |
| dynamic | 1/2 | 0/2 | 0/2 | -0.012298 [-0.026063, +0.001467] | -0.008471 [-0.011302, -0.005639] |
| full_open | 0/2 | 0/2 | 0/2 | -0.064269 [-0.090467, -0.038071] | -0.091523 [-0.127604, -0.055441] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
