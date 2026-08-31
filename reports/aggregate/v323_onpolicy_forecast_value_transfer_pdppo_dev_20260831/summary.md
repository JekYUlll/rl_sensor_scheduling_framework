# V323 on-policy forecast-value transfer PD-PPO development diagnostic

The protocol completed 2 corrected-scene seed/policy pairs. Context-alert and one-step forecast-greedy rows were not rerun in this diagnostic.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 1/2 | 0/2 | -0.073108 [-0.145077, -0.001139] | -0.054085 [-0.217120, +0.108950] |
| dynamic | 1/2 | 1/2 | 1/2 | -0.050041 [-0.116555, +0.016474] | -0.020140 [-0.089494, +0.049213] |
| full_open | 0/2 | 0/2 | 0/2 | -0.062742 [-0.104533, -0.020950] | -0.044396 [-0.081541, -0.007251] |

Behavior and feasibility gate: 1/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
