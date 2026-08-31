# V328 joint action utility PD-PPO development aggregate

V328 retains the V325/V327 clean PD-PPO scaffold and enables both candidate interaction and online quality-context action utility; no bandit-dependent component is used.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 1/2 | 1/2 | 1/2 | -0.026332 [-0.065106, +0.012443] | -0.063097 [-0.198189, +0.071996] |
| dynamic | 1/2 | 1/2 | 1/2 | -0.003264 [-0.036584, +0.030055] | -0.029152 [-0.070563, +0.012259] |
| full_open | 0/2 | 0/2 | 0/2 | -0.015965 [-0.024562, -0.007369] | -0.053408 [-0.062610, -0.044205] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
