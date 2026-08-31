# V327 quality-context action PD-PPO development aggregate

V327 retains V325 except it disables candidate interaction and enables the existing online quality-context candidate utility head; two development seeds, no bandit-dependent component.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 1/2 | 1/2 | 1/2 | -0.004218 [-0.042344, +0.033908] | -0.023905 [-0.195079, +0.147269] |
| dynamic | 1/2 | 1/2 | 1/2 | +0.018849 [-0.013822, +0.051520] | +0.010040 [-0.067452, +0.087533] |
| full_open | 1/2 | 1/2 | 1/2 | +0.006148 [-0.001800, +0.014096] | -0.014216 [-0.059500, +0.031068] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
