# V322 temporal hard-pretraining PD-PPO development diagnostic

The protocol completed 2 corrected-scene seed/policy pairs. Context-alert and one-step forecast-greedy rows were not rerun in this diagnostic.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 1/2 | 1/2 | 1/2 | -0.034538 [-0.069745, +0.000669] | -0.063906 [-0.222869, +0.095057] |
| dynamic | 1/2 | 1/2 | 1/2 | -0.011471 [-0.041223, +0.018281] | -0.029961 [-0.095242, +0.035320] |
| full_open | 0/2 | 0/2 | 0/2 | -0.024172 [-0.029201, -0.019143] | -0.054217 [-0.087290, -0.021144] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
