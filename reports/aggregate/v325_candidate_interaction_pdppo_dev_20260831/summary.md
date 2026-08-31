# V325 candidate interaction PD-PPO development aggregate

V325 enables the existing candidate-interaction state-action head while retaining the V324 context-aware masked PPO training configuration and disabling the forecast-value trust gate; two development seeds, no bandit-dependent component.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 1/2 | 0/2 | -0.033257 [-0.061946, -0.004567] | -0.019057 [-0.123630, +0.085515] |
| dynamic | 1/2 | 2/2 | 1/2 | -0.010190 [-0.033424, +0.013045] | +0.014887 [+0.003997, +0.025778] |
| full_open | 0/2 | 1/2 | 0/2 | -0.022891 [-0.024379, -0.021402] | -0.009368 [-0.030686, +0.011950] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
