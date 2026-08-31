# V324 state-dependent teacher-trust PD-PPO development aggregate

State-dependent forecast-value trust gate; same feature-parity context-aware PD-PPO scaffold; development seeds 6811/6812; no bandit-dependent objective or prior.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 0/2 | 0/2 | -0.019251 [-0.032951, -0.005552] | -0.058589 [-0.073104, -0.044074] |
| dynamic | 1/2 | 1/2 | 1/2 | +0.003816 [-0.015338, +0.022969] | -0.024644 [-0.103811, +0.054523] |
| full_open | 1/2 | 1/2 | 1/2 | -0.008885 [-0.052762, +0.034991] | -0.048900 [-0.160275, +0.062476] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
