# V280 matched-source event-balanced ordinary-forecast PD-PPO development

V280 repeats V278 with same-seed control assets and active validation-only checkpoint selection.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 1/2 | 0/2 | -0.068899 [-0.068971, -0.068827] | -0.039221 [-0.099261, +0.020819] |
| dynamic | 0/2 | 2/2 | 0/2 | -0.015089 [-0.029268, -0.000909] | +0.031311 [+0.013017, +0.049605] |
| full_open | 0/2 | 0/2 | 0/2 | -0.059650 [-0.093513, -0.025787] | -0.032143 [-0.038394, -0.025892] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
