# V278 event-balanced ordinary-forecast PD-PPO development

The matched two-seed event-balanced training protocol completed 2 scene/policy seed pairs.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 1/2 | 0/2 | -0.021626 [-0.037640, -0.005611] | +0.000314 [-0.066175, +0.066804] |
| dynamic | 1/2 | 1/2 | 1/2 | -0.001235 [-0.013077, +0.010608] | +0.023106 [-0.015876, +0.062089] |
| full_open | 0/2 | 0/2 | 0/2 | -0.038540 [-0.040602, -0.036479] | -0.024024 [-0.026976, -0.021073] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
