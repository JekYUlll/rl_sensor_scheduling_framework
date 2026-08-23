# V104 frequency-cost PD-PPO development evaluation

The bounded protocol completed 5 frozen development scene/policy seed pairs.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 3/5 | 3/5 | 2/5 | +0.005510 [-0.046444, +0.047628] | -0.010522 [-0.174705, +0.117256] |
| dynamic | 4/5 | 4/5 | 4/5 | +0.035403 [-0.036738, +0.096038] | +0.098719 [-0.091005, +0.227150] |
| context | 4/5 | 3/5 | 3/5 | -0.013132 [-0.061944, +0.015241] | -0.057986 [-0.229489, +0.041551] |
| exact_label | 4/5 | 3/5 | 3/5 | -0.012871 [-0.058115, +0.014146] | -0.058839 [-0.220791, +0.038814] |
| full_open | 2/5 | 3/5 | 2/5 | -0.008167 [-0.079332, +0.056988] | -0.032929 [-0.240690, +0.124415] |

Behavior and feasibility gate: 3/5 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
