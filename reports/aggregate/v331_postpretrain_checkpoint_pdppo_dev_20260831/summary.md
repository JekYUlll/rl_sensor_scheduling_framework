# V331 post-pretraining checkpoint-selection PD-PPO development aggregate

V331 evaluates two scene/policy seed pairs after excluding pretraining-only checkpoints below PPO update 5.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 1/2 | 1/2 | 1/2 | -0.022988 [-0.065838, +0.019862] | +0.000728 [-0.114689, +0.116144] |
| dynamic | 1/2 | 2/2 | 1/2 | +0.000079 [-0.037316, +0.037474] | +0.034673 [+0.012938, +0.056407] |
| full_open | 1/2 | 1/2 | 0/2 | -0.012622 [-0.025294, +0.000050] | +0.010417 [-0.000057, +0.020891] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
