# V330 plain forecast reward PD-PPO development aggregate

V330 retains V328 joint action representation and teacher initialization but replaces static subtype-normalized training reward with unnormalised forecast-loss reward to align training with the ordinary endpoint; no bandit-dependent signal is used.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 1/2 | 1/2 | 1/2 | -0.031024 [-0.082554, +0.020506] | -0.016650 [-0.151715, +0.118416] |
| dynamic | 1/2 | 1/2 | 1/2 | -0.007957 [-0.054032, +0.038118] | +0.017295 [-0.024089, +0.058680] |
| full_open | 1/2 | 1/2 | 1/2 | -0.020658 [-0.042010, +0.000694] | -0.006961 [-0.016136, +0.002215] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
