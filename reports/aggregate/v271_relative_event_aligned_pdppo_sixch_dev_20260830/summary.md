# V271 relative block-gain event-start alignment

The protocol completed 2 fresh physical six-channel scene/policy seed pairs.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 1/2 | 1/2 | 1/2 | -0.019849 [-0.049950, +0.010251] | +0.036350 [-0.007021, +0.079722] |
| dynamic | 0/2 | 0/2 | 0/2 | -0.010182 [-0.012883, -0.007480] | -0.020801 [-0.032270, -0.009333] |
| full_open | 0/2 | 0/2 | 0/2 | -0.046834 [-0.062347, -0.031322] | -0.090452 [-0.140160, -0.040744] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
