# V329 decaying forecast AWBC PD-PPO development aggregate

V329 retains the V328 clean PD-PPO actor and forecast-loss objective and adds only a small forecast-greedy advantage-weighted behavior-cloning term during PPO, linearly decayed to zero by 30,000 timesteps; no bandit-dependent signal is used.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 0/2 | 0/2 | -0.026713 [-0.040446, -0.012979] | -0.059096 [-0.106320, -0.011872] |
| dynamic | 1/2 | 1/2 | 0/2 | -0.003646 [-0.011925, +0.004634] | -0.025151 [-0.071608, +0.021306] |
| full_open | 1/2 | 1/2 | 1/2 | -0.016347 [-0.032790, +0.000097] | -0.049407 [-0.128073, +0.029259] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
