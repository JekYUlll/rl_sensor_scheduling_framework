# V326 macro checkpoint PD-PPO development aggregate

V326 retains V325 candidate-interaction context-aware masked PPO and changes only validation checkpoint selection to static-normalized macro loss; two development seeds, no bandit-dependent component.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 1/2 | 0/2 | -0.028177 [-0.045665, -0.010689] | -0.029264 [-0.078095, +0.019567] |
| dynamic | 1/2 | 1/2 | 0/2 | -0.005110 [-0.017144, +0.006924] | +0.004681 [-0.040170, +0.049531] |
| full_open | 0/2 | 1/2 | 0/2 | -0.017811 [-0.030501, -0.005121] | -0.019575 [-0.096634, +0.057484] |

Behavior and feasibility gate: 1/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
