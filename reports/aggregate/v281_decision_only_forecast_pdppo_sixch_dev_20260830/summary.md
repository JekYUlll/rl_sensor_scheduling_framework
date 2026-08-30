# V281 decision-only ordinary-forecast PD-PPO development

V281 tests decision-only PPO policy credit with ordinary forecast reward, matched to V280 scene and same-seed validation assets.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 1/2 | 0/2 | -0.035775 [-0.070090, -0.001460] | -0.059817 [-0.140588, +0.020954] |
| dynamic | 1/2 | 1/2 | 1/2 | +0.018035 [-0.002172, +0.038243] | +0.010715 [-0.028310, +0.049739] |
| full_open | 0/2 | 0/2 | 0/2 | -0.026526 [-0.027050, -0.026002] | -0.052739 [-0.067218, -0.038259] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
