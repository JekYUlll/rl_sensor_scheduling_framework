# V269 event-start-balanced absolute block-gain

The protocol completed 2 fresh physical six-channel scene/policy seed pairs.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 0/2 | 1/2 | 0/2 | -0.036897 [-0.055600, -0.018195] | +0.004600 [-0.014271, +0.023470] |
| dynamic | 0/2 | 0/2 | 0/2 | -0.025397 [-0.047241, -0.003553] | -0.056956 [-0.085286, -0.028627] |
| full_open | 0/2 | 0/2 | 0/2 | -0.070981 [-0.094365, -0.047597] | -0.099194 [-0.123886, -0.074503] |

Behavior and feasibility gate: 2/2 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
