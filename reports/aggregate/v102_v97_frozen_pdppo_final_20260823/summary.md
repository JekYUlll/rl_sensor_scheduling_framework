# Frozen V97 flexible-subset confirmation

The locked protocol completed 22 fresh scene/policy seed pairs.
Positive margins indicate lower PD-PPO loss than the comparator.

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% CI) | Mean macro margin (95% CI) |
|---|---:|---:|---:|---:|---:|
| static | 15/22 | 12/22 | 11/22 | +0.009987 [-0.003529, +0.023311] | +0.023076 [-0.018575, +0.065800] |
| dynamic | 21/22 | 19/22 | 18/22 | +0.027555 [+0.019873, +0.035666] | +0.054944 [+0.035895, +0.074272] |
| context | 7/22 | 9/22 | 5/22 | -0.008641 [-0.017496, -0.001367] | -0.032302 [-0.053128, -0.012547] |
| forecast_greedy | 18/22 | 20/22 | 18/22 | +0.022473 [+0.008565, +0.037050] | +0.098269 [+0.057676, +0.135653] |
| exact_label | 7/22 | 7/22 | 4/22 | -0.008918 [-0.018008, -0.001596] | -0.034350 [-0.054511, -0.015457] |
| full_open | 6/22 | 6/22 | 4/22 | -0.008214 [-0.015507, -0.000356] | -0.038898 [-0.067084, -0.011226] |

Behavior and feasibility gate: 22/22 seeds.
Invalid actions: 0; per-step power violations: 0; startup-peak violations: 0; warm-up aborts: 0.

The one-step forecast-greedy and exact-label rows are privileged offline diagnostics. The context row uses only supplied noisy warning scores and validation-calibrated actions.
