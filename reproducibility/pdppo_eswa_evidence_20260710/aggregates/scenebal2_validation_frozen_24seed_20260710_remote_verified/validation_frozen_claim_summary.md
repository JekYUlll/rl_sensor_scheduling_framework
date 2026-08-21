# Validation-Frozen Macro Evidence

This report recomputes every macro score from final-test rollout losses using subtype denominators fixed by the validation static-candidate table. No model, forecaster, or reference action is fitted or selected from final-test losses for the primary comparison.

## Primary Comparisons

| Comparison | Wins | Mean margin | 95% bootstrap CI | Minimum |
| --- | ---: | ---: | ---: | ---: |
| PD-PPO vs validation-selected static | 24/24 | 0.077820 | [0.066468, 0.089608] | 0.031532 |
| PD-PPO vs AoI | 24/24 | 0.157928 | [0.137544, 0.180333] | 0.084342 |
| PD-PPO vs round robin | 24/24 | 0.158553 | [0.137021, 0.181519] | 0.075575 |
| PD-PPO vs random | 24/24 | 0.160050 | [0.138115, 0.183246] | 0.083389 |
| PD-PPO vs post-hoc strongest rule dynamic | 24/24 | 0.149832 | [0.129810, 0.171092] | 0.075575 |

## Diagnostic Boundaries

The fixed static replay is labelled post-hoc because it ranks constant actions on held-out loss. The event-label diagnostic is privileged because it has access to simulator event labels. Neither diagnostic defines the primary result.

- Primary seed count: 24
- Action-trace operational rows available: 24
- Privileged event-label rows available: 24
