# Validation-Frozen Macro Evidence

This report recomputes every macro score from final-test rollout losses using subtype denominators fixed by the validation static-candidate table. No model, forecaster, or reference action is fitted or selected from final-test losses for the primary comparison.

## Primary Comparisons

| Comparison | Wins | Mean margin | 95% bootstrap CI | Minimum |
| --- | ---: | ---: | ---: | ---: |
| PD-PPO vs validation-selected static | 22/22 | 0.077969 | [0.066958, 0.089596] | 0.036312 |
| PD-PPO vs AoI | 22/22 | 0.157635 | [0.136529, 0.180931] | 0.084342 |
| PD-PPO vs round robin | 22/22 | 0.158842 | [0.136656, 0.182947] | 0.075575 |
| PD-PPO vs random | 22/22 | 0.159751 | [0.137234, 0.183623] | 0.083389 |
| PD-PPO vs post-hoc strongest rule dynamic | 22/22 | 0.149453 | [0.128956, 0.171437] | 0.075575 |

## Diagnostic Boundaries

The fixed static replay is labelled post-hoc because it ranks constant actions on held-out loss. The event-label diagnostic is privileged because it has access to simulator event labels. Neither diagnostic defines the primary result.

- Primary seed count: 22
- Action-trace operational rows available: 22
- Privileged event-label rows available: 22
