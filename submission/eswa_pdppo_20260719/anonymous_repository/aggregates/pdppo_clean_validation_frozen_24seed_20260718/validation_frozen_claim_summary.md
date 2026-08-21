# Validation-Frozen Macro Evidence

This report recomputes every macro score from final-test rollout losses using subtype denominators fixed by the validation static-candidate table. No model, forecaster, or reference action is fitted or selected from final-test losses for the primary comparison.

## Primary Comparisons

| Comparison | Wins | Mean margin | 95% bootstrap CI | Minimum |
| --- | ---: | ---: | ---: | ---: |
| PD-PPO vs validation-selected static | 24/24 | 0.080126 | [0.067398, 0.093035] | 0.019128 |
| PD-PPO vs AoI | 24/24 | 0.160234 | [0.138627, 0.183879] | 0.090648 |
| PD-PPO vs round robin | 24/24 | 0.160859 | [0.137866, 0.185297] | 0.083312 |
| PD-PPO vs random | 24/24 | 0.162356 | [0.139157, 0.186777] | 0.087948 |
| PD-PPO vs post-hoc strongest rule dynamic | 24/24 | 0.152138 | [0.130964, 0.174692] | 0.083312 |

## Diagnostic Boundaries

The fixed static replay is labelled post-hoc because it ranks constant actions on held-out loss. The event-label diagnostic is privileged because it has access to simulator event labels. Neither diagnostic defines the primary result.

- Primary seed count: 24
- Action-trace operational rows available: 24
- Privileged event-label rows available: 0
