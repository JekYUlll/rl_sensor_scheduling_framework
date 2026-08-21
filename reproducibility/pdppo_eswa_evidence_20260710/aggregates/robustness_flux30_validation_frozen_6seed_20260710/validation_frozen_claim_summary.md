# Validation-Frozen Macro Evidence

This report recomputes every macro score from final-test rollout losses using subtype denominators fixed by the validation static-candidate table. No model, forecaster, or reference action is fitted or selected from final-test losses for the primary comparison.

## Primary Comparisons

| Comparison | Wins | Mean margin | 95% bootstrap CI | Minimum |
| --- | ---: | ---: | ---: | ---: |
| PD-PPO vs validation-selected static | 6/6 | 0.076388 | [0.056886, 0.101325] | 0.050716 |
| PD-PPO vs AoI | 6/6 | 0.186447 | [0.125035, 0.250167] | 0.093791 |
| PD-PPO vs round robin | 6/6 | 0.196678 | [0.129877, 0.264168] | 0.091938 |
| PD-PPO vs random | 6/6 | 0.188403 | [0.131994, 0.248715] | 0.104838 |
| PD-PPO vs post-hoc strongest rule dynamic | 6/6 | 0.180922 | [0.123734, 0.240677] | 0.091938 |

## Diagnostic Boundaries

The fixed static replay is labelled post-hoc because it ranks constant actions on held-out loss. The event-label diagnostic is privileged because it has access to simulator event labels. Neither diagnostic defines the primary result.

- Primary seed count: 6
- Action-trace operational rows available: 6
- Privileged event-label rows available: 6
