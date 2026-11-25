# Validation-Frozen Macro Evidence

This report recomputes every macro score from final-test rollout losses using subtype denominators fixed by the validation static-candidate table. No model, forecaster, or reference action is fitted or selected from final-test losses for the primary comparison.

## Primary Comparisons

| Comparison | Wins | Mean margin | 95% bootstrap CI | Minimum |
| --- | ---: | ---: | ---: | ---: |
| PD-PPO vs validation-selected static | 24/24 | 0.079260 | [0.064229, 0.095031] | 0.013825 |
| PD-PPO vs AoI | 0/0 | nan | [nan, nan] | nan |
| PD-PPO vs round robin | 0/0 | nan | [nan, nan] | nan |
| PD-PPO vs random | 0/0 | nan | [nan, nan] | nan |
| PD-PPO vs post-hoc strongest rule dynamic | 0/0 | nan | [nan, nan] | nan |

## Diagnostic Boundaries

The fixed static replay is labelled post-hoc because it ranks constant actions on held-out loss. The event-label diagnostic is privileged because it has access to simulator event labels. Neither diagnostic defines the primary result.

- Primary seed count: 24
- Action-trace operational rows available: 24
- Privileged event-label rows available: 0
