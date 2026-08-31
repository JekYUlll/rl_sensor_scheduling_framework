# V334 forecast-gain guard matched diagnostic

## Protocol

- Scene/control pairs: 6811 and 6812, using the corresponding V294 control
  directories for checkpoint validation.
- Fresh policy seeds: 7221 and 7222.
- Budget: 1.75; minimum dwell: 6 steps.
- Intervention: guard the one-step `forecast_gain` reward so hold steps with
  `decision_available=0` receive no decision credit.
- Checkpoints: selected update 30 (seed 6811) and update 40 (seed 6812)
  from non-empty validation scenes.

## Predictive result

Loss and macro metrics are lower-is-better. The value below is
`baseline - custom_ppo`; positive values indicate a PD-PPO win.

| Comparator | Ordinary-loss wins | Mean ordinary margin | Macro wins | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 0/2 | -0.060895 | 1/2 | -0.116130 |
| feasible static projected | 0/2 | -0.036759 | 1/2 | -0.030213 |
| AoI | 0/2 | -0.037828 | 0/2 | -0.082185 |
| round-robin | 1/2 | -0.014997 | 0/2 | -0.033709 |
| random | 1/2 | -0.020341 | 1/2 | -0.007864 |
| full-open unconstrained | 0/2 | -0.050529 | 0/2 | -0.106441 |

## Operational result

- Warm-up aborts: 0/2.
- Constant-on channels: 0/2.
- Constant-off channels: 1/2 (seed 6811 had one).
- Mid-duty channel counts: 5 and 6.
- Switching rates: 0.025763 and 0.034448 per step.

## Decision

The corrected reward-credit semantics did not improve predictive transfer.
V334 is rejected as a primary improvement and no final evaluation is launched
from this variant. The valid matched checkpoint-selection protocol is retained
as evidence that the failure is not caused by an empty validation selector.
