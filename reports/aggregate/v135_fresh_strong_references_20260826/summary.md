# V135 frozen strong-reference comparison

## Protocol

- Fresh scene seeds 1401--1424 and the frozen V132 method configuration.
- The baseline runner replayed `context_alert_bandit_t0p5` and
  `forecast_greedy_one_step` on each frozen scene.
- Baseline raw losses were joined externally to the final V133/V134 PD-PPO
  metrics. The baseline runner's embedded placeholder-PPO margin columns were
  not used.
- Positive margins indicate lower PD-PPO loss than the comparator.

## Results

| Comparator | Ordinary wins | Macro wins | Joint wins | Mean ordinary margin (95% bootstrap CI) | Mean macro margin (95% bootstrap CI) |
|---|---:|---:|---:|---:|---:|
| Strongest static subset per endpoint | 13/24 | 13/24 | 12/24 | +0.009518 [-0.009162, +0.026156] | +0.030914 [-0.018206, +0.082883] |
| Best conventional dynamic | 21/24 | 21/24 | 21/24 | +0.039625 [+0.026168, +0.051434] | +0.082467 [+0.049063, +0.113617] |
| Online warning-context policy | 15/24 | 17/24 | 14/24 | +0.012670 [+0.000005, +0.024645] | +0.038143 [-0.000077, +0.074823] |
| Privileged one-step forecast-greedy | 16/24 | 20/24 | 16/24 | +0.014895 [-0.000799, +0.030563] | +0.083304 [+0.034192, +0.131640] |
| Unconstrained full-open reference | 7/24 | 5/24 | 5/24 | -0.014144 [-0.030208, +0.000249] | -0.058181 [-0.093273, -0.024881] |

One-sided sign-test p-values were `0.153728/0.031957` for ordinary/macro
comparison with the online warning-context policy and `0.075795/0.000772` for
the privileged one-step forecast-greedy reference.

## Interpretation

PD-PPO retains robust evidence against conventional dynamic policies and is
competitive with the validation-calibrated online warning-context policy. It
also has positive mean margins against the privileged one-step greedy diagnostic,
with clear macro evidence but an ordinary-loss interval that still crosses zero.
The result does not establish stable two-endpoint dominance over either the
strongest selected static subset or every strong reference.

All 24 PD-PPO runs passed the basic behavior and feasibility checks: no invalid
action, per-step power violation, startup-peak violation, or warm-up abort was
observed. The separate V134 complexity audit records stricter state-dependence
results.
