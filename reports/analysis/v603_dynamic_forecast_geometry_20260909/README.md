# V603 Dynamic Resource / Forecast Geometry (2026-09-09)

V603 evaluates all 32 optional masks with the frozen V527 forecaster and then
filters each fixed-mask loss row using the V602 effective-resource trace. The
resource trace is not passed to the forecaster. The resulting dynamic
row-wise minimum is a replay diagnostic, not an executable policy.

| Seed | Candidates | Common static masks | Dynamic opportunity gap | Positive |
|---:|---:|---:|---:|:---:|
| 7181 | 32 | 8 | 0.192819 | yes |
| 7182 | 32 | 8 | 0.064878 | yes |
| 7183 | 32 | 8 | 0.226531 | yes |
| 7184 | 32 | 8 | 0.089142 | yes |

The mean gap is `0.143338`, and the minimum is `0.064878`. Each seed has
32768 matched loss rows and 28 dynamically feasible candidates represented in
the sampled rows. This establishes development-level alignment between the
proxy resource frontier and frozen forecast loss. It does not establish
chronological online transfer, dwell-aware execution, or PD-PPO performance.

## Next gate

Build a deployable transfer probe using only resource states observable before
the decision, with explicit dwell/startup execution and a validation-selected
static comparator. Do not use the row-wise replay minimum as a policy target or
as training supervision.
