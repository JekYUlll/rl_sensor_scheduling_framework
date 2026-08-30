# V311 corrected-scene masked-teacher hard forecast-value diagnostic

- The teacher label was the minimum frozen-forecaster cost among the feasible candidate masks. The fallback was used only when no feasible candidate existed.
- Training used 16,384 training-partition pretraining steps and zero PPO updates. Scene seeds were 6811/6812; policy seeds were 6961/6962.
- The result is a diagnostic of hard forecast-value pretraining, not a final PD-PPO claim.

## Results

- Seed 6811: custom ordinary `0.427300`, static-normalized macro `0.928934`, zero aborts, zero always-on/off channels, five mid-duty channels, switching `0.007526` per step.
- Seed 6812: custom ordinary `0.481154`, static-normalized macro `0.955633`, zero aborts, one always-on and two always-off channels, three mid-duty channels, switching `0.009697` per step.

## Win counts (lower loss is better)

| Reference | Ordinary wins | Macro wins | Mean ordinary margin | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 1/2 | 1/2 | -0.021405 | -0.039809 |
| feasible static projected | 1/2 | 1/2 | +0.002732 | +0.046107 |
| full-open unconstrained | 1/2 | 1/2 | -0.011039 | -0.030120 |
| AoI | 1/2 | 1/2 | +0.001662 | -0.005865 |
| random | 1/2 | 1/2 | +0.019149 | +0.068457 |
| round-robin | 1/2 | 1/2 | +0.024493 | +0.042612 |

## Decision

The feasible-only teacher-label fix is valid, but hard forecast-value pretraining still does not transfer reliably to closed-loop performance: it loses to validation-selected static in one seed and to the full-open reference in one seed, while seed 6812 retains constant-channel behavior. The branch is rejected as a primary improvement. The evidence points to state-distribution or closed-loop credit mismatch, so additional label-volume or teacher-target variants should not be expanded without a new hypothesis.
