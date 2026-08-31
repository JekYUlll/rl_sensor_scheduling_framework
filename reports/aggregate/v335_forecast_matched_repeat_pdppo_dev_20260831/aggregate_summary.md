# V335 matched forecast-objective repeat

## Protocol

- Same corrected scenes and V294 control assets as V334: scene seeds 6811/6812.
- Fresh policy seeds: 7231/7232.
- Budget 1.75, minimum dwell 6, 50,000 training timesteps.
- Same network, pretraining, auxiliary heads, and valid checkpoint selection
  protocol as V334; only `REWARD_PROXY_MODE=forecast` was restored.
- Selected update 49 for both seeds from non-empty validation scenes.

## Predictive result

The value is `baseline - custom_ppo`; positive values indicate a PD-PPO win.

| Comparator | Ordinary-loss wins | Mean ordinary margin | Macro wins | Mean macro margin |
|---|---:|---:|---:|---:|
| validation-selected static | 0/2 | -0.035151 | 1/2 | -0.064737 |
| feasible static projected | 0/2 | -0.011015 | 2/2 | +0.021180 |
| AoI | 0/2 | -0.012084 | 0/2 | -0.030792 |
| round-robin | 2/2 | +0.010747 | 2/2 | +0.017685 |
| random | 2/2 | +0.005403 | 2/2 | +0.043530 |
| full-open unconstrained | 0/2 | -0.024785 | 0/2 | -0.055048 |

## Operational result

- Warm-up aborts: 0/2.
- Constant-on channels: 0/2.
- Constant-off channels: 0/2.
- Mid-duty channel counts: 6 and 4.
- Switching rates: 0.032783 and 0.009263 per step.

## Decision

Restoring the original forecast reward improved over V334 for several dynamic
comparisons but did not recover static or AoI performance. V335 is rejected as
a primary improvement and is not suitable for final evaluation.
