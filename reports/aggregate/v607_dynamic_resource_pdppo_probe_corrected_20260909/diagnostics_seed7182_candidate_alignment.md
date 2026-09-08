# V607 seed 7182 candidate-action alignment

This diagnostic replays the trained seed-7182 checkpoint on six held-out
evaluation starts and compares the deterministic policy's selected candidate
with the frozen-forecaster one-step candidate ranking. The ranking is a
diagnostic only; it is not used as a training target or evaluation baseline.

## Result

The audit contains 6,144 rollout rows. Across all rows, the selected action is
the lowest forecast-loss candidate in 39.8% of rows. The mean selected-action
rank is 4.197, the mean candidate-loss regret is 0.07787, and the mean policy
probability assigned to the selected action is 0.552.

The result is conditional on the number of currently feasible candidates:

| Feasible candidates | Rows | Rank-1 rate | Mean rank | Mean candidate-loss regret | Mean selected probability |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,031 | 100.0% | 1.000 | 0.00000 | 1.000 |
| 8 | 3,417 | 10.5% | 5.026 | 0.09846 | 0.359 |
| 16 | 559 | 5.7% | 9.331 | 0.20758 | 0.197 |
| 20 | 137 | 16.8% | 9.971 | 0.18925 | 0.150 |

For event subtype IDs 0--3, rank-1 rates were 34.8%, 35.6%, 38.1%, and
61.5%, respectively. The policy therefore performs best for subtype 3, but
its action selection is substantially less aligned with the frozen forecast
ranking when the resource frontier contains many alternatives.

## Interpretation

This is evidence of a policy-selection/value-estimation bottleneck in the
current V607 configuration. It is not evidence of action collapse: the saved
rollouts use 14--16 unique masks, and the constraint audit reports zero
optional-resource violations. The diagnostic also does not justify replacing
the PPO objective with the oracle ranking. The next clean experiment should
therefore test ordinary long-horizon value/credit improvements, while keeping
the forecast-loss reward, feasible-action masking, and online resource state
unchanged.
