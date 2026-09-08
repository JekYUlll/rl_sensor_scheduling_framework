# V609 resource-state value-transfer diagnostic

## Purpose

This is a read-only diagnostic on the corrected V607 seed-7182 candidate
alignment artifact. It asks whether the observed heater/resource state can
predict the downstream forecast-optimal feasible subset. No PPO training,
policy selection, or test feedback was used.

## Inputs

- `reports/aggregate/v607_dynamic_resource_pdppo_probe_corrected_20260909/diagnostics_seed7182_candidate_alignment.csv`
- `reports/analysis/v600_v602_heater_frontier_20260909/resource_trace_seed7182.csv`

The alignment artifact contains 6,144 evaluated rows. Resource heater flags
were joined by `truth_step_idx`.

## Results

The resource trace produced only three observed heater-state combinations:

| resource state | rows | distinct best actions | best-action entropy (bits) | rank-1 policy rate | mean candidate regret |
|---|---:|---:|---:|---:|---:|
| `00000` | 145 | 9 | 3.037 | 0.214 | 0.179 |
| `01000` | 637 | 16 | 3.549 | 0.173 | 0.182 |
| `11000` | 5362 | 8 | 2.790 | 0.430 | 0.063 |

The state code follows the five optional channels in the resource trace. The
observed resource state therefore does not determine the forecast-optimal
subset. Conditioning additionally on the event subtype still leaves high
best-action entropy, approximately `2.40--2.87` bits in the populated groups.

## Decision

The dynamic resource model changes feasibility, but the current resource
observables do not provide a sufficient standalone predictor of downstream
subset value. This explains why adding the same resource-state features to
V607/V608 did not resolve the static shortcut. A further PPO-only intervention
is not authorized. Any next scene intervention must create a predeclared,
deployable relation between observed resource state and forecast-relevant
measurement value, then pass the same no-RL transfer gate before training.
