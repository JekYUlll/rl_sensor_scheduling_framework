# V245 on-policy action-value auxiliary: development summary

- Configuration: clean V243 scaffold, balanced quality scene, B=1.85, minimum dwell=6, 100k PPO steps, seeds 3601--3605.
- Variant: candidate-conditioned on-policy action-value head; coefficient 0.10 and actor logit scale 0.50.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 2/5 | -0.253446 | 1/5 | -0.322637 |
| feasible_static_projected | 3/5 | -0.009292 | 2/5 | 0.075185 |
| aoi | 2/5 | -0.091159 | 2/5 | -0.063817 |
| round_robin | 2/5 | -0.080149 | 2/5 | -0.044656 |
| random | 2/5 | -0.085948 | 1/5 | -0.071405 |
| full_open_unconstrained | 1/5 | -0.144240 | 1/5 | -0.128526 |
| best_static | 2/5 | -0.261340 | 1/5 | -0.335968 |
| best_original_dynamic | 2/5 | -0.101001 | 1/5 | -0.083200 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 3601 | 0 | 2 | 3 | 0.000000 | 0 |
| 3602 | 0 | 0 | 5 | 0.000000 | 0 |
| 3603 | 0 | 0 | 5 | 0.000000 | 0 |
| 3604 | 0 | 2 | 3 | 0.000000 | 0 |
| 3605 | 2 | 1 | 2 | 0.008684 | 0 |

## Per-sensor duty

| seed | flowcapt_fc4 | gmx500_weather_station | lps10_pyranometer | parsivel2_disdrometer | si111_surface_ir |
|---:|---:|---:|---:|---:|---:|
| 3601 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 3602 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3603 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3604 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3605 | 0.9249 | 0.0000 | 0.0751 | 1.0000 | 0.0000 |

## Interpretation

- The new auxiliary term is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result must be compared with V243/V244 before any confirmation or final-evaluation decision.
