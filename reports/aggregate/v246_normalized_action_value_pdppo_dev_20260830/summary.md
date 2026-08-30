# V246 normalized action-value auxiliary: development summary

- Configuration: clean V243 scaffold, balanced quality scene, B=1.85, minimum dwell=6, 100k PPO steps, seeds 3606--3610.
- Variant: candidate-conditioned on-policy action-value head; coefficient 0.10 and actor logit scale 0.50.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 1/5 | -0.087226 | 2/5 | -0.072267 |
| feasible_static_projected | 1/5 | -0.065612 | 3/5 | -0.030256 |
| aoi | 2/5 | -0.051776 | 3/5 | -0.030007 |
| round_robin | 3/5 | -0.019578 | 3/5 | 0.035529 |
| random | 3/5 | -0.027913 | 3/5 | -0.003030 |
| full_open_unconstrained | 1/5 | -0.065831 | 3/5 | -0.036086 |
| best_static | 0/5 | -0.095166 | 2/5 | -0.073904 |
| best_original_dynamic | 2/5 | -0.052972 | 3/5 | -0.034562 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 3606 | 0 | 2 | 3 | 0.000000 | 0 |
| 3607 | 0 | 1 | 4 | 0.000000 | 0 |
| 3608 | 0 | 2 | 3 | 0.000000 | 0 |
| 3609 | 0 | 2 | 3 | 0.000000 | 0 |
| 3610 | 0 | 2 | 3 | 0.000000 | 0 |

## Per-sensor duty

| seed | flowcapt_fc4 | gmx500_weather_station | lps10_pyranometer | parsivel2_disdrometer | si111_surface_ir |
|---:|---:|---:|---:|---:|---:|
| 3606 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 3607 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 3608 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 3609 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3610 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |

## Interpretation

- The new auxiliary term is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- V246 is rejected: normalization did not recover state-dependent behavior or comparison performance; no confirmation is authorized.
