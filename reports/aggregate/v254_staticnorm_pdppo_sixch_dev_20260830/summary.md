# V254 no-teacher PD-PPO: development summary

- Configuration: online context-aware PD-PPO with subtype auxiliary loss and validation static-normalized forecast reward, but no AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 4401--4402.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.035171 | 0/2 | -0.070360 |
| feasible_static_projected | 1/2 | 0.011954 | 2/2 | 0.003684 |
| aoi | 1/2 | 0.006519 | 1/2 | 0.002039 |
| round_robin | 2/2 | 0.016317 | 1/2 | 0.009386 |
| random | 2/2 | 0.018827 | 1/2 | 0.002254 |
| full_open_unconstrained | 0/2 | -0.028526 | 0/2 | -0.017540 |
| best_static | 0/2 | -0.035171 | 0/2 | -0.070360 |
| best_original_dynamic | 1/2 | 0.006519 | 0/2 | -0.009929 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4401 | 6 | 0 | 0 | 0.061224 | 0 |
| 4402 | 5 | 0 | 1 | 0.031843 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4401 | 0.1372 | 0.1141 | 0.5547 | 0.1957 | 0.3247 | 0.6280 |
| 4402 | 0.1905 | 0.6059 | 0.8954 | 0.1979 | 0.0790 | 0.0078 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests policy learnability without training-time teacher computation on the V249-calibrated six-channel scene.
