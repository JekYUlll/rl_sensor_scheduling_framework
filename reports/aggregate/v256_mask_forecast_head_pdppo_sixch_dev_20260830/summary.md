# V256 candidate forecast-head PD-PPO: development summary

- Configuration: online context-aware PD-PPO with subtype auxiliary loss but candidate-level mask forecast head, with no AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 4601--4602.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.012231 | 0/2 | -0.009822 |
| feasible_static_projected | 0/2 | -0.007782 | 1/2 | 0.003979 |
| aoi | 0/2 | -0.017144 | 0/2 | -0.015253 |
| round_robin | 1/2 | -0.011524 | 1/2 | -0.006551 |
| random | 1/2 | -0.000927 | 1/2 | 0.001408 |
| full_open_unconstrained | 0/2 | -0.020744 | 0/2 | -0.016322 |
| best_static | 0/2 | -0.012231 | 0/2 | -0.009822 |
| best_original_dynamic | 0/2 | -0.017144 | 0/2 | -0.015253 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4601 | 6 | 0 | 0 | 0.067304 | 0 |
| 4602 | 6 | 0 | 0 | 0.051238 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4601 | 0.1493 | 0.1146 | 0.4076 | 0.5968 | 0.5091 | 0.2070 |
| 4602 | 0.0560 | 0.2157 | 0.0781 | 0.8030 | 0.5304 | 0.2279 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests policy learnability without training-time teacher computation on the V249-calibrated six-channel scene.
