# V257 candidate forecast-head PD-PPO: development summary

- Configuration: online context-aware PD-PPO with subtype auxiliary loss but candidate-level mask forecast head, with no AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, dense auxiliary stride 8, seeds 4701--4702.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.107579 | 0/2 | -0.148706 |
| feasible_static_projected | 1/2 | -0.044490 | 1/2 | -0.048330 |
| aoi | 0/2 | -0.010576 | 0/2 | -0.017178 |
| round_robin | 0/2 | -0.006900 | 0/2 | -0.018287 |
| random | 1/2 | -0.005363 | 0/2 | -0.014196 |
| full_open_unconstrained | 0/2 | -0.076402 | 0/2 | -0.096382 |
| best_static | 0/2 | -0.107579 | 0/2 | -0.148706 |
| best_original_dynamic | 0/2 | -0.012541 | 0/2 | -0.028302 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4701 | 6 | 0 | 0 | 0.059922 | 0 |
| 4702 | 6 | 0 | 0 | 0.049718 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4701 | 0.3728 | 0.1910 | 0.6289 | 0.3329 | 0.3303 | 0.1337 |
| 4702 | 0.2253 | 0.6120 | 0.3216 | 0.1619 | 0.5339 | 0.0838 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests policy learnability without training-time teacher computation on the V249-calibrated six-channel scene.
