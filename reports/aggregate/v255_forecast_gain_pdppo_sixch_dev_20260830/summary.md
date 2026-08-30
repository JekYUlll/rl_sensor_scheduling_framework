# V255 forecast-gain PD-PPO: development summary

- Configuration: online context-aware PD-PPO with subtype auxiliary loss but forecast-gain reward, with no AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 4501--4502.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.056256 | 1/2 | -0.042994 |
| feasible_static_projected | 0/2 | -0.044263 | 0/2 | -0.022150 |
| aoi | 0/2 | -0.027374 | 0/2 | -0.020225 |
| round_robin | 1/2 | -0.010786 | 2/2 | 0.006517 |
| random | 1/2 | -0.003343 | 2/2 | 0.016674 |
| full_open_unconstrained | 0/2 | -0.053067 | 1/2 | -0.024562 |
| best_static | 0/2 | -0.065885 | 0/2 | -0.057904 |
| best_original_dynamic | 0/2 | -0.027374 | 0/2 | -0.020225 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4501 | 5 | 0 | 0 | 0.036185 | 0 |
| 4502 | 6 | 0 | 0 | 0.062672 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4501 | 0.0703 | 0.9379 | 0.1345 | 0.3615 | 0.0195 | 0.0938 |
| 4502 | 0.2869 | 0.2361 | 0.2396 | 0.3355 | 0.2027 | 0.6654 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests policy learnability without training-time teacher computation on the V249-calibrated six-channel scene.
