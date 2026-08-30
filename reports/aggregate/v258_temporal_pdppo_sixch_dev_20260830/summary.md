# V258 temporal-encoder PD-PPO: development summary

- Configuration: online context-aware PD-PPO with a 64-unit GRU over the 20-step value/mask history, with no candidate forecast head, AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 4801--4802.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.066675 | 0/2 | -0.070976 |
| feasible_static_projected | 0/2 | -0.055718 | 0/2 | -0.049654 |
| aoi | 1/2 | 0.007168 | 1/2 | 0.015039 |
| round_robin | 1/2 | -0.001874 | 1/2 | 0.000787 |
| random | 1/2 | 0.006887 | 1/2 | 0.002187 |
| full_open_unconstrained | 0/2 | -0.073163 | 0/2 | -0.080046 |
| best_static | 0/2 | -0.066675 | 0/2 | -0.070976 |
| best_original_dynamic | 1/2 | -0.001874 | 1/2 | -0.000785 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4801 | 5 | 0 | 0 | 0.048777 | 0 |
| 4802 | 6 | 0 | 0 | 0.076422 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4801 | 0.3559 | 0.7001 | 0.0339 | 0.0964 | 0.2700 | 0.5308 |
| 4802 | 0.2821 | 0.5477 | 0.1953 | 0.0803 | 0.3585 | 0.5200 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests policy learnability without training-time teacher computation on the V249-calibrated six-channel scene.
