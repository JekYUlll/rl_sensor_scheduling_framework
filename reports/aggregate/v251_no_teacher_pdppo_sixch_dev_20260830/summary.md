# V251 no-teacher PD-PPO: development summary

- Configuration: online context-aware PD-PPO with subtype auxiliary loss but no AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 200k PPO steps, seeds 4101--4105.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 1/5 | -0.027859 | 1/5 | -0.035112 |
| feasible_static_projected | 1/5 | -0.017934 | 3/5 | -0.003284 |
| aoi | 0/5 | -0.006144 | 0/5 | -0.006958 |
| round_robin | 4/5 | 0.009287 | 4/5 | 0.013771 |
| random | 4/5 | 0.014142 | 5/5 | 0.020089 |
| full_open_unconstrained | 0/5 | -0.027851 | 1/5 | -0.017312 |
| best_static | 1/5 | -0.030275 | 1/5 | -0.035112 |
| best_original_dynamic | 0/5 | -0.006144 | 0/5 | -0.007225 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4101 | 6 | 0 | 0 | 0.068172 | 0 |
| 4102 | 6 | 0 | 0 | 0.075337 | 0 |
| 4103 | 6 | 0 | 0 | 0.062238 | 0 |
| 4104 | 6 | 0 | 0 | 0.073527 | 0 |
| 4105 | 6 | 0 | 0 | 0.068968 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4101 | 0.1536 | 0.4006 | 0.1376 | 0.1645 | 0.5052 | 0.4826 |
| 4102 | 0.4488 | 0.1237 | 0.3542 | 0.4253 | 0.1966 | 0.2756 |
| 4103 | 0.3650 | 0.3121 | 0.4618 | 0.1337 | 0.4336 | 0.0781 |
| 4104 | 0.2604 | 0.2174 | 0.3051 | 0.1788 | 0.4835 | 0.3537 |
| 4105 | 0.1380 | 0.1949 | 0.3776 | 0.2947 | 0.4562 | 0.3511 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests policy learnability without training-time teacher computation on the V249-calibrated six-channel scene.
