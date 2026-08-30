# V259 dwell-aware temporal-encoder PD-PPO: development summary

- Configuration: online context-aware PD-PPO with a 64-unit GRU over the 20-step value/mask history, with executable candidate masking during active minimum dwell and no candidate forecast head, AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 4901--4902.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 1/2 | -0.014743 | 1/2 | 0.016215 |
| feasible_static_projected | 0/2 | -0.041338 | 1/2 | 0.001889 |
| aoi | 0/2 | -0.045751 | 1/2 | -0.024087 |
| round_robin | 0/2 | -0.041428 | 1/2 | -0.020625 |
| random | 0/2 | -0.040688 | 1/2 | -0.022290 |
| full_open_unconstrained | 0/2 | -0.060126 | 1/2 | -0.041976 |
| best_static | 0/2 | -0.041338 | 1/2 | -0.004997 |
| best_original_dynamic | 0/2 | -0.045751 | 1/2 | -0.026234 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4901 | 4 | 0 | 0 | 0.028224 | 0 |
| 4902 | 2 | 1 | 3 | 0.033145 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4901 | 0.3659 | 0.1029 | 0.6324 | 0.0365 | 0.6007 | 0.0386 |
| 4902 | 0.0000 | 0.0000 | 0.4783 | 0.5191 | 0.0000 | 0.9974 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests policy learnability after aligning candidate feasibility with environment-level minimum-dwell execution on the V249-calibrated six-channel scene.
