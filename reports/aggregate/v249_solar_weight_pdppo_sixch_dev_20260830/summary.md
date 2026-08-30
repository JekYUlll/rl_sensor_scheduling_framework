# V249 solar-weight PD-PPO: development summary

- Configuration: complete training-only guide/AWBC/subtype scaffold with solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 100k PPO steps, seeds 3901--3905.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 2/5 | -0.004638 | 2/5 | -0.005653 |
| feasible_static_projected | 2/5 | 0.002543 | 5/5 | 0.033387 |
| aoi | 3/5 | 0.003505 | 3/5 | 0.005041 |
| round_robin | 3/5 | 0.009635 | 3/5 | 0.002942 |
| random | 3/5 | 0.011171 | 4/5 | 0.005030 |
| full_open_unconstrained | 1/5 | -0.011142 | 3/5 | -0.005743 |
| best_static | 1/5 | -0.007565 | 2/5 | -0.005653 |
| best_original_dynamic | 3/5 | 0.002144 | 3/5 | -0.001998 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 3901 | 5 | 0 | 1 | 0.021132 | 0 |
| 3902 | 5 | 0 | 1 | 0.039876 | 0 |
| 3903 | 5 | 0 | 1 | 0.014474 | 0 |
| 3904 | 4 | 0 | 2 | 0.013171 | 0 |
| 3905 | 4 | 0 | 2 | 0.008612 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 3901 | 0.0000 | 0.2834 | 0.3685 | 0.3685 | 0.4557 | 0.1723 |
| 3902 | 0.0000 | 0.3967 | 0.4045 | 0.4045 | 0.1988 | 0.3602 |
| 3903 | 0.3416 | 0.2925 | 0.3659 | 0.0000 | 0.7075 | 0.2925 |
| 3904 | 0.3442 | 0.7118 | 0.0625 | 0.0000 | 0.5933 | 0.0000 |
| 3905 | 0.0000 | 0.0000 | 0.5911 | 0.5911 | 0.3173 | 0.4089 |

## Interpretation

- The complete scaffold is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result must be interpreted together with the V247 scene admission; V247 established latent headroom but not policy learnability.
