# V248 complete six-channel PD-PPO: development summary

- Configuration: complete training-only guide/AWBC/subtype scaffold, balanced quality scene, B=1.75, minimum dwell=6, 100k PPO steps, seeds 3801--3805.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/5 | -0.029549 | 1/5 | -0.031311 |
| feasible_static_projected | 1/5 | -0.017547 | 2/5 | -0.014191 |
| aoi | 4/5 | 0.016151 | 4/5 | 0.015471 |
| round_robin | 4/5 | 0.026136 | 4/5 | 0.029347 |
| random | 4/5 | 0.016860 | 4/5 | 0.014034 |
| full_open_unconstrained | 1/5 | -0.029077 | 2/5 | -0.032659 |
| best_static | 0/5 | -0.030088 | 1/5 | -0.033121 |
| best_original_dynamic | 4/5 | 0.013856 | 4/5 | 0.010058 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 3801 | 5 | 0 | 1 | 0.037922 | 0 |
| 3802 | 4 | 0 | 2 | 0.014329 | 0 |
| 3803 | 3 | 1 | 2 | 0.007961 | 0 |
| 3804 | 3 | 1 | 2 | 0.010276 | 0 |
| 3805 | 4 | 0 | 2 | 0.014184 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 3801 | 0.1914 | 0.3971 | 0.6029 | 0.0000 | 0.2057 | 0.6029 |
| 3802 | 0.6719 | 0.0000 | 0.0694 | 0.0000 | 0.9306 | 0.3281 |
| 3803 | 0.4388 | 0.2422 | 0.3190 | 0.0000 | 1.0000 | 0.0000 |
| 3804 | 0.2782 | 0.0000 | 0.2075 | 0.0000 | 0.5143 | 1.0000 |
| 3805 | 0.0000 | 0.7799 | 0.2031 | 0.0000 | 0.5100 | 0.2201 |

## Interpretation

- The complete scaffold is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result must be interpreted together with the V247 scene admission; V247 established latent headroom but not policy learnability.
