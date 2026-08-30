# V260 long-credit PD-PPO: development summary

- Configuration: V259 dwell-aware executable-mask correction plus a 64-unit GRU over the 20-step value/mask history; gamma=0.997 and GAE lambda=0.98; no candidate forecast head, AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 5001--5002.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.048236 | 0/2 | -0.037783 |
| feasible_static_projected | 1/2 | -0.023456 | 1/2 | -0.000374 |
| aoi | 1/2 | -0.013902 | 2/2 | 0.005433 |
| round_robin | 1/2 | -0.013318 | 1/2 | 0.006064 |
| random | 2/2 | 0.004051 | 2/2 | 0.028879 |
| full_open_unconstrained | 0/2 | -0.054831 | 0/2 | -0.034128 |
| best_static | 0/2 | -0.048236 | 0/2 | -0.037783 |
| best_original_dynamic | 1/2 | -0.019179 | 1/2 | -0.002043 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 5001 | 2 | 1 | 2 | 0.012158 | 0 |
| 5002 | 4 | 0 | 2 | 0.027862 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 5001 | 0.8542 | 0.0000 | 1.0000 | 0.0104 | 0.1354 | 0.0000 |
| 5002 | 0.0000 | 0.7930 | 0.0000 | 0.2405 | 0.1237 | 0.4154 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests whether a longer standard PPO credit horizon improves learning after the dwell-aware action-interface correction.
