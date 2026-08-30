# V262 semi-Markov dwell-credit PD-PPO: development summary

- Configuration: V259 dwell-aware executable-mask correction plus a 64-unit GRU with semi-Markov dwell-block credit over the 20-step value/mask history; gamma=0.99 and GAE lambda=0.95; no candidate forecast head, AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 5101--5102.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.133392 | 0/2 | -0.206837 |
| feasible_static_projected | 0/2 | -0.083695 | 0/2 | -0.102975 |
| aoi | 0/2 | -0.058165 | 0/2 | -0.073359 |
| round_robin | 0/2 | -0.046724 | 0/2 | -0.062848 |
| random | 0/2 | -0.035400 | 0/2 | -0.042075 |
| full_open_unconstrained | 0/2 | -0.080614 | 0/2 | -0.094364 |
| best_static | 0/2 | -0.139380 | 0/2 | -0.206837 |
| best_original_dynamic | 0/2 | -0.058165 | 0/2 | -0.073359 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 5201 | 5 | 0 | 0 | 0.033869 | 0 |
| 5202 | 6 | 0 | 0 | 0.045737 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 5201 | 0.5512 | 0.6315 | 0.2344 | 0.2483 | 0.3086 | 0.0234 |
| 5202 | 0.1654 | 0.0903 | 0.0790 | 0.1437 | 0.7279 | 0.5890 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests whether accumulating epoch-level forecast loss over a dwell block improves learning after the dwell-aware action-interface correction.
