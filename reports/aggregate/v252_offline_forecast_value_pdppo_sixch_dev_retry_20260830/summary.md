# V252 offline forecast-value pretraining: development summary

- Configuration: context-aware PD-PPO with one-time offline forecast-value pretraining, no on-policy AWBC/oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 4201--4202.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.076373 | 0/2 | -0.125622 |
| feasible_static_projected | 0/2 | -0.032778 | 0/2 | -0.030898 |
| aoi | 0/2 | -0.032621 | 0/2 | -0.043070 |
| round_robin | 0/2 | -0.034887 | 0/2 | -0.049434 |
| random | 0/2 | -0.026732 | 0/2 | -0.038166 |
| full_open_unconstrained | 0/2 | -0.106788 | 0/2 | -0.130273 |
| best_static | 0/2 | -0.076373 | 0/2 | -0.125622 |
| best_original_dynamic | 0/2 | -0.038183 | 0/2 | -0.052582 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4201 | 6 | 0 | 0 | 0.059053 | 0 |
| 4202 | 6 | 0 | 0 | 0.062093 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4201 | 0.1931 | 0.2700 | 0.3832 | 0.4141 | 0.5299 | 0.1050 |
| 4202 | 0.6476 | 0.4553 | 0.2253 | 0.2969 | 0.3129 | 0.0516 |

## Interpretation

- The offline forecast-value targets are collected once during BC pretraining; no candidate forecast evaluation is performed during on-policy rollout collection.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This is a two-seed development result testing whether one-time forecast-value pretraining restores predictive direction without online teacher computation.
