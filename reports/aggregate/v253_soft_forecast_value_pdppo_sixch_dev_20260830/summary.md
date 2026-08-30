# V253 soft forecast-value pretraining: development summary

- Configuration: context-aware PD-PPO with one-time offline soft forecast-value pretraining (temperature 0.75), no on-policy AWBC/oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 4301--4302.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.042601 | 1/2 | -0.032048 |
| feasible_static_projected | 0/2 | -0.022263 | 1/2 | 0.010285 |
| aoi | 0/2 | -0.020322 | 1/2 | -0.003680 |
| round_robin | 1/2 | 0.004276 | 1/2 | 0.026767 |
| random | 0/2 | -0.012362 | 1/2 | 0.002032 |
| full_open_unconstrained | 0/2 | -0.059535 | 0/2 | -0.050706 |
| best_static | 0/2 | -0.042601 | 1/2 | -0.032048 |
| best_original_dynamic | 0/2 | -0.020322 | 1/2 | -0.004685 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 4301 | 5 | 0 | 0 | 0.087856 | 0 |
| 4302 | 6 | 0 | 0 | 0.052974 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 4301 | 0.4340 | 0.2917 | 0.5061 | 0.0104 | 0.5373 | 0.2005 |
| 4302 | 0.1037 | 0.1081 | 0.3472 | 0.1771 | 0.6211 | 0.5907 |

## Interpretation

- The offline forecast-value targets are collected once during BC pretraining; no candidate forecast evaluation is performed during on-policy rollout collection.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This is a two-seed development result testing whether soft one-time forecast-value pretraining restores predictive direction without online teacher computation.
