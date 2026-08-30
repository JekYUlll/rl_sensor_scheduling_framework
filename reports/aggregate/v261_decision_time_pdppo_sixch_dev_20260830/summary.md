# V261 decision-time-update PD-PPO: development summary

- Configuration: V259 dwell-aware executable-mask correction plus a 64-unit GRU with decision-only actor updates over the 20-step value/mask history; gamma=0.99 and GAE lambda=0.95; no candidate forecast head, AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 5101--5102.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 1/2 | -0.012148 | 0/2 | -0.012396 |
| feasible_static_projected | 1/2 | 0.013534 | 1/2 | 0.052695 |
| aoi | 1/2 | -0.001526 | 1/2 | -0.008358 |
| round_robin | 2/2 | 0.011315 | 1/2 | 0.005006 |
| random | 1/2 | 0.004861 | 1/2 | -0.003233 |
| full_open_unconstrained | 1/2 | -0.019932 | 1/2 | -0.004889 |
| best_static | 1/2 | -0.012148 | 0/2 | -0.012396 |
| best_original_dynamic | 1/2 | -0.001526 | 1/2 | -0.012050 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 5101 | 4 | 0 | 0 | 0.036619 | 0 |
| 5102 | 5 | 0 | 1 | 0.062238 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 5101 | 0.0286 | 0.5130 | 0.2201 | 0.1385 | 0.9028 | 0.0382 |
| 5102 | 0.0000 | 0.4479 | 0.1250 | 0.3407 | 0.8203 | 0.2609 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests whether a decision-time policy updates improves learning after the dwell-aware action-interface correction.
