# V263 semi-Markov dwell-credit PD-PPO: development summary

- Configuration: V259 dwell-aware executable-mask correction plus a 64-unit GRU with terminal-only semi-Markov dwell-block credit over the 20-step value/mask history; gamma=0.99 and GAE lambda=0.95; no candidate forecast head, AWBC, BC pretraining, or oracle teacher; solar target weight 3, balanced quality scene, B=1.75, minimum dwell=6, 50k PPO steps, seeds 5301--5302.
- Physical geometry: six independent channels with arbitrary feasible subsets; final execution uses online alert context and no event labels.
- Lower `oracle_loss_mean` and lower `oracle_loss_macro_subtype_event` are better.

## Comparison

| baseline | ordinary wins | mean delta (baseline - PD-PPO) | macro wins | mean macro delta |
|---|---:|---:|---:|---:|
| validation_selected_static | 0/2 | -0.003734 | 2/2 | 0.017766 |
| feasible_static_projected | 0/2 | -0.019644 | 1/2 | -0.004499 |
| aoi | 0/2 | -0.013478 | 0/2 | -0.010950 |
| round_robin | 1/2 | -0.005343 | 1/2 | -0.003316 |
| random | 1/2 | -0.007411 | 0/2 | -0.011721 |
| full_open_unconstrained | 0/2 | -0.055512 | 0/2 | -0.065031 |
| best_static | 0/2 | -0.019644 | 1/2 | -0.004499 |
| best_original_dynamic | 0/2 | -0.013478 | 0/2 | -0.012745 |

## Behavior

| seed | mid-duty | always-on | always-off | switches/step | warmup aborts |
|---:|---:|---:|---:|---:|---:|
| 5301 | 5 | 0 | 0 | 0.078014 | 0 |
| 5302 | 6 | 0 | 0 | 0.081777 | 0 |

## Per-sensor duty

| seed | fc4_flux | laser_disdrometer | met_station_core | radiometer_basic | shielded_thermo_hygro | surface_temp_ir |
|---:|---:|---:|---:|---:|---:|---:|
| 5301 | 0.6684 | 0.1997 | 0.4288 | 0.0104 | 0.5282 | 0.1541 |
| 5302 | 0.1914 | 0.4149 | 0.4080 | 0.2930 | 0.1198 | 0.4974 |

## Interpretation

- The no-teacher configuration is evaluated from seed-level rollout metrics; the legacy `forecast_value_aux` log field is a separate feature.
- `full_open_unconstrained` is an upper-bound reference, not a fair constrained comparator.
- This development result tests whether a terminal-only semi-Markov dwell-block credit improves learning after the dwell-aware action-interface correction.
