# SOC Auxiliary + Abort-Control Gate Report

## Run

- Protocol: corrected semi-Markov energy-account split protocol.
- Seed: `41`.
- Training: `200000` PPO timesteps.
- Controls: `soc_aux_horizon=16`, `soc_aux_coef=0.1`,
  `lambda_warmup_abort=0.16`.
- Output:
  `reports/energy_account_split_protocol_socaux_abort2x_200k/budget1p20_seed41`.
- Status: completed with `exit_code=0`; collector completed with `complete_seeds=1/1`.

## Same-Run Final-Test Ranking

Lower oracle loss is better:

| Policy | Oracle loss | Warm-up aborts |
|---|---:|---:|
| full_open_unconstrained | 0.46240 | 0 |
| validation_selected_static | 0.47617 | 0 |
| custom_ppo | 0.47950 | 81 |
| round_robin | 0.48098 | 768 |
| aoi | 0.48660 | 6 |
| feasible_static_projected | 0.48710 | 0 |
| random | 0.49873 | 1956 |

## Decision

Do not scale this setting to `n=5`.

The gate reduced custom PPO warm-up aborts relative to the previous strict-protocol
seed-41 run (`206 -> 81`), but it failed the required comparator test:
`custom_ppo` is worse than `validation_selected_static` by `+0.00334` oracle loss
(`+0.70%`).

## Mechanism Notes

Custom PPO became more event-biased for some low/no-warmup channels:

| Sensor | Event/non-event selected ratio |
|---|---:|
| fc4_flux | 1.32x |
| snow_particle_counter | 2.57x |
| radiometer_basic | 1.04x |
| laser_disdrometer | 0.52x |

The laser channel remains anti-event: selected less often during event windows
than non-event windows. The change therefore improves abort control but does not
repair the core learned event-laser gating mechanism.

## Interpretation Caveat

Absolute oracle-loss differences against the earlier strict-protocol seed-41 run
are not a clean ablation because `scripts/61_energy_account_split_protocol_run.py`
retrained a fresh frozen TCN oracle for this run. Static and full-open losses also
shift. The defensible judgment is the same-run final-test ranking plus behavior
diagnostics above.

## Next-Step Cost Assessment

Low-cost useful diagnostics:
- Extract training curves and action usage from existing runs.
- Report the actual timing parameters: oracle horizon `8`, generator lead `5`,
  laser warm-up `3`.

Medium-cost next experiment, only if continuing optimization:
- Add explicit forecast/context features such as event probability or time-to-event.
- Or run a cleaner ablation that reuses a fixed oracle across controller variants.

High-cost changes not recommended for the current CRST manuscript:
- Full CMDP/CPO rewrite.
- SAC/TD3/off-policy replacement.
- Large lambda/horizon/architecture grid.
