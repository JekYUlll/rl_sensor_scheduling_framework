# V3.1 Complex-Cost Pilot Results

Date: 2026-05-22

## Question

Does a more physically heterogeneous sensor-cost vector break the V3.1 static trap and make PD-PPO learn useful event-conditioned scheduling?

## Runs

| Run | Output | Key change |
| --- | --- | --- |
| complex prior pilot | `reports/v31_complex_pilot/` | `windblown_sensors_complex.yaml`, `B=1.70`, seed `41`, candidate prior + AWBC enabled |
| no-prior ablation | `reports/v31_complex_no_prior_seed41/` | same truth/cost/budget, `50k` PPO steps, candidate prior disabled, `awbc_coef=0`, `prior_kl_coef=0` |

## Main Findings

The complex-cost prior pilot should not be scaled. `custom_ppo` and `feasible_static_projected` are exactly identical at rollout level:

- `selected_masks`, `mode_ids`, `powers`, `peaks`, `oracle_losses`, `observations`, and observed masks are equal elementwise.
- Both select `met_station_core + radiometer_basic + surface_temp_ir + snow_particle_counter`.
- Both have `8` near-constant sensors, `4` constant-active sensors, and `4` constant-off sensors.
- `laser_disdrometer` and `fc4_flux` are never selected.
- `forecast_weighted_mae_overall = 0.122425`, `power_mean = 1.30`, `warmup_abort_rate = 0`.

The no-prior/no-AWBC ablation confirms that the policy can become dynamic, but the dynamics are not useful:

- `custom_ppo` switch rate rises to `2.049` switches/step.
- `warmup_abort_rate` rises to `0.0664`.
- `forecast_weighted_mae_overall` worsens to `0.145684`, below static, full-open, round-robin, AoI, and random.
- `laser_disdrometer` is selected less during events than non-events: event lift `-0.0206`.
- `fc4_flux` remains unused.

## Interpretation

Physical cost heterogeneity alone does not solve the static-trap problem. With candidate prior/AWBC, PD-PPO collapses exactly to the strong fixed-priority static subset. Without that regularization, the policy becomes dynamic but mostly noisy and warmup-inefficient, not event-conditioned in the intended direction.

The next design decision should therefore not be a larger complex-cost sweep. The failure points toward reward/candidate design: the current oracle reward and action prior do not make high-latency snow sensors valuable at event onset, and the unconstrained exploration path does not discover a better warmup strategy.

## Decision

Do not scale `v31_complex_pilot` to more seeds or budgets as a paper result. Treat it as a mechanism probe showing that:

1. cost-vector realism changes the preferred fixed subset;
2. candidate-prior/AWBC can recover a strong static policy;
3. removing those stabilizers produces dynamic but poor scheduling;
4. the paper should not claim adaptive event-conditioned warmup from this configuration.

## Paper-Claim Triage

The following claim types should be softened or reframed before the next paper revision:

- Claims that PD-PPO adds an "adaptive mechanism for event-conditioned warm-up decisions".
- Claims that the scheduler performs warm-up-aware selection of particle and flux channels in event-heavy windows.
- Claims that adaptive scheduling is empirically necessary in the current V3.1 configuration.
- Claims that the complex or physical cost structure alone supports dynamic scheduling value.

Safer framing:

- PD-PPO is a prediction-driven scheduler that approaches a strong static projection and outperforms deployable heuristic schedulers in the current V3.1 sweep.
- The strong static baseline reveals that much of the current benchmark can be solved by a robust fixed allocation.
- Dynamic switching without the oracle prior/AWBC is possible, but in the current setting it is warm-up-inefficient and degrades forecast quality.
- Event-conditioned high-latency activation remains an unresolved limitation and a target for reward/candidate redesign.
