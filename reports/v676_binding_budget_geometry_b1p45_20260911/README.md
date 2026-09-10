# V676 binding-budget geometry audit

## Decision

Closed before online transfer and PPO. The lower normalized budget `1.45`
reduced the feasible action surface, but did not create stable downstream
forecast-value variation across the four development seeds.

## Frozen protocol

- Truth/evaluator assets: V675 split-affinity assets, unchanged.
- Candidate actions: all 32 optional-channel subsets under the mandatory
  backbone.
- Budgets: steady, startup, and dynamic resource budgets all `1.45`.
- Starts: `82600`, `83000`, `84800`, and `85500`.
- Geometry only: no policy training, online transfer, or final-test selection.
- Acceptance gate: every seed must have an operating opportunity gap above
  `0.01` and an empty 1% operating near-optimal static intersection.

## Results

| Seed | condition gap | operating gap | operating best | 1% operating intersection |
|---:|---:|---:|---|---|
| 7177 | 0.0001624003 | 0.0000377744 | candidate_004 | 5 candidates |
| 7178 | 0.0000028163 | 0.0000000000 | candidate_002 | 5 candidates |
| 7179 | 0.0000000000 | 0.0000000000 | candidate_004 | 5 candidates |
| 7180 | 0.0011787489 | 0.0002088025 | candidate_016 / candidate_000 by state | 2 candidates |

The operating best candidate remains unchanged across all four heater states
for seeds 7178 and 7179. Seed 7180 has a state-specific best in one state,
but the absolute gap remains far below the materiality threshold.

## Interpretation

The resource constraint is mechanically binding, but the forecast objective
still admits a near-optimal static subset. This distinguishes action-space
restriction from useful adaptive opportunity. V676 therefore does not support
an online-transfer or PD-PPO claim.

## Source artifacts

- `seed7177/subset_forecast_geometry_seed7177.json`
- `seed7178/subset_forecast_geometry_seed7178.json`
- `seed7179/subset_forecast_geometry_seed7179.json`
- `seed7180/subset_forecast_geometry_seed7180.json`
- Remote launcher: `scripts/run_v676_binding_budget_geometry_20260911.sh`
