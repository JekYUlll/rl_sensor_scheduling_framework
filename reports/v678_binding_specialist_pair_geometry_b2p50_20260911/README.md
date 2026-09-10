# V678 binding specialist-pair geometry audit

## Decision

Closed before online transfer and PPO. Budget `2.50` makes the three
specialist pairs startup-feasible while excluding their three-channel union,
but the downstream forecast geometry is not stable across seeds.

## Results

| Seed | condition gap | operating gap | 1% operating intersection |
|---:|---:|---:|---|
| 7177 | 0.0031816321 | 0.0370164451 | empty |
| 7178 | 0.0003820463 | 0.0000063937 | 11 candidates |
| 7179 | 0 | 0.0001480981 | 11 candidates |
| 7180 | 0.0051178549 | 0.0026746816 | 2 candidates |

Operating winners changed between heater states in seeds 7177 and 7180, but
the change was not a cross-seed property: three seeds failed the material
`0.01` operating-gap gate, and two of them retained broad static intersections.

## Interpretation

The resource geometry is now genuinely pair-capable, but the V675 target and
quality process does not yield stable condition-specific forecast value. This
separates the resource bottleneck from the remaining scene-design bottleneck.
No online observations, policy checkpoints, or final-test results are inferred
from this audit.

## Source artifacts

- `seed7177/subset_forecast_geometry_seed7177.json`
- `seed7178/subset_forecast_geometry_seed7178.json`
- `seed7179/subset_forecast_geometry_seed7179.json`
- `seed7180/subset_forecast_geometry_seed7180.json`
- Remote launcher: `scripts/run_v678_binding_specialist_pair_geometry_20260911.sh`
