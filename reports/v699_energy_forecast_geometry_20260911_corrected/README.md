# V699 Corrected Cumulative-Energy Forecast Geometry

This report contains the corrected policy-free geometry audit for the
entity-compatible six-channel system. It uses the documented 24 V, 8640 Wh
battery, trace-backed PV harvest, a 1728 Wh reserve, four resource-selected
512-step windows per seed, and all 32 optional channel subsets. The selection
of windows uses only SOC and forward harvest ranks; forecast losses are not
used to select windows.

## Correction history

The first audit was invalid because the environment's historical
zero-means-full compatibility default replaced an explicitly supplied zero SOC
with full capacity. The trace-backed initialization semantics were corrected
in `src/v2/env.py`, and the audit was rerun. The earlier output is retained in
the sibling non-corrected report directory and must not be used as evidence.

## Results

| Seed | Opportunity gap | 1% static intersection | Conditions | Min--max candidate support rows |
|---:|---:|---:|---:|---:|
| 7177 | 0.007659 | 0 | 6 | 1536--2048 |
| 7178 | 0.004258 | 1 | 6 | 1536--2048 |
| 7179 | 0.007456 | 0 | 6 | 1536--2048 |
| 7180 | 0.023342 | 0 | 6 | 1536--2048 |

The resource frontier is genuinely state-dependent, and the energy guard
removes candidate support in low-resource windows. However, the predeclared
forecast-geometry gate requires an opportunity gap above 0.01 across the
screened seeds and no persistent 1% near-optimal static subset. That gate is
not satisfied: only one of four gaps exceeds 0.01 and seed 7178 retains one
1%-near-optimal candidate.

## Decision

V699 is not promoted to online transfer or PPO training. It is retained as a
valid partial result showing that the documented energy account creates a
dynamic feasible frontier, while the current frozen forecast does not provide
robust enough downstream value separation for a policy claim.

The authoritative files are the four `energy_subset_geometry_seed*.json`
files and their corresponding `energy_subset_losses_seed*.csv` files in this
directory.
