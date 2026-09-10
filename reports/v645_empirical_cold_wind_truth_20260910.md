# V645 Empirical Cold-Wind Truth Screen

## Scope

This is a truth-only continuation of the V614 empirical cold-availability
route. It does not train a forecaster or policy. The V614 decision-time
quality proxy for the weather backbone is used to scale a frozen AR(1) local
wind innovation after a six-step delay. Power budgets, heater traces, target
weights, partitions, and policy architecture are unchanged.

## Verified output

Four seeds (`7401--7404`) produced 90,000-row CSVs on `remote-gpu`. The
innovation standard deviations were `0.643`, `0.673`, `0.691`, and `0.654 m/s`;
the 95th absolute magnitudes were `1.492--1.571 m/s`. Mean absolute
innovation was `0.96--1.03 m/s` for rows with quality below `0.5`, compared
with `0.18--0.19 m/s` for the remaining rows. Wind clipping occurred in at
most two rows per seed at the lower bound and never at the upper bound.

The generated audit column is `generator_cold_wind_innovation`. It is not part
of the declared sensor-quality/context columns, and exact hardware-test labels
are not exposed to the policy. The quality relation is gradual: quality below
one occurs in about 99% of the extreme-scene rows, while quality below `0.5`
occurs in about 22.5%; the former must not be described as binary failure.

## Provenance

Truth files and metadata are under
`reports/v645_empirical_cold_wind_truth_20260910/` on `remote-gpu`.
Generation used `scripts/183_build_empirical_cold_wind_innovation_truth.py`
and `scripts/run_v645_empirical_cold_wind_truth_20260910.sh`. The first tmux
wrapper wrote status `1` because the outer shell expanded `$?` before tmux
execution; output completeness was independently verified. The subsequent
asset run is the first stage that can determine whether the target coupling
creates downstream subset-value separation.

## Geometry result

Matched assets were evaluated over the same four resource-state starts per
seed as V616. The downstream opportunity gaps were `0.000000`, `0.015164`,
`0.000944`, and `0.004598` for seeds `7401--7404`. Only one seed exceeded the
predeclared `0.01` materiality threshold. A 1% near-optimal static candidate
remained for seeds 7401 and 7403, and seed7401 selected the same candidate in
all heater conditions. The route therefore closed before online transfer and
PPO. The compact geometry artifacts are in
`reports/v645_empirical_cold_wind_subset_crossover_audit_20260910/`.
