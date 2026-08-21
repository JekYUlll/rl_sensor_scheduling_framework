# Anonymous evidence repository

This archive supports the manuscript "Reinforcement learning with forecast-loss
rewards for sensor scheduling under power constraints." It contains the core
PD-PPO scheduler implementation, matched controls, evaluation collectors,
focused tests, and the aggregate evidence used by the manuscript.

## Scope

- Pilot/model-selection seeds: 117--118.
- Post-selection evaluation seeds: 119--140.
- The combined 117--140 results are retained only as a 24-seed descriptive
  aggregate.
- Primary forecaster: fixed residual temporal convolutional network (TCN).
- Alternative evaluator: multi-output ridge model fitted on the same partition.
- Final policy input excludes exact condition labels.

The archive contains simulated-data aggregates rather than private field data.
Large simulated time series, rollout tensors, and the external
meteorological anchor files are omitted. The aggregate evidence and seed
accounting are self-contained; full simulation regeneration additionally requires
the anchor data described in the manuscript.

## Layout

- `src/v2/`: scheduler, environment, feasibility, forecasting, and control code.
- `src/data_sources/`: simulated time-series construction used by the experiment.
- `configs/`: the reported sensing-system channel configuration.
- `scripts/`: training, evaluation, collection, and paper-asset scripts.
- `tests/v2/`: focused scheduler and evaluator tests.
- `aggregates/`: seed-level CSV plus protocol and summary files.
- `verify_aggregates.py`: checks seed coverage and the headline paired results.
- `pyproject.toml`: Python dependency declaration.

## Quick verification

```bash
python verify_aggregates.py
python -m pytest -q tests/v2
```

Formal training is computationally intensive. The aggregate verifier provides a
lightweight check of the pilot/post-selection seed accounting and
manuscript-facing result directions without retraining a policy.

## Provenance

The archive was assembled from the frozen 2026-07-18 evidence package and the
2026-08-03 submission package. Paths in the original run metadata are not needed
for aggregate verification and are excluded from this anonymous release. Public
repository identifiers are omitted during double-anonymous review. The repository
will be made publicly available upon acceptance.
