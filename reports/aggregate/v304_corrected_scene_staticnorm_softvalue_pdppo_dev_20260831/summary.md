# V304 corrected-scene soft forecast-value development result

## Configuration

- Corrected six-channel quality scene, budget 1.75, startup peak budget 2.15.
- Scene seeds 6811--6812; independent policy seeds 6901--6902.
- V302 settings retained: validation-frozen subtype-normalized forecast reward, decision-only PPO updates, temporal/context encoders, and validation-only checkpoint selection.
- Added only the existing soft forecast-value action-ranking auxiliary (coefficient 0.20, six-step lookahead, temperature 0.75).

## Outcome

- ordinary vs validation_selected_static: 0/2 wins, mean margin `-0.023097` (positive means lower loss than the comparator).
- macro_staticnorm vs validation_selected_static: 1/2 wins, mean margin `-0.030787` (positive means lower loss than the comparator).
- ordinary vs feasible_static_projected: 1/2 wins, mean margin `+0.001039` (positive means lower loss than the comparator).
- macro_staticnorm vs feasible_static_projected: 2/2 wins, mean margin `+0.055129` (positive means lower loss than the comparator).
- ordinary vs full_open_unconstrained: 1/2 wins, mean margin `-0.012731` (positive means lower loss than the comparator).
- macro_staticnorm vs full_open_unconstrained: 1/2 wins, mean margin `-0.021098` (positive means lower loss than the comparator).
- ordinary vs aoi: 1/2 wins, mean margin `-0.000030` (positive means lower loss than the comparator).
- macro_staticnorm vs aoi: 1/2 wins, mean margin `+0.003158` (positive means lower loss than the comparator).
- ordinary vs random: 1/2 wins, mean margin `+0.017457` (positive means lower loss than the comparator).
- macro_staticnorm vs random: 2/2 wins, mean margin `+0.077479` (positive means lower loss than the comparator).
- ordinary vs round_robin: 2/2 wins, mean margin `+0.022801` (positive means lower loss than the comparator).
- macro_staticnorm vs round_robin: 2/2 wins, mean margin `+0.051634` (positive means lower loss than the comparator).

## Behavior

- Seed 6811: warm-up aborts 0, always-on 0, always-off 0, mid-duty 5, switches/step `0.043422`.
- Seed 6812: warm-up aborts 0, always-on 0, always-off 0, mid-duty 5, switches/step `0.057172`.

## Decision

V304 does not pass the static-transfer gate: it is below validation-selected static on both seeds in the ordinary endpoint. It should not be promoted to a primary configuration or expanded to fresh confirmation. The auxiliary is retained as an isolated diagnostic; the corrected scene still provides a valid dynamic-value test, but the learner does not robustly recover it.

## Provenance

- Raw runs: `reports/v304_corrected_scene_staticnorm_softvalue_pdppo_dev_seed6811_b1p75_20260822/` and `reports/v304_corrected_scene_staticnorm_softvalue_pdppo_dev_seed6812_b1p75_20260822/`.
- Aggregate source: `seed_metrics.csv` in this directory.
- Launcher: `scripts/run_v304_corrected_scene_staticnorm_softvalue_pdppo_dev_20260831.sh`.
