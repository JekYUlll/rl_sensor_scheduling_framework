# V305 corrected-scene soft forecast-value pretraining result

## Configuration

- Corrected six-channel quality scene, budget 1.75, startup peak budget 2.15.
- Scene seeds 6811--6812; policy seeds 6911--6912.
- V302 settings retained; added only training-partition soft forecast-value pretraining (4096 steps, temperature 0.75).

## Outcome

- ordinary vs validation_selected_static: 0/2 wins, mean margin `-0.046643`.
- macro_staticnorm vs validation_selected_static: 1/2 wins, mean margin `-0.116440`.
- ordinary vs feasible_static_projected: 0/2 wins, mean margin `-0.022507`.
- macro_staticnorm vs feasible_static_projected: 1/2 wins, mean margin `-0.030523`.
- ordinary vs full_open_unconstrained: 0/2 wins, mean margin `-0.036277`.
- macro_staticnorm vs full_open_unconstrained: 0/2 wins, mean margin `-0.106751`.
- ordinary vs aoi: 1/2 wins, mean margin `-0.023576`.
- macro_staticnorm vs aoi: 1/2 wins, mean margin `-0.082495`.
- ordinary vs random: 1/2 wins, mean margin `-0.006089`.
- macro_staticnorm vs random: 1/2 wins, mean margin `-0.008174`.
- ordinary vs round_robin: 1/2 wins, mean margin `-0.000745`.
- macro_staticnorm vs round_robin: 1/2 wins, mean margin `-0.034019`.

## Behavior

- Seed 6811: aborts 0, always-on 0, always-off 0, mid-duty 5, switches/step `0.030540`.
- Seed 6812: aborts 0, always-on 0, always-off 0, mid-duty 6, switches/step `0.030612`.

## Decision

V305 is rejected as a primary improvement because validation-static ordinary wins remained below the required gate. The pretraining path is retained as a diagnostic only.

## Provenance

- Raw runs: `reports/v305_corrected_scene_staticnorm_softpretrain_pdppo_dev_seed6811_b1p75_20260822/` and `reports/v305_corrected_scene_staticnorm_softpretrain_pdppo_dev_seed6812_b1p75_20260822/`.
- Launcher: `scripts/run_v305_corrected_scene_staticnorm_softpretrain_pdppo_dev_20260831.sh`.
