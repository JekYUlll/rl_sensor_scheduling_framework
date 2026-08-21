# Flexible-subset v2 unmatched 100k diagnostic

Extending seed 403 from 30k to 100k steps did not recover performance. PD-PPO
lost to the validation-selected static schedule by 0.008091 in mean loss and
0.031861 in macro loss. It executed four masks, left FC4 inactive, and switched
at 0.013026 per step.

This run is not a strict training-duration comparison. Although its truth seed
and final start indices match the 30k run, it refitted the stochastic frozen
forecaster and regenerated validation candidate scores. The selected static
mask and all baseline losses consequently changed. The result diagnoses
continued policy collapse, but it cannot isolate the effect of training length.
A matched rerun must reuse the 30k control-source assets.
