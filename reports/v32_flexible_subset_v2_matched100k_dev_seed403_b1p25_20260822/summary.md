# Flexible-subset v2 matched 100k diagnostic

The matched control reused the seed-403 30k truth, frozen forecaster, candidate
action surface, validation selection, and final windows. Every non-PD-PPO
baseline metric was exactly unchanged.

Longer training did not fix the failed seed. PD-PPO lost to the selected static
schedule by 0.019041 in mean loss and 0.088967 in macro loss, and lost to the
best conventional dynamic reference by 0.024986 and 0.136727. It executed seven
masks, switched at 0.031698 per step, and never activated the laser channel.

The undertraining hypothesis is rejected. The next bounded variant keeps the
v2 scene and unrestricted 24-mask action surface, normalizes subtype forecast
loss using validation-static scales, and replaces noisy automatic teacher-mask
selection with the existing physically specified training prototypes. These
prototypes remain auxiliary training information and do not restrict online
actions.
