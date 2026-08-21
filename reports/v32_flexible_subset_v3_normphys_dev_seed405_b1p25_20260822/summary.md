# Flexible-subset v3 normalized physical-teacher pilot

Seed 405 beat the validation-selected static schedule by 0.017557 in mean loss
and 0.045745 in macro loss. It did not beat the best conventional dynamic
reference, trailing by 0.003114 and 0.047103 respectively.

All six channels were used and no channel crossed the always-on or always-off
threshold, but the policy used only five masks and kept the meteorological core
active for 95.1% of steps. Subtype analysis showed correct met-plus-laser use on
all particle steps and partial met-plus-FC4 use on flux steps. During thermal
steps, the policy retained the calm prototype for most epochs and activated IR
for only 28.7%, despite the online thermal alert being cleanly separable.

The result supports the flexible action surface and physically aligned
prototypes but fails the dynamic-baseline gate. The next matched run freezes all
v3 evaluator and split assets and strengthens only the existing BC pretraining
so the actor can learn the observable context-to-action mapping before PPO.
