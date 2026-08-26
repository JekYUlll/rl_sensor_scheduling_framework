# V151 leave-one-scene-out result

All five folds completed. PD-PPO beat validation-selected static on ordinary
loss in 1/5 folds and on macro loss in 2/5 folds. Behavior passed in 3/5, while
the joint prediction, dynamic, and behavior gate passed in 0/5. Mean
ordinary/macro static margins were -0.009487/-0.014571.

PD-PPO still beat the strongest AoI, round-robin, or random schedule in 4/5
folds with mean ordinary margin +0.026678. The result closes generic multi-scene
transfer and checkpoint selection. A new scene must make relevant state changes
observable online before further PPO training.
