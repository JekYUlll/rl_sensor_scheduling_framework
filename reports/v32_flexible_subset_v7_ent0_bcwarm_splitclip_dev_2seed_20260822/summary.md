# Flexible-subset v7 zero-entropy diagnostic

V7 changed only the PPO entropy coefficient from 0.005 to zero on development
seeds 406 and 407. Relative to the validation-selected static subset, average
mean and macro margins were -0.014901 and -0.056149. Relative to the strongest
conventional dynamic policy, they were -0.003140 and -0.044832. The policies
executed 8 and 19 subsets, so removing the entropy bonus neither stabilized
action coverage nor improved performance. This hypothesis is rejected.
