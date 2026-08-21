# Flexible-subset v12 no-static-prior development result

Disabling the state-independent per-action prior increased executed coverage to
12, 17, and 18 of 20 subsets. It did not improve forecast performance: only
seed 405 beat static, and no seed beat the strongest conventional dynamic policy
on both endpoints. Average margins were -0.025519/-0.126514 against static and
-0.017208/-0.115053 against dynamic. The static-prior hypothesis is rejected.

An online context-alert diagnostic achieved lower mean and macro loss than
PD-PPO in all three seeds. It beat static in seed 405 but remained slightly
behind static in seeds 406 and 407. Thus online context is informative, while
the current smooth subtype dynamics do not provide stable dynamic-over-static
value across development seeds.
