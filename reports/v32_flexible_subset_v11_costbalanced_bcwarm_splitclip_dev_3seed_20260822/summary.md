# Flexible-subset v11 cost-balanced development result

V11 raised only the fixed per-epoch effective costs of the four lower-cost
channels. The action set contains the empty mask, all six singleton masks, and
13 arbitrary pairs, with no required channel or explicit cardinality cap.
PD-PPO executed 11, 15, and 15 subsets without always-on or always-off channels,
but lost both endpoints to static in all three seeds. Average margins were
-0.026523/-0.098774 against static and -0.019486/-0.063568 against the strongest
conventional dynamic policy. Cost balancing improved behavioral generality but
did not improve forecast performance.
