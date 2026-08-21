# Flexible-subset v3 matched strong-BC pilot

The matched seed-405 run changed only BC pretraining strength relative to v3.
All non-PD-PPO comparator metrics were identical. BC accuracy increased from
0.110 to 0.786.

PD-PPO beat the validation-selected static schedule by 0.031659 in mean loss and
0.097111 in macro loss. It beat the best conventional dynamic reference by
0.010987 and 0.004263. The policy used ten masks, all six channels had
intermediate duty, switches per step were 0.021421, and no warm-up abort or
feasibility failure occurred.

The configuration passes the single-seed development gate. It is frozen for
replication on development seeds 406 and 407 before any final-seed decision.
