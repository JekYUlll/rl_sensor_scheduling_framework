# V102 evidence decision

The locked 22-seed confirmation establishes three robust properties of the
flexible-subset method. PD-PPO beats the conventional dynamic family on both
endpoints in 18/22 seeds, beats the one-step frozen-forecaster greedy diagnostic
in 18/22 seeds, and passes the complete behavior/feasibility gate in 22/22
seeds. All executed masks and power traces are feasible, switching is nonzero,
and event-versus-calm channel duty changes materially in every seed.

The strongest-static gate is not confirmed. Ordinary and macro mean margins
are positive (`+0.009987/+0.023076`), but joint wins are 11/22 and both paired
bootstrap intervals include zero. The result supports competitive average
performance, not stable static dominance.

This limitation is primarily structural. On the same 22 scenes, the
validation-calibrated context reference reaches only 14/22 joint wins against
static, and the exact-label reference reaches 13/22. The present effective-cost
geometry admits 13 three-channel subsets, allowing one fixed subset to cover
multiple regime-specific information sources. Further PPO tuning on V102 is
therefore not justified.

The next development screen changes only fixed effective per-epoch costs. It
will retain arbitrary power-feasible subset enumeration and make all one- and
two-channel subsets feasible while three-channel subsets become infeasible due
to the physical power budget. No cardinality rule, duty quota, sampling-rate
action, or test-label input will be introduced. New development seeds must pass
deployable context and exact-receding scene gates before policy training.
