# Flexible-subset v6 BC-warm-start grouped-gradient pilot

The matched seed-405 run used strong physical-prototype BC only for
initialization, disabled AWBC during PPO, and retained grouped actor/critic
gradient clipping.

PD-PPO beat selected static by 0.030085 in mean loss and 0.106974 in macro loss,
and beat the best conventional dynamic reference by 0.009414 and 0.014126. It
also narrowly improved over the unconstrained full-open reference on both
reported losses in this seed.

The policy executed 19 of 24 feasible masks. All six channels had intermediate
duty, switches per step were 0.039079, and no feasibility or warm-up failure
occurred. This is the first pilot to combine positive performance gates with
broad arbitrary-subset use. The configuration is frozen for seeds 406 and 407.
