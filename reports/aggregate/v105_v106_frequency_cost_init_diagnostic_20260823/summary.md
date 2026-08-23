# V104--V106 initialization diagnostic

This bounded diagnostic tests whether the V104 seed-1304 failure is primarily
caused by one unfavorable policy initialization. Both policies use the frozen
V103 scene, objective, feasible action geometry, training configuration, and
test trajectory. Only the policy-training seed differs. Validation losses are
computed without test feedback.

| Policy initialization | Policy seed | Validation ordinary loss | Test ordinary loss | Test macro loss | Static ordinary margin | Static macro margin | Behavior |
|---|---:|---:|---:|---:|---:|---:|---|
| V104 init 1 | 4304 | 0.392805 | 0.620068 | 1.684553 | -0.091556 | -0.323363 | pass |
| V105 init 2 | 5304 | 0.387165 | 0.592573 | 1.627153 | -0.064061 | -0.265963 | pass |

Positive margins indicate lower learned-policy loss than the strongest static
schedule selected on validation. The second initialization ranks better on the
validation ordinary loss and improves both test endpoints in the same direction,
but it still loses substantially to the static comparator. It also loses to the
best conventional dynamic comparator by `-0.065179` ordinary and `-0.212757`
macro margin.

Initialization contributes measurable variance but is not the main blocker.
The prespecified expansion rule is therefore not met, and multi-initialization
selection is closed. V103's exact-receding headroom remains evidence of a
learnable sequential opportunity, not evidence that the current PPO scaffold
can recover it.

Full artifacts are archived at:

- `/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v105_init2_seed1304_full`
- `/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v106_init1_validation_replay_full`
