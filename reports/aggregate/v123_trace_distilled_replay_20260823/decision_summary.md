# V123 closed-loop trace-distillation decision

The ExtraTrees action-value model was fitted on policy-training receding traces
only. Final execution used the same online scheduler state and hard feasibility
projection as the main method.

The policy beat the validation-selected static schedule in 3/5 scenes on both
ordinary forecast loss and the static-normalized subtype macro. Mean margins
were -0.047145 and -0.157208. Seed 1301 dominated the negative mean, with
ordinary loss 0.898986 and macro 2.024480. Seed 1304 also collapsed to two
always-on and four always-off channels; seed 1305 had two always-off channels.

The preceding one-step audit was optimistic because its candidate costs were
evaluated on states visited by the receding policy. Closed-loop deployment
changes that state distribution. Direct offline tree-policy deployment is
therefore rejected. A subsequent learner may use dense forecast-value targets
only on states visited by its current policy; forecast-loss PPO and hard
feasibility masking remain the method core.
