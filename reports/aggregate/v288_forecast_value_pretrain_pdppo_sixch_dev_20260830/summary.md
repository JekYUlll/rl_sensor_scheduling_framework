# V288 forecast-value regression pretraining

Configuration: frozen V279 scene/evaluator, ordinary decision-time forecast
reward, arbitrary feasible subsets, six-step forecast-derived candidate-cost
targets, `4096` training-partition samples, `20` pretraining epochs, followed
by `30,000` requested PPO timesteps (`30,720` collected per seed). Seeds
`6801--6802` ran remotely on GPU1/GPU2. No bandit action, test label, or test
feedback was used.

Custom ordinary losses were `0.393409` and `0.396219`. Relative ordinary
loss wins were `1/2` against AoI, `1/2` against random, `2/2` against
round-robin, `0/2` against validation-selected static, `0/2` against feasible
static, and `0/2` against full-open. Macro static-normalized wins were `0/2`
against every reported comparator. The pretraining stage therefore did not
improve the static shortcut or the macro endpoint.

Behavior passed warm-up with zero aborts and no always-on sensors. Seed6801
had five mid-duty and no always-off sensors; seed6802 had four mid-duty and
two always-off sensors. Switching rates were `0.039224` and `0.043422` per
step.

Decision: reject forecast-value regression pretraining as a mainline change;
retain the run as a negative learner-transfer diagnostic.
