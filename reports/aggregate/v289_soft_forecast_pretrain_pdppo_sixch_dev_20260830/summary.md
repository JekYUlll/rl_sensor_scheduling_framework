# V289 soft forecast-value policy pretraining

Configuration: frozen V279 scene/evaluator, ordinary decision-time forecast
reward, arbitrary feasible subsets, six-step forecast-derived candidate costs,
`4096` training-partition samples, `20` pretraining epochs with soft
cross-entropy targets at temperature `0.75`, followed by `30,000` requested
PPO timesteps (`30,720` collected per seed). Seeds `6801--6802` ran remotely
on GPU1/GPU2. No bandit action, event label, or test feedback was used.

Custom ordinary losses were `0.376347` and `0.398989`. Relative ordinary loss
wins were `2/2` against AoI, random, and round-robin, `0/2` against
validation-selected static and feasible static, and `0/2` against full-open.
Macro static-normalized wins were `2/2` against AoI, random, and round-robin,
and `0/2` against validation-selected static, feasible static, and full-open.
The soft initialization improves dynamic comparisons but does not overcome the
static shortcut.

Behavior passed warm-up with zero aborts and no always-on or always-off
sensors. Both seeds used five mid-duty sensors. Switching rates were `0.083080`
and `0.050659` per step.

Decision: reject as a mainline promotion; retain as the strongest tested
initialization diagnostic in this sequence.
