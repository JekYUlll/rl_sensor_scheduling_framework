# V93 event-only retained-guidance development summary

V93 changes only the scope of the retained training guidance relative to V86:
AWBC and subtype-action supervision are applied to event samples, while calm
samples remain governed by the original forecast-loss PPO objective. The V83a
truth, forecaster, candidate set, costs, budget, policy seeds, and remaining
training controls are frozen.

## Gate result

- Strongest-static joint wins: 4/5.
- Conventional-dynamic joint wins: 5/5.
- Mean strongest-static margins: +0.010409 ordinary and +0.072432 macro.
- Mean conventional-dynamic margins: +0.049779 ordinary and +0.190395 macro.
- Feasibility failures and warm-up aborts: 0/5.
- Complete behavior-gate passes: 3/5.

The performance gate passes, but the complete gate does not. Seed 1101 leaves
the met-station and FC4 channels unused; seed 1105 leaves the met-station,
radiometer, and FC4 channels unused. These channels remain unused during both
event and non-event epochs, so restricting guidance to event samples did not
repair deterministic channel collapse.

The failure is not explained by insufficient switching alone. Every run has a
nonzero switch rate, while the two failed policies repeatedly choose a small
set of masks. Seed 1105 additionally executes the empty subset for 27.9% of test
epochs. V93 is therefore retained as positive performance evidence and rejected
as the frozen primary configuration.

## Next bounded test

V94 retains the V93 objective and event-only guidance and changes only the
subset representation from nonlinear to linear additive action embeddings.
This test is motivated by the earlier V80 result, where the additive
representation passed the behavior gate in all five matched scenes. V94 must
retain at least 4/5 strongest-static wins, 5/5 conventional-dynamic wins, and
pass the behavior gate in all five V83a scenes before supplementary baselines
or fresh confirmation are authorized.
