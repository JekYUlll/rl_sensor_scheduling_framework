# V344 Direct Mask-Action PD-PPO Development Run

## Method change

V344 added a direct state--candidate-mask utility head to the masked PPO
actor. The head receives the encoded online state and the raw six-channel
candidate mask. The forecast-loss reward, arbitrary feasible-subset action
space, hard feasibility mask, V338 scene, dwell protocol, and training-only
forecast-value initialization were unchanged. No bandit output, comparator
action, or final-test label was used.

## Verified results

Both seeds passed the operational behavior gate: zero warm-up aborts, zero
always-on channels, zero always-off channels, and six intermediate-duty
channels. Switching rates were `0.060501` and `0.032494` per step. BC action
accuracy increased to approximately `0.831` on both training batches, versus
`0.108--0.143` for V342.

Predictive transfer did not improve. PD-PPO ordinary loss was `0.512627` and
`0.355076` for scenes 6871 and 6872, while the best static losses were
`0.420823` and `0.276648`. Static-normalized macro losses were `1.142025` and
`0.628645` for PD-PPO versus `0.914900` and `0.426915` for the corresponding
best static schedules. PD-PPO therefore lost to the static shortcut on both
seeds and both endpoints.

## Decision

The direct mask head fixes the supervised candidate-representation capacity
problem but not closed-loop forecast transfer. It is rejected as a primary
method improvement. The evidence supports a mismatch between local
candidate-cost supervision and long-horizon executed-return optimization.
The head remains available as a reproducible diagnostic; no positive claim is
assigned to V344.

Raw per-seed metrics and training histories are stored in the two seed
subdirectories. The implementation entry point is
`--direct-mask-action-score`.
