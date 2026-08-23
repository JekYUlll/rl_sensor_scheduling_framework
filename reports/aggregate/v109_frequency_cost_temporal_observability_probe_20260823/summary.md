# V109 temporal online-observability probe

This diagnostic repeats V107's training-partition-only hard eight-step
forecast-value supervision on frozen V103 scene seed `1304`. The sole model
change is a GRU encoder over the existing 20-step carried-value and observation-
mask history. PPO updates and subtype auxiliary labels remain disabled, and
final evaluation receives no future labels.

- The teacher covers all `22` feasible actions. Training action accuracy is
  `0.2078`, below V107's `0.2493`; the improvement therefore does not come from
  fitting the privileged labels more closely.
- Held-out ordinary and macro losses are `0.519187` and `1.311281`, improving
  over V107 by `0.020462` and `0.089038`, respectively.
- V109 beats the strongest validation-selected static schedule by
  `+0.009325` ordinary and `+0.049909` macro margin. It also beats the best
  conventional dynamic endpoint by `+0.008207` ordinary and `+0.103115` macro
  margin.
- Five channels have intermediate duty, no channel is always on, one channel
  is always off, switching is `0.007816` per step, and there are no warm-up
  aborts. This passes the prespecified behavior gate of at most one constant-on
  and at most one constant-off channel.

Structured encoding of the deployable temporal history improves held-out
action transfer on the principal V104 failure scene. This authorizes one
matched complete-PPO control with no other scene, reward, supervision, or
optimization change.

Full artifacts are archived at
`/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v109_frequency_cost_temporal_observability_probe_full`.
