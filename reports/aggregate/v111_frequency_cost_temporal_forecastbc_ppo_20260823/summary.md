# V111 PPO transfer from temporal forecast-value initialization

V111 starts from V109's temporal architecture and hard eight-step forecast-
value BC design, then adds 40,960 forecast-loss PPO steps. Continuing AWBC,
subtype auxiliary classification, and subtype-action supervision are disabled.

- Ordinary/macro losses are `0.539342/1.399040`. Relative to V109, PPO degrades
  the endpoints by `0.020154/0.087759`.
- V111 trails strongest static by `-0.010830/-0.037850`. It trails the best
  conventional dynamic ordinary endpoint by `-0.011948`, while retaining a
  `+0.015356` macro margin.
- The policy executes 13 subsets, all six channels have nonzero duty, five have
  intermediate duty, switching is `0.026198` per step, and there are no aborts.
  Unlike V110, its failure is not empty-action or channel-coverage collapse.
- Final actor entropy remains `2.8564`, so the degradation is not explained by
  a nearly deterministic categorical distribution.

The forecast-loss PPO updates do not preserve the superior held-out ranking of
the BC checkpoint. The existing validation checkpoint mechanism evaluates only
post-PPO updates and therefore cannot retain the BC model. V112 repairs this
general model-selection omission and compares the step-zero BC checkpoint with
every fifth PPO update on the calibration/validation partition before one test
evaluation.

Full artifacts are archived at
`/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v111_frequency_cost_temporal_forecastbc_ppo_full`.
