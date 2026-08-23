# V110 matched temporal complete-PPO control

V110 repeats V104 on frozen V103 scene seed `1304` and policy seed `4304`.
The only configuration change is the V109 GRU encoder over the existing online
20-step value and observation-mask history.

- Ordinary/macro losses improve only slightly from V104's
  `0.620068/1.684553` to `0.610400/1.652073`.
- V110 remains below the strongest static reference by
  `-0.081889/-0.290883` and below the best conventional dynamic endpoints by
  `-0.083007/-0.237678`.
- Execution uses only five subsets. The empty subset is selected in
  `963/2304` steps, FC4 is always off, mean power is `0.582743`, and only three
  channels have intermediate duty. The prespecified constant-channel behavior
  gate still passes, but the policy is operationally too sparse.
- The initial subtype-static BC accuracy is `0.4634`, while action entropy falls
  from `3.0681` after pretraining to `1.9762` at 40,960 PPO steps.

Temporal encoding alone does not repair the complete V104 training scaffold.
The contrast with V109 localizes the remaining failure to the training target
and PPO transfer: hard forecast-value supervision learns useful pair schedules,
whereas subtype-static initialization followed by PPO returns to empty and
single-channel actions.

Full artifacts are archived at
`/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v110_frequency_cost_temporal_pdppo_control_full`.
