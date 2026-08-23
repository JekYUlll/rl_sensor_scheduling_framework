# V108 soft forecast-value probe

V108 matches V107 on frozen V103 scene seed `1304`, 4096 policy-training
states, network, final windows, and disabled PPO/auxiliary updates. It replaces
hard argmin labels with `softmax(-cost / 0.05)` over feasible eight-step
forecast-value costs.

- Training argmax accuracy is `0.2375`; all 22 teacher actions remain present.
- Held-out ordinary/macro losses are `0.737331/1.807278`, substantially worse
  than V107 (`0.539649/1.400319`) and strongest static
  (`0.528512/1.361190`).
- Five channels have intermediate duty, switching is `0.030974` per step, and
  no warm-up abort occurs.

Low-temperature soft targets do not improve transfer and are closed. V107's
hard-label gain with limited accuracy instead motivates a structured temporal
encoder for the online history and observation-mask sequence.

Full artifacts are archived at
`/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v108_frequency_cost_soft_value_probe_full`.
