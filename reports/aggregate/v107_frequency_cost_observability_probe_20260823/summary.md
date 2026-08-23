# V107 online observability probe

This diagnostic trains an online policy only from hard eight-step
forecast-value action labels generated on the policy-training partition of
frozen V103 scene seed `1304`. PPO updates, subtype auxiliary labels, and
subtype-action supervision are disabled. Final evaluation uses no future label.

- The teacher spans all `22` feasible actions. Training action accuracy is
  `0.2493`, compared with `0.0455` uniform chance.
- Held-out ordinary loss improves from V104's `0.620068` to `0.539649`, and
  macro loss improves from `1.684553` to `1.400319`.
- The probe still trails the strongest validation-selected static schedule by
  `-0.011138` ordinary and `-0.039129` macro margin. It trails the best
  conventional dynamic ordinary endpoint by `-0.012255`, while exceeding its
  macro endpoint by `+0.014077`.
- All six channels have intermediate duty, switching is `0.019540` per step,
  and there are no warm-up aborts.

The online state contains useful information about future-value actions, but a
hard argmin classification target does not fully transfer the privileged
receding policy. Continuous action-value learning remains justified.

Full artifacts are archived at
`/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v107_frequency_cost_observability_probe_full`.
