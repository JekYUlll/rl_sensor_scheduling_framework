# V112 validation-selected temporal forecast-value PPO

V112 evaluates the forecast-value BC checkpoint at update 0 and every fifth
PPO update on the calibration/validation partition. Selection uses ordinary
forecast loss, and the selected checkpoint is evaluated once on test.

- Validation selects update 35 (`35,840` PPO steps), with ordinary loss
  `0.370136`. The BC checkpoint has `0.385357`. The static validation reference
  is still stronger at `0.364751`.
- A static-normalized joint score computed after completion also selects update
  35; changing to a two-endpoint selector would not alter this run.
- Test ordinary/macro losses are `0.561536/1.504863`, trailing strongest static
  by `-0.033024/-0.143673` and the best conventional dynamic endpoints by
  `-0.034142/-0.090467`.
- Five channels have intermediate duty, all channels have nonzero duty,
  switching is `0.050297` per step, and there are no aborts.

The checkpoint mechanism is now complete, but validation selection does not
transfer on this scene. No tested online model beats static on the validation
partition, while the positive V109 test result reverses that ordering. V103
therefore does not pass an online-transfer gate and should not receive further
PPO tuning. The next work unit must improve deployable online identifiability or
scene observability before retraining.

Full artifacts are archived at
`/data/zhangzhuyu/experiment_artifacts/pdppo/20260823_v112_frequency_cost_temporal_forecastbc_ckptppo_full`.
