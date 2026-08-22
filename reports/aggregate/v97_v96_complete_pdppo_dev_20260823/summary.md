# V97 complete PD-PPO on frozen V96 scenes

V97 trains the complete nonlinear PD-PPO configuration on the frozen V96
equal-base-weight scene assets with policy seeds 3101--3105. No action prior,
duty regularizer, hard duty guard, stochastic final execution, or scene change
is used.

- Strongest-static ordinary wins: 4/5; mean margin `+0.020304`.
- Strongest-static macro wins: 5/5; mean margin `+0.086369`.
- Conventional-dynamic joint wins: 5/5; mean ordinary/macro margins
  `+0.022752/+0.047480`.
- Warm-up aborts: zero in every seed; switching is nonzero in every seed.
- No-constant-channel behavior passes: 4/5.

The only ordinary static miss is seed 1105 (`-0.002231`), which retains a
positive macro margin (`+0.089544`). The only constant channel is FC4 in seed
1101. This is traceable to the calibration-selected four-state teacher map:
none of its four actions contains FC4. V97 otherwise exhibits strong
state-dependent allocation; several channels change event versus non-event duty
by more than 0.5.

V97 passes the predictive-performance gate but not the stronger behavior gate.
V98 changes only the teacher action map to the fixed physical-function map,
whose four states cover all six channels, while retaining V97's AWBC coefficient
and every policy, reward, scene, and evaluation setting.
