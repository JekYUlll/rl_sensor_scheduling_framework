# V98 physical-function teacher result

V98 retains the frozen V96 scenes and all V97 learner, reward, and evaluation
settings. It changes only the four teacher actions from seed-specific calibrated
maps to the fixed physical-function map whose union covers all six channels.

- Strongest-static ordinary/macro wins: 4/5 and 5/5.
- Mean strongest-static ordinary/macro margins: `+0.015746/+0.091400`.
- Conventional-dynamic joint wins: 5/5.
- Mean conventional-dynamic ordinary/macro margins: `+0.018194/+0.052511`.
- Strict no-constant-channel behavior: 4/5; zero aborts in every seed.

Seed 1101 still gives FC4 zero duty even though the flux teacher action contains
FC4. The fixed-map replacement therefore does not explain or repair the
collapse. Training records 31 greedy actions near the end of optimization, but
the final seed1101 test rollout uses only six actions, indicating that a
four-state static action teacher does not transfer the within-state temporal
variation needed by the test distribution.

Further static-teacher weighting is closed. V99 tests a prediction-driven
training teacher on seed1101 only: it chooses feasible actions by eight-step
frozen-forecaster value within the training partition. Subtype action CE is
disabled so that it cannot conflict with the dynamic teacher.
