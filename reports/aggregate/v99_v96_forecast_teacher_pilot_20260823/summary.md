# V99 forecast-greedy teacher pilot

V99 replaces V97's four validation-calibrated static teacher actions with an
eight-step frozen-forecaster greedy teacher on the policy-training partition.
The frozen V96 seed-1101 scene, objective, action geometry, budget, policy
architecture, forecast reward, and deterministic final execution are unchanged.

- PD-PPO ordinary/macro losses: `0.453898/1.317553`.
- Margin against the strongest static family: `-0.019192/+0.003126`.
- Margin against the strongest conventional dynamic family:
  `-0.006112/-0.024157`.
- Behavior: zero always-on channels, zero always-off channels, six
  intermediate-duty channels, 24 executed subsets, switching `0.043566`, and
  zero warm-up aborts.
- Training AWBC cross-entropy remained near the 36-action random scale despite
  the teacher producing all 35 non-empty feasible actions.

The dynamic teacher broadens channel use but fails the prediction gate. Its
fine-grained target depends on forecast consequences that are not recoverable
reliably from the policy's current online state. V99 is not expanded. The
four-state validation-calibrated V97 guidance remains the frozen primary method.
