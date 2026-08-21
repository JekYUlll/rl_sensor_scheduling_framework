# Final manuscript evidence outline

This outline fixes the argument order for the canonical English manuscript
after the clean actor-only evidence is complete. It is not a result draft and
contains no claims inferred from partial runs.

## Central method claim

PD-PPO is a prediction-driven, feasibility-masked PPO framework for selecting
executable channel subsets under a power budget and minimum activation rule.
Its identity comes from the downstream forecast objective, the constrained
action geometry, and sequential policy learning. It must not be presented as a
new PPO optimizer or as a benchmark-only contribution.

## Evidence hierarchy for Results

1. **Clean held-out policy result.** Compare the frozen actor-only PD-PPO with
   the validation-selected fixed schedule and named conventional schedules over
   seeds 117--140. Report the 24-seed aggregate and the unchanged 22-seed
   post-pilot replication separately where useful. State ordinary forecast loss,
   validation-normalized regime macro score, paired margins, confidence
   intervals, and win counts.
2. **Objective and learning-method isolation.** Compare the same masked-PPO
   architecture trained with forecast, AoI, and diagonal-uncertainty rewards.
   Then compare forecast-trained PD-PPO with the same-mask Double-DQN. This
   subsection must report the observed ranking even if the forecast reward does
   not dominate both proxies; no reward-specific retuning is permitted.
3. **Policy behavior and regime allocation.** Use frozen final trajectories to
   report mask entropy, subtype--mask mutual information, fixed/cycle checks,
   switching, aborts, and per-channel duty. Event-type grouping is an offline
   interpretation only. The mandatory weather backbone and the unused or
   low-duty specialist must be explained explicitly.
4. **Claim-boundary references.** Report one-step forecast greedy, the
   handcrafted context-alert rule, and the privileged event-label replay in
   separate roles. Only the first two are online-feasible references; the exact
   event-label replay is an opportunity diagnostic and never the learned policy.
5. **Forecaster-family sensitivity.** Rescore the same frozen final trajectories
   with the ridge forecaster fitted only on the forecaster-fitting partition and
   with its static schedule selected only on validation. This is a robustness
   check, not a second training result.

## Discussion logic

- Start from what the clean result establishes about adaptive specialist
  allocation under a hard capacity bottleneck.
- Interpret same-architecture reward controls as an empirical test of objective
  alignment. If forecast, AoI, and uncertainty are statistically close, retain
  the theoretical non-equivalence claim but state that the auxiliary training
  signals and small action surface can reduce empirical separation.
- Use Double-DQN to distinguish the contribution of stable masked policy
  optimization from the existence of a learnable six-action problem.
- Explain why the context-alert rule is strong: the benchmark supplies noisy
  leading warning scores derived from the synthetic event process. Do not call
  them exact event labels or a field-validated detector.
- Explain sequential value through the six-epoch mask hold and delayed forecast
  consequences; do not describe the rule as a generic deployment constraint.
- Keep simulator transfer, fixed-forecaster dependence, one-specialist scope,
  and synthetic warnings in a concise limitations subsection after the positive
  interpretation.

## Summary-layer update order

Update Results and all generated tables/figures first. Then update Discussion,
Conclusion, Introduction, and Abstract in that order. The abstract must mention
only evidence present in the final aggregate and must not use old hard-router
win counts. The Introduction contributions remain formulation, feasible-mask
PPO implementation, and evidence protocol, with the reward controls and matched
learner strengthening the second contribution.

## Evidence-dependent wording branches

- Forecast reward clearly wins both same-architecture controls: report direct
  empirical support for forecast-objective alignment.
- Forecast reward is comparable to one or both controls: state that all three
  objectives learn valid adaptive schedules, while the forecast reward directly
  optimizes the stated downstream task; do not claim proxy inferiority.
- Forecast reward loses a control: retain PD-PPO performance and matched-DQN
  claims, report the control result prominently, and narrow empirical reward
  claims without weakening the constrained scheduling framework itself.
- Context-alert or event-label reference wins: treat it as a strong handcrafted
  or privileged boundary, never as evidence that the main policy failed its
  fixed/conventional/learned-policy comparisons.

## Final rejection scan

The production pass must reject: hard-router results, exact online event labels,
old values `0.0778`, `0.0824`, `0.0054`, `0.0314`, old `19/24` and `7/24`
counts, historical mechanism-ablation claims, and changed-mixture claims not
regenerated for the selected actor-only policy. Every surviving number must map
to one frozen aggregate CSV or a configuration asset whose checksum is recorded.
