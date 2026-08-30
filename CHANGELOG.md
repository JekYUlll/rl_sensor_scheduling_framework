# PD-PPO Scene Recalibration Changelog

## 2026-08-30 - V245 tests candidate-conditioned on-policy values

- Added a five-seed development launcher for the existing candidate-conditioned
  on-policy value head. The head is trained from PPO returns and contributes to
  candidate-mask logits; the forecast-loss reward, online context, feasible
  subset geometry, and no-teacher configuration are unchanged.
- V245 uses no bandit signal, residual action, counterfactual label, or final
  execution event label. It tests whether an explicit state-action value
  representation can recover action consequences that the shared embedding did
  not learn.
- Results will be compared with V243 and V244 after raw metrics and behavior
  diagnostics are complete.

## 2026-08-30 - V245 rejects candidate-conditioned on-policy values

- V245 completed on seeds `3601--3605` with the V243 clean forecast-loss PPO
  scaffold and the existing candidate-conditioned on-policy value head
  (`coef=0.10`, actor logit scale `0.50`). No bandit signal, teacher action,
  counterfactual label, or test-time event label was used.
- The variant failed the behavior gate: four of five runs had zero mid-duty
  channels, and the runs contained 2--5 always-off or always-on channels. All
  runs had zero warm-up aborts; switching was zero in four runs and `0.008684`
  in the fifth.
- Relative to the V245 PD-PPO row, the mean baseline-minus-PD-PPO margins were
  `-0.253446` for validation-selected static, `-0.009292` for feasible static,
  `-0.101001` for the best original dynamic family, and `-0.128526` for the
  unconstrained full-open reference. Macro wins were `1/5`, `2/5`, `1/5`, and
  `1/5`, respectively. The auxiliary value head therefore did not improve the
  action-value mapping and caused policy concentration.
- Raw and recomputed aggregates are stored under
  `reports/aggregate/v245_onpolicy_action_value_pdppo_dev_20260830/`. No
  confirmatory expansion is authorized for this variant.

## 2026-08-30 - V246 rejects normalized candidate values

- V246 reran the V245 candidate-conditioned on-policy value head after
  standardizing candidate values per state before detached logit fusion. All
  other settings were unchanged; fresh seeds were `3606--3610`.
- The correction did not recover state-dependent behavior. Mid-duty counts were
  `0/5`; always-off counts were `3,4,3,3,3`; always-on counts were `2,1,2,2,2`;
  switching was `0` in every seed; warm-up aborts remained zero.
- The variant failed to beat the best static family (`0/5` ordinary and `2/5`
  macro wins; mean margins `-0.095166/-0.073904`) and the best original
  dynamic family (`2/5` ordinary and `3/5` macro wins; mean margins
  `-0.052972/-0.034562`). The normalized action-value route is closed.
- Recomputed evidence is stored under
  `reports/aggregate/v246_normalized_action_value_pdppo_dev_20260830/`. No
  confirmatory expansion is authorized for this route.

## 2026-08-30 - V244 result closes the direct quality-action route

- V244 completed on fresh seeds `3501--3505` with the existing forecast-loss
  reward, hard feasible-subset action space, online context, and no AWBC/BC or
  teacher action guidance. The five quality forecasts were placed in the
  leading positions required by the action-conditioned quality module.
- Behavior was valid in all five runs: zero warm-up aborts, zero always-on and
  always-off channels, five mid-duty channels, and switching rates
  `0.063048--0.095441`.
- Against validation-selected static, V244 achieved `1/5` ordinary wins and
  `0/5` macro wins, with mean baseline-minus-PPO margins `-0.051067/-0.069200`.
  It also failed to beat AoI or round-robin consistently. The explicit quality
  action score therefore improves neither static performance nor the central
  action-value mapping under this scene.
- No confirmatory expansion is authorized for this route. Aggregates are stored
  under `reports/aggregate/v244_quality_action_context_pdppo_dev_20260830/`.

## 2026-08-30 - V244 rejects direct quality-action scoring

- V244 evaluated the existing action-conditioned quality representation on fresh
  seeds `3501--3505`, retaining the V243 clean forecast-loss PPO configuration
  and placing the five quality forecasts first in the context vector.
- Behavior passed in all five runs: zero warm-up aborts, zero always-on/off
  channels, five mid-duty channels, and switching rates of `0.0630--0.0954`.
- The representation did not recover performance: PD-PPO beat
  validation-selected static in `1/5` ordinary and `0/5` macro comparisons,
  with mean margins `-0.051067/-0.069200`. It also failed to beat AoI or
  round-robin consistently.
- V244 closes the direct quality-action-score route. The next diagnosis must
  examine whether the forecast-loss reward provides an identifiable action
  consequence under partial observations, rather than adding another context
  feature or teacher prior.
- Aggregates are stored under
  `reports/aggregate/v244_quality_action_context_pdppo_dev_20260830/`.

## 2026-08-30 - V244 tests action-conditioned quality representation

- Added a five-seed development launcher that reuses the V243 clean PPO
  configuration while enabling the existing quality-context action score.
  Quality forecasts are placed in the leading five context positions required
  by the model interface; weather nowcasts remain available to its context
  encoder.
- V244 changes no scene, reward, feasibility rule, teacher, bandit signal, or
  test-time label. It tests whether explicit candidate-mask quality utility
  improves prediction-driven action selection.
- Results will be interpreted against V243 only after all five raw metrics and
  behavior diagnostics are present.
- The launcher records V244 logs under a separate prefix to preserve experiment
  provenance when reusing the V243 execution scaffold.

## 2026-08-30 - V243 removes the teacher collapse but not the static gap

- V243 reran the corrected balanced quality scene on seeds `3401--3405` for
  `100,000` PPO timesteps. It retained forecast-loss reward, online weather
  and alert context, subtype auxiliary prediction, arbitrary feasible subsets,
  and execution constraints, while removing AWBC, behavior-cloning pretraining,
  and subtype teacher actions.
- All five runs passed the behavior gates: zero warm-up aborts, zero always-on
  channels, zero always-off channels, and 4--5 mid-duty channels per rollout.
  The mean switching rate was `0.071906`.
- The policy did not beat validation-selected static on ordinary or macro loss
  (`0/5` for both; mean margins `-0.071337/-0.100279`). It beat random in
  `4/5` on both endpoints, while AoI and round-robin were each `3/5`.
  V243 therefore confirms that the V242 permanent-off behavior was induced by
  the training teacher, but removing that teacher alone does not solve the
  forecast-value-to-action mapping or static-shortcut problem.
- Aggregates are stored under
  `reports/aggregate/v243_no_awbc_pdppo_balanced_quality_dev_20260830/`.

## 2026-08-28 - V233 closes joint weather-and-health state admission

- V233 is a read-only information gate on the frozen V232 physical-group
  scenes. It combines the legal noisy weather-nowcast tail with online
  instrument-health diagnostics, fits action-value diagnostic models on
  chronological policy-training and validation traces, and replays the selected
  trace-distilled policy only on final partitions. It does not retrain PD-PPO
  or expose event labels, heuristic actions, or future losses at execution.
- The strongest offline complete-state model is a ridge candidate-cost regressor
  fitted on policy-training plus validation traces. It has mean static gain
  `+0.005214` with `4/5` positive scenes, recovering `17.81%` of the privileged
  receding gain. The alert-only ExtraTrees probe is positive in `5/5` scenes
  but has lower mean gain (`+0.004326`).
- The mandatory closed-loop replay rejects admission: it beats the
  validation-selected static schedule on ordinary and macro loss in only `2/5`
  scenes, with mean margins `-0.001087/-0.008071`, and passes the broad-use
  behavior gate in `0/5` scenes. It collapses to one or two always-on groups
  and two or three always-off groups per scene, despite zero warmup aborts.
  Thus the joint legal state does not survive policy-induced observation and
  dwell-state shift. No new PD-PPO training is authorized on V232; the next
  scene revision must strengthen the physically interpretable,
  forecast-observable instrument-value relation without changing fixed
  effective costs, sampling interval, or arbitrary feasible-subset geometry.

## 2026-08-28 - V232 rejects health-only control despite physical-group dynamic headroom

- V232 adds distinct, fixed weather-exposure reliability profiles for the five
  verified physical instrument groups while preserving V227's fixed effective
  loads, unrestricted feasible subsets, and eight-step noisy weather nowcast.
  Health reports are online diagnostics; they do not expose event labels or
  change the sampling interval.
- The paired receding control retains broad dynamic opportunity: it wins against
  validation-selected static on ordinary loss in `5/5` scenes and macro loss in
  `4/5`, with mean margins `+0.032218/+0.072460`. Every replay uses `15`
  actions, has zero aborts, and keeps all five instrument groups at intermediate
  duty.
- The calibrated health-only reference does not clear static (`0/5` ordinary,
  `1/5` macro; mean margins `-0.003922/-0.017156`) and remains fixed-like, with
  one always-on group in every scene. This closes health-only scene admission.
  The next bounded gate must test a joint online weather-nowcast plus health
  diagnostic mapping before any primary PPO training.

## 2026-08-28 - V231 closes online-state learnability for the physical-group nowcast scene

- V231 is a read-only learnability audit of the completed V229 physical-group
  runs. It fits diagnostic classifiers and candidate-cost regressors on
  chronological policy-training and validation receding traces, then evaluates
  them only on held-out final traces. It does not retrain PD-PPO, alter the
  simulator, or introduce event labels, heuristic actions, or privileged inputs
  into the deployed state.
- The strongest legal probe uses the alert-context tail with an ExtraTrees
  candidate-cost regressor fitted on policy-training plus validation traces. It
  improves over the validation-selected static action in all five held-out
  scenes, but recovers only `17.90%` of the paired eight-step receding gain on
  average (`+0.003090` gain versus `+0.017293` receding gain). Its mean top-1
  action agreement is `31.96%`.
- Adding the complete carried-state observation does not improve this result:
  the strongest complete-state probe recovers `13.02%` of receding gain. The
  physical-group scene therefore has real but weak legal predictive signal for
  dynamic action selection. Further PPO losses, teachers, or actor priors are
  not justified on this scene; the next design step is to strengthen the
  physically interpretable condition-to-measurement-value relation while
  retaining fixed effective costs and unrestricted feasible subsets.

## 2026-08-28 - V230 closes actor-value initialization on physical groups

- V230 keeps V229's physical action surface and legal l8 weather forecast, but
  initializes and regularizes the actor with frozen-forecaster candidate values
  produced solely on policy-training starts. It is the direct, method-consistent
  test of the credit-assignment explanation; no rule action or execution-time
  privileged signal is added.
- The configuration does not pass. Mean validation-static-minus-PPO
  ordinary/macro margins are `-0.027334/-0.083835`, with `1/5` and `0/5` wins.
  Best-conventional-dynamic means are `-0.005215/-0.034238`, with `2/5` and
  `1/5` wins. Four runs retain all five groups at intermediate duty; no run has
  an always-on group.
- The actor-value route is closed for this scene. Before changing either the
  generator or the method, the next step is an online-nowcast policy audit to
  determine whether the available forecast inputs themselves support a
  non-RL context decision under the same feasibility and dwell rules.

## 2026-08-28 - V229 closes the nowcast-only physical-group screen

- V229 restores only the legal noisy eight-step wind, humidity, and
  temperature forecast to the V227 physical-group state. It keeps the same
  clean masked PPO objective and forbids labels, heuristic actions, priors,
  imitation, and forecast-value targets.
- The resulting policy remains behaviorally broad (five intermediate-duty
  groups in four runs; no always-on group) but does not clear prediction gates.
  Mean validation-static-minus-PPO ordinary/macro margins are
  `-0.031111/-0.061114`, with `1/5` and `2/5` respective wins. Against the
  best conventional dynamic reference, means are `-0.011688/-0.034604` with
  `3/5` and `2/5` wins.
- This isolates the remaining gap to action-value credit assignment: legal
  forecasts are present but scalar forecast reward alone does not reliably
  train their mapping to candidate masks. The next bounded configuration uses
  the existing training-partition frozen-forecaster l8 actor-value targets;
  no execution-time privileged signal is added.

## 2026-08-28 - V228 closes the no-forecast-input policy screen

- V228 trains the unassisted, no-teacher masked PPO configuration for 50,176
  steps on five fresh physical-group scenes (`2471--2475`). It retains all five
  groups at intermediate duty in four runs and four in the fifth, with no
  always-on or always-off channel; this is not an action-collapse failure.
- It nevertheless loses to validation-selected static on ordinary and macro
  loss in every run. Ordinary/static-minus-PPO margins are `-0.052617`,
  `-0.067621`, `-0.026967`, `-0.018801`, and `-0.016574`; it beats the best
  conventional dynamic baseline only on seed `2474`.
- Metadata audit shows the policy was given no agent-context columns and no
  alert tail. Thus it saw current and historical measurements but no legal
  forward weather estimate, whereas the V227 admission diagnostic scores
  eight-step forecast value. V228 closes the no-forecast-input configuration;
  it does not reject the physically grouped scene or prediction-driven method.

## 2026-08-28 - V227 admits the verified physical instrument grouping

- V227 replaces the unverified sixth logical sensor with five independently
  powered physical instrument groups: GMX500, LPS10, SI-111, Parsivel2, and
  FlowCapt FC4. Their configured values are fixed per-epoch effective loads,
  not controllable sampling frequencies or literal watt measurements.
- At budget `1.85` and startup budget `2.25`, the unrestricted feasible action
  surface contains all five singleton actions and nine legal pairs (`15` masks
  including the empty action). It therefore has no fixed three-way selection
  rule, while preserving the physical power trade-off.
- The exact l8 receding diagnostic clears the static gate on all fresh scene
  seeds `2461--2465`. Validation-static-minus-receding ordinary-loss margins
  are `+0.033381`, `+0.017326`, `+0.032904`, `+0.022065`, and `+0.019124`.
  Every physical group has nonzero replay duty across all seeds; aggregate
  per-seed duties range from `0.088` to `0.805`, with switching rates
  `0.0916--0.1411` per step.
- This admits one clean, no-teacher PD-PPO training screen on fresh policy
  development seeds. The diagnostic's minimal 1,024-step PPO artifacts remain
  excluded from performance evidence.

## 2026-08-28 - V219--V220 Establish a Meteorological Nowcast Input Path

- V219 audits a label-free four-step proxy built solely from noisy forecasts of
  wind speed, relative humidity, and air temperature. Across the frozen V213
  final partitions it attains mean four-class macro-F1 `0.4808` and event
  recall `0.6481`; no policy, event label, target value, or candidate loss is
  consumed by the proxy.
- V220 provides the corresponding perfect-forecast information bound. The same
  three meteorological quantities attain mean macro-F1 `0.5898` and event
  recall `0.7726`. Thus the quantities contain useful anticipatory information;
  the V219 reduction is attributable to forecast error rather than an absent
  physical relationship.
- The next scene implementation will expose a fixed-error exogenous weather
  nowcast through the scheduler context interface and remove event-label-derived
  alert context from the primary online input. It must pass the existing online
  and receding gates before any PD-PPO training.

## 2026-08-28 - V217--V218 Close the Short-Horizon Distillation Route

- V217 reuses V216's frozen l4 final traces and generates only training and
  validation l4 traces. Complete-state ExtraTrees prediction of l4 receding
  actions reaches positive held-out action-value gain in `4/5` scenes with mean
  `+0.003486`, modestly better than the corresponding l8 probe.
- V218 is the required closed-loop check. The ExtraTrees policy is fitted only
  to policy-training l4 costs and replayed once under the normal final
  environment transitions. It loses ordinary loss to validation-selected static
  in all five scenes (mean static-minus-policy `-0.020707`) and macro loss in
  four scenes (mean `-0.040001`).
- Runtime feasibility remains intact (zero aborts), but the policy averages
  `2.4` always-off channels and only `2.8` intermediate-duty channels. Thus
  better open-loop action prediction is not sufficient under policy-induced
  observation and dwell-state shift.
- V213 is closed. No CA-PD-PPO, feature-parity PPO, alert-rule extension, or
  additional trace-distillation variant is authorized on this scene. The next
  scene family must expose a physically interpretable, exogenous forecast or
  nowcast precursor, not a stronger event-label-derived alert.

## 2026-08-28 - V216 Establishes the Receding Lead-Time Threshold

- On the frozen V213 final partitions, the exact all-action receding diagnostic
  has mean validation-static-minus-receding margins of `-0.005028`, `+0.001811`,
  `+0.018678`, and `+0.032106` for privileged lookaheads `0`, `2`, `4`, and `8`
  respectively. The corresponding ordinary-loss win counts are `1/5`, `3/5`,
  `5/5`, and `5/5`.
- Dynamic value therefore begins reliably at four future scheduling intervals.
  This is an offline information bound, not a deployable policy result: each
  receding policy sees future target-dependent loss unavailable at execution.
- The existing synthetic warning tail was configured with a nominal 12-step
  lead, yet thresholded final-window alerts identify only `29.6%` of subtype
  onsets at least four steps early (median observed lead `2` steps). V217 tests
  whether the simpler four-step receding target is learnable before changing
  warning generation or the primary PD-PPO method.

## 2026-08-28 - V215 Closes the Full Online-State Sufficiency Check

- V215 first corrected the diagnostic trace path so that receding traces retain
  every legal online scheduler feature, including the complete 20-value alert
  context. The repaired training, validation, and final traces each contain
  `533` online-state values; earlier `514`-value traces are not used.
- Training-partition probes do not show a reliable advantage from the complete
  state over alert context alone. The best complete-state ExtraTrees probe has
  mean training gain `+0.003123` versus static in only `3/5` seeds, below the
  alert-only probe's `+0.004640` in `3/5` seeds.
- The final closed-loop trace-distilled replay is negative against the
  validation-selected static schedule: ordinary static-minus-policy margin is
  positive in `1/5` seeds with mean `-0.003199`; macro margin is positive in
  `2/5` seeds with mean `-0.005198`. It has zero aborts but still leaves an
  average `2.6` channels always off.
- This diagnostic neither trains nor modifies PD-PPO. It closes feature
  stacking and context-rule expansion on V213: the existing legal online state
  does not recover the eight-step receding advantage. The next bounded step is
  an offline `0/2/4/8`-step lookahead sweep to quantify the required warning
  lead before modifying the synthetic precursor process.

## 2026-08-28 - V212 Closes the Coverage-Balanced Cost Geometry Screen

- V212 applies one physical effective-load recalibration before any result is
  inspected: the multi-variable GMX500 core channel receives its bundled
  acquisition/interface cost, yielding an arbitrary-subset surface of six
  singles and eleven feasible pairs (`18` masks total). No cardinality rule,
  sampling-frequency action, or PPO change is introduced.
- The geometry improves alert-only online margins on average
  (`+0.007626/+0.035088` ordinary/macro) and reduces persistent core-channel
  use, but it passes both endpoints in only `2/5` development scenes. The
  quality-aware rule is behaviorally broader but passes only `1/5` scenes;
  its best mean margins are `-0.016160/-0.022491`.
- The exact eight-step receding diagnostic is positive in `3/5` scenes with
  mean static-minus-receding ordinary margin `+0.030065`, but it loses in
  seeds `2202` and `2205`. Thus the cost geometry does not provide robust
  dynamic opportunity independently of the learner, and V212 is rejected
  before PPO training.
- This closes fixed effective-cost rebalancing as a standalone remedy. The next
  analysis must inspect failed-scene event coverage and validation-selected
  masks before changing either the target process or the physical observation
  relationship; another budget or penalty scan is not justified.

## 2026-08-28 - V211 Rejects the First Condition-Dependent Reliability Scene

- V211 introduces a development-only, condition-dependent channel-quality
  generator over the physical six-channel, 22-mask geometry. It preserves
  exogenous event generation and derives noisy online quality diagnostics from
  continuous wind, humidity, particle, flux, and thermal exposure. No PPO
  training is authorized or run in this screen.
- The privileged eight-step receding diagnostic confirms substantial dynamic
  opportunity in all five fresh development scenes: validation-static minus
  receding ordinary loss is positive in `5/5` seeds with mean `+0.027530`.
  This is an upper diagnostic only, not a deployable comparison.
- The initial alert-only context rule is not a valid reliability gate because
  it omits the new diagnostics. The corrected quality-aware online rule was
  therefore evaluated at three prespecified quality penalties (`0.25`, `1.0`,
  and `4.0`). It records `0/5` joint ordinary/macro wins at every penalty; the
  least-negative mean margins are `-0.014917/-0.048673` at penalty `0.25`.
- The quality-aware policy also concentrates on a few pairs in several seeds,
  despite zero aborts. V211 is rejected before PPO training: its dynamic
  opportunity is not recoverable by the specified online context-and-quality
  gate. The next scene design must examine why validation regime maps collapse
  to robust pairs before changing reliability magnitudes again.

## 2026-08-28 - V210 Closes the Selected-Action Q-Head Diagnostic

- V210 replays the frozen V208 checkpoint over the same `2,304` final-window
  states as V209 and records the on-policy action-value head for every
  currently feasible mask. The diagnostic remains offline-only: privileged
  eight-step candidate costs are used solely to rank the frozen predictions.
- The Q head improves local-best identification over the categorical actor,
  but not enough to support an actor-side scale sweep. Its top mask matches
  the privileged local best in `17.14%` of states and assigns the local best a
  mean rank of `8.16` among feasible masks, compared with the actor's
  `1.04%` and `9.68`, respectively. The median Q rank is `6.00`.
- The Q and actor orderings are not aligned: the Q top mask never equals the
  actor's deterministic selected mask in this audit. Q top-match rates remain
  low in calm/particle/flux/thermal states (`16.45%/12.26%/15.37%/27.52%`).
  A larger Q-logit scale would therefore amplify a still inaccurate ordering,
  not establish a clean repair.
- Together with V208's failed closed-loop result, this closes the
  selected-action return-head route. Rewards observed only for executed masks
  do not provide sufficient candidate coverage for reliable arbitrary-subset
  ordering in this scene. No further Q-head scaling or replication is run;
  any next primary design must address the information structure without
  importing oracle candidate labels or heuristic-dependent priors.

## 2026-08-28 - V209 Confirms V208 Candidate-Ranking Failure

- V209 performs a zero-update, frozen-policy replay of V208 on its six final
  windows. Before each of `2,304` policy actions, all currently feasible masks
  are evaluated over the same privileged eight-step diagnostic horizon. The
  resulting costs remain offline-only and never enter training or execution.
- V208 selects the local best candidate in only `1.04%` of states. Its mean
  selected rank is `9.68` among the feasible masks, with mean local regret
  `0.036870`; its mean selected-action probability is `0.07349` and entropy is
  `3.05362`. The return-head policy therefore does not learn useful conditional
  candidate ordering.
- Mean selected ranks are `9.33` in calm, `10.92` in particle, `8.22` in flux,
  and `10.84` in thermal states. The ranking failure is broad and is not
  confined to one event subtype. V210 now replays the same checkpoint with a
  Q-head rank audit to distinguish an unlearned Q representation from a failure
  to transfer Q scores into categorical PPO logits.

## 2026-08-28 - V208 Rejects On-Policy Action-Value Augmentation

- V208 adds an action-conditioned return head to the clean V199 actor. The
  head is trained only on the PPO rollout's executed `(state, feasible mask,
  discounted forecast return)` tuples and contributes detached scores to masked
  logits. It does not enumerate counterfactual candidate costs, use oracle
  action labels, or receive heuristic output.
- On frozen V193 seed 1901, the result fails both static endpoints: ordinary
  loss is `0.249197` versus `0.237045` (static-minus-PPO `-0.012152`) and
  macro loss is `0.735298` versus `0.712537` (margin `-0.022762`).
- Runtime feasibility has zero aborts and no always-on channel, but one channel
  is always off, only four are intermediate-duty, and switching drops to
  `0.013605` per step. The single-seed variant is closed without expansion.
- This rules out a lightweight selected-action return critic as a sufficient
  repair for sparse policy-induced action coverage. The primary path must not
  add further actor-side value heads without a new evidence-based design.

## 2026-08-28 - V207 Rejects Additive Quality-Context Pooling

- V207 tests one candidate-geometry correction on frozen V193 seed 1901. The
  quality-context actor now sums, rather than averages, the learned online
  utilities of sensors included in each feasible mask. This preserves the
  22-mask arbitrary-subset action space and changes no physical cost,
  forecast-loss reward, feasibility rule, online feature, or heuristic input.
- The hypothesis does not pass the single-seed gate. PD-PPO ordinary loss is
  `0.257584` versus validation-selected static `0.237045`
  (static-minus-PPO `-0.020539`), and macro loss is `0.713316` versus
  `0.712537` (margin `-0.000780`). It is therefore not expanded to more seeds.
- Runtime behavior is valid: zero warm-up aborts, zero always-on/off channels,
  six intermediate-duty channels, and switching `0.061514` per step. The
  failure is predictive rather than a collapse to a non-deployable allocation.
- The configurable pooling implementation and its unit test remain because
  sum pooling is a valid action-set primitive and the default `mean` preserves
  prior behavior. V207 closes the claim that mean pooling alone caused the
  candidate-value failure.

## 2026-08-28 - V206 Validation Checkpoint Selection Does Not Generalize

- V206 retains the frozen V205 architecture, physical six-channel V193 scene,
  22-mask action surface, forecast-loss reward, and `163,840` PPO-step budget.
  Its only change is standard model selection on the disjoint
  calibration/validation partition: every 10 updates, the behavior-valid
  checkpoint minimizing the worse of the ordinary/static and macro/static loss
  ratios is retained. Final windows are not consulted for checkpoint choice.
- On the frozen five-seed development set (`1901--1905`), V206 beats the
  validation-selected static schedule on both endpoints only `1/5` times
  (seed 1905). Mean static-minus-PPO margins are `-0.018041` for ordinary
  forecast loss and `-0.038951` for macro loss. Per-seed ordinary/macro margins
  are `-0.004085/-0.003153`, `-0.038683/-0.104069`,
  `-0.036786/-0.083576`, `-0.015415/-0.038473`, and
  `+0.004764/+0.034518`.
- The selected update varies across seeds (`40`, `140`, `80`, `80`, `90`), so
  the failure is not caused by always replaying the terminal checkpoint.
  All selected validation checkpoints satisfy the predeclared behavior check.
- Runtime feasibility remains good: all five rollouts have zero warm-up aborts
  and zero always-on channels; four have all six channels at intermediate duty,
  while seed 1901 leaves one channel always off. Mean switching is `0.050007`
  per step. Thus validation-only selection can preserve feasibility but does
  not repair cross-seed forecast-value ranking against the strong static mask.
- V206 is closed. It must not be expanded to more seeds or used as positive
  evidence. The next method decision will target the candidate-score geometry,
  whose current mean pooling can discard the additive value of jointly selected
  physical channels, while retaining the same online observations, forecast
  reward, and hard feasibility mask.

## 2026-08-28 - V205 Long-Horizon PPO Fails Cross-Seed Confirmation

- The V205 configuration was frozen after seed 1901 and evaluated unchanged on
  seeds `1902--1905`. Across all five development seeds it wins both static
  endpoints only `2/5` times, with mean static-minus-PPO margins
  `-0.033139` ordinary and `-0.065044` macro.
- Seeds 1902 and 1903 fail substantially (`-0.081126/-0.186885` and
  `-0.077927/-0.157607` ordinary/macro margins); seed 1904 also fails both
  endpoints. Seed 1905 passes narrowly (`+0.000388/+0.016973`). The apparent
  V205 seed-1901 success is therefore not reproducible and cannot support a
  main result.
- Runtime feasibility remains robust across all seeds: zero aborts and zero
  always-on/off channels. One seed has only five intermediate-duty channels,
  so the full intended six-channel behavior gate also fails in aggregate.
- Longer training is closed as a standalone remedy. V206 will retain the
  frozen configuration but use existing calibration/validation-only checkpoint
  selection with the `max_static_ratio` dual-endpoint score and valid-behavior
  requirement. This tests standard validation model selection, not a new
  reward, heuristic, or final-partition adaptation.

## 2026-08-28 - V205 Passes the Long-Horizon PPO Gate

- V205 restores V199's clean online quality/context actor and changes only
  standard PPO duration from `40,960` to `163,840` steps on frozen V193
  seed 1901. It retains the forecast-loss reward, 22 feasible masks, physical
  six-channel costs, and all deployment rules.
- The longer run passes both static endpoints: ordinary forecast loss is
  `0.235582` versus validation-selected static `0.237045`, and macro
  static-normalized loss is `0.693146` versus `0.712537`. It also improves on
  random (`0.237214`) and round-robin (`0.242842`) in ordinary loss. AoI
  (`0.220819`) and full-open (`0.199993`) remain stronger references on this
  single seed; the latter is infeasible under the power budget.
- Deployment behavior passes: zero warm-up aborts, zero always-on/off
  channels, six intermediate-duty channels, and switching `0.0683` per step.
  The policy uses all six physical channels under the fixed per-step costs.
- This is the first V193 PPO configuration to pass the dual static and
  behavior gate. Its architecture and hyperparameters are now frozen for
  confirmation on fresh development seeds `1902--1905`; no further tuning is
  permitted before that aggregate is inspected.

## 2026-08-27 - V203 Rejects Nonlinear Candidate Interaction as a Standalone Fix

- V203 retains V199's online quality/context action score, frozen V193
  seed-1901 scene, forecast-loss reward, 22 feasible-mask action geometry,
  and standard PPO configuration. It adds only a nonlinear state--candidate
  interaction head inside the masked actor; it receives no candidate cost,
  oracle label, heuristic output, or action prior.
- The variant preserves the required operational behavior: zero warm-up
  aborts, zero always-on channels, zero always-off channels, and six
  intermediate-duty channels. Its switching rate is `0.0465` per step.
- It nevertheless fails both static endpoints. Ordinary/macro losses are
  `0.254322/0.753630`, versus validation-selected static
  `0.237045/0.712537`. This is worse than V199 on both endpoints, so the
  nonlinear scorer is rejected without seed expansion.
- The next bounded step is a frozen policy-alignment audit of V203. It will
  determine whether the added interaction changes conditional feasible-mask
  ranking at all before any further standard-PPO training adjustment is made.

## 2026-08-28 - V204 Closes the Nonlinear-Scorer Diagnostic

- V204 reloads V203 with zero policy updates and replays the same six frozen
  final windows. It exactly reproduces V203's ordinary/macro loss
  `0.254322/0.753630` and its valid six-channel behavior.
- The interaction head does not improve conditional candidate selection. Its
  best-candidate match rate is `5.25%`, mean selected rank is `10.87/22`, and
  mean eight-step candidate regret is `0.033574`. V202's base V200 policy had
  `6.73%`, `10.28/22`, and `0.035654`, respectively; the small regret change
  is not a ranking improvement because V203 visits a different policy
  trajectory.
- The nonlinear interaction scorer is closed. The next single-factor test
  returns to V199's best clean quality/context representation and increases
  only standard PPO training duration from `40,960` to `163,840` steps. It
  retains the forecast-loss reward, online features, and hard feasibility mask.

## 2026-08-27 - V202 Localizes the Candidate-Value Credit Gap

- V202 reloads V200's checkpoint with `total_timesteps=0` and regenerates the
  same frozen V193 final replay. The policy reproduces V200 exactly:
  ordinary/macro losses `0.266148/0.741188`, static `0.237045/0.712537`, zero
  aborts, and six intermediate-duty channels.
- Before every executed PPO action, V202 computes all feasible eight-step
  candidate costs on the policy's own state trajectory. These costs are
  privileged offline diagnostics only and are not supplied to the actor,
  reward, action mask, or checkpoint.
- The frozen policy selects the locally best feasible candidate in only `6.73%`
  of 2,304 states. Its mean candidate rank is `10.28/22`, mean local regret is
  `0.035654`, and mean policy entropy is `2.7119`. This directly attributes
  the V200 failure to learned candidate-value/temporal-credit mismatch, not to
  missing dynamic headroom or an invalid six-channel rollout.
- The next clean architecture variant will strengthen the state-conditioned
  candidate scorer inside the masked actor. It will retain the forecast-loss
  reward, online observations, and feasibility mask, and will not use oracle
  candidate costs, bandit outputs, or counterfactual labels during training.

## 2026-08-27 - V201 Confirms Dynamic Headroom in the Frozen V193 Final Windows

- V201 is a diagnostic-only exact eight-step candidate replay on V200's six
  final evaluation windows. It uses the frozen V193 truth, evaluator, 22-mask
  geometry, and operational dwell rules; it does not train, select, or alter a
  policy.
- The validation-selected static action is action `5`. Across all 2,304
  replayed states, the privileged receding choice improves its local eight-step
  forecast cost by `0.020488` on average and is better in `95.57%` of states.
  The receding diagnostic uses all `22` feasible masks.
- The static-minus-receding headroom is positive in calm, particle, flux, and
  thermal states (`0.016720`, `0.025318`, `0.020812`, and `0.024288`,
  respectively). This verifies that V200's failure is not caused by a
  static-only scene after the physical six-channel correction.
- Exact candidate costs remain privileged offline diagnostics. The next audit
  will calculate the same rankings along the PPO policy's own trajectory to
  locate the residual action-value or temporal-credit mismatch without using
  those costs as runtime input or training labels.

## 2026-08-27 - V200 Low-Entropy Candidate-Aligned PPO Does Not Pass

- V200 combines V199's candidate-aligned online quality/context actor with the
  single standard-PPO entropy setting already tested in V196
  (`ent_coef=0.005`). The V193 seed-1901 truth, frozen forecaster,
  validation-selected static reference, reward, and feasibility rules remain
  unchanged; no heuristic output, execution-time event label, or
  counterfactual cost enters the policy.
- The configuration preserves the intended operational behavior: all six
  physical channels have intermediate duty, no channel is always on or off,
  switching is `0.0554` per step, and warm-up aborts are zero.
- It nevertheless degrades both forecast endpoints to ordinary/macro losses
  `0.266148/0.741188`, above static `0.237045/0.712537`. Lower entropy does
  not combine constructively with the candidate-aligned actor, so V200 fails
  without seed expansion.
- Entropy tuning is now closed for the candidate-aligned architecture. The
  next investigation must diagnose why the learned long-horizon forecast
  policy does not select the value-aligned masks supported by the same online
  observations, rather than adding another scalar hyperparameter variant.

## 2026-08-27 - V194/V195 Clean Context-Aware PPO Diagnostics Rejected

- V194 trained a full 40,960-step forecast-reward masked PPO on the frozen
  V193 seed-1901 scene with the online 20-feature alert context encoded in a
  dedicated branch. It used no bandit prior, residual action, imitation of the
  context policy, or counterfactual bandit label. The learned policy was
  feasible but lost to validation-selected static: ordinary/macro losses were
  `0.285028/0.798889` versus `0.237045/0.712537`.
- A feature-parity audit showed that the six online channel-quality values were
  already present in the shared observation but were outside V194's dedicated
  context branch. V195 expanded that branch from 20 to 26 trailing online
  context features, with all other training choices unchanged.
- V195 removed the residual behavior defect (`0` always-on, `0` always-off,
  and `6` intermediate-duty channels), but not the forecast failure. Its
  ordinary/macro losses were `0.286925/0.758300`, again worse than the frozen
  static reference.
- These two configurations are rejected without seed expansion. The next
  bounded diagnostic changes only the ordinary PPO entropy coefficient on the
  same frozen scene; it does not add a heuristic-dependent module or alter the
  forecast-loss objective.

## 2026-08-27 - V196 Lower-Entropy PPO Improves but Does Not Pass

- V196 repeats V195 on the identical frozen V193 seed-1901 scene with only
  the standard PPO entropy coefficient reduced from `0.02` to `0.005`.
- Lower entropy improved ordinary loss from `0.286925` to `0.254028`, but it
  still did not reach the validation-selected static reference (`0.237045`).
  Macro loss was `0.749825` versus `0.712537` for static.
- The rollout remained feasible with zero warm-up aborts, but one channel was
  always off. V196 therefore fails both the prediction and six-channel
  behavioral gates.
- Standard entropy tuning is closed for this frozen scene. The evidence now
  isolates a reward/temporal credit-assignment bottleneck: the same online
  quality/context state supports a strong explicit policy, while three clean
  forecast-reward PPO variants do not convert it into a better schedule.

## 2026-08-27 - V197 Training-Only Subtype Action Supervision Does Not Pass

- V197 adds only the framework's existing training-partition subtype-action
  cross-entropy auxiliary (`0.05`) to V195's feature-parity configuration.
  The final policy still sees only online alert, quality, and scheduler state;
  no simulator label is supplied during execution.
- The auxiliary reduced policy entropy to about `2.0` and improved ordinary
  loss to `0.247557`, but it remained above static `0.237045`; macro loss was
  `0.743873` versus `0.712537`.
- The more decisive subtype routing also made the behavior less acceptable:
  two channels were always off and switching fell to `0.0152` per step.
- This auxiliary encourages a subtype-static action target and does not encode
  online within-subtype channel-quality variation. It is rejected without
  tuning its weight or expanding seeds.

## 2026-08-27 - V198 Validation Checkpoint Selection Does Not Recover PPO

- V198 retained V195's clean feature-parity context policy and selected among
  every fifth update using only the independent validation partition and the
  existing behavior gate. No final-partition quantity entered selection.
- The selected update-10 checkpoint was already worse than the validation
  static schedule (ordinary ratio `1.1253`), and final replay remained worse:
  ordinary/macro losses `0.254751/0.769658` versus `0.237045/0.712537`.
- Its final behavior was invalid for the intended six-channel setting (`1`
  always-on, `3` always-off, and switching `0.0039` per step).
- Checkpoint selection therefore does not expose a hidden viable CA-PD-PPO
  policy. Plain context concatenation, entropy tuning, subtype-action
  supervision, and validation checkpoint selection are closed on V193.

## 2026-08-27 - V199 Candidate-Aligned Context Actor Restores Dynamic Behavior

- V199 enables a learned candidate-aligned actor term: online alert context is
  mapped to per-channel utilities, modulated by the corresponding online
  quality values, and pooled over each feasible mask before masked PPO action
  selection. It uses no baseline output, event label at execution, or
  counterfactual candidate cost.
- On frozen seed 1901, V199 improves the macro endpoint to `0.708799` versus
  `0.712537` for the validation-selected static schedule while retaining all
  six channels at intermediate duty, zero always-on/off channels, and zero
  warm-up aborts.
- Ordinary loss remains `0.247760` versus static `0.237045`; therefore V199
  is a partial pass, not a scene or policy confirmation. A single matched
  low-entropy combination is warranted because V196 improved ordinary loss
  under the same frozen scene.

## 2026-08-27 - V193 Removes Sensor-Dependent Event Generation

- Corrected the generic physical generator so that particle-event assignment is
  independent of Parsivel availability. Environmental events must not be caused
  by a channel's reporting state.
- On fresh seeds 1901--1905, the original warning-context policy improved on
  validation-selected static schedules in only `2/5` joint comparisons. Its
  mean ordinary/macro margins were `+0.007617/+0.021805`.
- Adding online channel-quality observations to the same replay-calibrated
  context policy produced `4/5` joint static wins at the prespecified mild
  quality penalty. The corresponding mean margins were
  `+0.008367/+0.034202`, with zero always-on channels and five intermediate
  channels on average.
- This passes the scene's bounded online-value gate. The 1,024-step scene-gate
  PPO is not interpreted as a learned-policy result; V194 reuses each frozen
  V193 truth, forecaster, and static selection for a full 40,960-step clean
  context-aware masked PPO diagnostic.

## 2026-08-26 - V137 Invalidated; V138 Repairs State Inputs

- Removed simulator-only subtype latent variables from the generic physical
  channel configuration and disabled subtype-weighted objectives and auxiliaries.
- V137 revealed that the shared scheduler/forecaster state list still appended
  the three latent columns, causing TCN loss saturation up to the clip value.
  V137 is invalid and contributes no scene evidence.
- Made state columns explicit run metadata and propagated them through training,
  strong-reference replay, and receding diagnostics. Legacy runs retain their
  original default state surface.
- V138 repeats the unchanged V137 scene gate with the corrected 12-column
  physical state. The complete test suite passes `125/125` before launch.

## 2026-08-26 - V136 Multi-Initialization Control Failed

- Added two independent policy initializations to each frozen V132 development
  scene and selected among three policies using validation score only.
- Selected policies achieved ordinary/macro/joint static wins of `4/5`, `3/5`,
  and `3/5`; conventional-dynamic wins were `5/5` and behavior passed `4/5`.
- Although every scene contained a joint-positive candidate, validation selected
  a poor held-out candidate in two scenes. Random restarts therefore expose but
  cannot reliably select the desired behavior.
- Closed multi-initialization selection and withheld fresh confirmation. The next
  development stage targets condition-dependent sensor information value in the
  simulator, without changing arbitrary-subset feasibility.

## 2026-08-26 - V135 Fresh Strong-Reference Replay

- Completed online warning-context and privileged one-step forecast-greedy
  replays on all 24 frozen confirmation scenes.
- Rejoined raw reference losses to the final V133/V134 policy metrics, avoiding
  stale placeholder-policy margin fields emitted by the baseline runner.
- PD-PPO beat the online warning-context policy in `15/24` ordinary and `17/24`
  macro comparisons, with positive mean margins of `+0.012670/+0.038143`.
- PD-PPO beat privileged forecast-greedy in `16/24` ordinary and `20/24` macro
  comparisons, with positive means of `+0.014895/+0.083304`; only the macro
  bootstrap interval remained clearly above zero.
- The result supports competitiveness with strong context-aware references, not
  stable two-endpoint dominance. No confirmation seed is used for further tuning.

## 2026-08-26 - V134 Fresh Behavior-Complexity Audit

- Audited all 24 frozen arbitrary-subset PD-PPO rollouts for fixed-mask,
  deterministic-cycle, action-diversity, and event/subtype-dependence behavior.
- No rollout was fixed-like or cyclic; 23/24 showed event- or subtype-dependent
  behavior and 22/24 passed the complete prespecified complexity gate.
- Recorded seed1407's slightly low mask entropy and seed1415's weak conditional
  dependence as frozen confirmation exceptions; neither seed is used for tuning.
- Added the seed-level audit and aggregate interpretation under
  `reports/aggregate/v134_fresh_behavior_complexity_20260826/`.

## 2026-08-22 - V61 Stratified Subtypes Failed the Online Gate

### Result

- Changed only subtype assignment from random to stratified on new seeds
  901--905, retaining V51 sensor physics, costs, budget, warnings, and noise.
- The physical online context policy passed the static joint gate in `3/5`
  seeds; the validation-selected all-subset mapping passed `2/5`. Both had
  negative mean ordinary and macro margins.

### Interpretation and Decision

- Subtype balance does not fix online learnability. At `B=1.25`, the feasible
  surface is dominated by two-channel subsets; switching to a specialist drops
  broad-variable continuity and can sharply worsen thermal forecasts.
- V62 reuses V61 truth and evaluator while increasing only the natural power
  and startup budgets to `1.65/2.0`. This admits three-channel subsets without
  a required core, cardinality rule, or full-open action. Screen the online gate
  before PPO.

## 2026-08-22 - V51 Online Context Gate Failed but Localized the Scene Defect

### Result

- Replayed two online-only context policies on frozen V51. Validation-selected
  context actions passed the static joint gate in `2/5` seeds. The prespecified
  physical context mapping improved this to `3/5`.
- Under the physical mapping, macro margins were positive in all five seeds,
  while ordinary margins failed only in seeds883 and 884. All six channels had
  intermediate duty on average and warm-up aborts were zero.

### Interpretation and Decision

- V51 contains online-responsive value, but random subtype composition makes
  ordinary-loss transfer unstable. It therefore fails the strengthened scene
  gate even though the privileged all-action upper passes.
- V61 keeps V51 physics, costs, warnings, and noise unchanged and alters only
  subtype assignment from random to stratified on new development seeds
  901--905. Generate and screen online context before any PPO training.

## 2026-08-22 - V60 BC-Only Diagnostic Rejected V51 Learnability Gate

### Result

- Evaluated the soft forecast-value warm start before any PPO update on all
  five V51 development seeds. Joint wins were `1/5` against static and `3/5`
  against dynamic references. Behavior passed in only `1/5`; most policies
  became effectively fixed with switching below `0.0014` per step.
- Mean ordinary/macro margins were `-0.006702/+0.035849` against static and
  `+0.010055/+0.028079` against conventional dynamic policies.

### Interpretation and Decision

- PPO is not solely responsible for the transfer failure. The privileged
  future-loss upper can identify dynamic actions, but the resulting mapping is
  not robustly recoverable from online state across the final partition.
- Future scene screens require two gates before PPO: an all-action privileged
  headroom gate and a deployable online-context policy gate. V51 is rejected if
  its existing context-alert replay does not pass the latter.

## 2026-08-22 - V59 Forecast-Gain Replication Failed

### Result

- Frozen 200,704-step V59 was replicated on all V51 development seeds. Joint
  wins were `1/5` against static and `3/5` against conventional dynamic
  references. Mean ordinary/macro margins were `-0.018184/-0.000430` against
  static and `-0.001427/-0.008200` against dynamic policies.
- Behavior passed in `4/5` seeds. Reward and critic scales remained stable, so
  numerical instability does not explain the failed transfer.

### Interpretation and Decision

- Reject longer forecast-gain exploration and stop this reward branch. The
  matched counterfactual is useful diagnostically but does not preserve the
  absolute forecast-loss ordering across seeds.
- Run a BC-only soft-value transfer diagnostic. It will determine whether PPO
  updates destroy an initially useful online mapping or whether the privileged
  V51 dynamic upper is intrinsically unavailable from online observations.

## 2026-08-22 - V59 Longer Forecast-Gain PPO Passed Seed881

### Result

- Extended the V57 no-BC forecast-gain policy from 40,960 to 200,704
  transitions without changing scene, reward, architecture, constraints, or
  evaluation.
- PD-PPO beat static by `+0.028736/+0.043429` ordinary/macro margin and AoI by
  `+0.000320/+0.004320`. All six channels had intermediate duty, no channel was
  always on or off, and warm-up aborts were zero.

### Interpretation and Decision

- With the exogenous forecast-difficulty component removed from the reward,
  longer on-policy exploration can learn the flexible 20-action surface. The
  dynamic-reference advantage is still small on this seed and requires
  replication.
- Freeze V59 exactly and run seeds882--885. Require at least 4/5 joint wins
  against both static and conventional dynamic references, positive mean
  margins on both endpoints, and at least 4/5 behavior passes.

## 2026-08-22 - V58 Soft Initialization plus Forecast Gain Failed

### Result

- Combined V56's soft forecast-value warm start with V57's forecast-gain PPO
  reward on the frozen seed881 screen.
- PD-PPO loss rose to `0.399985`, worse than static `0.381965` and AoI
  `0.353549`; the macro score was also worse. One channel became always off and
  only five channels retained intermediate duty.

### Interpretation and Decision

- The two training signals do not compose cleanly. The soft target optimizes
  absolute future cost ordering while the gain reward optimizes incremental
  measurement value, and PPO moves away from the useful initialization.
- Reject the combined design. Run one final no-BC forecast-gain duration test
  at 200k transitions on seed881; this distinguishes exploration time from
  objective mismatch without another architecture or scene change.

## 2026-08-22 - V57 Forecast-Gain Reward Fixed Credit Scale but Missed AoI

### Result

- Trained masked PPO from scratch with reward equal to forecast improvement
  over a same-epoch no-new-measurement counterfactual. No BC, action labels,
  baseline policy, or final-test information entered training.
- Rollout advantage standard deviation fell from roughly `9` under absolute
  loss to `0.38--0.69` early in training. On seed881, ordinary/macro margins
  were `+0.014006/+0.011919` against static but `-0.014410/-0.027190` against
  AoI.
- All six channels had intermediate duty, no channel was always on or off, and
  warm-up aborts were zero.

### Interpretation and Decision

- The matched counterfactual corrects reward scale and temporal attribution,
  but random exploration does not learn the 20-action surface within 40,960
  transitions.
- V58 will combine only the two forecast-derived components already tested:
  V56 soft forecast-value initialization and V57 forecast-gain PPO reward.
  The scene, constraints, policy architecture, and final evaluation remain
  frozen.

## 2026-08-22 - V56 Soft Forecast-Value Replication Failed

### Result

- Frozen V56 was replicated on all V51 development seeds. Joint wins were
  `2/5` against static and `3/5` against conventional dynamic references.
  Mean ordinary/macro margins were `-0.015741/-0.022948` against static and
  `+0.001016/-0.030718` against conventional dynamic policies.
- Behavior passed in `4/5` seeds with no always-on or always-off channels.
  Soft-target argmax fit rose to `0.631--0.829`, but seed883 and seed884 still
  had large negative prediction margins.

### Interpretation and Decision

- Soft action-value supervision improves label learnability and channel usage,
  but it does not repair the scalar PPO credit signal. Do not tune its
  temperature or pretraining duration.
- The next bounded method test will express the same forecast objective as
  information gain over a no-new-measurement counterfactual at the same epoch.
  This removes exogenous event difficulty from the reward while retaining
  online inputs, masked PPO, hard feasibility constraints, and absolute
  forecast-loss evaluation.

## 2026-08-22 - V56 Soft Forecast-Value Warm Start Passed Seed881

### Result

- Replaced the all-action teacher's noisy hard argmin target with a normalized
  soft distribution over every feasible action's eight-step forecast cost.
  The frozen V51 scene, online observations, constraints, PPO reward, and final
  evaluation remained unchanged.
- On the bounded seed881 screen, PD-PPO beat the strongest static reference by
  `+0.037001/+0.065688` ordinary/macro margin and the strongest conventional
  dynamic reference by `+0.008585/+0.026580`.
- All six channels had intermediate duty, no channel was always on or off,
  warm-up aborts were zero, and switching was `0.06079` per step.

### Interpretation and Decision

- Preserving the complete forecast-value ordering is more learnable than
  classifying a single future-dependent winner. This is a clean training-only
  representation of the existing prediction objective, not a new baseline or
  deployed input.
- Freeze V56 settings and replicate on the remaining V51 development seeds
  882--885. Require the existing 4/5 joint performance and behavior gates
  before declaring fresh confirmation seeds.

## 2026-08-22 - V55 Stronger All-Action BC Failed

### Result

- Kept the frozen V51 scene, evaluator, action geometry, all-action teacher,
  forecast-loss reward, and 40,960-step PPO protocol unchanged. The bounded
  test increased one-time BC fitting from 12 to 40 epochs and reduced the PPO
  entropy coefficient from `0.005` to `0.001`.
- Joint wins fell to `1/5` against both the strongest static reference and the
  strongest conventional dynamic reference. Mean ordinary/macro margins were
  `-0.032479/-0.024542` against static and `-0.015722/-0.032312` against
  conventional dynamic policies.
- The strict behavior gate passed in `3/5` seeds. Seed884 collapsed to one
  always-on and three always-off channels. BC accuracy improved to
  `0.240--0.299`, but this did not transfer to forecast performance; final
  entropy remained `1.711--2.273`.

### Interpretation and Decision

- Reject additional BC-epoch and entropy tuning. The privileged receding
  teacher ranks actions using future targets, and its 20-action choice is not
  reliably identifiable from the scheduler's online observation. Better label
  fitting therefore does not solve the sequential prediction-credit problem.
- Stop the all-action imitation learner line as predeclared. Preserve the V51
  scene and the central prediction-driven masked-PPO method. The next work unit
  will audit return attribution and action-value separation using online
  information before implementing any bounded structural learner change.

## 2026-08-22 - V54 All-Action Warm Start Improved Coverage but Failed

### Result

- Replaced the obsolete prototype action supervision with the framework's
  all-action, eight-step forecast-greedy teacher on the policy-training
  partition. The teacher was used only for BC initialization; continuing AWBC
  and action cross-entropy were disabled during PPO.
- Joint wins improved to `2/5` against static and `3/5` against conventional
  dynamic policies, but the gate still failed. Mean ordinary/macro margins were
  `-0.014789/+0.004939` against static and `+0.001968/-0.002831` against
  conventional dynamic policies.
- Behavior passed in `3/5` seeds and no channel was always on. The teacher used
  all 20 actions, but BC accuracy was only `0.105--0.170`; final policy entropy
  remained high at `2.102--2.551`.

### Interpretation and Decision

- Action coverage is repaired, but the actor has not learned a sharp observable
  state-to-action map from the one-time teacher data. Do not modify the frozen
  scene, reward, or teacher.
- V55 is one bounded ordinary-training test: increase BC fitting epochs on the
  same 2,000-label design (`12 -> 40`) and reduce entropy coefficient
  (`0.005 -> 0.001`). Require the original 4/5 performance and behavior gates;
  stop this learner line if the improvement does not transfer.

## 2026-08-22 - V53 Frozen V51 PD-PPO Transfer Failed

### Result

- Trained the corrected 20-feature PD-PPO architecture for forty complete
  1,024-step rollouts on each frozen V51 seed. Truth, evaluator, test starts,
  baseline selection, action geometry, costs, and scene parameters were copied
  from the V51 control assets.
- Joint wins were `1/5` against the strongest static reference and `2/5`
  against conventional dynamic policies. Mean ordinary/macro margins were
  `-0.012256/+0.050620` against static and `+0.004501/+0.042850` against
  conventional dynamic policies.
- Behavior also failed: several seeds left two or three channels unused, and
  seed884 had one always-on and three always-off channels. All runs had zero
  warm-up aborts.

### Interpretation and Decision

- The scene has a verified 20-action continuous upper, but V53's initialization
  and action auxiliary expose only three or four physical subtype prototypes.
  This supervision geometry is inconsistent with the flexible scheduling goal.
- V54 keeps the frozen V51 scene and complete PD-PPO method, removes prototype
  action cross-entropy, and uses the existing all-action forecast-greedy teacher
  only for BC initialization. Continuing AWBC remains disabled. This is a clean
  training-design correction, not a bandit-dependent module.

## 2026-08-22 - V51 All-Action Continuous Dynamic Gate Passed

### Diagnostic Correction

- Added a privileged receding-horizon structural diagnostic that evaluates all
  20 currently feasible masks at every epoch by snapshotting the complete
  environment, replaying each mask for eight steps, restoring constraint and
  sensor runtime state, and executing only the first action before replanning.
- The diagnostic preserves power, startup, warm-up, and six-step dwell rules.
  It uses future forecast loss and is therefore an upper diagnostic, not a
  deployable comparator or a source of final policy inputs.

### Result

- On the frozen V51 seeds `881--885`, all-action dynamic-over-static margins
  were `+0.056414`, `+0.037329`, `+0.018914`, `+0.070042`, and `+0.072553`.
  The result passes the strengthened gate with `5/5` wins and mean `+0.051050`.
- Every seed used all 20 actions, all six channels had intermediate duty, no
  channel was always on or always off, warm-up aborts were zero, and switching
  rates were `0.03899--0.04724` per step.

### Interpretation and Decision

- V51 contains strong continuous state-dependent scheduling value. The earlier
  `4/5` subtype-auto result was a limitation of the four-bucket, top-two-mask
  diagnostic, not evidence that V51 remained intrinsically static.
- Freeze V51 truth, evaluator, starts, action geometry, costs, constraints, and
  sensor-quality configuration. Train the corrected V48 PD-PPO architecture on
  the same assets before any fresh confirmation seeds are declared.

## 2026-08-22 - V52 Forecast-Horizon Lag Screen Rejected

### Result

- Retained V51's channels, costs, action geometry, state-dependent measurement
  quality, event process, evaluator protocol, and online context. Changed only
  specialist-latent target lag from four to eight steps on fresh development
  seeds `891--895`.
- Exact dynamic-over-static margins were `+0.004239`, `+0.005831`, `-0.004535`,
  `-0.015881`, and `-0.011356`; only `2/5` were positive and the mean was
  `-0.004340`.

### Interpretation and Decision

- Reject direct horizon alignment. Increasing the target lag reduces the value
  of the current measurement over much of the prediction window and does not
  produce stable dynamic scheduling value.
- Stop scalar generator tuning. Audit and replace the subtype-bucket upper
  diagnostic with an arbitrary-mask, continuous state-dependent oracle
  diagnostic before drawing another scene conclusion. The current diagnostic
  considers only the top two aggregate masks for calm and each of three event
  subtypes, which is narrower than the redesigned flexible scheduler.

## 2026-08-22 - V51 Specialist-Specific Observation Screen Improved but Failed

### Result

- Kept V42's six channels, 20 feasible subsets, fixed costs, budget, event
  process, target amplitudes, and frozen-TCN protocol. Reduced only the ability
  of a specialist sensor to substitute for a different event state's primary
  measurement; cross-state measurements remain available with finite noise.
- On fresh development seeds `881--885`, exact dynamic-over-static margins were
  `+0.036090`, `-0.006879`, `+0.002882`, `+0.019608`, and `+0.016084`.
  The mean improved to `+0.013557`, but the `4/5` win count failed the gate.

### Interpretation and Decision

- Specialist-specific observation quality is a productive physical mechanism,
  but it is insufficient by itself. No PPO policy is trained on V51.
- V52 retains V51 unchanged and tests one forecast-specific mechanism: align
  the specialist-latent target lag with the eight-step forecast horizon
  (`4 -> 8`) on new seeds `891--895`. This asks whether an early specialist
  observation has value across the complete prediction window. The exact upper
  gate remains `5/5` with mean margin at least `+0.02`.

## 2026-08-22 - V50 Persistence-Only Screen Rejected

### Result

- Restored the V42 specialist amplitudes and removed shared subtype proxies,
  then changed only specialist-latent persistence (`alpha 0.15 -> 0.08`) on
  fresh development seeds `871--875`; no PPO policy was trained.
- Exact-geometry privileged dynamic schedules beat hindsight static in `4/5`
  seeds. Per-seed ordinary margins were `+0.002771`, `+0.022561`, `+0.003055`,
  `-0.005636`, and `+0.012574`; the mean was `+0.007065`.

### Interpretation and Decision

- Reject persistence alone as the next scene. It does not reproduce V42's
  `5/5` upper-bound wins and its mean headroom is smaller than V42's
  `+0.010995`, so PPO training would test a learner near an even lower ceiling.
- Keep V48's complete 20-feature context encoder as the corrected method. The
  next scene hypothesis must change a physically interpretable online
  observability mechanism that affects the value and timeliness of specialist
  measurements. Raw amplitude, event duration, lookback, and latent alpha are
  closed as isolated tuning directions.

## 2026-08-22 - V49 Specialist-Amplitude Screen Rejected

### Result

- Increased only particle and flux specialist-specific latent amplitudes by
  1.5x on fresh development seeds `861--865`; no PPO policy was trained.
- Exact-geometry privileged dynamic schedules beat hindsight static in `2/5`
  seeds. Per-seed ordinary margins were `-0.014808`, `-0.078140`, `-0.023027`,
  `+0.101064`, and `+0.005382`; the mean was `-0.001906`.

### Interpretation and Decision

- Reject amplitude scaling. With `target_scales=null`, the frozen TCN correctly
  normalizes each target using fitting-partition standard deviations. Increasing
  target amplitude also increases its normalization scale and does not reliably
  increase specialist information value.
- V50 returns to V42 amplitudes and changes only specialist-latent persistence
  (`alpha 0.15 -> 0.08`) on new seeds `871--875`. This tests whether a current
  specialist observation remains informative across the eight-step forecast
  horizon. No PPO training occurs before the exact upper gate passes.

## 2026-08-22 - V48 Complete Context Encoding Improved Dynamic Baselines

### Result

- Assigned all 20 online alert-context features to the dedicated context branch
  and retained the corrected full-rollout V46 training protocol.
- PD-PPO jointly beat the strongest conventional dynamic reference in `5/5`
  seeds. Mean ordinary/macro margins were `+0.015078/+0.041426`.
- Against the strongest static reference, joint wins remained `2/5`; mean
  ordinary/macro margins were `-0.002139/+0.035182`. Behavior passed `5/5`,
  with zero always-on channels, one unused channel, five intermediate-duty
  channels, and zero warm-up aborts.

### Decision

- Retain complete context encoding as the corrected architecture. Do not switch
  to calibration-selected action labels, whose cross-partition transfer was
  previously unstable.
- The exact dynamic upper bound averages only `+0.010995` over static, leaving
  insufficient room for a learned online policy. Screen a new development-only
  scene that increases particle and flux specialist-specific target innovations
  by 1.5x while preserving thermal dynamics, online warnings, costs, budget,
  action geometry, and deployment constraints. No PPO training occurs before
  a stronger exact-geometry upper pass.

## 2026-08-22 - V47 Checkpoint Selection Failed; Context Split Audited

### Result

- Validation-only checkpoint selection every five full PPO updates produced
  `2/5` static joint wins and `3/5` conventional-dynamic joint wins. Mean
  ordinary/macro margins were `-0.012827/+0.000100` against static and
  `+0.004390/+0.006343` against conventional dynamic policies.
- All five behavior gates passed. Validation-selected updates varied from 10 to
  40, but early selection did not improve final-partition transfer.

### Architecture Finding

- The environment appends 20 online alert-context features: three scores, three
  threshold flags, a four-class argmax one-hot vector, maximum confidence,
  alert age, three trends, previous-specialist one-hot, and remaining dwell.
- The launcher configured the dedicated context encoder for only the last 10
  entries. Alert scores, flags, and argmax indicators were consequently left in
  the large history encoder instead of the context branch.

### Decision

- Reject checkpoint selection as the remedy. V48 returns to final full-rollout
  training and assigns the complete 20-dimensional online context tail to the
  context encoder. No privileged labels, bandit dependence, scene changes, or
  final-test selection are introduced.

## 2026-08-22 - V46 Full-Rollout Correction Did Not Pass

### Result

- Repeated V43 on the same frozen V42 assets after eliminating the 64-sample
  tail update. Every seed used forty complete 1,024-step PPO rollouts.
- Joint wins were `2/5` against both strongest static and conventional dynamic
  references. Mean ordinary/macro margins were `-0.001583/+0.014512` against
  static and `+0.015633/+0.020756` against conventional dynamic policies.
- Behavior passed `5/5`, with zero always-on channels, one unused channel, five
  intermediate-duty channels, and zero warm-up aborts.

### Decision

- Retain the full-rollout implementation correction, but do not treat V46 as a
  performance pass. It reduced the mean ordinary static deficit without fixing
  cross-seed endpoint instability.
- V47 adds only validation-partition checkpoint selection every five complete
  updates. Test metrics remain unavailable to selection, and all scene,
  evaluator, reward, action, and optimization settings remain fixed.

## 2026-08-22 - V45 Reward-Weight Flag Audit and PPO Tail Fix

### Result

- V45 disabled oracle-fitting subtype weights while restoring the complete V43
  policy configuration. Its training log and final metrics reproduced V43
  exactly. The flag is correctly recorded as disabled, but a reused frozen
  evaluator makes it irrelevant to the online scalar reward.
- Code audit found a separate training defect. A nominal 40k run with
  `n_steps=1024` performed 39 full updates followed by one 64-transition PPO
  update. In seed855, that short update reduced greedy action coverage from 20
  actions to 3 and entropy from `0.8732` to `0.4957`.

### Correction

- PPO now treats `total_timesteps` as a minimum sample budget and trains only on
  complete rollout batches. A 40k request therefore executes 40 full updates
  and 40,960 transitions. Added a unit test for the exact rollout schedule.
- V46 will rerun the V43 configuration on frozen V42 assets with this training
  correction. No scientific configuration parameter is changed.

## 2026-08-22 - V44 No-Action-CE Control Failed

### Result

- Removed only the event-action cross-entropy term while retaining the V43
  scene, frozen evaluator assets, behavior-cloning initialization, reward,
  action geometry, and 40k-step PPO protocol.
- Joint wins fell to `1/5` against both strongest static and conventional
  dynamic references. Mean ordinary/macro margins were
  `-0.038535/-0.063711` against static and `-0.021318/-0.057467` against
  conventional dynamic policies.
- Four of five runs passed the behavior gate. No channel was always on or off
  in any seed, but seed852 had only four intermediate-duty channels.

### Decision

- Reject removal of event-action cross-entropy. The physical guide is helpful
  but does not resolve the V43 endpoint tradeoff.
- The next matched control restores V43 supervision and disables subtype-
  dependent reward weights. This tests the observed mismatch between a
  subtype-weighted training objective and the ordinary forecast-loss endpoint;
  all scene, evaluator, action, cost, and PPO-budget settings remain frozen.

## 2026-08-22 - V43 Matched Policy Development Gate Failed

### Result

- Trained the frozen V36b policy configuration for 40k steps on the five V42
  truth/evaluator assets. Control-source hashes, final-test starts, action
  geometry, costs, reward, and scene parameters were held fixed.
- PD-PPO jointly beat the strongest static reference on both endpoints in `3/5`
  seeds and the strongest conventional dynamic reference in `3/5` seeds. Mean
  ordinary/macro margins were `-0.006348/+0.017032` against static and
  `+0.010868/+0.023276` against conventional dynamic policies.
- Operational behavior passed in `5/5` seeds. Every run had zero always-on
  channels, exactly one unused channel, five intermediate-duty channels, zero
  warm-up aborts, and switching rates from `0.0120` to `0.0269` per step.

### Decision

- The policy development gate requires at least `4/5` joint wins against both
  baseline families and therefore failed. No fresh confirmation seeds are
  authorized.
- Because the matched V42 privileged upper gate passed `5/5`, the remaining
  failure is learner/supervision transfer, not action feasibility or lack of
  dynamic scene value. Test one matched no-action-cross-entropy policy control
  on the same assets. Do not change the scene, reward, costs, or PPO budget.

## 2026-08-22 - V42 Exact-Geometry Scene Gate Passed

### Change

- Retained the unrestricted six-channel, power-feasible subset geometry and the
  fixed effective per-epoch sensor costs.
- Removed subtype-specific perturbations from shared humidity, wind, and air-
  temperature channels. Specialist-specific latent observations and forecast
  targets remain unchanged. This prevents low-cost shared channels from acting
  as substitute subtype sensors while preserving the physical channel mapping.
- Used fresh development seeds `851--855`, a 20-step history, and independently
  fitted frozen TCN evaluators. No PPO policy was selected or trained in this
  scene screen.

### Result

- Under the exact no-required-channel action geometry, the best privileged
  subtype-dependent dynamic schedule beat the best hindsight static subset in
  all `5/5` seeds.
- Ordinary forecast-loss margins were `+0.004466`, `+0.011498`, `+0.004429`,
  `+0.011042`, and `+0.023542`; the mean margin was `+0.010995`.
- The predeclared scene gate of `5/5` positive margins and mean margin at least
  `+0.01` passed. The V42 truth sequences, evaluators, starts, action geometry,
  and scene configuration are now frozen for matched policy training.

### Decision

- Train the previously selected clean V36b policy configuration on the same
  V42 assets. Do not alter the reward, action geometry, effective costs, scene,
  evaluator, or evaluation starts during this development replication.
- Fresh confirmation seeds remain prohibited until the learned policy passes
  the static, conventional-dynamic, behavior, and feasibility gates on at least
  four of these five development seeds.

## 2026-07-03 - CA-PD-PPO Bounded Dev2 Failure-Guided Wave

Objective: improve CA-PD-PPO against `context_alert_bandit_t0p5` without using
bandit-dependent patchwork. The primary method identity remains prediction-
driven masked PPO with hard feasibility masking.

### Diagnostics

- Completed failure analysis for the previous CA-PD-PPO dev run. Losses against
  the context-alert bandit concentrate in flux windows and in lower-confidence
  or alert-boundary regions, not in high-confidence alert regions.
- The analysis does not support bandit imitation: seed-level agreement with the
  bandit is negatively correlated with CA-PD-PPO macro margin.

### Clean Dev2 Variants

- Added bounded variants only: larger context encoder (`ctx128`), gated context
  fusion (`gated`), larger gated fusion (`gated_ctx128`), and longer PPO rollout
  (`nsteps2048`).
- Excluded from the main method: residual bandit actions, bandit-margin rewards,
  counterfactual bandit labels, bandit imitation losses, and bandit actor priors.

### Results

- `ctx128`: failed the fresh-final gate. Against `context_alert_bandit_t0p5`,
  macro wins were `14/24` and mean macro margin was `0.004083`; against
  `forecast_greedy_one_step`, macro wins were `23/24`.
- `gated`: failed the fresh-final gate. Against `context_alert_bandit_t0p5`,
  macro wins were `13/24` and mean macro margin was `0.002763`; against
  `forecast_greedy_one_step`, macro wins were `24/24`.
- `gated_ctx128`: failed the fresh-final gate. Against
  `context_alert_bandit_t0p5`, macro wins were `13/24`, mean macro margin was
  `0.006706`, and mean step margin was slightly negative; against
  `forecast_greedy_one_step`, macro wins were `24/24`.
- `nsteps2048`: failed the fresh-final gate. Against
  `context_alert_bandit_t0p5`, macro wins were `10/24`, mean macro margin was
  `0.002962`, and mean step margin was `-0.000719`; against
  `forecast_greedy_one_step`, macro wins were `24/24`.

### Decision

- No bounded dev2 variant passed the predeclared fresh-final gate. Fresh final
  seeds `301--324` were not launched.
- The strongest current interpretation is unchanged: CA-PD-PPO is a clean
  method-consistent improvement that remains competitive with the strong
  context-alert bandit and robustly beats forecast-greedy, but it does not yet
  support a stable-superiority claim over the bandit.

## 2026-06-20 - Strong-Claim Multiseed Extension Started

Objective: upgrade the met+specialist-pair result from a single-seed validated
candidate to paper-safe robustness evidence.

### Final Experiment Design

- Fixed scenario and method: use the seed45 metpair contract without further
  tuning drift:
  `configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml`,
  `budget=0.75`, `startup_peak_budget=0.95`, `max_active=2`,
  `truth_steps=70000`, event subtype latent lag `4`, context lead `8`,
  subtype-aware AWBC/auxiliary PPO, no candidate-prior KL, and router-confidence
  evaluation at `min_confidence=0.8`.
- Per-seed evidence must pass all three gates:
  learned PPO beats `validation_selected_static` under `eval_router_conf08`;
  strict subtype replay beats replay-local true fixed static with no duty guard;
  corrected behaviour audit rejects fixed-subset and simple-cycle explanations.
- Strong paper claim threshold:
  at least `10` complete seeds, at least `8/10` full seed-gate passes, positive
  mean learned margin, and positive mean strict-replay margin.
- Moderate fallback claim:
  at least `5` complete seeds with at least `80%` full seed-gate passes. Anything
  below that remains a replicated pilot or single-seed mechanism demonstration.
- Optional robustness after the main 10-seed result:
  nearby budgets `B=0.73` and `B=0.77` on a smaller seed subset, reported as
  sensitivity rather than the main claim.

### New Automation

- Added `scripts/run_v31_metpair_strongclaim_seed_sweep_20260620.sh`, which
  runs the fixed seed protocol plus standard eval, router-confidence eval,
  strict no-duty-guard replay, and behaviour-complexity audit.
- Added `scripts/72_v31_collect_metpair_strongclaim.py`, which aggregates per
  seed learned margins, strict replay margins, behaviour gates, and classifies
  the resulting claim strength.

### Launched

- Remote smoke collection on existing seed45 correctly reports
  `claim_strength=single_seed_only` with all three gates passing.
- Replication batch launched on `remote-gpu`:
  `metpair_s41` through `metpair_s44` on GPU2-GPU5, plus `metpair_s46` on
  GPU0 and `metpair_s47` on GPU1. Together with completed seed45, this will
  produce the first 7-seed evidence pool.

### Result

- Seven-seed collection finished:
  `reports/aggregate/metpair_strongclaim_7seed_20260620/`.
- Outcome: `complete_seeds=7`, `seed_gate_pass_count=1`,
  `learned_gate_pass_count=1`, `replay_gate_pass_count=3`,
  `behavior_gate_pass_count=2`, `claim_strength=not_supported`.
- Mean learned margin versus `validation_selected_static` was negative
  (`-0.022343`), so the old metpair branch is not a strong-claim candidate.
- Diagnosis: seed45 is a useful mechanism demonstration, but other seeds expose
  two shortcuts: static baselines can choose non-backbone pairs such as
  `shielded_thermo_hygro + laser_disdrometer`, and the learned PPO did not
  consistently infer subtype context strongly enough to follow the explicit
  dynamic teacher.
- New branch started: `v31_metpair_backbone_context_*`, which makes
  `met_station_core` a required backbone, exposes the generated subtype-alert
  context columns to the agent, balances subtype probabilities, and uses
  subtype-balanced transport-rich final-test windows.
- First backbone-context pilot result:
  `reports/aggregate/metpair_backbone_context_pilot_20260620/`.
  Seeds 41 and 42 both pass learned, replay, and behaviour gates:
  `complete_seeds=2`, `seed_gate_pass_count=2`,
  `mean_learned_margin_abs=0.020020`, `claim_strength=replicated_pilot`.
- Backbone-context seeds 43, 44, 45, 46, and 47 were launched next. If at least
  `4/5` complete context seeds pass, expand to the full 10-seed strong-claim
  run.
- Seven-seed backbone-context collection:
  `reports/aggregate/metpair_backbone_context_7seed_20260620/`.
  Result: `complete_seeds=7`, `seed_gate_pass_count=3`,
  `learned_gate_pass_count=5`, `replay_gate_pass_count=3`,
  `behavior_gate_pass_count=7`, `claim_strength=not_supported`.
- Diagnosis after backbone-context:
  the agent behaviour problem is largely fixed (`7/7` behaviour gates), but
  the simulator still permits fixed-static shortcuts in some seeds. In
  particular, explicit subtype replay fails or has too small a margin when the
  matching specialist does not materially improve future target loss.
- Strong-latent backbone-context branch launched on the failed seeds 43 and
  44:
  `v31_metpair_backbone_context_stronglatent_seed{43,44}_h075ctxsl_20260620`.
  It strengthens hidden subtype latents and specialist-dependent future target
  effects while keeping the met backbone and context-alert observation model.

## 2026-06-20 - Static-Shortcut Recalibration Closed On Met+Specialist Pair Scene

Objective: find and verify a PD-PPO scheduling scene that breaks the fixed-static
shortcut, run the necessary TCN-oracle gate and reduced PPO experiment, and
record enough evidence for paper-mainline decision making.

### Failed / Superseded Branches

- V25 low-budget static squeeze created structural headroom in one TCN gate, but
  split replay failed against replay-local raw/static references.
- V26 calm-selective scene also produced an apparent structural pass, but split
  replay again lost to replay-local raw static.
- V27 subtype-auto replay showed privileged dynamic headroom, but learned PPO
  variants did not clear the strict raw-static gate:
  subtype-aux PPO lost to replay-local raw static by `0.001088`, and strongBC2
  was only a single-seed near-threshold candidate.
- Context-power and decoy/fusion V31 variants did not provide paper-safe
  evidence. The best fusion/decoy run was close to source selected static
  (`custom_ppo=44.286885`, `validation_selected_static=44.288191`), but strict
  no-duty-guard replay found a much stronger true fixed static subset
  (`44.037335`). This branch is internal diagnostic evidence only.

### Framework / Scenario Changes

- Fixed duplicate-observation handling in `src/v2/env.py`: selected sensors that
  observe the same variable now fuse measurements by inverse noise variance,
  with circular fusion for `wind_direction_deg`, instead of overwriting in
  sensor-list order.
- Added strict no-duty-guard static replay support to
  `scripts/70_v31_split_replay_gate.py` so fixed static references are evaluated
  as true fixed masks rather than duty-guard rotations.
- Added `scripts/71_v31_explicit_replay_fast.py` for faster explicit subtype
  replay screening.
- Corrected `scripts/71_v31_behavior_complexity_audit.py` so state-dependent
  four-regime policies are not misclassified as fixed/simple merely because
  they use four masks or persist with period 1.
- Added `configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml`.
  The key contract is `met_station_core + one specialist` feasible under
  `budget=0.75`, while two specialists remain infeasible.

### Final Candidate

- Run: `reports/v31_metpair_stronglatent_seed45_h075_20260620` on `remote-gpu`.
- TCN-oracle gate artifact: `v2_tcn_oracle.pt` exists in the run directory.
- Reduced PPO artifact: `custom_ppo.pt` exists; training log reached
  `120000` timesteps.
- Split protocol: `truth_steps=70000`, split ratios `[0.35, 0.50, 0.075, 0.075]`,
  final-test event-rich windows with mean event rate `0.768799`.

### Forecast Gate Evidence

- Source learned PPO:
  `custom_ppo=0.487083`,
  `validation_selected_static=0.491597`,
  `full_open_unconstrained=0.508449`,
  `feasible_static_projected=0.511365`,
  `aoi=0.525733`,
  `round_robin=0.530911`,
  `random=0.559486`.
- Router-confidence re-evaluation (`eval_router_conf08`) improved learned PPO:
  `custom_ppo=0.485635` vs `validation_selected_static=0.491597`, absolute
  margin `0.005962`, relative margin about `1.21%`.
- Strict explicit replay with no static duty guard passed:
  best dynamic `split_metpair_subtype_explicit_l4=0.482174`,
  replay-local true fixed static `static_action10=0.492351`,
  absolute margin `0.010177`, relative margin about `2.07%`,
  `gate_pass=true`.

### Behaviour Gate Evidence

- Correct audit file: `behavior_audit_v2/behavior_complexity_summary.json`.
  The older `eval_router_conf08/behavior_audit` file used the pre-fix audit
  logic and should not be used for final behaviour claims.
- Best learned policy under `eval_router_conf08`:
  `unique_mask_count=4`,
  `top1_mask_fraction=0.412354`,
  `top3_mask_fraction=0.913574`,
  `mask_entropy_bits=1.806220`,
  `transition_entropy_bits=1.998642`,
  `event_sensor_l1=1.579090`,
  `event_mask_mi_bits=0.520959`,
  `state_dependent=true`,
  `fixed_like=false`,
  `simple_cycle_like=false`,
  `behavior_complexity_gate_pass=true`.
- Learned deployment pattern:
  `met_station_core` is always on as the meteorological backbone; the second
  slot switches by context among `laser_disdrometer`, `fc4_flux`,
  `shielded_thermo_hygro`, and `surface_temp_ir`. This is a
  state-dependent specialist scheduler, not a fixed subset or simple cycle.

### Decision

- Migrate the met+specialist-pair scene to the PD-PPO paper mainline as the
  current validated candidate.
- Present the method as forecast-oriented contextual specialist scheduling with
  a stable meteorological backbone.
- Keep V20+/V25/V26/V27/context-decoy failures as diagnostics or appendix
  material, not as the main result.
- Remaining robustness work before final submission: reproduce the candidate
  on additional seeds or nearby budgets if time permits. This is a robustness
  extension, not a blocker for the 2026-06-20 recalibration objective.

## 2026-06-20 Macro-Subtype Evidence Audit

- Added event-subtype macro loss reporting to
  `scripts/70_v31_split_replay_gate.py`. Replay/static tables and summary JSON
  now include `oracle_loss_macro_subtype_event`, computed as the unweighted
  mean over particle, flux, and thermal event subtype losses.
- Extended `scripts/72_v31_collect_metpair_strongclaim.py` to backfill macro
  losses from saved rollout NPZ files and replay/static CSVs, so existing runs
  can be re-aggregated without retraining.
- Backbone-context 7-seed aggregate:
  `reports/aggregate/metpair_backbone_context_7seed_macro_20260620/`.
  Strict claim remains unsupported (`seed_gate_pass_count=3/7`), but
  event-subtype macro evidence improves (`macro_seed_positive_count=5/7`).
- Strong-latent probes:
  `reports/aggregate/metpair_backbone_context_stronglatent_2seed_macro_20260620/`.
  Seeds 43 and 44 are both macro-positive, but neither passes the strict
  step-weighted 1% replay gate.
- Paper implication: the current evidence does not support a broad
  step-weighted "PD-PPO always beats static" claim. The viable direction is a
  narrower regime-macro robustness claim, pending a larger multi-seed
  strong-latent run.

## 2026-06-20 Ortholinear / Strong-Teacher Follow-Up

- Treat `remote-gpu` as the only valid remote GPU target. Older internal-address
  or tunnel-based connection notes are obsolete for current experiments.
- Strong-latent continuation failed as a final-claim branch:
  `reports/aggregate/metpair_backbone_context_stronglatent_partial4_macro_20260620/`
  has `seed_gate_pass_count=0/4` and `macro_seed_positive_count=2/4`.
- Added an orthogonal-linear event generator branch:
  `event_subtype_flux_latent_linear_scale`, `offset`, and `clip`. This replaces
  unstable exponential flux-latent amplification with a bounded linear term
  while reducing thermal shortcut strength.
- Ortholinear seed41 fixed the structural replay problem:
  explicit dynamic replay `split_metpair_subtype_explicit_l10=5.142764` beats
  replay-local static `static_action5=5.212586`; behaviour also passes.
- Learned deployment audit:
  raw PPO is step-weighted positive (`4.956431` vs selected static `5.233835`),
  but router-confidence deployment is negative (`5.330788`). Learned macro is
  still negative because the raw policy underperforms static on the flux
  subtype.
- Exposed teacher/curriculum controls in
  `scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh`:
  AWBC strength, BC pretrain length, entropy coefficient, subtype lookahead,
  dwell, and switch penalty.
- Added
  `scripts/run_v31_metpair_backbone_context_ortholinear_strongteacher_seed_sweep_20260620.sh`
  to test whether stronger subtype imitation and aligned lookahead can convert
  ortholinear structural headroom into a learned macro-positive PD-PPO policy.

## 2026-06-20 Remote Hygiene and Strong-Teacher 10-Seed Expansion

- Cleaned local operational instructions and scripts so `remote-gpu` is the
  only valid remote GPU entry point. Removed stale hardcoded remote-host paths
  from active local context, the smoke-result fetch script, and the local
  microclimate experiment skill. Removed password-based sync helpers from the
  fetch script.
- Strong-teacher 3-seed aggregate is now available:
  `reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_3seed_macro_20260620/`
  and raw counterpart
  `reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_3seed_raw_macro_20260620/`.
  Strict step-weighted gate, learned gate, replay gate, and behaviour gate are
  all `3/3`.
- Macro-subtype robustness is not supported yet: macro-positive full seeds are
  only `1/3`.
- Launched strong-teacher extension seeds `44--50` on `remote-gpu` across GPUs
  `0/1/2/4/5`. The final claim audit should aggregate seeds `41--50` and judge
  the strong claim on the strict step-weighted static/replay/behaviour gate.

## 2026-06-20 Static-Normalized Macro and Reward-Aligned Balanced Objective

- Re-audited the previously reported strong-teacher evidence. The 14-seed pool
  does not support a strong main-paper claim: behaviour is solved (`14/14`),
  but strict step-weighted and raw macro gates remain below the required
  robustness threshold.
- Diagnosed the remaining failure as objective-scale dominance. Feasible fixed
  static candidates frequently choose `met_station_core + fc4_flux`; flux loss
  is much larger numerically than particle and thermal losses, so raw aggregate
  objectives can reward a static flux shortcut even when the dynamic specialist
  scheduler is better balanced across regimes.
- Added `oracle_loss_macro_subtype_event_staticnorm`: subtype losses are divided
  by the median feasible fixed-static subtype loss before averaging particle,
  flux, and thermal regimes. PPO metrics, replay gates, and the multiseed
  collector can now use this score.
- Added `--reward-loss-normalization staticnorm_subtype` to PPO training. The
  normalizers are computed only from validation static candidates and are stored
  in `v2_ppo_metadata.json`, so the training reward is aligned with the
  static-normalized macro gate without using final-test data.
- Added and queued the balanced-objective runner
  `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh`.
  First target seeds are `41` and `42`; expansion depends on learned PPO,
  strict replay, and behaviour gates under the static-normalized macro contract.
- Completed the posthoc static-normalized replay/collector audit for the
  ortholinear strong-teacher 14-seed pool:
  `reports/aggregate/metpair_ortholinear_strongteacher_14seed_staticnorm_replay_20260620/`.
  Result: `complete_seeds=14`, strict step `seed_gate_pass_count=10`,
  `behavior_gate_pass_count=14`, `macro_seed_gate_count=13`,
  `one_sided_sign_test_p_macro_seed_gate=0.00091552734375`, and
  `macro_claim_strength=strong_macro_multiseed`.
- The only static-normalized macro failure is seed `48`; strict step gate also
  fails on seeds `44`, `46`, `48`, and `52`. The paper claim should therefore
  be written as a static-normalized event-regime macro claim, not as broad
  step-weighted forecast optimality.

## 2026-06-20 Paper Mainline Rewrite for Static-Normalized Macro Claim

- Archived the previous paper source/PDF before rewriting:
  `paper/_archive/pre_staticnorm_macro_rewrite_20260620_232507/`.
- Rewrote the canonical manuscript entry point `paper/main.tex` and main
  sections to remove the stale eight-channel, ten-seed, dynamic-baseline
  narrative.
- Replaced the main claim with the supported backbone-plus-specialist result:
  PD-PPO improves static-normalized event-regime macro forecast loss in `13/14`
  seeds, passes behaviour complexity in `14/14` seeds, and has one-sided
  macro-gate sign-test `p=0.00091552734375`.
- Added `paper/tables/metpair_staticnorm_macro_summary.tex` and replaced
  `paper/tables/sensor_specs.tex` with the six-channel met+specialist contract.
- The rewritten paper explicitly states the limitation that the strict
  step-weighted fixed-static gate passes only `10/14` seeds, so broad
  average-loss dominance over true fixed static is not claimed.
- Local and `remote-gpu` paper builds both complete with `latexmk`; the only
  remaining LaTeX issue is a minor `1.79993pt` overfull hbox warning.
### 2026-06-30 17:34:12 UTC | session `20260625_020` | model `gpt-5.5` | interrupted
**Tools:** shell, write
**Files:**
  - `<framework-root>/docs/07-01-01-LEMMA.md`
Commands:
  - `stat -c '%n %s bytes %y' docs/07-01-01-LEMMA.md && sha256sum docs/07-01-01-LEMMA.md`

### 2026-06-30 17:48:08 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/findings.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/progress.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/task_plan.md`
  - `<framework-root>/paper/tables/pdppo_training_hyperparameters.tex`
Commands:
  - `date '+%Y%m%d_%H%M%S %Y-%m-%d %H:%M:%S %Z'; git status --short`
  - `set -euo pipefail
stamp=$(date '+%Y%m%d_%H%M%S')
mkdir -p paper/backups paper_archives
pdf_backup="paper/backups/main_before_lemma0701_${stamp}.pdf"
archive="paper_archives/paper_before_lemma0701_${st…`
  - `git diff --check -- main.tex sections/01_introduction.tex sections/03_problem_formulation.tex sections/04_framework_protocol.tex sections/05_simulation_setup.tex sections/06_results.tex sections/07_di…`
  - … and 30 more

### 2026-06-30 18:13:39 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell
**Files:**
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/findings.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/progress.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/task_plan.md`
  - `<framework-root>/paper/sections/appendix_theory.tex`
Commands:
  - `python - <<'PY'
from pathlib import Path
import subprocess,re
base=Path('<framework-root>/paper')
text=subprocess.check_output(['pdftotext','main.pdf'…`
  - `set -euo pipefail
stamp=$(date '+%Y%m%d_%H%M%S')
mkdir -p backups ../paper_archives
pdf_backup="backups/main_before_nfix_${stamp}.pdf"
archive="../paper_archives/paper_before_nfix_${stamp}.tar.gz"
cp …`
  - `git diff --check -- main.tex sections/03_problem_formulation.tex sections/04_framework_protocol.tex sections/05_simulation_setup.tex sections/appendix_theory.tex tables/regime_balanced_24seed_summary.…`
  - … and 54 more

### 2026-06-30 18:33:23 UTC | session `20260701_022` | model `gpt-5.5` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/.planning/2026-07-01-supervisor-ppt-easy/findings.md`
  - `<framework-root>/.planning/2026-07-01-supervisor-ppt-easy/progress.md`
  - `<framework-root>/.planning/2026-07-01-supervisor-ppt-easy/task_plan.md`
  - `<framework-root>/reports/supervisor_update_20260701/build_supervisor_ppt_20260701.py`
Commands:
  - `pdftotext supervisor_report_20260629.pdf - | sed -n '1,220p'`
  - `python - <<'PY'
from pathlib import Path
base=Path('<framework-root>')
for d in [base/'paper/figures', base/'reports/supervisor_update_20260629', base…`
  - `python - <<'PY'
from PIL import Image
from pathlib import Path
base=Path('<framework-root>/reports/supervisor_update_20260701/figures')
for p in sorte…`
  - … and 41 more

### 2026-07-01 16:07:03 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell
**Files:**
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/progress.md`
  - `<framework-root>/paper/references.bib`
Commands:
  - `git status --short`
  - `cp references.bib backups/references_before_refaudit_$(date '+%Y%m%d_%H%M%S').bib
ls -t backups/references_before_refaudit_*.bib | head -1 | xargs -I{} sh -c 'stat -c "%n %s bytes %y" "{}"; sha256sum …`
  - `python - <<'PY'
from pathlib import Path
import re, sys
bib=Path('references.bib').read_text()
keys=re.findall(r'^@\w+\s*\{\s*([^,]+),', bib, flags=re.M)
from collections import Counter
dups=[k for k,…`
  - … and 12 more

### 2026-07-01 19:35:09 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/audit_extract.py`
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/findings.md`
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/progress.md`
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/task_plan.md`
  - `<framework-root>/paper/figures/data_split_timeline_tikz.tex`
  - `<framework-root>/paper/figures/gen_fig_event_type_diagnostics.py`
  - `<framework-root>/paper/figures/gen_fig_framework_and_support.py`
  - `<framework-root>/paper/figures/gen_fig_scenebal2_diagnostics.py`
  - `<framework-root>/paper/figures/gen_fig_scenebal_evidence.py`
  - `<framework-root>/paper/figures/gen_fig_training_diagnostics.py`
  - … and 11 more
Commands:
  - `pwd; git status --short; stat -c '%n %s bytes %y' paper/main.pdf 2>/dev/null || true; sha256sum paper/main.pdf 2>/dev/null || true; pdfinfo paper/main.pdf | awk '/^Pages:/ {print}' 2>/dev/null || true`
  - `pdftotext -f 1 -l 1 main.pdf -`
  - `pdftotext -f 2 -l 2 main.pdf -`
  - … and 313 more

### 2026-07-01 20:17:14 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** shell, write
**Files:**
  - `<framework-root>/.planning/2026-07-02-reference-audit/findings.md`
  - `<framework-root>/.planning/2026-07-02-reference-audit/progress.md`
  - `<framework-root>/.planning/2026-07-02-reference-audit/task_plan.md`
  - `<framework-root>/docs/07-02-02-reference-audit.md`
Commands:
  - `pwd && git status --short && date '+%Y-%m-%d %H:%M:%S %Z'`
  - `python - <<'PY'
from pathlib import Path
import re, json
root=Path('<framework-root>')
paper=root/'paper'
plan=root/'.planning/2026-07-02-reference-au…`
  - `python - <<'PY'
from pathlib import Path
import json, urllib.parse, urllib.request, time, re
root=Path('<framework-root>')
plan=root/'.planning/2026-0…`
  - … and 49 more
### 2026-07-08 19:28:02 UTC | session `20260625_020` | model `deepseek-v4-pro` | completed
**Tools:** shell, write
**Files:**
  - `<framework-root>/docs/07-07-01-HERMES.md`
Commands:
  - `# Backup the original
cp <framework-root>/docs/07-07-01-HERMES.md <framework-root>/docs/backups/07-0…`
  - `mkdir -p <framework-root>/docs/backups && cp <framework-root>/docs/07-07-01-HERMES.md <user-home>/_c…`
  - `echo "=== ORIGINAL (backup) ===" && wc -l -c <framework-root>/docs/backups/07-07-01-HERMES_orig_20260709_032649.md && echo "=== REFINED ===" && wc -l …`
  - … and 2 more

### 2026-07-19 18:06:36 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_072001_wording/task_plan.md`
Commands:
  - `set -euo pipefail
paper=<framework-root>/paper
printf '%s\n' '=== codex ==='
ps -eo pid=,ppid=,stat=,etime=,time=,cmd= | grep -E '(codex --yolo resume…`
  - `set -euo pipefail
fw=<framework-root>
paper="$fw/paper"
plan=<temporary-workspace>/pdppo_072001_wording
files=("$paper/main.tex" "$paper/sections/01_i…`
  - `set -euo pipefail
fw=<framework-root>
paper="$fw/paper"
pkg="$fw/submission/eswa_pdppo_20260719"
plan=<temporary-workspace>/pdppo_072001_wording
stamp…`
  - … and 24 more

### 2026-07-20 17:02:32 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
Commands:
  - `set -euo pipefail
python - <<'PY'
from pathlib import Path
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmd=(proc/'cmdline').read_bytes().repl…`
  - `set -euo pipefail
conda run -n darts python scripts/95_v31_build_clean_paper_assets.py --framework-root .
python -m py_compile scripts/95_v31_build_clean_paper_assets.py`
  - `set -euo pipefail
python - <<'PY'
from pathlib import Path
import subprocess,json
script=Path('<hermes-home>/skills/writing/technical-manuscript-editing/scripts/compare_latex_invariants.py')
bef…`
  - … and 21 more

### 2026-07-20 17:26:59 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
Commands:
  - `set -euo pipefail
root=<framework-root>; paper="$root/paper"; pkg="$root/submission/eswa_pdppo_20260719"; plan=<temporary-workspace>/pdppo_072002_corr…`
  - `set -euo pipefail
root=<framework-root>; paper="$root/paper"; pkg="$root/submission/eswa_pdppo_20260719"; plan=<temporary-workspace>/pdppo_072002_corr…`
  - `set -euo pipefail
python - <<'PY'
from pathlib import Path
import subprocess,json
script=Path('<hermes-home>/skills/writing/technical-manuscript-editing/scripts/compare_latex_invariants.py'); be…`
  - … and 9 more

### 2026-07-23 20:04:15 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_drift_check_final_preintegration.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_ignored_png_integration.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_integration_hashes.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/ci_recomputation_hermes.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_roundtrip_verification.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_sync_report.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/zip_rebuild_report.json`
Commands:
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python -c 'from pathlib import Path
import pandas as pd, numpy as np, json
p=Path("reports/aggregate/pdppo_clean_validat…`
  - `python -c 'from pathlib import Path
p=Path("<hermes-home>/cache/delegation/live/deleg_f4750d7d/task-1.log")
s=p.read_text().splitlines()[101]
msg=s.split("assistant|",1)[1]
print(msg.replace(" -…`
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python -c 'from pathlib import Path
import re,ast,subprocess
s=Path("paper/main.tex").read_text(); a=s.split("\\begin{ab…`
  - … and 27 more

### 2026-07-23 20:41:15 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/paper_archives/plain_language_final_20260724_043256.sha256`
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_final_verification.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_roundtrip_verification_final.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/upload_artifact_filelist.txt`
Commands:
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python scripts/95_v31_build_clean_paper_assets.py`
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python scripts/95_v31_build_clean_paper_assets.py`
  - `set -euo pipefail
for f in main.tex anonymous_manuscript.tex supplementary_material.tex title_page.tex; do
  echo "== $f =="
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$f"
done`
  - … and 49 more

### 2026-07-23 21:19:54 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/paper_archives/plain_language_final_20260724_051549.sha256`
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_final_verification.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/late_async_audit_mapping.md`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_roundtrip_verification_final.json`
Commands:
  - `set -euo pipefail
command -v drawio || command -v draw.io || command -v diagrams-net || true
printf 'DISPLAY=%s\n' "${DISPLAY:-}"`
  - `drawio --help | sed -n '1,160p'`
  - `pdfinfo figure_pdppo_framework_drawio.pdf | sed -n '/Page size/p'; identify -format '%w x %h\n' figure_pdppo_framework_drawio.png`
  - … and 127 more

### 2026-07-28 20:57:01 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/scripts/96_v31_analyze_warning_information.py`
Commands:
  - `python - <<'PY'
import csv, hashlib, json, pathlib
root=pathlib.Path('<framework-root>')
rows=[]
for seed in range(117,141):
    meta_path=root/f'repr…`
  - `python - <<'PY'
import pandas as pd
p='<framework-root>/reproducibility/pdppo_eswa_evidence_20260718/aggregates/pdppo_clean_validation_frozen_24seed_2…`
  - `python - <<'PY'
import pandas as pd
d=pd.read_csv('<framework-root>/reports/aggregate/pdppo_framework_baselines_clean_24seed_20260718/framework_baseli…`
  - … and 10 more

### 2026-08-02 13:44:53 UTC | session `20260802_213` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/docs/PD-PPO-TERMINOLOGY.md`
  - `<temporary-workspace>/pdppo_terminology_contract/findings.md`
  - `<temporary-workspace>/pdppo_terminology_contract/glossary_validation.json`
  - `<temporary-workspace>/pdppo_terminology_contract/progress.md`
  - `<temporary-workspace>/pdppo_terminology_contract/task_plan.md`
Commands:
  - `git status --short && printf '\nPAPER\n' && git -C paper status --short && printf '\nBRANCH\n' && git branch --show-current && git -C paper branch --show-current`
  - `set -euo pipefail
if command -v markdownlint >/dev/null 2>&1; then markdownlint docs/PD-PPO-TERMINOLOGY.md; else echo 'markdownlint: unavailable (structural Python audit used)'; fi
python -m py_compil…`
  - `set -euo pipefail
if command -v codespell >/dev/null 2>&1; then codespell docs/PD-PPO-TERMINOLOGY.md; else echo 'codespell: unavailable'; fi
set +e
git diff --no-index --check /dev/null docs/PD-PPO-TE…`

### 2026-08-02 19:06:31 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/paper_archives/eswa_pdppo_main65_upload_final_20260803_030029.zip.sha256`
  - `<framework-root>/paper_archives/paper_phase9_final_20260803_030029.tar.gz.sha256`
Commands:
  - `set -euo pipefail
for f in main.pdf anonymous_manuscript.pdf supplementary_material.pdf title_page.pdf; do echo "--- $f"; pdfinfo "$f" | grep -E '^(Title|Subject|Keywords|Author|Creator|Producer|Creat…`
  - `set -euo pipefail
conda run -n darts pytest -q -o addopts='' tests | tee <temporary-workspace>/pdppo_full_refinement_20260802/phase9_pytest_explicit_final.log`
  - `set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=<temporary-workspace>/pdppo_full_refinement_20260802/backups/phase9_pre_package_final_${STAMP}.tar.gz
mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT…`
  - … and 33 more

### 2026-08-22 | Flexible-subset PD-PPO development pilot

- Replaced the mandatory-core plus one-specialist geometry in an independent
  experiment track with 29 power-feasible subsets over six physical-system channels.
- Seed 401 at 30k steps improved over the selected static schedule on macro
  normalized loss (`0.656944 < 0.723909`) but lost on mean forecast loss
  (`0.171239 > 0.161443`) and remained slightly behind AoI.
- Operational feasibility passed with zero violations and zero warm-up aborts,
  but behaviour failed the diversity gate: one channel was always active, two
  were always inactive, and only three masks were executed.
- Diagnosis: the validation teacher selected the laser mask for both particle
  and flux conditions. The run was stopped at one development seed; v2 targets
  this measured identifiability and cost-geometry failure before further scaling.

### 2026-08-22 | Flexible-subset v2 bounded pilot

- Seed 402 passed both performance endpoints against the validation-selected
  static schedule: mean loss `0.165710 < 0.197650` and macro normalized loss
  `0.786622 < 0.996109`.
- It also beat AoI, round-robin, and random on both endpoints, with zero power
  violations and zero warm-up aborts.
- Behaviour improved from three to eleven executed masks. No channel was always
  active, five channels had intermediate duty, and only FC4 remained effectively
  inactive. The frozen v2 configuration advanced to seeds 403 and 404.

### 2026-08-22 | Flexible-subset v2 30k replication

- Seeds 403 and 404 did not reproduce seed 402's two-endpoint advantage. Across
  the three development seeds, PD-PPO beat the validation-selected static subset
  on both mean and macro forecast loss in `1/3` seeds.
- Seed 403 lost to the selected static subset by `0.008999` in mean loss and
  `0.060160` in macro loss. Seed 404 lost mean loss by `0.005471` while retaining
  a small macro advantage of `0.004232`.
- All three runs remained feasible with no always-on channel, but seed 403 used
  only four subsets and switched at `0.007237` per step. The 30k configuration
  failed the replication gate and was not expanded to confirmation seeds.
- A bounded 100k-step rerun of seed 403 was launched with the v2 scene and all
  other settings fixed to isolate training duration from scene calibration.

### 2026-08-22 | Flexible-subset v2 unmatched 100k diagnostic

- The seed-403 100k policy still used only four subsets, left FC4 inactive, and
  lost to selected static by `0.008091` in mean loss and `0.031861` in macro loss.
- Audit found that the longer run regenerated the frozen forecaster and
  validation candidate scores. Its final start indices were unchanged, but its
  selected static mask and all comparator losses differed from the 30k run.
- The launcher now exposes the existing control-source validation path so that
  training-duration comparisons can reuse byte-identical truth, evaluator,
  action surface, validation selection, and final windows. This unmatched run
  is retained as a collapse diagnostic, not a causal duration comparison.

### 2026-08-22 | Flexible-subset v2 matched 100k diagnostic

- A corrected seed-403 duration comparison reused byte-identical truth and
  frozen-forecaster assets. All six non-PD-PPO comparator rows matched the 30k
  run exactly, confirming the matched-control path.
- At 100k steps, PD-PPO lost to selected static by `0.019041` in mean loss and
  `0.088967` in macro loss. It used seven masks but never activated the laser.
  Longer training therefore does not resolve the failure.
- The next bounded variant keeps the scene and full feasible action surface but
  applies validation-static subtype normalization to the forecast-loss reward
  and uses physically specified subtype prototypes for auxiliary supervision.
  The prototypes do not constrain executable actions at training or evaluation.

### 2026-08-22 | Flexible-subset v3 normalized physical-teacher pilot

- Seed 405 beat selected static on mean loss (`+0.017557`) and macro loss
  (`+0.045745`), used all six channels, and had no always-on or always-off count.
- It still trailed the best conventional dynamic reference by `0.003114` in
  mean loss and `0.047103` in macro loss, so v3 did not pass the full gate.
- Particle scheduling followed the intended met-plus-laser prototype, but IR
  duty reached only `0.287` during thermal windows even though the online
  thermal alert was fully separable. The actor's BC pretraining accuracy was
  only `0.110`; the next matched run changes only BC pretraining strength.

### 2026-08-22 | Flexible-subset v3 matched strong-BC pilot

- Stronger pretraining raised BC accuracy from `0.110` to `0.786` while all
  frozen comparator metrics remained exactly unchanged.
- PD-PPO beat selected static by `0.031659` in mean loss and `0.097111` in macro
  loss. It also beat the best conventional dynamic reference by `0.010987` and
  `0.004263`.
- The policy used ten masks; all six channels had intermediate duty, switches
  per step were `0.021421`, and no feasibility or warm-up failure occurred.
  This configuration advanced to development seeds 406 and 407.

### 2026-08-22 | Flexible-subset v3 strong-BC replication

- Across seeds 405--407, PD-PPO beat selected static on both endpoints in `2/3`
  seeds and beat the best conventional dynamic reference on both in `2/3`.
- Mean margins against static remained positive (`+0.004613` mean and
  `+0.026351` macro), but the mean macro margin against the best dynamic
  reference was `-0.018517`. The configuration did not advance to final seeds.
- All runs used every channel at intermediate duty and executed 4--10 masks.
  The next matched variant replaces four-prototype BC labels with frozen-
  forecast greedy labels over the complete feasible action surface.

### 2026-08-22 | Flexible-subset v4b forecast-greedy warm start

- Full-surface forecast-greedy BC raised action coverage to 17 masks but failed
  both performance gates. PD-PPO lost to selected static by `0.042258` mean and
  `0.134527` macro, and one channel remained inactive.
- Training logs showed mean value loss `103.96` versus mean policy-loss
  magnitude `0.00836`. Whole-model gradient clipping therefore allowed critic
  gradients to suppress actor updates despite separate network modules.
- Added an opt-in actor/critic/auxiliary grouped gradient-clipping mode. It is
  enabled only for the flexible-subset experiment track and leaves historical
  configurations unchanged.

### 2026-08-22 | Flexible-subset v5 grouped-gradient diagnostic

- Grouped clipping reduced final entropy from about `3.11` to `1.99`, confirming
  that whole-model clipping had suppressed actor optimization.
- The policy beat selected static on both endpoints and beat the best dynamic
  mean loss by `0.002737`, but lost dynamic macro by `0.004705`.
- Strong BC plus grouped clipping collapsed exactly to the four physical
  prototypes. Grouped clipping is retained; the next matched run restores weak
  BC so PPO can use non-prototype feasible subsets.

### 2026-08-22 | Flexible-subset v5b grouped-gradient weak-BC diagnostic

- Weak BC plus continuing AWBC still produced exactly four prototype masks.
- PD-PPO beat selected static on both endpoints and the best dynamic mean by
  `0.003651`, but lost the best dynamic macro by `0.009019`.
- The next matched run uses strong BC only as initialization and disables AWBC
  during PPO, isolating whether continuing imitation causes prototype lock-in.

### 2026-08-22 | Flexible-subset v6 BC-warm-start grouped-gradient pilot

- Disabling AWBC after strong BC initialization produced the first broad-action
  passing pilot: 19 of 24 feasible masks and all six channels at intermediate
  duty.
- PD-PPO beat selected static by `0.030085` mean and `0.106974` macro, and beat
  the best conventional dynamic reference by `0.009414` and `0.014126`.
- No feasibility or warm-up failure occurred. The configuration was frozen for
  development replication on seeds 406 and 407.

### 2026-08-22 | Flexible-subset v6 development replication

- Across seeds 405--407, PD-PPO beat the validation-selected static subset on
  both endpoints in `2/3` seeds. Mean margins were `+0.008988` for mean loss
  and `+0.089263` for macro loss.
- PD-PPO beat the strongest conventional dynamic reference on both endpoints
  in `2/3` seeds. Mean margins were `+0.004541` and `-0.000391`, so v6 did not
  pass the frozen final-evaluation gate.
- Executed subset coverage was `19`, `8`, and `11` of 24. Seed 406 never used
  the laser and used FC4 at only `0.016` duty, localizing the remaining problem
  to cross-seed action-coverage stability rather than feasibility or switching.
- No warm-up abort occurred. The next bounded change will preserve the scene,
  action geometry, reward, and comparators while addressing actor coverage
  stability; fresh final seeds remain untouched.

### 2026-08-22 | Flexible-subset v7 zero-entropy diagnostic

- Removing the PPO entropy bonus did not stabilize the v6 replication failures.
  Across seeds 406 and 407, mean margins were `-0.014901/-0.056149` against
  static and `-0.003140/-0.044832` against the strongest dynamic reference.
- The policies executed 8 and 19 subsets, showing that entropy removal changed
  individual trajectories without reducing cross-seed coverage variance. This
  hypothesis is rejected.
- Added an opt-in linear AWBC decay schedule. Historical behavior is unchanged
  by default; the next bounded test uses existing physical-prototype guidance
  only during early PPO updates and decays it to zero, avoiding both immediate
  BC forgetting and permanent four-prototype lock-in.

### 2026-08-22 | Flexible-subset v8 decaying-AWBC diagnostic

- Linear decay from AWBC `0.15` to zero over 10k PPO steps degraded both
  representative development seeds. Average margins were
  `-0.020409/-0.083357` against static and `-0.008042/-0.061643` against the
  strongest conventional dynamic policy.
- Seed 406 recovered six-channel coverage, but seed 407 collapsed to a
  low-switching met-plus-FC4 policy. Short-lived teacher guidance does not solve
  cross-seed optimization drift and is rejected.
- The next bounded diagnostic restores v6 and lowers only the PPO learning rate
  from `3e-4` to `1e-4`. The launcher now exposes this existing hyperparameter;
  its historical default remains unchanged.

### 2026-08-22 | Flexible-subset v9 low-learning-rate diagnostic

- Lowering the PPO learning rate to `1e-4` increased executed subset coverage
  to 13 and 16 on seeds 406/407, but average margins remained negative:
  `-0.012654/-0.004511` against static and `-0.001211/-0.032193` against the
  strongest conventional dynamic reference.
- Entropy, teacher-transition, and learning-rate diagnostics have now all
  failed to produce stable cross-seed performance. Ordinary hyperparameter
  retries stop here.
- The next structural correction selects checkpoints only on the existing
  calibration/validation partition and restores the selected policy before
  independent test evaluation. The current final-update-only behavior can
  discard better intermediate PPO policies and amplify seed variance.

### 2026-08-22 | Flexible-subset v10 calibration-checkpoint diagnostic

- Calibration-only selection chose PPO updates 5 and 30 for seeds 406/407, but
  average margins remained mixed: `-0.009463/+0.038832` against static and
  `+0.000694/-0.001590` against the strongest dynamic reference.
- Checkpoint selection is retained as an optional, methodologically valid
  facility, but final-update selection is not the primary blocker.
- Cost-geometry audit showed that three low-cost channels can run together at
  `B=1.25`, recreating a compact static shortcut. The next scene calibration
  increases only fixed per-epoch effective costs for low-power channels. It
  keeps every single channel and most arbitrary pairs feasible, with no required
  channel and no explicit cardinality cap.

### 2026-08-22 | Flexible-subset v11 cost-balanced development result

- The calibrated power geometry contains 20 feasible actions: empty, all six
  singletons, and 13 arbitrary pairs. PD-PPO executed 11/15/15 subsets with no
  always-on or always-off channel, satisfying the flexible-behavior objective.
- Forecast performance failed: all three seeds lost both endpoints to static;
  average margins were `-0.026523/-0.098774` against static and
  `-0.019486/-0.063568` against the strongest conventional dynamic policy.
- Actor audit found that every candidate subset receives a trainable,
  state-independent action-prior parameter even when no external prior is used.
  This is a direct static shortcut. The next bounded variant disables that term
  while retaining state-conditioned additive sensor scoring and all 20 actions.

### 2026-08-22 | Flexible-subset v12 no-static-prior and context diagnostic

- Removing the state-independent per-action prior expanded execution to
  12/17/18 of 20 subsets but did not improve prediction: static two-endpoint
  wins were `1/3`, and strongest-dynamic wins were `0/3`.
- The online context-alert diagnostic beat PD-PPO in all three seeds, confirming
  that warning context is informative. It beat static only in seed 405 and was
  slightly weaker in seeds 406/407, so the scene still lacks stable
  dynamic-over-static value.
- The next bounded calibration changes only subtype latent update speed from
  `0.22` to `0.55`. Faster event-specific evolution should make stale specialist
  observations costly while preserving the physical channel model and online
  warning lead.

### 2026-08-22 | Flexible-subset v13 fast-latent diagnostics

- Raising latent update alpha from `0.22` to `0.55` degraded PD-PPO; average
  margins were `-0.034975/-0.098079` against static and
  `-0.020109/-0.065995` against conventional dynamic policies.
- Warning thresholds from 0.3 to 0.7 did not create stable dynamic-over-static
  value. Privileged exact-label replay also failed, including when actions were
  fixed to the prespecified physical specialist pairs.
- Faster latent innovations reduce future predictability from current
  specialist observations. V13 is rejected. The next bounded calibration
  restores alpha `0.22` and increases only the subtype-latent target amplitudes
  to strengthen specialist information value without changing action geometry.

### 2026-08-22 | Flexible-subset v14--v15 specialist-value calibration

- Stronger subtype target amplitudes improved v14 relative to fast-latent v13,
  but PD-PPO still reached only `1/3` two-endpoint static wins and `0/3`
  strongest-dynamic wins.
- Replacing the thermal physical pair with `{shielded thermo-hygro, IR}` made
  both online-warning and privileged exact-label physical policies beat static
  in seeds 405 and 407. Seed 406 remained slightly negative.
- Validation-selected, physical, hybrid, and validation-guarded action mappings
  were evaluated. None passed 3/3 because the seed-406 validation preference did
  not transfer to test. Further action-map tuning is stopped to avoid post-hoc
  adaptation.
- The flexible formulation itself is validated behaviorally: 20 naturally
  power-feasible actions, no required channel, no cardinality cap, broad subset
  use, and no forced duty quotas. Stable prediction superiority remains an open
  algorithm/scenario-transfer problem and is not claimed from these dev runs.

### 2026-08-22 | V16 invalid precursor-alias diagnostic

- A source audit initially used the state definition from the wrong pipeline
  helper and incorrectly concluded that the three subtype latent observations
  were absent from custom-PPO execution. V15 rollout artifacts instead confirm
  a 15-dimensional state with all three latent columns observed conditionally.
- V16 duplicated the same latent variables under new precursor aliases. Its
  frozen-oracle losses saturated near the clipping ceiling for every policy, so
  the run is classified as an invalid implementation diagnostic and is excluded
  from scientific comparisons.
- The alias configuration and state-column extension are removed. Exploration
  resumes from V15 with event-window and transition-level analysis of the
  seed-406 transfer failure.

### 2026-08-22 | Flexible-subset launcher teacher defaults

- The flexible-subset launcher still defaulted to infeasible three-channel calm
  and thermal teacher masks inherited from the earlier action geometry. A V17
  preflight rejected the launch before oracle fitting or policy training.
- Defaults now reproduce the feasible V15 teacher actions: `{met station,
  radiometer}` for calm and `{thermo-hygro, surface IR}` for thermal. Particle
  and flux teacher actions are unchanged.

### 2026-08-22 | Flexible-subset v17 no-action-prior signal

- With the V15 scene and the state-independent trainable action prior disabled,
  seed 406 PD-PPO improved over its within-run validation-selected static
  reference on mean loss (`0.24219` versus `0.24721`) and normalized subtype
  macro loss (`1.06079` versus `1.26149`). All six channels had intermediate
  duty, the switch rate was `0.05710` per step, and there were no aborts.
- This run retrained the stochastic TCN oracle. Its selected static action
  changed from the V15 action 8 to action 1, so it is positive development
  evidence but not a strict single-variable ablation. V18 will reuse the frozen
  V15 control artifacts to isolate the action-prior effect.

### 2026-08-22 | Flexible-subset v18 frozen no-action-prior control

- Reusing the exact V15 seed-406 truth, oracle, windows, and static reference,
  disabling the state-independent action prior reduced PD-PPO mean loss from
  `0.29292` to `0.27352` and normalized subtype macro loss from `0.96089` to
  `0.91173`.
- The controlled variant still lost to static (`0.25439`, `0.85606`) and left
  one channel effectively off. The prior is therefore a verified source of
  static bias, but removing it alone does not pass the prediction or behavior
  gates. The next bounded control adds validation-only checkpoint selection.

### 2026-08-22 | Flexible-subset v19 frozen checkpoint control

- Validation-only checkpoint selection chose update 30. On the frozen seed-406
  test windows it improved normalized subtype macro loss to `0.84910`, narrowly
  better than static `0.85606`, while mean loss remained worse (`0.27675`
  versus `0.25439`).
- Checkpoint selection therefore trades the two co-primary endpoints on this
  seed and does not pass the joint gate. The next bounded control aligns the
  training reward with validation-normalized subtype losses while retaining the
  no-prior actor and frozen evidence path.

### 2026-08-22 | Flexible-subset v20 normalized-reward control

- Subtype-normalized forecast reward with the frozen V15 evidence path reached
  macro loss `0.85721`, effectively tying but not beating static `0.85606`.
  Mean loss remained worse (`0.26286` versus `0.25439`), and only four channels
  retained intermediate duty.
- Reward normalization is rejected as the primary correction. Architecture
  inspection found that candidate embeddings were linear sums of sensor
  embeddings, which cannot represent non-additive complementarity or redundancy
  between channels. A shared nonlinear subset encoder is added as the next clean
  arbitrary-subset actor variant.

### 2026-08-22 | Flexible-subset v21 nonlinear subset encoder

- On the frozen seed-406 evidence path, the nonlinear subset encoder preserved
  intermediate duty for all six channels and improved normalized subtype macro
  loss from V18's `0.91173` to `0.88322`. Mean loss was `0.27579`; both values
  remained worse than static (`0.25439`, `0.85606`).
- Actor entropy declined much faster than in the linear encoder, indicating
  premature concentration after adding subset interaction capacity. The next
  bounded control raises the existing entropy coefficient and uses
  validation-only checkpoint selection; no new supervision or baseline prior is
  introduced.

### 2026-08-22 | Flexible-subset v22 nonlinear entropy control

- Raising entropy regularization to `0.02` restored broad six-channel use but
  degraded both endpoints to mean `0.30077` and normalized subtype macro
  `0.95556`. Higher actor entropy is rejected.
- The bounded actor controls are exhausted. Since the privileged physical
  dynamic reference itself passed only two of three V15 seeds, exploration
  moves to a prespecified stratified subtype generator that stabilizes event
  coverage across chronological partitions without using outcome feedback.

### 2026-08-22 | Flexible-subset v23 stratified-subtype gate

- Stratified event assignment prevented subtype-count drift, but checkpointed
  PD-PPO collapsed to a fixed two-channel action. It beat static mean loss by
  only `0.000118` and lost normalized subtype macro by `0.06382`; the learned
  policy therefore failed the dynamic-behavior and joint-performance gates.
- Frozen-oracle privileged replay confirmed that event adaptation is useful but
  not yet sufficient overall. Its subtype-average event loss was about `0.2454`
  versus the best static action's `0.2576`, while overall mean loss was narrowly
  worse (`0.155844` versus `0.155428`) because calm and transition periods
  offset the event gain.
- The next scene-only gate increases the prespecified event coverage before any
  PPO training. This tests whether an event-monitoring workload can support a
  positive dynamic upper bound without changing actions or using test feedback.

### 2026-08-22 | Flexible-subset v24 event-coverage upper bound

- Increasing the prespecified synthetic event coverage from `0.45` to `0.55`
  made privileged subtype adaptation beneficial on both endpoints. The best
  dynamic replay reached mean loss `0.115125`, ahead of the best static action
  at `0.123864`; its raw three-subtype average was about `0.1604` versus static
  `0.1792`.
- The dynamic upper bound used the same frozen TCN, final windows, power budget,
  and dwell constraint. It had zero aborts and only `0.00362` switches per step.
  The scenario gate therefore passes, and V25 will train PD-PPO on the frozen
  V24 evidence path before any seed expansion.

### 2026-08-22 | Flexible-subset v25 trained policy on passed scene

- Despite the positive V24 dynamic upper bound, checkpointed PD-PPO reached mean
  `0.15812` and normalized subtype macro `0.63718`, losing to static
  `0.14540/0.56057`.
- The selected policy again became nearly static, with one always-on channel,
  three always-off channels, and only `0.00608` switches per step. V26 removes
  validation checkpoint restoration while keeping the frozen scene, oracle,
  reward, and no-prior nonlinear actor unchanged.

### 2026-08-22 | Flexible-subset v26 final-policy control

- Removing checkpoint restoration recovered dynamic execution with all six
  channels at intermediate duty, `0.0453` switches per step, and zero aborts.
  Prediction still failed: mean `0.15987` and normalized subtype macro `0.57814`
  remained worse than static `0.14540/0.56057`.
- The actor learns broad behavior but does not reliably map its high-accuracy
  subtype representation to the corresponding feasible subset. V27 activates
  the existing training-only subtype action cross-entropy auxiliary without an
  execution-time label or hard subtype router.

### 2026-08-22 | Flexible-subset v27 subtype-action auxiliary

- Adding subtype-action cross-entropy at coefficient `0.1` degraded the frozen
  V24 seed-406 evaluation to mean loss `0.18254` and normalized subtype macro
  loss `0.71743`, compared with static `0.14540/0.56057` and AoI
  `0.14180/0.51523`.
- The policy regressed to one always-on and two always-off channels with only
  `0.0220` switches per step. High subtype-classification accuracy therefore
  did not translate into forecast-value action ranking. Hard subtype-action
  supervision is rejected; the next diagnostic checks whether its prototype
  labels agree with per-window forecast-loss-optimal feasible subsets before
  any further policy training.

### 2026-08-22 | Flexible-subset v28 physical event prototypes

- Replacing calibration-selected subtype actions with prespecified physical
  prototypes improved custom PPO to mean loss `0.14291` and normalized subtype
  macro loss `0.49512`. It beat static on both endpoints
  (`0.14540/0.56057`) and beat AoI on macro loss (`0.51523`), while using all
  six channels at intermediate duty with zero aborts.
- The remaining miss was narrow and localized: AoI retained lower mean loss
  (`0.14180`) because its non-event loss was lower, although PD-PPO was better
  during events. A backward-compatible event-only option is added to the
  existing subtype-action auxiliary so calm actions remain governed by the
  forecast objective instead of a fixed prototype.

### 2026-08-22 | Flexible-subset v29 event-only action guidance

- Restricting physical subtype-action guidance to event samples reduced mean
  loss to `0.13356`, ahead of AoI `0.14180` and validation-selected static
  `0.14540`. Normalized subtype macro loss was `0.49633`, also ahead of AoI
  `0.51523` and static `0.56057`.
- The policy had zero aborts, `0.0439` switches per step, five intermediate-duty
  channels, no always-on channel, and one unused channel. This is the first
  learned flexible-subset configuration to pass both performance comparisons
  without fixed-mask degeneration. The configuration is frozen for independent
  development-seed replication; no further seed-406 tuning is permitted.

### 2026-08-22 | Flexible-subset v29 development replication

- Independent scenes for seeds 405--407 gave mean PD-PPO losses
  `0.18213/0.59862` for the ordinary and normalized subtype-macro endpoints,
  compared with static `0.18353/0.78110` and AoI `0.19377/0.65420`.
- PD-PPO beat AoI on both endpoints in `3/3` seeds and static on both in `2/3`;
  seed 405 won static macro but not ordinary mean loss. Every seed had five
  intermediate-duty channels, no always-on channel, one unused channel, and
  zero aborts. The configuration and fresh confirmation seeds 501--505 are now
  locked; no further method or scene calibration is allowed.

### 2026-08-22 | Flexible-subset v29 frozen confirmation

- On locked seeds 501--505, PD-PPO beat AoI on both endpoints in `4/5` seeds
  and in aggregate (`0.20643/0.73332` versus `0.21414/0.76253`). Against static,
  it won both endpoints in `1/5`; aggregate macro remained better
  (`0.73332` versus `0.74957`) but ordinary mean loss was worse (`0.20643`
  versus `0.19792`). The full confirmation gate therefore did not pass.
- All five runs had zero aborts, no always-on channel, five intermediate-duty
  channels, and one effectively unused radiometer. Static frequently selected
  met-plus-laser or shield-plus-laser, while PD-PPO spent capacity across FC4,
  IR, and thermal channels. A bounded development-only test halves the
  event-action auxiliary weight to determine whether the macro-versus-mean
  tradeoff can be reduced without changing the scene or action space.

### 2026-08-22 | Flexible-subset v30 guidance-weight control

- Halving event-action guidance to `0.05` did not improve the three development
  seeds. Aggregate mean loss remained slightly worse than static
  (`0.18354` versus `0.18166`), although macro loss remained better
  (`0.59857` versus `0.63984`); static joint wins fell to `1/3`.
- A frozen-seed privileged subtype scheduler beat the strongest hindsight
  static mask on ordinary mean loss in only `3/5` confirmation scenes, with
  mean margin `+0.00169`. The scene therefore lacks a robust ordinary-loss
  dynamic upper bound. Further PPO coefficient tuning stops. The next bounded
  screen increases physically plausible event-process persistence and reduces
  shared microstructure correlation before any policy training.

### 2026-08-22 | Flexible-subset v31 persistent-microstructure screen

- Two development-only scenes increased event-process persistence and reduced
  shared particle microstructure correlation. The moderate V31a setting
  (`sigma=0.12`, `alpha=0.60`, correlation `0.10`, subtype-latent alpha `0.55`)
  matched the more aggressive alternative's upper margin and was retained.
- Under frozen truth and forecasters, privileged dynamic schedules beat the
  strongest hindsight static mask on ordinary mean loss in all three V31a
  development seeds. Margins were `+0.00497`, `+0.00154`, and `+0.00173` for
  seeds 601--603. The scene upper-bound gate passes; V29 training is now tested
  without further scene changes.

### 2026-08-22 | Flexible-subset v31 learned-policy check

- V29 training on the frozen V31a scenes improved aggregate mean/macro losses
  to `0.19544/0.68022`, ahead of static `0.19815/0.75647` and AoI
  `0.20193/0.70926`. Seed-level joint wins remained only `1/3` against static
  and `2/3` against AoI, so the replication gate did not pass.
- Privileged upper schedules used different subtype combinations across these
  seeds, while V31 retained fixed physical event prototypes. V32 tests
  calibration-selected prototype actions only on event samples. Calm remains
  controlled by forecast reward, and no final-partition information is used.

### 2026-08-22 | Flexible-subset v32 calibrated event actions

- Calibration-selected event actions improved aggregate losses to
  `0.18999/0.65486`, ahead of static `0.19815/0.75647` and AoI
  `0.20193/0.70926`, with `2/3` joint wins against each family.
- Behavior failed in two seeds: the policy used only two intermediate-duty
  channels and converged to one always-on plus three always-off channels.
  Calibration labels improved action value but coefficient `0.1` dominated
  exploration. V33 tests the same labels at `0.05`; no new module, scene, or
  action rule is introduced.

### 2026-08-22 | Flexible-subset v33 calibrated-guidance interpolation

- Halving calibrated event guidance restored broader execution but did not
  preserve the V32 performance gain. Aggregate ordinary loss `0.20242` was
  worse than static `0.19815` and AoI `0.20193`; paired joint wins were only
  `1/3` against each.
- The V31a scene is rejected for learned-policy confirmation. Its privileged
  ordinary-loss headroom was only `0.0015--0.0050`, insufficient relative to
  policy variation. The next scene-only upper-bound screen raises event
  coverage from `0.55` to `0.65` while holding costs, actions, constraints,
  persistent dynamics, and subtype generation fixed.

### 2026-08-22 | V31--V34 persistence-semantics correction

- Source inspection corrected the interpretation of generator `alpha`:
  `_lowpass` updates as `x[t] = x[t-1] + alpha * (raw[t] - x[t-1])`, so raising
  alpha from `0.22` to `0.60` shortened memory instead of increasing it. V31--V34
  remain valid negative diagnostics, but their directory label "persistent"
  does not describe the implemented dynamics.
- Raising event coverage to `0.65` also left the privileged ordinary-loss
  margin at only `+0.00144` on seed 604 and is rejected. V35 lowers both event
  microstructure and subtype-latent alpha to `0.08`, matching the 8-step
  forecast horizon, while restoring coverage `0.55` and retaining low shared
  correlation.

### 2026-08-22 | Flexible-subset v35 long-memory upper bound

- Correctly lowering event and subtype-latent alpha to `0.08` produced a robust
  dynamic upper bound. Privileged dynamic ordinary-loss margins over hindsight
  static were `+0.01570`, `+0.03241`, and `+0.01880` on seeds 607--609; all
  subtype-macro comparisons also favored dynamic schedules.
- Calibration-selected subtype actions remained inconsistent with final-window
  rankings, so automatic action labels are not used. A seed607 method choice
  compares forecast-reward learning without action CE against prespecified
  physical event-only CE; both retain the same masked PPO and frozen evidence.

### 2026-08-22 | Flexible-subset v36 clean method selection

- On the frozen long-memory scene, prespecified physical event-only action CE
  improved seed607 over the no-action-CE control and then replicated on seeds
  608--609. Across seeds 607--609, PD-PPO achieved mean ordinary/macro losses
  of `0.20741/0.62545`, compared with `0.22717/0.75909` for the strongest
  validation-selected static schedule and `0.23280/0.72110` for the strongest
  conventional dynamic schedule on each endpoint.
- PD-PPO won both endpoints in `3/3` seeds against both comparison families.
  Every run had zero always-on channels, one unused channel, five
  intermediate-duty channels, `0.0175--0.0352` switches per step, and zero
  warm-up aborts. The V36b method and V35 scene therefore pass the development
  gate. All configuration choices are frozen, and seeds 701--705 are declared
  as the untouched confirmation set before launch; no tuning may use their
  results.

### 2026-08-22 | Flexible-subset v36 frozen confirmation

- The untouched seeds 701--705 did not confirm static superiority. PD-PPO's
  aggregate ordinary/macro losses were `0.21954/0.71903`, compared with
  `0.20686/0.71529` for the strongest static schedule. It jointly won both
  endpoints in only `1/5` seeds; mean static margins were `-0.01268` and
  `-0.00374`.
- Against the strongest conventional dynamic schedule, PD-PPO retained positive
  aggregate margins of `+0.01640/+0.04997` and `3/5` joint wins. All five runs
  passed the behavior gate with no always-on channel, one unused channel, five
  intermediate-duty channels, `0.0124--0.0203` switches per step, and zero
  warm-up aborts. The configuration is rejected for a static-dominance claim.
  Seeds 701--705 are frozen and may be used only for post-hoc diagnosis, never
  for subsequent scene or policy selection.

### 2026-08-22 | Flexible-subset upper-bound geometry correction

- The V35 upper-bound screen inherited the diagnostic script's default
  `required_sensor_ids=[met_station_core]`, while the flexible mainline has no
  required channel. Its reported dynamic margins are reproducible but do not
  establish an upper bound over the mainline's 20-action feasible geometry.
- A post-hoc exact-geometry audit on frozen seeds 701--705 removed the required
  channel and enumerated the same feasible subsets as the learned policy.
  Privileged subtype schedules beat the hindsight best static subset in only
  `2/5` seeds. Ordinary-loss margins were `+0.01600`, `-0.00324`, `+0.02032`,
  `-0.00863`, and `-0.00435` (mean `+0.00402`). This confirms that the current
  scene still admits a cross-seed static shortcut. Further PPO tuning stops;
  future scene screens must pass an exact-geometry upper gate before training.

### 2026-08-22 | Flexible-subset v37 innovation-timescale screen

- V37a tested a mechanistic scene-only change on new development seeds
  801--805: event and subtype innovations used `alpha=0.25`, event
  microstructure sigma was `0.18`, and particle/flux microstructure correlation
  was zero. Costs, targets, constraints, and the 20-action geometry were held
  fixed; no PPO training was used for selection.
- Exact-geometry privileged dynamic schedules beat hindsight static in only
  `2/5` seeds. Margins were `+0.00896`, `-0.01305`, `-0.03034`, `-0.00055`,
  and `+0.01206` (mean `-0.00458`). Faster shared innovations are rejected.
  Candidate rankings show that thermal specialization is stable while particle
  and flux specialist value is not, motivating a bounded screen of stronger
  specialist-specific latent effects rather than another global perturbation.

### 2026-08-22 | Flexible-subset v38 specialist-latent screen

- V38 doubled the particle-specific latent effects, raised the flux-specific
  latent scale, and used moderate `alpha=0.15` with zero shared particle/flux
  microstructure correlation. New development seeds 811--815 were screened
  only with the exact 20-action privileged upper diagnostic.
- Dynamic schedules beat hindsight static in `4/5` seeds, with margins
  `+0.00577`, `-0.00711`, `+0.00947`, `+0.00164`, and `+0.01169` (mean
  `+0.00429`). This improves subtype specialization but does not provide enough
  headroom for policy training. V39 keeps these amplitudes and tests longer
  event residence so specialist observations can amortize warm-up and forecast
  history transitions. The launcher now exposes duration controls while
  preserving all historical defaults.

### 2026-08-22 | Flexible-subset v39 long-regime screen

- V39 retained V38 specialist effects but extended event duration to 48--96
  steps, minimum gaps to 24 steps, and event lead to 12 steps on new seeds
  821--825. Exact-geometry dynamic upper margins were `+0.00052`, `+0.01556`,
  `+0.00621`, `-0.00309`, and `-0.01257` (3/5; mean `+0.00133`).
- Longer residence did not overcome static history retention and is rejected.
  V40 instead shortens the 20-step observation/forecaster lookback to the
  8-step forecast horizon while restoring V38 event durations. The launcher
  exposes `LOOKBACK` with historical default 20; no PPO run is authorized until
  the exact upper gate passes.

### 2026-08-22 | Flexible-subset v40 matched-horizon lookback screen

- V40 restored V38 durations and reduced observation/forecaster lookback from
  20 to 8 on seeds 831--835. An initial launch exited at argument parsing before
  truth fitting; the split-protocol entry point was then updated to propagate
  the parameter, tested, and the same unused seeds were rerun successfully.
- Exact upper margins were `+0.00147`, `-0.03994`, `+0.00909`, `+0.01936`, and
  `+0.01057` (4/5; mean `+0.00011`). Shorter history did not stabilize the
  gate and is rejected. Because four seeds remain positive while one frozen
  TCN ranking is a large outlier, V41 holds the V38 scene fixed and increases
  evaluator candidate coverage and fitting epochs before any policy training.

### 2026-08-22 | Flexible-subset v41 evaluator-stability screen

- V41 restored lookback 20, doubled candidate-mask forecaster rollouts, raised
  subtype-teacher repeats from 4 to 6, and trained the frozen TCN for 20 epochs
  on new seeds 841--845. Exact upper margins were `-0.01104`, `-0.00912`,
  `+0.01195`, `+0.00071`, and `+0.03493` (3/5; mean `+0.00549`). Evaluator
  capacity alone does not remove the static shortcut.
- Environment reset inspection found no truth-history leakage: histories start
  at fitting-partition means with zero masks, and inactive variables are carried
  forward. The generator does, however, inject subtype-specific changes into
  shared humidity, wind, and air-temperature channels. V42 removes those three
  cross-channel subtype proxies while retaining specialist latents and physical
  targets; launcher defaults remain unchanged for historical runs.

### 2026-08-22 | Flexible-subset v62 natural-budget geometry screen

- V62 retained the V61 stratified scene, six-channel effective-cost model, and
  seeds 901--905, but increased the per-step and startup budgets from
  `1.25/1.70` to `1.65/2.00`. The matched geometry contains 28 executable
  subsets, including all six single channels and six three-channel subsets;
  full-open operation remains infeasible. Regenerated truth files are
  byte-identical to V61, while the oracle/evaluator mixture was refitted for the
  enlarged action surface. No PPO training was run.
- The validation-selected online warning policy beat the strongest static
  reference on both endpoints in `3/5` seeds. Its mean static-minus-policy
  margins were `+0.007610` for ordinary loss and `+0.068168` for macro loss.
  The physically specified warning policy also passed jointly in `3/5`, with
  mean margins `-0.008923` and `+0.090644`. It exercised all six channels at
  intermediate duty in every seed with zero aborts, but did not meet the
  prespecified `5/5` online gate. V62 is rejected before privileged replay or
  PPO training; increasing budget alone does not remove the static shortcut.

### 2026-08-22 | Flexible-subset v63 specialist-triple geometry screen

- A deterministic geometry audit showed that V62 did not admit laser plus two
  broad-observation channels. V63 tested the first buffered geometry above that
  discrete threshold at steady/startup budgets `1.75/2.15`. It contains 35
  executable actions, including 13 triples and no four-channel action; all five
  truth files remain byte-identical to V61/V62. No PPO training was run.
- The validation-selected warning policy passed both endpoints against the
  strongest static reference in `3/5` seeds, with mean ordinary and macro
  margins `+0.023569` and `+0.032368`. The physical warning policy also passed
  `3/5`, but its means were `-0.005040` and `-0.024229`. Physical behavior used
  every channel at intermediate duty with zero aborts. Because both online
  mappings failed in seeds 902 and 905, the natural-budget branch is closed;
  subsequent screens must improve online regime identifiability, not budget.

### 2026-08-22 | Variance-consistent carried-state estimator

- Added an opt-in `variance_weighted` measurement update to the warm-up
  scheduling environment. It uses the existing normalized process and sensor
  variances to update both the carried state mean and posterior variance with a
  matched predict--update step; wind direction uses a circular innovation.
  Historical runs retain the backward-compatible `direct` update default.
- Propagated the mode through split-protocol training, baseline selection,
  evaluation, metadata, and the flexible-subset launcher. Static candidate
  replay now also inherits the uncertainty configuration instead of silently
  reverting to defaults. The complete test suite passes (`110` tests).

### 2026-08-22 | Flexible-subset v64 estimator screen

- V64 held the V63 scene, seeds, costs, and 35-action geometry fixed and enabled
  the variance-consistent carried-state estimator. Full-open improved from
  `2/5` to `4/5` joint wins over static, with mean ordinary and macro margins
  `+0.021641` and `+0.046879`.
- The validation-selected online warning policy likewise improved from `3/5`
  to `4/5` joint wins, with mean margins `+0.023191` and `+0.119193`. Seed905
  remained negative on both checks, so PPO training stays blocked.
- The frozen evaluator fitting mixture contained about 70 static candidate
  groups but only 6 subtype-dynamic and 3 full-open groups. The launcher now
  exposes `ORACLE_FULL_OPEN_REPEAT`; V65 will rebalance fitting coverage without
  changing the simulator, action space, estimator, reward, or online policy.

### 2026-08-22 | Flexible-subset v65 evaluator-mixture screen

- V65 changed only frozen-evaluator fitting coverage to 12 full-open, 12
  subtype-dynamic, and one repeat of each of the 35 feasible static masks. The
  recorded full-open fraction rose to 23.5%, confirming that the requested
  mixture was applied.
- Full-open nevertheless beat static jointly in only `2/5` seeds; mean ordinary
  and macro margins were `+0.014249` and `+0.013195`. Reweighting trajectory
  families is rejected as an evaluator-stability fix, and no online-policy or
  PPO evaluation was authorized.
- The split protocol and flexible launcher now expose the existing `linear` and
  `tcn` frozen-oracle implementations. V66 will isolate predictor-family
  sensitivity under the unchanged V64 scene and estimator.

### 2026-08-22 | Flexible-subset v66 linear-predictor diagnostic

- V66 held the V64 truth generator, six-channel costs, `1.75/2.15` budgets,
  35-action geometry, variance-weighted estimator, and seeds 901--905 fixed;
  only the frozen forecast evaluator changed from TCN to linear ridge. No PPO
  training or online-policy selection was run.
- Full-open operation lost to the strongest static schedule on both endpoints
  in every seed (`0/5` ordinary, `0/5` macro, `0/5` joint). Mean
  static-minus-full-open margins were `-0.176308` and `-0.177406`.
- The linear evaluator therefore worsens, rather than repairs, information
  ordering. Predictor-family substitution is rejected as a mainline fix. A
  fixed evaluator's loss cannot be used as a physical monotonicity axiom; the
  next scene gate must compare feasible online dynamic schedules directly with
  static schedules under the same evaluator and observable information.

### 2026-08-22 | V64 continuity-guarded online scene audit

- Added a calibration-only diagnostic that selects the best calm mask, requires
  the relevant physical specialist during each alert, retains the maximum
  number of calm-mask channels, and uses subtype forecast loss only to break
  calibration ties. This diagnostic does not alter PD-PPO or use test labels.
- On the frozen V64 runs it beat static jointly in `3/5` seeds. Mean ordinary
  and macro margins improved to `+0.018754` and `+0.114463`, but seeds 902 and
  905 still failed. Continuity-aware mask selection is therefore useful but
  insufficient as a scene repair.
- The truth audit found that the fixed `-30 C` Parsivel operating threshold
  makes `54%--61%` of particle-subtype target epochs unavailable and zero-valued
  across these seeds. Subtype assignment was independent of this physical
  availability. Added an opt-in run-level eligibility condition so particle
  subtypes can be assigned only when the instrument is operational for a
  specified fraction of the event; historical generation remains unchanged.
- The first V67 launch generated truth but stopped before evaluator fitting
  because the downstream training entry point did not yet declare the new
  option. No metric was produced; the partial outputs are excluded from
  evidence and archived before the corrected rerun.
- A second propagation audit found that the split wrapper's initial truth-build
  command had omitted both the new availability option and the pre-existing
  subtype-assignment option. Consequently, V61--V66 requested `stratified` in
  manifests but their generated truth used the generator's `random` default.
  Those runs remain valid random-assignment diagnostics but are no longer
  described as stratified evidence. V67 is regenerated after passing both
  options to the authoritative truth builder.

### 2026-08-22 | Flexible-subset v67c corrected scene gate

- V67c is the first run whose generator metadata verifies both stratified
  subtype assignment and an `0.8` run-level Parsivel availability requirement.
  Particle-subtype availability rose to `97.2%--98.6%`, while all three
  subtype populations remained substantial. The TCN evaluator, 35-action
  geometry, costs, budgets, estimator, and all latent amplitudes were retained.
- The validation-selected online warning schedule beat the strongest static
  reference jointly in `4/5` seeds. It beat the deployable validation-selected
  static schedule on ordinary loss in `4/5` and macro loss in `5/5`; mean
  strongest-static margins were `+0.026954/+0.161580`.
- Seed901 missed ordinary loss by `0.010997` while retaining a positive macro
  margin. Some diagnostic mappings also left channels unused, so this result
  establishes a strong but not final scene gate. The next experiment trains
  the complete context-aware PD-PPO on the frozen V67c assets and evaluates
  policy behavior directly; no further scene tuning is selected from V67c test
  windows.

### 2026-08-22 | Flexible-subset v68 clean context-PPO control

- Trained the clean 20-feature context-aware masked PPO for 40 complete
  1,024-step updates on each frozen V67c seed. It retained the forecast-loss
  reward and used no behavior-cloning warm start, action auxiliary, bandit
  imitation, trainable action prior, or scene change.
- The learner beat the strongest static and conventional dynamic families
  jointly in only `1/5` seeds. Mean ordinary/macro margins were
  `-0.030244/-0.011349` against static and `-0.034723/-0.094993` against
  dynamic references.
- Four runs used all six channels at intermediate duty; seed903 used five.
  All runs had zero always-on channels, zero warm-up aborts, and nontrivial
  switching. The failure is context-to-action learning, not policy activity.
- V69 tests the previously established complete PD-PPO training configuration
  on the same frozen assets: training-only BC initialization and event-only
  physical action supervision, with no execution-time labels and no change to
  the 35-action policy surface.

### 2026-08-22 | Flexible-subset v69 complete-training control

- Added the previously validated training-only BC warm start and event-only
  physical action auxiliary to the frozen V67c context-aware policy. Runtime
  inputs, forecast reward, hard feasibility mask, action surface, and scene
  were unchanged.
- Static joint wins improved to `4/5`, with mean ordinary/macro margins
  `+0.027142/+0.144113`. Conventional dynamic joint wins reached `3/5`, with
  positive mean margins `+0.022662/+0.060468`.
- Sustained prototype guidance narrowed execution: only seed901 retained all
  six channels at intermediate duty, while the remaining seeds left one to
  three channels unused. V69 is therefore not the flexible main method despite
  its strong static result.
- V70 retains the BC initialization, removes ongoing action CE, and restores
  entropy `0.02`. This isolates whether a broad feasible-subset policy can keep
  the learned performance without prototype lock-in.

### 2026-08-22 | Flexible-subset v70 BC-initialized control

- V70 retained the V69 behavior-cloning initialization but removed ongoing
  event-action cross-entropy and restored entropy `0.02`; the frozen V67c
  scene, evaluator, 35-action surface, power model, and evaluation starts were
  unchanged.
- The policy beat the strongest static family jointly in `3/5` seeds, with
  mean ordinary/macro margins `+0.000545/+0.032104`. It beat the conventional
  dynamic family jointly in only `1/5`, with mean margins
  `-0.003935/-0.051540`.
- Three seeds used all six channels at intermediate duty. Seeds 901 and 905
  used five; every run had zero always-on channels and zero warm-up aborts.
  BC initialization alone therefore restores neither stable performance nor
  complete flexible coverage.
- V71 is the single bounded interpolation between V69 and V70: event-only
  action CE `0.05` with entropy `0.02`. All other assets and settings remain
  frozen. It must pass both performance families and six-channel behavior to
  proceed to fresh confirmation.

### 2026-08-22 | Flexible-subset v71 supervision interpolation

- V71 used event-only action CE `0.05` and entropy `0.02` between the V69 and
  V70 endpoints. It improved conventional-dynamic joint wins to `4/5`, with
  positive mean ordinary/macro margins `+0.019979/+0.080215`.
- Static joint wins remained `3/5`; mean static margins were
  `+0.024459/+0.163859`. Only seed901 used all six channels at intermediate
  duty, while the other policies left one to three channels unused. All runs
  remained free of always-on channels and warm-up aborts.
- A configuration audit found that V71 also disabled the nonlinear subset
  encoder, so it is not a one-variable interpolation from V70. V68--V70 had
  already used the nonlinear encoder. V72 accidentally duplicated V70 and
  reproduced its training histories and metrics exactly; it is excluded as a
  duplicate-configuration audit, not interpreted as a new experiment.
- V71c restores the nonlinear encoder and changes only event-only action CE
  from `0` to `0.05` relative to V70. This corrected run is the valid bounded
  supervision interpolation.

### 2026-08-22 | Flexible-subset v71c corrected supervision interpolation

- V71c restored the nonlinear subset encoder and changed only event-only
  action CE from `0` to `0.05` relative to V70. It beat both strongest-static
  and conventional-dynamic families jointly in `4/5` seeds. Mean
  ordinary/macro margins were `+0.016935/+0.111200` against static and
  `+0.012456/+0.027556` against dynamic references.
- The deterministic policy showed clear subtype-conditioned switching, but
  seed904 left one channel unused and seed905 left two nearly unused. The other
  three seeds had no always-off channels; all five had no always-on channels or
  warm-up aborts.
- V73 enables the existing calibration-only checkpoint selection every five
  PPO updates. Selection uses validation forecast loss only, without final-test
  metrics or channel-duty criteria, to test whether last-update collapse is
  responsible for the remaining low-use channels.

### 2026-08-22 | Flexible-subset v73 validation checkpoint selection

- Selecting checkpoints by validation mean forecast loss produced static joint
  wins `3/5` and conventional-dynamic joint wins `4/5`. All mean margins stayed
  positive, but seed903 lost its macro advantage and seed905 retained three
  nearly unused channels.
- The selector exposed an objective mismatch: it optimized ordinary validation
  loss while subtype-balanced macro loss is a co-primary endpoint. Added a
  backward-compatible checkpoint score option and full per-checkpoint endpoint
  logging. The default remains ordinary loss.
- V74 changes only the calibration checkpoint score to the existing
  static-normalized subtype macro loss. Focused tests and the complete test
  suite pass before remote deployment.

### 2026-08-22 | Flexible-subset v74c macro checkpoint selection

- The first V74 launch stopped at update 5 because checkpoint replay had not
  yet applied the existing subtype aggregation before reading the macro score.
  No final metrics were produced. The callback now reuses the authoritative
  subtype aggregation and derives fixed normalizers from control-source
  validation candidates; all tests pass after the repair.
- Corrected V74c achieved `4/5` joint wins against both static and conventional
  dynamic families, with positive mean margins. Four seeds avoided multiple
  always-off channels; seed905 retained three. Macro-only selection also chose
  a poor final checkpoint for seed901, so it does not resolve the dual-endpoint
  selection problem.
- V75 is the final supervision interpolation on the nonlinear actor. It uses
  event-only action CE `0.025`, no checkpoint selection, and otherwise exactly
  the V70/V71c configuration.

### 2026-08-22 | Flexible-subset v75 exact-action CE midpoint

- V75 reached `4/5` joint wins against static but only `2/5` against the
  conventional dynamic family. Four seeds avoided multiple always-off channels;
  seed905 still left three nearly unused. Coefficient interpolation is closed.
- Exact-action CE penalizes every feasible subset except one prototype, even
  when another subset contains the same physically relevant channels. This is
  structurally misaligned with arbitrary feasible-subset scheduling.
- Added a backward-compatible `positive_sensor_inclusion` supervision mode.
  It raises the total probability of all feasible actions containing the
  training-time guide channels and does not penalize extra channels. Historical
  `exact_action` behavior remains the default. V76 applies inclusion guidance
  to calm and event subtype examples with no runtime labels or duty constraint.

### 2026-08-22 | Flexible-subset v76 positive-inclusion development gate

- V76 beat the strongest static and conventional dynamic families jointly in
  `4/5` seeds. Mean ordinary/macro margins were
  `+0.030283/+0.164848` against static and `+0.025803/+0.081204` against
  conventional dynamic references.
- Every seed had zero always-on channels, no more than one always-off channel,
  nonzero switching, and zero warm-up aborts. Seed901 was the sole performance
  loss but retained valid dynamic behavior.
- The V76 method is frozen. Framework replay against context-alert,
  one-step forecast-greedy, and privileged event-label references is running
  before launching new-seed confirmation.

### 2026-08-22 | Flexible-subset v76 framework baseline gate

- PD-PPO beat the privileged one-step forecast-greedy diagnostic on both
  endpoints in all five development seeds. Mean ordinary/macro margins were
  `+0.026394/+0.238765`.
- Against the supplied-warning context-alert policy, PD-PPO won `2/5` on both
  endpoints but retained small positive mean margins
  `+0.003329/+0.003268`. This supports competitiveness, not stable dominance,
  against the strong handcrafted context reference.
- Locked fresh confirmation seeds `1001--1005` before launch. Each run rebuilds
  truth and the frozen evaluator from the prespecified V76 configuration; no
  development control assets or final feedback are reused.

### 2026-08-22 | Flexible-subset v76 fresh confirmation

- The frozen V76 policy retained valid dynamic operation in all five fresh
  seeds: every run had zero always-on channels, at most one always-off channel,
  nonzero switching, and zero warm-up aborts. The mean switching rate was
  `0.036648` per step.
- PD-PPO beat the strongest conventional dynamic reference jointly on both
  endpoints in `5/5` seeds. Mean ordinary/macro margins were
  `+0.030479/+0.106572`.
- The prespecified strongest-static confirmation gate did not pass. Joint wins
  were `3/5`; mean ordinary/macro margins were `+0.000104/+0.059654`, with the
  ordinary mean effectively tied. Seeds1002 and 1005 lost both endpoints to
  the validation-selected static subset.
- These fresh results are retained as confirmatory evidence and will not be
  used to retune V76. Frozen replay against forecast-greedy, context-alert, and
  privileged event-label references is being completed to close the evidence
  boundary before deciding a new development-only hypothesis.

### 2026-08-22 | Flexible-subset v76 fresh supplementary references

- Against the privileged one-step forecast-greedy diagnostic, PD-PPO won both
  endpoints in `4/5` fresh seeds. Mean ordinary/macro margins were
  `+0.005727/+0.099330`, confirming a useful sequential advantage but not the
  development set's `5/5` stability.
- The supplied-warning context-alert policy remained stronger in three seeds.
  PD-PPO ordinary/macro wins were `2/5`, with mean margins
  `-0.004275/-0.010749`. The exact-event-label reference produced the same
  `2/5` win count and mean margins `-0.007753/-0.024272`.
- V76 therefore confirms feasible, nondegenerate dynamic scheduling and strong
  gains over conventional dynamic policies, but it does not meet the required
  comprehensive baseline-superiority claim. The next operation is a privileged
  all-action upper diagnostic to separate scene headroom from policy transfer;
  no fresh-seed result will be used to retune V76.

### 2026-08-23 | Flexible-subset v76 fresh dynamic-headroom diagnostic

- The first diagnostic invocation inherited the historical
  `met_station_core` required-channel default in script49 and enumerated only
  15 actions. Those outputs were archived as invalid and excluded. The
  corrected invocation explicitly used no required channel and reproduced all
  35 candidate actions from the primary geometry.
- The privileged eight-step receding-horizon diagnostic beat the best static
  candidate in all five fresh scenes. Ordinary-loss margins ranged from
  `+0.036500` to `+0.104118`, with mean `+0.065336`.
- Every corrected run used all 35 actions; all six channels had intermediate
  duty, with zero always-on/off channels and zero warm-up aborts. The fresh
  scenes therefore contain substantial dynamic headroom. V76's failure is an
  online context-to-action transfer problem, not a remaining static scene.
- The next bounded development comparison will retain V76's frozen scene,
  reward, masked PPO, constraints, and positive-inclusion auxiliary objective,
  while replacing the single prototype BC target with the existing soft
  all-action forecast-value warm start. This combination is newly justified by
  the online context features added after the earlier V56 experiment.

### 2026-08-23 | Flexible-subset v77 soft forecast-value context test

- V77 combined V76's online context encoder and positive-inclusion auxiliary
  with the existing soft all-action forecast-value BC initialization. It used
  the same development truth, evaluator, starts, reward, constraints, and PPO
  protocol as V76.
- The variant failed: strongest-static joint wins were `2/5` and conventional-
  dynamic joint wins were `1/5`. Mean ordinary/macro margins were
  `-0.003851/+0.028943` against static and `-0.008329/-0.054697` against
  dynamic references.
- Behavior remained broad, with all six channels at intermediate duty in every
  seed and no always-on/off channels or aborts. Future-value soft imitation
  therefore improves coverage but does not solve online policy transfer; this
  teacher line is closed without temperature or duration tuning.
- A conditional-duty audit found that V76 already uses near-maximal subset
  cardinality, so unused power is not the blocker. The weaker seeds instead
  fail to retain guided event-to-channel mappings; for example, seed901 has
  zero FC4 duty in flux windows despite valid subtype supervision. V78 tests a
  single evidence-driven change, positive-inclusion weight `0.05` to `0.10`,
  while preserving set-valued extra-channel freedom and every other V76 choice.

### 2026-08-23 | Flexible-subset v78 stronger inclusion guidance

- Raising positive-inclusion weight from `0.05` to `0.10` did not improve the
  complete development gate. Strongest-static joint wins were `3/5` and
  conventional-dynamic wins were `4/5`; V76 remained better at `4/5` for both.
  Mean margins remained positive, but seed901 lost both families and seed904
  lost the ordinary static endpoint.
- Channel use narrowed again: one seed left two channels unused and two seeds
  left one unused. All runs retained zero always-on channels and zero aborts.
  Stronger scalar guidance therefore recreates the performance/coverage trade-
  off and is rejected; no further inclusion coefficient is tested.
- Added a `subtype_moe` actor fusion mode. Online context features predict soft
  subtype routing weights that blend four small action-representation experts;
  simulator labels supervise the router only during training. Runtime inputs,
  forecast reward, 35-action feasibility masking, critic, and V76 auxiliary
  loss remain unchanged. V79 is the matched development test.

### 2026-08-23 | Flexible-subset v79 subtype mixture-of-experts actor

- V79 did not improve the complete gate. Strongest-static and conventional-
  dynamic joint wins were both `3/5`; aggregate margins were positive, but
  seeds901 and 904 retained ordinary static losses and seed903 lost the dynamic
  macro endpoint. Behavior passed in all five seeds.
- The added context specialization is therefore not adopted as the primary
  method. The implementation remains available as an explicit architecture
  ablation, with unit tests confirming label-free runtime routing and feasible
  action masking.
- V80 tests a representation simplification motivated by set-valued guidance.
  It removes only the nonlinear subset encoder so that sensor value is shared
  compositionally across every subset containing that sensor. All V76 context,
  reward, supervision, PPO, and constraint settings remain fixed.

### 2026-08-23 | Flexible-subset v80 linear action representation

- V80 improved the aggregate margins but did not improve paired stability.
  Strongest-static and conventional-dynamic joint wins were `3/5` and `4/5`;
  mean ordinary/macro margins were `+0.026406/+0.170481` against static and
  `+0.021928/+0.086841` against dynamic references.
- All five seeds passed the behavior gate, with no always-on channel, at most
  one always-off channel, nonzero switching, and zero warm-up aborts. The
  linear representation is therefore viable but does not replace V76, whose
  static joint win count remains higher at `4/5` on the matched scenes.
- Architecture and scalar-objective tuning are closed. The next experiment
  separates the policy/training random seed from the frozen scene seed and
  selects one of three independent initializations using validation losses
  only. This gives the learned policy the same validation-selection discipline
  already used by the strongest static baseline without inspecting test loss.

### 2026-08-23 | Flexible-subset v81-v82 validation-only initialization selection

- Added independent data and policy seeds and trained three V76 policy
  initializations for each of the five frozen development scenes. The locked
  selector minimized the worse of the two validation loss ratios relative to
  the strongest validation static candidate; test metrics were read only after
  the selection manifest and input hashes had been written.
- V81 exposed an aggregation mismatch: its selector used unnormalized subtype
  macro loss, while the frozen co-primary endpoint is static-normalized subtype
  macro loss. V81 is retained as a diagnostic and is not used for a gate.
- V82 recorded the correct normalized macro validation metric. Selected
  policies jointly beat strongest static in `3/5` scenes and conventional
  dynamic references in `4/5`; mean ordinary/normalized-macro margins were
  `+0.022101/+0.136017` against static and `+0.017622/+0.052372` against
  dynamic references. Behavior passed in `4/5` scenes.
- Validation ranking matched the post-hoc test-best initialization in `4/5`
  scenes, but one scene had no successful initialization and only `4/5` scenes
  contained any post-hoc static joint winner. Initialization selection is
  therefore closed. The next scene gate must require a deployable online-
  context policy, not only a privileged future-loss upper bound, to beat static
  robustly before any further PPO training.
### 2026-08-23 | Flexible-subset V83a deployable-context scene gate

- Generated five new development scenes (`1101--1105`) with the frozen
  35-action physical geometry, changing only online warning lead from 8 to 12
  steps and warning noise from `0.05` to `0.02`. No full PD-PPO training was
  authorized during this screen.
- The validation-calibrated context-alert policy beat the strongest static
  family on both co-primary endpoints in `4/5` scenes. Its mean ordinary and
  static-normalized macro margins were `-0.001153` and `+0.030614`; seed1104
  caused the ordinary-mean failure. Thresholds `0.35--0.60` did not repair that
  seed, although threshold `0.35` made the aggregate margins
  `+0.001838/+0.041896`.
- Exact-label replay with the same validation-derived action map also failed
  seed1104 and passed only `4/5`. Physical, guarded-hybrid, and
  continuity-guarded maps performed worse. Replacing validation action scores
  with the larger policy-training candidate table reduced joint wins to `3/5`.
- V83a therefore fails the prespecified robust online-identifiability gate.
  The warning signal is not the primary blocker; subtype-to-subset rankings and
  transition-aware execution do not transfer stably across partitions. The
  next diagnostic must select the complete context policy by constrained
  calibration replay before any further scene amplification or PPO training.

### 2026-08-23 | V83a constrained-calibration and upper-bound gates

- Added `replay_calibrated` context-policy selection. It evaluates complete
  state-conditioned action maps on calibration/validation windows through the
  same power, startup, warm-up, and dwell-constrained environment, then locks
  the selected map before test replay. Unit and full test suites pass.
- The deployable warning-driven policy beat the strongest static family on
  both co-primary endpoints in all five V83a development scenes. Mean ordinary
  and static-normalized macro margins were `+0.016272/+0.096203`; every run had
  zero warm-up aborts and four or five intermediate-duty channels.
- The exact-geometry eight-step receding diagnostic independently passed
  `5/5`, with mean margins `+0.066626/+0.174036`. It exercised all 35 feasible
  actions, kept all six channels at intermediate duty, and had zero aborts.
- V83a therefore passes both prespecified scene-identifiability gates without
  changing the physical costs, arbitrary-subset geometry, or signal amplitudes.
  Full PD-PPO training is authorized on these frozen assets.

### 2026-08-23 | V84 complete PD-PPO learner-transfer result

- Trained the frozen complete PD-PPO configuration on all five V83a scenes with
  independent policy seeds 2101--2105. PD-PPO beat the conventional dynamic
  family jointly in `5/5` seeds and passed behavior checks in every seed.
- Strongest-static joint wins were only `2/5`; mean ordinary and
  static-normalized macro margins were `-0.009387/+0.004910`. No fresh
  confirmation is authorized.
- The experiment retained the historical per-subtype static/physical teacher,
  while the pre-training gate showed that only a complete policy selected by
  constrained calibration replay transfers robustly. V85 changes only the
  training-time teacher masks to those locked calibration-selected actions.

### 2026-08-23 | V85 calibrated-teacher warm start

- Replaced only the PD-PPO training guidance masks with each scene's
  constrained-calibration-selected context map; frozen oracle assets, reward,
  PPO settings, action geometry, and test protocol were unchanged.
- V85 regressed to `1/5` strongest-static and `4/5` conventional-dynamic joint
  wins, with mean static margins `-0.026645/-0.019071`. The variant is rejected.
- Test-rollout action audits show that the final policy exactly matches the
  calibrated online context action on only 0--57% of steps across seeds and is
  often near zero during event steps. The failure is PPO-stage retention of the
  warm start, not lack of deployable scene information. One bounded V86 run
  retains the same guidance with existing AWBC coefficient `0.05`.

### 2026-08-23 | V86 calibrated teacher with constant AWBC

- Constant AWBC `0.05` raised strongest-static joint wins to `4/5` and
  conventional-dynamic wins to `5/5`. Mean static margins became
  `+0.008486/+0.066459`, confirming that guidance retention was causal.
- Behavior did not pass: seeds 1101 and 1105 had two and three always-off
  channels, despite zero always-on channels, nonzero switching, and zero aborts.
- V87 keeps the same initial coefficient but linearly decays it to zero over
  40960 steps. This is the final bounded retention/diversity correction before
  reassessing the training formulation.

### 2026-08-23 | V87 decaying AWBC and online-label correction

- Linear AWBC decay reached only `3/5` strongest-static and `4/5`
  conventional-dynamic joint wins. Mean static margins were
  `-0.001906/+0.059765`, and the same two seeds retained multiple always-off
  channels. The subtype-look-ahead retention route is closed.
- Added a clean `context_alert` teacher mode. It derives warm-start/AWBC labels
  from the same online particle, flux, and thermal warning scores visible to the
  deployed policy, with a fixed threshold and calibration-selected feasible
  action map. No simulator event label is used by this teacher.
- V88 will test this observation-aligned teacher once, with subtype action CE
  disabled to avoid reintroducing the mismatched privileged timing signal.

### 2026-08-23 | V88 online-context teacher result

- Online-alert teacher labels preserved nondegenerate behavior in all seeds but
  reached `0/5` strongest-static and `2/5` conventional-dynamic joint wins.
  Mean static margins were `-0.028691/-0.069646`.
- The result rejects direct imitation of the deployable context heuristic as a
  sufficient training formulation. The `context_alert` implementation remains
  as a clean diagnostic option, not the primary method.
- V89 returns to V86's performance-positive guidance and changes only entropy
  coefficient `0.02 -> 0.04` to test whether action coverage can be restored
  without giving up forecast performance.
## 2026-08-23 - V89 rejects action-entropy correction

- Increased only the categorical policy entropy coefficient from 0.02 to 0.04
  on the V86 configuration and frozen V83a development scenes.
- PD-PPO beat the strongest static family jointly in 4/5 seeds and conventional
  dynamic policies in 5/5, with positive mean margins on both endpoints.
- The deployment-behavior gate passed only 3/5 seeds. Seeds 1101 and 1105 kept
  two and three channels always off, respectively, despite the higher action
  entropy.
- Closed entropy-coefficient tuning. The next bounded test must regularize
  channel-level marginal occupancy directly without imposing a hard duty quota
  or changing the arbitrary-subset action geometry.
## 2026-08-23 - V90 rejects channel-marginal entropy under argmax replay

- Added a soft, training-only entropy term over channel inclusion mass without
  changing feasible actions or imposing duty limits.
- V90 beat conventional dynamic policies in 5/5 seeds but reached only 3/5
  strongest-static wins and 2/5 behavior passes.
- The sampled training distribution retained high channel entropy, while the
  deterministic argmax rollout still produced always-on/off channels. This
  exposes an objective/execution mismatch rather than an exploration-coefficient
  problem.
- Closed coefficient tuning. The next step replays frozen checkpoints with
  stochastic feasible-action sampling and reports it separately from argmax.
## 2026-08-23 - V91 rejects unit-temperature stochastic deployment

- Replayed the five frozen V90 checkpoints three times with independent fixed
  sampling seeds, preserving all data, evaluators, constraints, and baselines.
- Stochastic feasible-action sampling restored six-channel dynamic behavior in
  every scene and replicate, with no always-on/off channels.
- Forecast performance degraded consistently: strongest-static joint wins were
  0/5 in every replicate and conventional-dynamic wins were 2/5, 3/5, and 2/5.
- Unit-temperature sampling is therefore diagnostic only. Any lower-temperature
  execution must be selected on calibration/validation before final replay.
## 2026-08-23 - V92 closes execution-temperature calibration

- Selected execution temperature per scene on validation replay from
  `0, 0.05, 0.1, 0.2, 0.5`, with temperature zero representing argmax.
- Validation selected argmax in three scenes and low temperatures in two.
- Final results remained at 3/5 strongest-static wins, 5/5 conventional-dynamic
  wins, and 2/5 behavior passes. Temperature selection did not resolve the
  performance/diversity conflict.
- Closed execution-temperature tuning. The next bounded training test retains
  calibrated guidance only in event windows and frees calm/transition actions.
### 2026-08-23 | V93 event-only guidance retains performance but not coverage

- Restricted retained AWBC to event samples while keeping subtype-action
  inclusion supervision at its V86 all-sample setting. A duplicate launcher
  assignment had overridden the requested event-only subtype setting; this is
  recorded explicitly rather than attributing V93 to a two-loss scope change.
  The V83a scene, forecaster, action geometry, reward, and policy seeds remained
  fixed. PD-PPO jointly beat the strongest static family in `4/5` seeds and
  conventional dynamic references in `5/5`; mean ordinary/macro margins were
  `+0.010409/+0.072432` and `+0.049779/+0.190395`, respectively.
- Behavior passed only `3/5`. Seed 1101 left the met-station and FC4 channels
  unused, and seed 1105 left the met-station, radiometer, and FC4 channels
  unused. The omissions occur in both event and non-event epochs; seed 1105
  also selected the empty subset for `27.9%` of test epochs.
- Event-only guidance is therefore not adopted as the frozen configuration.
  V94 changes only the nonlinear subset encoder to the previously tested linear
  additive representation, which directly shares learned channel value across
  every feasible subset containing that channel.
### 2026-08-23 | V94 rejects linear subset representation on V83a

- Replaced only the nonlinear subset encoder with a linear additive action
  representation while retaining the effective V93 guidance, objective, frozen
  scenes, and policy seeds.
- Joint wins fell to `3/5` against strongest static and `4/5` against
  conventional dynamic references. Behavior remained `3/5`; seeds 1101 and
  1105 still left multiple channels unused.
- Linear composition has now failed to meet the complete gate on two scene
  families and is closed. V95 restores the nonlinear encoder and performs the
  originally intended event-only scope test for both retained guidance losses,
  which V93 did not execute because of the corrected launcher overwrite.
### 2026-08-23 | V95 closes retained-guidance scope tuning

- Correctly restricted both retained AWBC and subtype-action inclusion to event
  samples, with all remaining V93 controls frozen. Joint wins were `3/5`
  against strongest static and `5/5` against conventional dynamic references;
  behavior passed `3/5`.
- Seed 1101 recovered full channel coverage, while seeds 1103 and 1105 each
  left two channels unused. The scope change therefore does not satisfy the
  complete gate and is closed.
- Variable-level decomposition identified a scene-objective defect: the base
  loss assigns mass-flux weight `18` and particle weights `8/8`, so snow targets
  dominate even in calm periods. The next screen uses equal weights after the
  existing physical target scaling while retaining event-subtype weights.
### 2026-08-23 | Configurable target-weight launcher

- Replaced the flexible-subset launcher's hard-coded base and event-subtype
  target-weight arguments with environment-overridable arrays. Defaults exactly
  preserve every historical run.
- This change permits auditable objective calibration without copying or
  editing the training command. Shell syntax and the complete test suite pass.
### 2026-08-23 | V96 passes the flexible-scene readiness gate

- Changed only the nine base forecast-target weights to equal values after the
  existing physical scaling. Event-subtype weights and all scene, sensor,
  constraint, warning, evaluator, and temporal settings remained fixed.
- The deployable context replay beat the strongest static subset on both
  endpoints in `4/5` seeds by ordinary loss and `5/5` by macro loss. Mean
  ordinary/macro margins were `+0.027521/+0.111938`; behavior passed `4/5`.
- The exact-label diagnostic reached `5/5` wins on both endpoints. More
  importantly, the exact-receding upper beat static in `5/5` seeds with mean
  ordinary margin `+0.064679`, exercised all 35 non-empty feasible actions,
  gave all six channels intermediate duty, and had zero warm-up aborts.
- V96 therefore has genuine dynamic headroom without required channels,
  cardinality limits, or duty quotas. Scene-weight tuning is closed; the next
  stage trains PD-PPO on the frozen V96 assets with fresh policy seeds.
### 2026-08-23 | V97 passes performance and localizes one teacher omission

- Trained the complete nonlinear PD-PPO configuration on frozen V96 assets
  with fresh policy seeds 3101--3105. Strongest-static ordinary/macro wins were
  `4/5` and `5/5`; conventional-dynamic joint wins were `5/5`.
- Mean margins were `+0.020304/+0.086369` against strongest static and
  `+0.022752/+0.047480` against conventional dynamic references. Every seed
  had nonzero switching and zero warm-up aborts.
- The strict no-constant-channel gate passed `4/5`. Seed 1101 never selected
  FC4 because its calibration-selected four-state teacher map contains no FC4
  action. This is a teacher-coverage defect, not missing dynamic scene value.
- V98 retains the V97 objective and learner but replaces only the teacher map
  with the six-channel-covering physical-function map. No runtime channel,
  cardinality, or duty constraint is added.
### 2026-08-23 | V98 rejects static teacher-map replacement

- Replaced the V97 seed-specific teacher actions with the original fixed
  physical-function map while retaining AWBC `0.05` and every other control.
- Performance remained strong: static ordinary/macro wins were `4/5` and
  `5/5`, dynamic joint wins were `5/5`, and all mean margins were positive.
- Strict behavior remained `4/5`; seed 1101 still gave FC4 zero duty even
  though its physical flux teacher contains FC4. Stronger static-teacher
  retention is therefore not justified.
- V99 replaces coarse static action labels with an eight-step frozen-forecaster
  greedy teacher on the training partition. It begins as a seed1101 mechanism
  test and retains the forecast reward and arbitrary feasible-action mask.
### 2026-08-23 | V100--V101 close broad-execution diagnostics

- V100 used soft forecast-value pretraining on frozen seed 1101. It exercised
  28 feasible subsets with all six channels at intermediate duty, but lost to
  the strongest static family by `-0.094912/-0.498743` on the ordinary/macro
  endpoints. The variant is rejected.
- V101 selected execution temperature from six candidates using validation
  replay only. Temperature `0.1` removed literal zero-duty channels, but final
  margins were `-0.003062/-0.012583` against static and
  `+0.010018/-0.039865` against conventional dynamic references. The variant
  is rejected.
- The original user requirement prohibits multiple always-on or always-off
  channels; it does not require every duty to be strictly interior. V97 meets
  that requirement in `5/5` development seeds, while retaining materially
  stronger prediction performance than V100 or V101. V97 is therefore the
  frozen confirmation configuration unless V99 reveals a performance-preserving
  mechanism improvement.
### 2026-08-23 | Fresh V97 confirmation protocol locked

- Locked 22 scene seeds `1201--1222` and independent policy seeds `4201--4222`
  before generating any confirmation output.
- The frozen protocol preserves the V96 scene/objective and complete V97 learner.
  Validation-only calibrated context maps provide training guidance; each test
  partition is evaluated once after all choices are frozen.
- The behavior gate follows the physical-system requirement: at most one
  always-on and at most one always-off channel per seed, nonzero switching,
  zero warm-up aborts, and state-dependent channel allocation.
### 2026-08-23 | V99 rejects the forecast-greedy training teacher

- Replaced only V97's four-state static teacher with an eight-step
  frozen-forecaster greedy teacher on the policy-training partition. The
  teacher exercised all 35 non-empty feasible actions, but AWBC cross-entropy
  remained near the 36-action random scale.
- Final execution used 24 subsets and gave all six channels intermediate duty,
  but margins were `-0.019192/+0.003126` against strongest static and
  `-0.006112/-0.024157` against conventional dynamic references.
- This teacher improves action breadth without preserving forecast quality and
  is not expanded. V97 remains frozen for fresh confirmation.
### 2026-08-23 | V102 confirms dynamic value but not static dominance

- Completed the locked 22-seed protocol with scene seeds `1201--1222` and
  independent policy seeds `4201--4222`. No configuration changed after lock.
- PD-PPO jointly beat conventional dynamic references in `18/22` seeds, with
  mean ordinary/macro margins `+0.027555/+0.054944` and both bootstrap
  intervals above zero. It also jointly beat the one-step forecast-greedy
  diagnostic in `18/22`, with margins `+0.022473/+0.098269`.
- Strongest-static ordinary/macro wins were `15/22` and `12/22`, with `11/22`
  joint wins. Mean margins were positive (`+0.009987/+0.023076`), but both
  confidence intervals included zero. Stable static dominance is not claimed.
- Behavior and feasibility passed `22/22`: no invalid masks, no power or startup
  violations, no warm-up aborts, nonzero switching, at most one constant-on/off
  channel, and strong event-versus-calm duty changes in every seed.
- The calibrated context and exact-label references themselves achieved only
  `14/22` and `13/22` joint static wins. The scene/action geometry remains the
  limiting factor; further V102 learner tuning is closed.
### 2026-08-23 | V103 exposes within-regime dynamic value

- Raised only the six fixed effective per-epoch costs, representing fixed
  within-epoch sampling load. Under the unchanged `1.75` budget, all 6 single
  and all 15 pair subsets remain feasible; triples become infeasible from power
  alone. No cardinality rule or frequency action was added.
- The exact-receding diagnostic beat strongest static in `5/5` development
  seeds with mean ordinary margin `+0.121575`, used all 22 feasible actions,
  gave all six channels intermediate duty, and had zero aborts.
- Four-state calibrated context reached only `3/5` joint static wins; the fixed
  physical map reached `2/5`. The gap is therefore not a lack of dynamic
  headroom. A single subtype-to-mask mapping cannot express useful variation
  within an event subtype.
- One bounded V104 run is authorized because the action set is smaller and the
  exact sequential headroom is substantially larger. It freezes the complete
  V97 learner and changes no training hyperparameter.

### 2026-08-23 | V104 rejects direct learner transfer

- The unchanged complete V97 learner reached only `2/5` joint wins against the
  strongest static family and `4/5` against conventional dynamic schedules.
  Mean static margins were `+0.005510` ordinary and `-0.010522` macro; the
  behavior/feasibility gate passed `3/5` seeds.
- Seed `1304` was the principal failure, with static margins
  `-0.091556/-0.323363`. Across all seeds there were no invalid actions, power
  violations, startup-peak violations, or warm-up aborts.
- The V103 exact-receding policy remains `5/5` over static with mean ordinary
  margin `+0.121575`. The scene contains substantial sequential headroom, but
  the existing PPO training scaffold does not recover it reliably.

### 2026-08-23 | V105--V106 close initialization selection

- A second policy initialization was tested only on frozen scene seed `1304`.
  Its validation ordinary loss (`0.387165`) was lower than the replayed V104
  initialization (`0.392805`), and its test losses improved in the same
  direction.
- The selected second initialization still lost to strongest static by
  `-0.064061` ordinary and `-0.265963` macro margin, and to conventional
  dynamic by `-0.065179/-0.212757`.
- Initialization variance is real but insufficient. The predeclared expansion
  condition is not met; no multi-initialization sweep is launched. Subsequent
  work must address sequential credit assignment instead of repeating random
  restarts or scalar hyperparameter tuning.

### 2026-08-23 | V107 establishes partial online identifiability

- A training-partition-only policy learned hard eight-step forecast-value
  action labels without PPO or subtype auxiliaries. Its action accuracy was
  `24.9%` over 22 represented actions, versus `4.5%` uniform chance.
- On frozen seed `1304`, ordinary/macro losses improved from
  `0.620068/1.684553` to `0.539649/1.400319`. All six channels had intermediate
  duty and no constraint or warm-up failure occurred.
- The probe still trailed strongest static by `-0.011138/-0.039129`. The online
  state is informative but hard privileged argmin labels do not fully transfer;
  a continuous action-value mechanism is the next clean learner hypothesis.

### 2026-08-23 | V108 rejects soft forecast-value targets

- Replaced V107's hard labels with a low-temperature (`0.05`) distribution over
  the same eight-step forecast-value costs. No PPO or auxiliary update was used.
- Training argmax accuracy remained `23.8%`, but held-out ordinary/macro losses
  deteriorated to `0.737331/1.807278`; the probe is substantially below both
  V107 and strongest static.
- Close target-temperature interpolation. The flat actor MLP currently receives
  a 20-step history and mask as one vector; V109 will test a structured temporal
  encoder under V107's hard-label observability protocol before any PPO run.

### 2026-08-23 | V109 validates structured temporal observation encoding

- Repeated V107's hard eight-step forecast-value supervision on frozen scene
  seed `1304`, replacing only the actor's flattened history MLP with a GRU over
  the existing 20-step value and observation-mask sequence.
- Held-out ordinary/macro losses improved to `0.519187/1.311281`, beating the
  strongest static reference by `+0.009325/+0.049909` and the best conventional
  dynamic endpoints by `+0.008207/+0.103115`.
- Training label accuracy decreased from `24.9%` to `20.8%`; the held-out gain
  is therefore consistent with improved temporal generalization, not closer
  memorization of privileged action labels.
- The behavior gate passed with zero always-on, one always-off, five
  intermediate-duty channels, switching `0.007816`, and zero aborts. One
  strictly matched complete-PPO temporal control is authorized before any
  multi-seed expansion.

### 2026-08-23 | V110 localizes failure beyond temporal representation

- Added the V109 temporal encoder to the otherwise unchanged V104 complete PPO
  run on matched scene seed `1304` and policy seed `4304`.
- Ordinary/macro losses were `0.610400/1.652073`, still below strongest static
  by `-0.081889/-0.290883`. The improvement over V104 was small and did not
  preserve V109's positive margins.
- The policy selected the empty subset in `963/2304` steps, never selected FC4,
  and had mean power `0.582743`. This reproduces V104's overly sparse execution
  despite the stronger history representation.
- V111 will isolate PPO transfer by starting from V109's forecast-value BC
  design, adding forecast-loss PPO, and disabling continuing imitation and
  subtype auxiliary losses. No scene, action, reward, or constraint changes are
  authorized.

### 2026-08-23 | V111 identifies post-BC PPO degradation

- Added 40,960 forecast-loss PPO steps to V109's temporal forecast-value BC
  initialization with continuing imitation and subtype auxiliary losses off.
- Ordinary/macro losses degraded from V109's `0.519187/1.311281` to
  `0.539342/1.399040`, leaving static margins `-0.010830/-0.037850`.
- The final policy used 13 subsets, all six channels had nonzero duty, and no
  abort occurred. The loss is therefore not caused by V110's sparse empty-
  action behavior; PPO updates changed a broadly dynamic but less predictive
  mapping.
- Checkpoint selection previously omitted the BC checkpoint because callbacks
  began after PPO update 1. The callback now exposes BC as update 0, with a
  regression test. V112 will select among update 0 and every fifth PPO update
  using calibration/validation data only.

### 2026-08-23 | V112 closes checkpoint selection on V103

- Added the forecast-value BC checkpoint as update 0 and compared it with every
  fifth PPO update on calibration/validation data. Update 35 was selected with
  validation ordinary loss `0.370136`; the static validation reference remained
  stronger at `0.364751`.
- The selected checkpoint reached test ordinary/macro losses
  `0.561536/1.504863`, trailing strongest static by
  `-0.033024/-0.143673`. All channels had nonzero duty and no abort occurred.
- A two-endpoint static-normalized score would also select update 35, so another
  selector rerun is not justified. V109's positive test result is a partition
  reversal, not stable validation-to-test evidence.
- Close further PPO tuning on V103. A deployable online method must first beat
  static on validation across development scenes; privileged exact-receding
  headroom alone is insufficient.

### 2026-08-23 | V113 retains dynamic headroom but rejects one-threshold context control

- Added a noisy lead forecast of continuous subtype intensity to the three
  online warning scores while preserving the V103 truth dynamics, costs,
  action geometry, evaluator, and test partitions.
- The exact eight-step receding diagnostic beat the strongest static action in
  all five scenes, with ordinary-loss margins from `+0.056991` to `+0.112564`.
- The validation-calibrated one-threshold context policy beat static in `3/5`
  scenes on ordinary loss and `5/5` on the static-normalized event macro, with
  mean margins `+0.038383/+0.106384`. An initial audit incorrectly reported the
  macro count by comparing unlike columns; the corrected values use the
  static-normalized reference column and filter the context-policy rows.
- Added a bounded low/high intensity context diagnostic. Its seven action
  mappings are selected only on calibration replay and final execution reads
  only the online warning scores. PPO remains blocked until this stronger
  information gate is evaluated.

### 2026-08-23 | V114--V115 validate magnitude use and close threshold tuning

- V114 used a fixed `0.75` high-intensity boundary and calibration-selected
  calm/low/high actions. It beat static in `3/5` scenes on ordinary loss and
  `5/5` on the static-normalized event macro, with mean margins
  `+0.028183/+0.074529`. Operational behavior remained feasible.
- V115 selected the high boundary from `0.65/0.75/0.85` using calibration
  replay only. It reached `3/5` wins on both endpoints and did not improve V114;
  threshold search is therefore closed.
- The information gate is sufficient for one clean learner experiment. V116
  freezes V113 and combines the existing
  temporal and online-context encoders with forecast-value initialization,
  forecast-loss PPO, and validation-only checkpoint selection.

### 2026-08-23 | V116 rejects hard argmin transfer across flexible masks

- Trained the clean temporal/context PD-PPO configuration on all five frozen
  V113 scenes. Validation selected BC step zero for seeds 1301--1302 and PPO
  updates 20, 35, and 5 for seeds 1303--1305.
- The selected policies reached only `2/5` ordinary and `0/5` macro wins over
  static, with mean margins `-0.060953/-0.170601`. Behavior remained feasible:
  zero always-on channels, at most one always-off channel, and zero aborts.
- Hard argmin forecast-value labels are unstable over 22 near-competing masks.
  V117 retains the matched scene and PPO design but pretrains the actor logits
  against standardized forecast values for all feasible actions. No context
  baseline, subtype label, or final-test feedback enters this target.

### 2026-08-23 | V117 improves mean transfer but does not pass the scene gate

- All-action forecast-value regression completed on all five matched scenes.
  It reached `1/5` ordinary and `0/5` macro wins over static, with mean margins
  `-0.036658/-0.106775`; this improves the V116 means but not robustness.
- All policies passed the behavior gate. The remaining failure is not caused by
  constant channels, excessive switching, warm-up aborts, or infeasible masks.
- V118 consolidates the clean context-aware method: a subtype mixture-of-experts
  encoder, the existing training-only context auxiliary, all-action forecast
  values, forecast-loss PPO, and a validation selector minimizing the worse of
  ordinary/static and macro/static loss ratios. Final execution remains label
  free.

### 2026-08-23 | V118 validates the cohesive CA-PD-PPO direction

- The context MoE, training-only context auxiliary, forecast-value regression,
  forecast-loss PPO, and joint validation selector reduced mean static gaps to
  `-0.007026/-0.046756`. Wins were `2/5` ordinary, `2/5` macro, and `1/5`
  jointly; conventional-dynamic wins improved to `3/5`.
- All five policies passed the execution behavior gate. The remaining pattern
  is dominated by policy-initialization variance, not a structural collapse.
- V119 freezes the complete V118 method and adds two policy initializations.
  Exactly one initialization per scene is selected by the frozen validation
  joint score before aggregating test metrics.

### 2026-08-23 | V119 closes policy-initialization expansion

- Validation-only selection among three frozen V118 initializations reached
  `3/5` ordinary, `1/5` macro, and `1/5` joint static wins. Mean margins were
  `-0.001688/-0.037760`; conventional-dynamic wins remained `3/5`.
- Seed 1305 selected a policy with two always-off channels, so the aggregate
  also fails the prespecified execution behavior gate. Additional restarts are
  closed.
- Failures remain concentrated in scenes 1301--1302, where the deployable
  context gate itself does not beat static on ordinary loss. V120 therefore
  tests full use of the noisy lead intensity forecast before any further
  learner run.

### 2026-08-23 | V120 preserves dynamic value but rejects full-intensity thresholding

- V120 increased only the online continuous warning-intensity contribution
  from `0.75` to `1.0`; truth dynamics, effective channel costs, feasible
  subsets, evaluator, partitions, and test seeds remained frozen.
- The calibration-defined context policy beat the validation-selected static
  schedule in `3/5` scenes on ordinary loss and `4/5` on the static-normalized
  subtype macro. Mean margins were `+0.005443/+0.031276`. Two scenes also had
  two always-off channels, so the policy did not pass the execution gate.
- The exact eight-step receding diagnostic beat the best fixed feasible subset
  in all five scenes. Its ordinary-loss margins were `+0.073649` to
  `+0.121511` (mean `+0.089451`), all six channels had intermediate duty, and
  switching remained near `0.05` per step.
- Dynamic value therefore remains present, while a fixed warning threshold and
  calibration action mapping do not expose it reliably. V121 training is
  blocked pending an online-observable versus receding-action learnability
  audit; no further threshold or PPO restart sweep is authorized.

### 2026-08-23 | V121--V122 quantify the online learnability gap

- Added trace-level receding diagnostics on disjoint policy-training,
  validation, and final partitions. Each trace records the complete online
  scheduler observation, the 20 alert-context features, all 22 counterfactual
  forecast costs, and the selected receding action.
- With validation-only fitting, the best cost-sensitive model recovered 23.1%
  of the receding diagnostic's one-step gain. Adding policy-training traces
  raised the descriptive combined-partition result to 33.0%.
- Under the strict policy-training-only fit, the best complete-state cost model
  produced positive one-step proxy gains in `5/5` scenes but recovered only
  25.5% of receding value. The warning-only model recovered 23.4%.
- The audit confirms that the online observations contain some transferable
  action-value information, while a large gap remains between privileged
  receding values and a deployable mapping. V123 tests the selected diagnostic
  model in an actual closed-loop rollout before any PPO integration.

### 2026-08-23 | V123 rejects direct offline tree-policy deployment

- A fixed ExtraTrees action-value model was fitted only on policy-training
  receding traces and executed from online observations on the final partition.
  It beat static in `3/5` scenes on both endpoints, but mean margins were
  `-0.047145/-0.157208` because the loss increased sharply in seed 1301.
- Operational behavior also failed: seed 1304 collapsed to two always-on and
  four always-off channels, while seed 1305 had two always-off channels.
- Positive one-step counterfactual estimates therefore do not survive the
  policy-induced state distribution shift. Offline tree deployment is closed;
  any further method must learn forecast values on states visited by the
  current policy and retain forecast-loss PPO as the primary objective.

### 2026-08-23 | V124 validates a sparse on-policy forecast-value auxiliary

- Added a dense-action forecast-value regression term to the masked PPO actor.
  Targets are computed from the frozen evaluator only at states visited by the
  current training policy; the primary reward, PPO objective, feasibility mask,
  and label-free final execution are unchanged.
- The bounded seed-1303 pilot used coefficient `0.05`, stride `64`, and 20,480
  PPO steps. It passed the behavior gate and beat AoI, random, and round-robin,
  but trailed static by `0.004276` on ordinary loss and `0.086195` on the
  static-normalized macro.
- Validation `max_static_ratio` improved from `1.303862` at initialization to
  `1.204061` at the selected final checkpoint. The auxiliary label rate was
  only `0.015625`, and its regression loss did not converge.
- V125 is the only authorized strength check: coefficient `0.5`, stride `16`,
  with scene, seed, architecture, PPO length, reward, and selection frozen.

### 2026-08-23 | V125 authorizes a frozen multi-scene check

- The denser seed-1303 pilot improved validation `max_static_ratio` from the
  V124 value `1.204061` to `1.105692`. The selected checkpoint was update 15.
- Final ordinary loss was `0.539172` versus static `0.548455`, a positive
  margin of `+0.009283`. The static-normalized macro remained weaker at
  `0.963249` versus `0.865771`, a margin of `-0.097478`.
- Execution remained feasible, with no always-on or always-off channel, five
  mid-duty channels, 0.040961 switches per step, and zero aborts. PD-PPO also
  beat all three conventional dynamic policies on ordinary loss.
- The frozen coefficient-0.5, stride-16 configuration is sufficiently improved
  for one four-seed expansion. No coefficient, stride, architecture, or scene
  changes are permitted during V126.

### 2026-08-23 | V126 rejects direct actor-logit value regression

- The frozen V125 configuration was expanded to seeds 1301, 1302, 1304, and
  1305 and aggregated with the completed seed-1303 pilot. It beat the selected
  static schedule on ordinary loss in `2/5` scenes and on the normalized event
  macro in `1/5`; only `1/5` scenes passed both endpoints.
- Mean margins versus static were `+0.001768` for ordinary loss and `-0.036311`
  for the event macro. PD-PPO beat the best conventional dynamic policy in
  `3/5` scenes, with mean ordinary margin `+0.021746`.
- All five scenes passed the operational gate, but seed 1305 approached a
  compact schedule with one always-on and one always-off channel. Endpoint
  failures were event-specific: thermal dominated seed 1301, while particle
  dominated seeds 1303--1305.
- Close direct squared-error regression from standardized action values to raw
  actor logits. The next bounded test preserves the forecast-loss PPO reward,
  observations, scene, and feasible actions, but represents the same on-policy
  forecast values as a masked soft action distribution and trains their ranking
  with cross-entropy. No confirmatory seed expansion is authorized yet.

### 2026-08-23 | V127 validates categorical forecast-value supervision

- V127 changed only the on-policy forecast-value auxiliary from raw-logit MSE
  to masked soft-target cross-entropy on seed 1303. It beat selected static by
  `+0.035874` on ordinary loss and `+0.014179` on the normalized event macro,
  and beat the best conventional dynamic policy by `+0.065721`.
- Execution passed with no always-on channel, one always-off channel, five
  intermediate-duty channels, 0.039369 switches per step, and zero aborts.
- Update 20 was selected by the frozen validation rule and had the lowest
  ordinary validation loss, so the result does not use test feedback for
  checkpoint selection. No validation checkpoint beat static on ordinary loss,
  however; V127 remains a development pilot rather than confirmatory evidence.
- Authorize a bounded frozen expansion to seeds 1301 and 1302. Complete the
  five-scene wave only if that two-scene check preserves positive aggregate
  evidence and valid behavior.

### 2026-08-23 | V128 authorizes completion of the development wave

- The frozen soft-target configuration was evaluated on seeds 1301 and 1302
  and aggregated with V127 seed1303. It beat static on ordinary loss in `2/3`
  scenes, on the normalized event macro in `3/3`, and jointly in `2/3`.
- Mean static margins were `+0.017156` ordinary and `+0.036321` macro. All
  three scenes beat their best conventional dynamic policy on ordinary loss,
  with mean margin `+0.053821`, and all passed the behavior gate.
- Seed1301 missed static ordinary loss by `-0.009274` while retaining a
  positive macro margin. The aggregate result authorizes a frozen completion
  on development seeds 1304 and 1305; no configuration changes are permitted.

### 2026-08-23 | V129 improves both means but misses the joint gate

- The complete five-scene soft-target wave beat static in `3/5` scenes on
  ordinary loss and `3/5` on the normalized event macro, with `2/5` joint
  wins. Mean margins were positive at `+0.012938/+0.013238`.
- It beat the best conventional dynamic policy in `4/5` scenes with mean
  ordinary margin `+0.032916`, and all five behavior gates passed.
- Remaining deficits are localized: seed1304 loses mainly on thermal windows,
  while seed1305 loses mainly on particle windows. The result does not
  authorize fresh confirmation.
- V130 is one bounded target-sharpness test on these two development scenes.
  It changes only soft-target temperature from `1.0` to `0.5`; no coefficient,
  architecture, reward, scene, or selector change is permitted.

### 2026-08-23 | V130 fixes prediction deficits but over-sharpens behavior

- Reducing soft-target temperature to `0.5` turned both failing scenes positive.
  Seed1304 margins became `+0.028408/+0.089846`; seed1305 reached
  `+0.041570/+0.141514`. Both also beat their best conventional dynamic policy.
- Seed1304 passed behavior, but seed1305 selected radiometer at 0% and the
  thermo-hygro channel at 0.87%, yielding two effectively always-off channels.
  The sharper target shifted 88.9% duty to surface IR in that scene.
- The prediction gain is real, but temperature0.5 fails the deployment-behavior
  gate. V131 is the final bounded interpolation at temperature `0.75` on the
  same two scenes; no wider scalar sweep is authorized.

### 2026-08-23 | V131 passes prediction and behavior at temperature 0.75

- The bracketed temperature `0.75` check passed both endpoints in both failing
  scenes. Seed1304 margins were `+0.041823/+0.170972`; seed1305 margins were
  `+0.026305/+0.110441`. Both beat their best conventional dynamic policy.
- Neither scene had an always-on or always-off channel. All six physical-system
  channels were used, with switching rates 0.023448 and 0.021856 per step.
- Temperature0.75 is frozen. V132 completes the same configuration on seeds
  1301--1303; the resulting consistent five-scene aggregate determines whether
  fresh confirmation is justified.

### 2026-08-24 | V132 passes the frozen five-scene development gate

- With temperature `0.75` fixed across all five scenes, PD-PPO beat selected
  static on ordinary loss in `5/5`, on the normalized event macro in `4/5`,
  and jointly in `4/5`. Mean margins were `+0.032846/+0.091491`.
- It beat the best conventional dynamic policy in `5/5`, with mean ordinary
  margin `+0.052825`. All behavior gates passed, with no always-on or always-off
  channel in any scene.
- Scene seeds `1401--1406` are locked as fresh confirmation seeds before their
  truth generation or policy training. Method, costs, scene generator,
  evaluator, target temperature, selector, and behavior thresholds are frozen.

### 2026-08-24 | V133 transfers directionally on six fresh scenes

- On locked seeds 1401--1406, the frozen method beat selected static on
  ordinary loss in `4/6`, on the normalized event macro in `4/6`, and jointly
  in `4/6`. Mean margins remained positive at `+0.023355/+0.050355`.
- It beat the best conventional dynamic policy in `5/6`, with mean margin
  `+0.038907`, and all six behavior gates passed.
- Six-seed bootstrap intervals versus static still include zero, driven mainly
  by seed1401. No parameter is changed. V134 expands the locked confirmation
  set through seed1424 to estimate stability with 24 fresh scenes.

### 2026-08-26 | V134 completes frozen 24-scene confirmation

- The unchanged temperature-0.75 method beat validation-selected static in
  `15/24` scenes on ordinary loss, `14/24` on the normalized event macro, and
  `13/24` jointly. Mean margins were `+0.017872/+0.033640`, but both 95%
  bootstrap intervals included zero.
- It beat fixed-priority feasible static in `20/24` with mean ordinary margin
  `+0.033916` and a positive 95% interval. It beat the best AoI, round-robin,
  or random policy in `21/24`, with mean `+0.039625` and a positive interval.
- All 24 scenes passed the behavior gate. No channel was always on; at most one
  was always off, and switching ranged from 0.008829 to 0.040527 per step.
- The frozen evidence supports robust gains over conventional dynamic and
  fixed-priority feasible references, but not stable superiority over the
  validation-selected static shortcut. V135 adds the missing online-context
  and privileged one-step forecast-greedy replays without retuning PD-PPO.

### 2026-08-26 | V138 validates dynamic value in the generic physical-channel scene

- V138 removes simulator subtype latents from both channel measurements and the
  scheduler/forecaster state. It also disables subtype-weighted reward and the
  subtype auxiliary objective, leaving twelve physical state variables and six
  independently selectable physical-system channels.
- The exact-geometry receding forecast reference beat the validation-selected
  static subset on both endpoints in `5/5` new development scenes. Mean margins
  were `+0.032692` for ordinary forecast loss and `+0.134500` for the
  validation-normalized event macro.
- The online context rule and one-step forecast-greedy reference each achieved
  only `2/5` joint wins. The scene therefore contains robust dynamic value, but
  the currently exposed online context does not yet identify that value
  reliably enough to authorize PPO training.
- Seed1505 contains no particle-subtype sample in the final partition. Its
  receding macro is computed over the two represented event strata, consistent
  with the main evaluator, and equals `0.794282` versus static `0.898495`.
  Future aggregation must exclude absent strata instead of averaging their
  sentinel `inf` values.

### 2026-08-26 | V139 exposes deterministic policy collapse

- Generic PD-PPO beat the best AoI, round-robin, or random policy in `5/5`
  development scenes, with mean ordinary margin `+0.032821`.
- It beat validation-selected static on ordinary/macro loss in `2/5` and `3/5`
  scenes, with only `2/5` joint wins. Mean margins were
  `-0.003344/+0.025271`, so fresh confirmation is not authorized.
- Only `1/5` scenes passed the basic six-channel behavior gate. Training action
  entropy remained high, but deterministic execution made one channel always on
  in four scenes and produced up to three always-off channels. The failure is
  localized to action-value separation and deterministic decoding, not the
  feasibility projector or conventional-dynamic comparison.

### 2026-08-26 | V140 rejects temperature sampling as a standalone repair

- Validation-only selection over deterministic execution and temperatures
  `0.1`, `0.25`, `0.5`, `0.75`, and `1.0` improved behavior to `3/5` scenes but
  reduced held-out performance to `1/5` joint static wins.
- Mean ordinary/macro margins became `-0.011795/-0.001873`; all five policies
  still beat the best conventional dynamic reference.
- Post-training sampling is closed. V141 instead sharpens the forecast-value
  target and weakens entropy regularization during training on the two hardest
  development scenes.

### 2026-08-26 | V141 sharpens action values but remains behavior-limited

- On seed1502, temperature `0.25` and entropy coefficient `0.005` changed the
  ordinary/macro static margins from `-0.014601/-0.000517` to
  `+0.000102/+0.017699`. The selected checkpoint moved from update 0 to update
  20 and no channel was always on.
- Seed1505 improved from `-0.012909/-0.032847` to
  `-0.011123/-0.013798`, but retained one always-on and one always-off channel.
  Both seeds continued to beat the best conventional dynamic reference.
- Target sharpness is directionally useful but insufficient. V142 performs one
  final same-objective density test on seed1505: on-policy forecast-value labels
  every four steps with direct standardized-value regression. Failure closes
  scalar tuning and requires a structural actor/value redesign.

### 2026-08-26 | V142 closes scalar forecast-value tuning

- Dense on-policy value regression reduced the auxiliary loss to `0.750971`
  with a 25% label rate and increased switching to `0.031119` per step.
- It nevertheless lost to selected static by `-0.019464/-0.043004` on
  ordinary/macro loss and retained one always-on and one always-off channel.
- Temperature, entropy, label density, and value-regression scaling are closed.
  V143 replaces diffuse value matching with masked hard-action supervision from
  the same frozen forecast evaluator on policy-visited training states. No
  subtype labels, bandit actions, or final-partition feedback are introduced.

### 2026-08-26 | V143 rejects hard teacher loss on shared actor logits

- Seed1505 lost to static by `-0.022025/-0.005542` on ordinary/macro loss.
  Hard forecast-teacher supervision reduced switching to `0.002605` and yielded
  one always-on, two always-off, and no mid-duty channels.
- Increasing teacher pressure on the shared PPO logits is closed. The actor now
  receives a separate forecast-value action head whose output is added to PPO
  residual logits before feasibility masking. Forecast supervision updates only
  this head; PPO gradients cannot overwrite its action-value output directly.

### 2026-08-26 | V144 isolates candidate representation as the next bottleneck

- The factorized forecast head reduced value-regression loss to `0.656085`, but
  seed1505 still lost to static by `-0.021223/-0.046992` and selected one
  always-on plus three always-off channels.
- A leave-one-scene-out capacity audit on the two hardest scenes found 33.7%
  top-1 teacher-action accuracy for the factorized mask embedding and 42.2% for
  an independent candidate head at dwell-free decisions. V145 changes only the
  forecast head to independent candidate outputs; the PPO actor retains its
  compositional mask embedding and hard feasibility mask.

### 2026-08-26 | V145 restores behavior but motivates multi-scene training

- The independent candidate head removed always-on channels on seed1505 and
  produced five mid-duty channels, one always-off channel, and a switching rate
  of `0.010855` per step.
- Forecast performance still trailed the validation-selected static subset by
  `-0.021706/-0.067269` on ordinary/macro loss. Deployable TCN forecast
  summaries also failed to improve cross-scene teacher-action prediction over
  the existing online state.
- Single-scene fitting is therefore the next controlled bottleneck. V146 carries
  one policy through training scenes 1501--1504 and evaluates the frozen result
  on held-out scene1505. No final-partition metric is used between stages.

### 2026-08-26 | V146 rejects sequential multi-scene curriculum

- The frozen policy carried through scenes1501--1504 failed on held-out
  scene1505. Ordinary/macro margins against selected static were
  `-0.044570/-0.109172`, and ordinary loss was `0.011141` worse than the best
  AoI, round-robin, or random schedule.
- Held-out behavior remained feasible and dynamic: no channel was always on or
  off, five channels had mid-range duty, switching was `0.018092` per step, and
  no warm-up abort occurred. The failure is cross-scene transfer, not action or
  constraint collapse.
- Sequential curriculum is closed because it exposes the policy to one scene at
  a time and permits catastrophic forgetting. V147 interleaves four training
  scenes at episode boundaries while sharing one model and optimizer, then
  freezes the result for scene1505. Rewards, online inputs, and feasibility
  rules remain unchanged.

### 2026-08-26 | V147 validates interleaving but misses one held-out endpoint

- Episode-level interleaving completed 81,920 PPO steps over scenes1501--1504
  with one shared model and optimizer. The frozen policy improved held-out
  seed1505 substantially over V146.
- Against selected static, the ordinary margin was `-0.008223` while the macro
  margin was `+0.014095`. PD-PPO beat the best conventional dynamic schedule by
  `+0.025206` ordinary loss.
- Behavior passed with zero always-on/off channels, five mid-duty channels,
  `0.026849` switches per step, and zero warm-up aborts. Interleaving is retained;
  V149 adds checkpoint selection aggregated over the four training scenes'
  calibration/validation partitions instead of freezing the final update.

### 2026-08-26 | V149 isolates behavior-aware validation as the missing gate

- Four-scene validation selected update 40 (`40,960` steps), minimizing the
  maximum of the mean ordinary and macro static ratios at `1.021842`.
- On held-out scene1505, ordinary/macro margins against validation-selected
  static were `-0.000315/+0.012189`; PD-PPO beat the best conventional dynamic
  schedule by `+0.033114` ordinary loss.
- The selected policy had one always-on channel, two always-off channels, only
  two mid-duty channels, `0.010494` switches per step, and zero warm-up aborts.
  It therefore failed the frozen six-channel behavior gate even though its
  predictive performance nearly passed.
- V150 retains V149 training, reward, online inputs, and feasibility rules. It
  selects checkpoints lexicographically by the number of validation scenes
  failing the predeclared behavior gate and then by the existing two-endpoint
  static-ratio score.

### 2026-08-26 | V150 rejects behavior-aware selection as a transfer repair

- Behavior-aware validation selected update 15, which passed the frozen gate
  on all four training-scene validation sets with score `1.035765`.
- On held-out scene1505, the selected policy lost to static by
  `-0.017231/-0.062708` on ordinary/macro loss. It retained a
  `+0.016197` ordinary margin over the best conventional dynamic schedule.
- Held-out execution had one always-on and three always-off channels, two
  mid-duty channels, `0.004632` switches per step, and zero aborts. Sparse
  validation behavior therefore does not transfer reliably across scenes.
- No threshold or held-out-specific tuning is authorized. V151 runs all five
  leave-one-scene-out folds under the identical behavior-aware protocol to
  measure whether multi-scene transfer is reproducible before fresh evaluation.

### 2026-08-26 | V151 closes generic multi-scene transfer

- Five leave-one-scene-out folds completed under identical four-scene training
  and behavior-aware validation selection. Ordinary/macro static wins were
  `1/5` and `2/5`; behavior passed `3/5`; no fold passed the joint gate.
- Mean ordinary/macro margins against validation-selected static were
  `-0.009487/-0.014571`. PD-PPO retained `4/5` wins over the best conventional
  dynamic schedule with mean margin `+0.026678`.
- Every selected checkpoint had zero behavior failures on its four validation
  scenes, but this did not predict held-out behavior or static-relative loss.
  Multi-scene training and checkpoint selection are therefore closed.
- The next scene must expose dynamic value through physically available online
  signals. Sensor self-diagnostic quality and condition-dependent reliability
  will be screened before any further PPO training.
## 2026-08-26 - Preserve sensor-quality configuration in receding diagnostics

- Stopped the first V152 receding-oracle wave after auditing its command line:
  the generic diagnostic environment had silently fallen back to unit sensor
  quality even though the frozen runs contained online quality signals.
- Replaced four partial `WarmupEnvConfig` reconstructions with dataclass copies,
  so every environment field is preserved while only the rollout seed and
  length change.
- Propagated frozen quality columns, noise scaling, and availability scaling
  from run metadata into the receding diagnostic. The complete 133-test suite
  passes. The invalid receding outputs were incomplete and are excluded; the
  completed V152 scene and one-step baseline artifacts remain valid.
## 2026-08-26 - Establish horizon-matched value in the V152 quality scene

- Corrected five-seed diagnostics show that the online one-step
  forecast-greedy policy does not beat the validation-selected static schedule
  (`0/5` ordinary and `1/5` macro wins).
- The eight-step receding forecaster diagnostic beats static in both endpoints
  for all five seeds. Mean ordinary and static-normalized macro margins are
  `+0.026327` and `+0.110197`; all six channels have intermediate duty in every
  seed, with no always-on/off channels or warm-up aborts.
- This separates horizon-matched dynamic headroom from myopic value. Added a
  reproducible V152 collector and authorized one bounded complete-PD-PPO pilot;
  no quality-specific reward, action prior, or heuristic imitation was added.
## 2026-08-26 - Complete the first V153 quality-aware PD-PPO pilot

- Seed 1601 completes with ordinary/macro margins versus the selected static
  schedule of `+0.000872/+0.031229`. Deployment behavior passes with zero
  always-on, one always-off, four mid-duty channels, nonzero switching, and no
  warm-up aborts.
- The pilot remains slightly behind AoI and round-robin in ordinary forecast
  loss. Checkpoint traces show stable validation selection but weak mapping of
  22 horizon-value action targets (14.8% pretraining top-1 accuracy and policy
  entropy near the 22-action maximum).
- Authorized the remaining four baseline seeds plus one bounded comparison
  using the existing independent forecast-value head. The comparison preserves
  the forecast reward, PPO objective, feasibility mask, and training-only
  information boundary.
## 2026-08-26 - Reject V153 and localize quality-context representation

- The complete five-seed V153 wave passes deployment behavior `5/5` but beats
  the selected static schedule in only `1/5` seeds. Mean ordinary/macro margins
  are `-0.036214/-0.122366`; ordinary wins against the best conventional
  dynamic policy are `2/5`.
- Selected channels have higher reported quality on average (`+0.062709`), so
  the policy detects quality but does not map it reliably to horizon-optimal
  masks. The factorized forecast-value-head comparison is also worse on the
  shared pilot seed and is rejected.
- The architecture audit found that the six quality signals were encoded by the
  generic runtime branch while the dedicated context encoder still consumed
  only the 20 alert features. A bounded representation wave now tests a
  26-feature quality-plus-alert context tail with three existing supervision
  forms; no new reward or baseline-dependent module is introduced.
## 2026-08-26 - Select the quality-context representation for behavior audit

- V155, which places six quality signals beside the 20 alert features in the
  dedicated context encoder, is the only bounded seed1601 variant to improve
  both endpoints over static (`+0.008424/+0.055835`) and ordinary loss over the
  best conventional dynamic (`+0.002307`).
- The independent forecast-value head and hard-classification variants are
  rejected because their ordinary margins remain negative. V155 itself is not
  accepted because the prediction-only selector chose a checkpoint with three
  always-off channels.
- The V155 validation ledger contains one behavior-valid checkpoint at update15.
  V158 repeats the identical deterministic training with the previously frozen
  behavior-valid checkpoint constraint; this tests selection transfer without
  adding a loss, guard, or action prior.
## 2026-08-26 - Pass the V158 behavior-selected pilot

- Repeating V155 with the frozen behavior-valid checkpoint rule selects update15
  exactly as predicted by the validation ledger. Seed1601 retains positive
  static margins (`+0.003464/+0.018083`) while improving deployment behavior to
  zero always-on, one always-off, four mid-duty channels, nonzero switching,
  and zero aborts.
- Ordinary loss remains `0.002654` behind the best conventional dynamic on this
  seed. The complete V158 configuration is now frozen while seeds1602--1605 run;
  no further variant selection is allowed from seed1601.
## 2026-08-26 - Reject V158 and audit validation headroom

- Frozen V158 beats static in only `1/5` seeds on both endpoints. It beats the
  best conventional dynamic in `3/5` seeds with mean ordinary margin
  `+0.001875`, while deployment behavior transfers in `4/5` seeds.
- Every selected checkpoint still has a validation max-static ratio above one.
  V159 therefore evaluates the same eight-step receding diagnostic on the
  validation partition. This distinguishes policy learnability from a missing
  model-selection signal before any further architecture change.
## 2026-08-26 - Confirm validation headroom and revise supervision ranking

- V159 evaluates the eight-step receding diagnostic on the validation
  partition. It beats that partition's best static candidate in both endpoints
  for all five seeds, with mean ordinary/macro margins
  `+0.051181/+0.104938`; behavior passes `5/5`.
- Dynamic value is therefore present in both validation and final partitions.
  The remaining failure is the policy's mapping from horizon costs to ranked
  actions. Full-vector squared-error supervision is not aligned with top-action
  selection when many candidate costs are close.
- V160--V161 use the existing soft forecast-value target with temperatures 0.25
  and 0.5, respectively, while retaining context-dim 26 and behavior-valid
  checkpoint selection. No new architecture, reward term, or information is
  introduced.

## 2026-08-26 - Reject soft action-ranking variants and repair comparison fairness

- V160 (temperature 0.25) retained valid behavior but lost ordinary forecast
  loss to static by `0.019139`; V161 (temperature 0.5) lost both endpoints and
  produced two always-off channels. Both soft-target variants are rejected.
- Auditing the candidate teacher exposed an experimental-protocol defect. With
  common random numbers disabled, different sensor masks consumed different
  random observation draws, so horizon-cost labels included action-dependent
  sampling-path noise. The static-candidate evaluator also reconstructed an
  environment config manually and omitted the new sensor-quality fields.
- Added explicit common-random-number propagation across training, frozen
  replay, operational baselines, and receding diagnostics. Static candidate
  evaluation now copies the complete frozen environment config. V152--V161
  remain useful architecture diagnostics, but their quality-scene rankings are
  not confirmatory evidence. V162 regenerates all five scenes under the
  corrected protocol before any further policy selection.

## 2026-08-26 - Pass the corrected V162 dynamic-value gate

- V162 regenerates five quality-varying scenes with common random numbers and
  complete sensor-quality configuration in static selection. The eight-step
  receding diagnostic beats the selected static schedule on both endpoints in
  all five validation and all five final partitions.
- Mean validation margins are `+0.047193` ordinary and `+0.107315` macro; mean
  final margins are `+0.032779` and `+0.120173`. All six channels have
  intermediate duty, nonzero switching, and zero warm-up aborts in every run.
  One-step forecast greedy passes only `2/5`, confirming that the headroom is
  horizon-dependent.
- An online learnability audit recovers positive alert-context gain in all five
  seeds but only about 14.6% of receding headroom. V163 therefore keeps the
  dedicated 26-feature alert-plus-quality encoder and behavior-valid checkpoint
  selection; it changes no reward, feasibility constraint, or runtime input.

## 2026-08-26 - Reject V163 after corrected-label policy training

- V163 passes the deployment-behavior gate in all five scenes and selects
  higher-quality channels on average (`+0.059931` selected-minus-unselected
  quality), but it does not recover the V162 dynamic headroom. It records
  `0/5` ordinary and `1/5` macro wins against static, with mean margins
  `-0.052749/-0.128358`.
- Seed1602 exhibits a large validation-to-final transfer failure, while the
  checkpoint ledger shows no behavior-valid validation checkpoint below the
  static reference. The failure is therefore not repaired by longer training
  or post-hoc checkpoint choice.
- V164 is a single-seed bounded retest of the existing factorized
  forecast-value action head. The retest is justified because V154 used the
  now-invalid action-dependent labels. Replication is allowed only if V164
  improves both static endpoints and retains valid behavior.

## 2026-08-26 - Reject V164 and revise quality-state persistence

- The corrected-label factorized forecast head fails seed1601, with ordinary
  and macro static margins `-0.047745/-0.297162`, two always-off channels, and
  weak quality-conditioned duty. The head is rejected without replication.
- The quality generator currently places abrupt, independently random
  degradation intervals. A receding diagnostic observes those future truth
  values during simulation, while an online scheduler cannot predict an unseen
  onset. This inflates upper headroom without making all of it learnable.
- V165 tests a physical persistence correction on fresh development seeds:
  degradation duration changes from 12--48 to 48--96 hourly steps and minimum
  separation to 24 steps. Coverage, quality severity, action geometry, reward,
  CRN protocol, and all weather/event settings remain fixed. No policy training
  is allowed until receding and online learnability gates are recomputed.

## 2026-08-26 - Close persistence-only tuning and add aligned quality scoring

- V165's longer quality states preserve dynamic headroom but do not solve
  learnability. The receding diagnostic records `5/5` ordinary and `4/5` macro
  wins; the best online ExtraTrees audit is positive in `4/5` seeds and
  recovers 20.7% of receding gain. Persistence alone is therefore closed.
- The actor audit identified a representation mismatch: six ordered channel
  health values were mixed as generic context although each value corresponds
  exactly to one bit of every candidate mask. Added an optional aligned
  mask-quality score that computes selected-channel health for each candidate
  and adds it to masked actor logits through a learned positive scale.
- The option preserves arbitrary feasible subsets, the forecast-loss reward,
  PPO updates, and hard feasibility masking. It is disabled by default and
  covered by the 136-test suite. V166 tests only seed1601 before replication.

## 2026-08-26 - Reject additive quality scoring and retest corrected AWBC labels

- V166 increases the selected-minus-unselected quality gap to `+0.177258` and
  passes behavior, but loses ordinary/macro forecast loss to static by
  `0.025820/0.066192`. A health-only additive preference cannot represent the
  interaction between channel condition and forecast-task value, so it is not
  replicated as the primary method.
- V167 returns to the unchanged context26 actor and enables the framework's
  existing on-policy oracle-guided AWBC at coefficient 0.1 and stride 4. This
  is a bounded retest because previous oracle-action labels were generated
  before CRN correction. Runtime inputs, reward, action space, and feasibility
  rules are unchanged; replication again requires both endpoints and behavior.

## 2026-08-27 - Reject corrected oracle-guided AWBC

- V167 completes the bounded seed1601 retest under common random numbers. It
  loses to the validation-selected static schedule by `0.034277` ordinary loss
  and `0.090708` macro loss. It also loses to the best conventional dynamic
  schedule by `0.012748` and to the eight-step receding diagnostic by
  `0.023669/0.067354` on the two endpoints.
- The rollout has zero warm-up aborts and nonzero switching, but one channel is
  always off (`0/1/5` always-on/always-off/intermediate-duty channels). Its
  selected-minus-unselected quality gap is only `+0.088604`, below the rejected
  aligned-score pilot despite the added supervision.
- The corrected teacher labels therefore do not rescue the existing AWBC
  pathway. V167 is not replicated, and bandit- or teacher-dependent residual
  modules remain excluded from the primary method. The next clean direction
  must model the interaction between channel condition and forecast-task
  context, or make quality transitions causally predictable from online
  diagnostics, while preserving arbitrary feasible subsets.

## 2026-08-27 - Launch gradual channel-quality gate

- The quality generator previously changed each channel directly between full
  and degraded quality. This made degradation onset invisible to an online
  scheduler before the first affected observation, while the receding
  diagnostic could inspect the future quality path.
- Added a default-off transition interval that linearly decreases and restores
  channel quality around a degraded plateau. It models a progressive
  self-diagnostic change without exposing future labels or changing the policy
  input contract. The option is recorded in truth metadata and propagated
  through the split protocol and training manifest.
- V168 evaluates five fresh seeds with an eight-step transition, 24--64-step
  quality episodes, and a 16-step minimum gap. All weather, event, cost,
  arbitrary-subset, reward, and common-random-number settings remain fixed.
  Policy training is blocked until receding headroom and online learnability
  are recomputed. The implementation passes the 137-test suite.

## 2026-08-27 - Calibrate gradual-quality integrated severity

- V168's eight-step quality transitions preserve macro dynamic value in `5/5`
  seeds but produce only `3/5` ordinary-loss wins over static. Mean receding
  margins are `+0.010567` ordinary and `+0.060221` macro, with valid six-channel
  dynamic behavior in all seeds. The scene therefore fails the dual-endpoint
  headroom gate and no PPO is trained on V168.
- Gradual quality is more learnable from online state. A complete-state ridge
  model fitted on policy-training plus validation traces has positive final
  gain in `5/5` seeds and recovers 32.2% of receding gain on average. Complete-
  state ExtraTrees also passes `5/5`, recovering 22.2%.
- The transition reduced integrated degradation severity: for the mean 44-step
  episode, two eight-step ramps are equivalent to about eight fewer fully
  degraded steps. V169 raises degraded coverage from 0.25 to 0.31, the rounded
  ratio needed to restore the original integrated deficit. All other V168
  settings remain fixed, and fresh seeds are used before any policy training.

## 2026-08-27 - Reject coverage-based severity compensation

- V169 restores dynamic headroom: the receding diagnostic records `5/5`
  ordinary and `4/5` scoreable macro wins over static, with mean margins
  `+0.029862/+0.131290` and valid six-channel behavior in every seed. One-step
  greedy remains below static on average.
- The higher degradation coverage destroys online learnability. On validation-
  only traces, no model reaches more than `3/5` positive-gain seeds; complete-
  state regressors have negative mean gain, and the best alert-only mean gain
  is only `+0.000626`. V169 is therefore rejected before PPO training.
- V170 returns to V168's 0.25 coverage and transition timing, preserving the
  more identifiable channel-state distribution. It compensates transition-
  diluted physical severity through the observation model instead: maximum
  noise amplification changes from 6.0 to 7.0 and the availability floor from
  0.20 to 0.05. This is the bounded alternative to increasing simultaneous
  degradation coverage; all other settings remain fixed on fresh seeds.

## 2026-08-27 - Pass V170 gates and launch clean context26 PD-PPO

- V170 passes the horizon-matched scene gate with `5/5` ordinary and `4/5`
  scoreable macro wins over static. Mean receding margins are
  `+0.026084/+0.090136`; all six channels have intermediate duty and all five
  seeds have nonzero switching with zero warm-up aborts. One-step greedy loses
  to static on average, preserving the sequential-decision requirement.
- Validation-only alert/context models achieve positive final gain in `5/5`
  seeds. ExtraTrees recovers 16.6% and histogram gradient boosting 15.2% of
  receding gain on average. V170 therefore passes the online-information gate.
- V171 trains the existing context26 PD-PPO on the five V170 scenes. It keeps
  forecast-loss reward, arbitrary feasible subsets, temporal/context encoders,
  forecast-value pretraining and auxiliary loss, and behavior-valid checkpoint
  selection. AWBC, bandit priors, residual actions, aligned quality scoring,
  subtype labels, and extra reward terms remain disabled.

## 2026-08-27 - Reject V171 and add task-conditioned channel utility

- V171 does not convert V170's learnable headroom into policy performance. It
  records `0/5` wins over static on both endpoints, with mean margins
  `-0.026309/-0.092166`; only `2/5` seeds pass the behavior gate. Higher-quality
  channels receive more duty on average, but this health preference is not
  conditioned strongly enough on the current forecast task.
- Added an optional task-conditioned channel-utility term to the masked actor.
  Online alert features produce six nonnegative channel utilities, each utility
  is modulated by its corresponding online quality value, and every arbitrary
  candidate subset receives the mean utility of its selected channels. The
  term is learned jointly with the existing PPO actor and is disabled by
  default.
- V172 is a bounded seed1701 pilot on the frozen V170 scene. It changes only
  this actor representation. Reward, forecast-value supervision, runtime
  information, action geometry, feasibility masking, and checkpoint rules are
  unchanged; replication requires both static margins and valid behavior.

## 2026-08-27 - Reject V172 and densify forecast-value supervision

- V172 improves seed1701 over V171 but remains below static by
  `0.007992/0.051692` on ordinary/macro loss and leaves one channel always off.
  It is not replicated. The best validation checkpoint is also above static,
  so final checkpoint transfer is not the primary failure.
- The task-conditioned utility increases the selected-minus-unselected quality
  gap to `+0.113456`, but forecast-value auxiliary cross-entropy remains near
  the 22-action uniform level (`2.999` versus `ln(22)=3.091`) and actor entropy
  remains `2.999`. Only 6.25% of states receive auxiliary labels at stride 16.
- V173 retains the V172 architecture and changes one training mechanism:
  forecast-value label stride decreases from 16 to 4 and its existing loss
  coefficient increases from 0.5 to 1.0. This fourfold denser supervision is a
  bounded seed1701 test; reward, PPO objective, runtime inputs, and constraints
  remain unchanged.

## 2026-08-27 - Reject dense soft targets and test standardized cost regression

- V173 narrows seed1701's ordinary static deficit to `0.002844`, but the macro
  deficit remains `0.052453` and one channel remains always off. Increasing the
  label rate to 25% leaves auxiliary cross-entropy at `2.989`; the soft target
  remains too close to uniform for accurate action ranking.
- The existing MSE path uses per-state standardized negative horizon costs,
  not raw unscaled costs. It therefore preserves relative action-value spacing
  that soft probabilities compress. V174 changes only the forecast-value
  auxiliary loss from soft cross-entropy to this standardized-cost regression,
  retaining V173's label stride, coefficient, actor, PPO reward, and seed1701
  gate.

## 2026-08-27 - Bound V174 and isolate the factorized value head

- V174 is the closest pilot in this wave: ordinary loss is `0.000061` better
  than static, while macro loss remains `0.018495` worse and one channel is
  always off. No validation checkpoint beats static on the macro selection
  ratio, so the pilot is not replicated.
- Standardized-cost MSE substantially changes the auxiliary signal, but direct
  regression on actor logits still couples PPO ranking and cost calibration in
  the same output. V175 enables the existing factorized forecast-value head,
  trained with the same MSE labels and candidate-mask embeddings. Its detached
  value logits guide the actor while PPO retains its own logits and reward.
  This single-seed test preserves arbitrary-subset generalization and changes
  no runtime information or feasibility rule.

## 2026-08-27 - Reject detached factorized guidance and reduce excess entropy

- V175 passes the six-channel behavior gate but sharply degrades prediction,
  losing to static by `0.079192/0.305252`. The detached factorized value logits
  overemphasize channel quality (`+0.157967` selected-quality gap) without
  preserving forecast-task ranking. The separate-head route is rejected.
- V174 remains the best policy structure, but its actor entropy is `2.999`,
  close to the 22-action maximum despite standardized-cost supervision. V176
  returns to direct V174 MSE supervision and changes only the ordinary PPO
  entropy coefficient from 0.02 to 0.002. This tests whether persistent
  near-uniform exploration prevents the learned ranking from controlling the
  deterministic policy; all evidence and replication gates remain unchanged.

## 2026-08-27 - Close entropy tuning and retain V170 as the scene candidate

- V176 loses to static by `0.027007/0.162898`; lowering the entropy coefficient
  does not reduce actor entropy, which remains `3.004`. Exploration weighting
  is therefore not the source of the near-uniform action distribution.
- Across V171--V176, V174 is closest to static but does not pass the macro or
  six-channel behavior requirement. None of these policy variants is eligible
  for replication or paper evidence. V170 remains the strongest calibrated
  scene candidate because it passes horizon headroom, online-information, and
  dynamic-behavior gates without constraining the action space to three choices.
- The remaining bottleneck is mask-level forecast-cost fitting. The 4096-state,
  20-epoch warm start ends at MSE `0.906` and top-action accuracy `0.110` on
  seed1701. Further work should redesign or separately validate the
  action-conditioned cost regressor and its sampling distribution before more
  PPO waves; ordinary entropy, teacher weighting, and scene severity tuning are
  closed.

## 2026-08-27 - Validate a shared mask-level cost model before PPO integration

- Added an offline action-conditioned forecast-cost regressor whose parameters
  are shared across sensor channels and candidate subsets. The model consumes
  online context, the proposed sensor mask, and normalized steady/startup cost;
  it does not assign an independent output parameter to each of the 22 actions.
- V177 used channel quality plus alert context. It improved mean top-action
  accuracy to `0.1381` and predicted 21 actions on average, but beat the
  validation-selected static action in only `3/5` scenes and recovered `7.5%`
  of receding headroom. It failed the frozen offline gate.
- V178 bounded four representation/target combinations. Alert context alone
  with per-state standardized costs and ranking weight `0.25` passed `5/5`,
  with mean static gain `+0.005547`, top-1/top-3 action accuracy
  `0.1565/0.3453`, and `17.8%` mean receding-gain recovery. Removing ranking
  loss reduced top-1 accuracy; global scaling reduced robustness to `4/5`;
  forcing channel quality into this teacher head again reduced transfer to
  `3/5`.
- V178 is a validation-to-final development diagnostic, not a legal policy
  training result. Policy-training receding traces must reproduce the mapping
  on validation and final partitions before the shared mask model can be
  connected to PPO.

## 2026-08-27 - Establish legal partition transfer for mask-cost supervision

- V179 generated 2,048 receding-action states on each seed's policy-training
  partition. These traces enable action-cost fitting without using validation
  or final targets as training labels.
- The first V180 aggregation incorrectly selected its static reference on the
  regressor-training partition. That comparison is invalid because the frozen
  protocol selects the static reference on validation. The audit now accepts
  an explicit static-selection partition; corrected evidence uses validation.
- With the V178 structure fitted only on policy-training data, corrected V180
  obtains positive final gain over validation-selected static in `5/5` seeds,
  with mean `+0.006541` and `21.2%` receding-headroom recovery. Validation is
  weaker at `3/5`, and only `3/5` scenes use all six channels, so this exact
  configuration is not selected for PPO.
- V181 varies only ranking weight, target scaling, and the already audited
  online quality input on validation. Ranking weight `0.1` is the strongest
  alert-only result (`4/5`, mean `+0.000558`), while direct quality fusion
  passes behavior `5/5` but loses prediction on average. Higher ranking weight,
  global scaling, and no ranking loss do not improve the joint gate.
- V182--V184 separate online warning context from a monotonic sensor-quality
  degradation term. The best validation result reaches prediction `4/5` and
  behavior `4/5`; the sole behavior failure is one channel at duty `0.00195`
  in seed1702, while the sole prediction loss is `-0.000671` in seed1704.
  Per-channel quality scales do not remove this trade-off. Offline quality
  tuning is closed.

## 2026-08-27 - Connect shared mask-cost supervision to complete PD-PPO

- Added a `mask_structured` forecast-value head to the existing PD-PPO actor.
  It predicts standardized action value from online quality/warning context,
  candidate sensor masks, and normalized steady/startup costs with parameters
  shared across subsets. Forecast-head logits remain detached when added to
  policy logits, preserving the PPO objective and gradient boundary.
- Existing factorized and independent heads remain unchanged. Focused gradient,
  feasibility-mask, arbitrary-action-order, and monotonic-quality tests pass;
  the full suite passes 142 tests.
- A 128-state, one-update remote dry run completed the full split-protocol
  training, checkpoint, rollout, and metrics chain with zero constraint
  violations or warm-up aborts. V185 is the bounded seed1701 full pilot; it is
  eligible for replication only if it beats static on both endpoints and passes
  the six-channel behavior gate.

## 2026-08-27 - Reject MSE-only mask-value integration and align ranking loss

- V185 improves the seed1701 pretraining fit from the old fixed-logit result:
  MSE falls from `0.906` to `0.8515`, top-action accuracy rises from `0.110` to
  `0.168`, and all 22 actions occur in the teacher batch. Actor entropy also
  falls from about `3.01` to `2.82` during PPO, confirming that the detached
  head changes policy preferences.
- Better fit does not transfer to prediction. V185 final ordinary loss is
  `0.313432` versus static `0.281203`; static-normalized macro is `1.122544`
  versus `0.972963`. One channel remains always off. Validation checkpoint
  scores are above static at update0 and every PPO checkpoint, so selection is
  not the failure source. V185 is rejected and not replicated.
- The passing offline regressor used Smooth-L1 cost fitting plus a best-action
  ranking term, whereas V185 used MSE alone. Added default-off
  `forecast_value_ranking_coef` and Smooth-L1 support to both pretraining and
  on-policy auxiliary updates. V186 changes only these matched loss terms
  (`smooth_l1`, ranking coefficient `0.1`) and is the final bounded test of the
  on-policy mask-value-head route.

## 2026-08-27 - Reject quality-coupled ranking and restore the validated alert-only head

- V186 improves the seed1701 static-normalized macro loss from V185's
  `1.122544` to `1.089740`, but remains worse than the validation-selected
  static schedule (`0.972963`). Its ordinary forecast loss is `0.308100`
  versus static `0.281203`, and one of six channels remains always off. V186
  therefore fails both prediction endpoints and the behavior gate and is not
  replicated.
- Every V186 validation checkpoint remains worse than static. The selected
  update20 score is `1.100183`, only marginally below the update0 score
  `1.101160`; checkpoint selection is not the failure source. Smooth-L1 plus
  ranking raises pretraining top-action accuracy to `0.196`, but policy entropy
  remains close to the 22-action maximum.
- The audit found a structural mismatch: the passing V178/V180 offline
  regressor used alert context only, whereas the embedded V185/V186 head was
  forced to consume six quality values with a monotonic degradation term.
  V181--V184 had already shown that this quality coupling harms prediction
  transfer. Added an opt-in `forecast-value-head-ignore-quality` path that
  removes quality only from the auxiliary mask-cost head while leaving the
  complete actor observation unchanged.
- V187 changes only this integration mismatch relative to V186. It is a bounded
  seed1701 pilot and may be replicated only if both forecast endpoints beat the
  validation-selected static schedule and all six channels have nondegenerate
  duty.

## 2026-08-27 - Close the detached mask-head route and sharpen direct policy ranking

- V187 fails more strongly than V186: seed1701 ordinary loss is `0.357604`
  and static-normalized macro loss is `1.270302`, versus static
  `0.281203/0.972963`. It leaves one channel effectively always off. The best
  validation checkpoint also remains above static, so the alert-only detached
  head is rejected without replication.
- The discrepancy between offline replay and V187 establishes that single-step
  action-cost transfer is insufficient under the closed-loop observation
  distribution. Quality coupling was a real integration mismatch, but removing
  it does not solve the policy problem. The detached mask-cost-head route is
  therefore closed.
- V174 remains the closest complete closed-loop policy: it narrowly beats
  static on ordinary loss (`+0.000061`) and misses the macro endpoint by
  `0.018495`, but its direct policy-logit MSE fit has only `0.110` top-action
  accuracy. V188 retains V174's actor, quality utility, PPO reward, scene,
  partitions, and 20-epoch warm start, changing only the direct auxiliary loss
  to Smooth-L1 plus a `0.1` best-action ranking term. Replication uses the same
  dual-endpoint and six-channel behavior gate.

## 2026-08-27 - Reject weak direct ranking and gate stronger imitation before PPO

- V188 raises pretraining top-action accuracy only from V174's `0.110` to
  `0.140`. Its final ordinary loss is `0.292141` and static-normalized macro
  loss is `1.041553`, versus static `0.281203/0.972963`; one channel remains
  always off. The `0.1` ranking term is rejected without replication.
- The teacher batch is generated by executing the eight-step receding teacher,
  so insufficient fit, not teacher-state covariate shift, remains measurable at
  initialization. V189 is a bounded pretraining-only sweep over ranking
  coefficient `0.25/1.0` and `20/80` epochs. No PPO update is run. A full
  training wave is allowed only if a validation-qualified variant materially
  improves teacher-action accuracy and the dual static ratios without behavior
  failure.

## 2026-08-27 - Reject stronger cost ranking and isolate hard teacher classification

- V189 increases teacher-action accuracy monotonically with training strength:
  `0.163/0.285` for ranking `0.25` at `20/80` epochs and `0.197/0.316` for
  ranking `1.0`. None beats static on either endpoint. The strongest fit has
  ordinary loss `0.343697` and macro `1.162808`; its six-channel behavior is
  valid, but the weaker fits leave one channel always off.
- Higher ranking accuracy does not improve forecast performance because the
  same logits are simultaneously calibrated to continuous standardized costs
  and classified by best action. Cost-regression/ranking strength tuning is
  closed.
- V190 isolates the existing hard teacher-action cross-entropy warm start at
  `80/200` epochs with no PPO updates or cost-regression loss. This is the final
  bounded teacher-fit diagnostic. Full PPO training is allowed only if the
  pretraining policy passes both static endpoints and six-channel behavior on
  validation-qualified evaluation.

## 2026-08-27 - Reject hard teacher fitting and invalidate the V170 deployability gate

- V190 hard teacher cross-entropy reaches recorded training accuracy
  `0.317/0.460` at `80/200` epochs and both policies use all six channels.
  Neither transfers: the 80-epoch policy has ordinary/macro loss
  `0.306487/1.086553`, and the 200-epoch policy has
  `0.324520/1.159181`, versus static `0.281203/0.972963`. No full PPO run is
  launched. Teacher-fit strength tuning is closed.
- The V170 gate had relied on a privileged receding oracle and one-step models
  evaluated on its visited states. V191 adds the missing closed-loop test: a
  validation-mapped context policy using only supplied noisy warning scores.
  It beats static in only `2/5` scenes on ordinary loss and `1/5` on macro,
  with mean margins `-0.008293/-0.032554`. Several scenes retain always-on or
  always-off channels.
- V170 therefore has privileged dynamic headroom but does not establish that
  the headroom is reachable by a deployable online policy. PPO architecture
  tuning on V170 is stopped. Subsequent scene candidates must first pass a
  closed-loop online context-and-quality policy gate on both endpoints and
  six-channel behavior before any PPO training.

## 2026-08-27 - Reject the V170 online gate and decouple events from sensor availability

- V192 evaluates a quality-aware context policy that combines validation-only
  regime costs, supplied warning scores, and reported online channel quality.
  Penalties `0.25/1.0/4.0` all record `0/5` wins over static on both endpoints;
  mean ordinary margins are `-0.014371/-0.021158/-0.023166` and mean macro
  margins are `-0.058579/-0.084665/-0.086860`. V170 is rejected as a deployable
  scene despite its privileged receding headroom.
- The audit identifies a physical confound in event generation. Particle-event
  assignment required Parsivel availability of at least `0.8`, so the simulated
  physical event subtype depended on whether an observation device happened to
  be available. Final evaluation then contained only `0--71` particle steps per
  seed, versus hundreds of flux and thermal steps.
- V193 removes this dependency by setting the particle-assignment availability
  threshold to zero and otherwise retains V170's costs, quality process,
  warning process, action geometry, and event parameters. Fresh seeds
  `1901--1905` must pass balanced subtype coverage, privileged receding
  headroom, and the closed-loop online context/quality gate before PPO training.

## 2026-08-28 - Close duration-balanced scene and alert-only mapping variants

- V213 replaces run-count subtype stratification with duration-balanced
  allocation while keeping complete event runs indivisible, exogenous, and
  physically interpreted. The all-action eight-step receding diagnostic beats
  validation-selected static in all five fresh development scenes, with mean
  ordinary-loss margin `+0.032106`.
- The deployable context-alert policy remains below static in four scenes:
  `1/5` joint wins and mean ordinary/macro margins `-0.002672/-0.009736`.
  It has zero warm-up aborts but averages `0.8` always-on and `2.8` always-off
  channels. The scene is not authorized for PD-PPO training.
- V214 is a zero-training, validation-replay diagnostic using the same online
  warnings but separate low/high alert actions at fixed thresholds `0.50/0.75`.
  It also reaches `1/5` joint wins and worsens mean ordinary/macro margins to
  `-0.006862/-0.023686`. Alert-bin rule expansion is closed; no additional
  heuristic thresholds or cost/budget sweeps follow from this result.

## 2026-08-28 - V219 rejects minimal PPO training with label-free nowcasts

- V219 adds only three noisy, four-step meteorological nowcasts (wind, relative
  humidity, and air temperature) to the policy state.  Synthetic event alerts,
  labels, and the inert zero-coverage channel-quality tail are excluded.  The
  context encoder receives exactly these three trailing state dimensions.
- Two initially launched configurations were stopped and excluded before
  aggregation because the generic runner overrode the intended context width.
  The accepted run records `context_feature_dim=3` in every seed metadata file
  and has no quality or alert context columns in its execution command.
- The bounded `1,024`-step PPO screen loses to validation-selected static in
  all five development scenes: mean static-minus-PPO ordinary-loss margin
  `-0.170605` and macro static-normalized margin `-0.481659`.  This closes the
  minimal-training variant; it is not evidence that the nowcast scene has no
  dynamic value.
- The paired all-action eight-step receding diagnostic confirms dynamic
  headroom in all five scenes: its executed policy beats validation-selected
  static by mean ordinary-loss margin `+0.028369` and uses `17` feasible
  actions in every trace. V221 therefore tests partition-normalized nowcasts
  with 50,000 PPO steps; it changes neither reward nor feasibility geometry.

## 2026-08-28 - V221 narrows but does not close the policy gap

- V221 standardizes the three nowcast inputs with statistics from the declared
  normalization partition and trains PPO for 50,000 steps on fresh development
  seeds `2411--2415`. It retains the same forecast reward, feasible-subset
  geometry, and label-free online state as V219.
- PPO still loses to validation-selected static in all five scenes: mean
  ordinary and macro static-minus-PPO margins are `-0.062611` and `-0.200999`.
  It is materially less unstable than V219 and has zero aborts, zero always-on
  channels, zero always-off channels, and five or six mid-duty channels in each
  seed, but these behavioral checks do not substitute for prediction wins.
- The matched l8 receding diagnostic again records five ordinary-loss wins,
  mean margin `+0.024957`, and 17 actions per trace. V222 is restricted to a
  label-free l4 trace-learnability audit before any new PPO objective or teacher
  component is considered.

## 2026-08-28 - V222 confirms label-free action-value learnability

- V222 reconstructs normalized nowcast state in receding traces, fits only on
  policy-training plus validation traces, and evaluates on held-out final
  traces. Alert-feature-only rows are diagnostic-only because they are not part
  of the deployed V221 state.
- For the actual complete online state, ridge action-cost regression recovers
  mean l4 gain `+0.014906` over the validation-selected static action in `4/5`
  final scenes (`25.95%` of receding gain). ExtraTrees recovers `+0.013522` in
  `4/5`. The normalized weather context is therefore informative enough to
  support a forecast-action-value training auxiliary; the prior PPO failure is
  not treated as a scene rejection.

## 2026-08-28 - V223 closes forecast-value pretraining as a primary path

- The first V223 launch was invalid before PPO training: forecast-value
  pretraining generated all-NaN teacher costs whenever the AWBC teacher was not
  `oracle_greedy`. The defect was fixed in `55849da` so that forecast-value
  targets are collected independently of the AWBC teacher; the full test suite
  passed before rerunning the same three development seeds.
- The corrected configuration keeps the label-free normalized three-nowcast
  state, forecast-loss reward, and 17-mask feasibility geometry. It adds only a
  frozen-forecaster factorized action-value head, 4,096 forecast-value
  pretraining samples, and an on-policy auxiliary target every 32 steps. It
  does not add alerts, event labels, heuristic actions, or a bandit-dependent
  prior.
- The rerun is behaviorally valid but does not clear static. Against the
  validation-selected static reference, ordinary and macro margins are
  `-0.041380/-0.086823`, `-0.025783/-0.065766`, and
  `-0.025656/-0.067967` for seeds `2421--2423`: `0/3` wins on both endpoints
  and mean margins `-0.030940/-0.073519`. It improves over round-robin in
  `2/3` and random in `2/3` ordinary-loss comparisons, but that does not meet
  the static gate.
- This closes forecast-value pretraining as a clean primary-method escalation.
  The remaining design question is scene-side: whether the physically grounded
  cost calibration and label-free meteorological precursor yield enough online
  action-value separation for a policy to surpass a validation-selected static
  schedule without importing a heuristic action or privileged test label.

## 2026-08-28 - V224 closes direct actor forecast-value warm start

- V224 corrects V223's separated-head limitation without changing the online
  state, candidate masks, feasibility rules, or forecast-loss reward. The same
  frozen-forecaster l4 targets now directly pretrain and regularize the
  categorical PPO actor logits on fresh seeds `2431--2433`; the configuration
  contains no bandit action, alert/event label, or test-time privileged input.
- The direct warm start improves conventional-dynamic comparisons: it beats
  round-robin and random on both endpoints in `3/3` seeds and AoI in `2/3`.
  However it remains below validation-selected static in every seed. Ordinary
  and macro margins are `-0.041550/-0.127899`, `-0.008765/-0.016285`, and
  `-0.023904/-0.091003`, with means `-0.024740/-0.078396` and `0/3` wins.
- All runs have zero warm-up aborts and zero always-on channels; one seed has
  one always-off channel. Direct forecast-value warm start is therefore closed
  as the final teacher-style escalation. The next screen changes no learner
  component and instead tests whether the operational forecast lead aligns with
  the l8 dynamic-value horizon established by the receding diagnostic.

## 2026-08-28 - V225 confirms l8 scene headroom but rejects unassisted PPO

- V225 changes only the legal weather-forecast lead from four to eight steps,
  with larger forecast errors `(1.4 m/s, 4.2 %, 1.0 C)`. It retains the same
  17-mask feasible-subset geometry, PPO configuration, and forecast-loss reward
  without a teacher, rule action, or event label.
- The unassisted PPO policy is below validation-selected static in all three
  seeds on both endpoints (mean ordinary/macro margins
  `-0.050207/-0.126608`) and also loses to all conventional dynamic references.
  All six channels retain intermediate duty in two runs and five in the third;
  there are no aborts or constant-on channels.
- The paired l8 receding diagnostic nevertheless beats the matched static
  reference by `+0.034489`, `+0.030323`, and `+0.058509` ordinary loss. The
  problem is therefore not absence of dynamic value but a mismatch between the
  l8 value horizon and the l4 value targets used in prior actor training. A
  single horizon-aligned actor-value screen is authorized before altering the
  physical scene or adding any new policy component.

## 2026-08-28 - V226 closes horizon-aligned actor supervision

- V226 aligns the label-free eight-step meteorological forecast, direct actor
  value pretraining, and on-policy frozen-forecaster target at l8. It keeps the
  same masked PPO policy, reward, and feasible-subset geometry, and introduces
  no rule, bandit, or event-label execution input.
- The policy beats AoI, round-robin, and random on both endpoints in `3/3`
  seeds. It is nevertheless only `1/3` against validation-selected static:
  mean ordinary/macro margins are `-0.012284/-0.013871`. Two seeds each retain
  two always-off channels, while all runs have zero aborts and no always-on
  channels.
- Horizon alignment materially improves the dynamic-policy comparison but does
  not satisfy the static or six-channel behavior gate. All teacher-style and
  forecast-horizon training escalations are closed. Further work must change
  the physical effective-cost calibration or the observable dynamic
  measurement-value process, not stack supervision onto PPO.

## 2026-08-30 - V234 rejects the physical crossover scene before PPO

- V234 introduces a development-only, weather-conditioned crossover reliability
  model for the five independently powered physical instrument groups. The
  action space, fixed effective costs, budget, evaluator, and training method
  remain unchanged; no policy is trained in this screen.
- The exact receding diagnostic confirms substantial latent dynamic opportunity:
  it beats the validation-selected static schedule on both endpoints in `5/5`
  scenes, with mean ordinary and macro margins of `+0.027655` and `+0.073701`.
  All runs use all 15 feasible masks, have five intermediate-duty groups, and
  have zero warm-up aborts.
- The deployable quality-only context policy fails the corresponding online
  gate (`0/5` ordinary and `1/5` macro wins; mean margins
  `-0.014230/-0.047484`). Its behavior is fixed-like, with two always-on and
  three always-off groups. V234 therefore does not establish an online
  information path and is closed without PPO training or parameter tuning.

## 2026-08-30 - V235 closes strong crossover calibration before PPO

- V235 strengthens only the weather-conditioned reliability profiles for the
  five physical instrument groups. Costs, budget, arbitrary-subset action
  geometry, evaluator, nowcast, and PPO configuration remain unchanged.
- Exact receding remains positive in `5/5` scenes, with mean ordinary/macro
  margins `+0.018448/+0.048317`; behavior passes `4/5`.
- The online quality policy wins only `1/5` ordinary and `1/5` macro scenes,
  with mean margins `-0.016178/-0.047618`. The stronger degradation lowers
  overall observation quality without yielding a stable online action path.
  V235 is closed without PPO; the next scene must preserve aggregate quality
  while creating relative, weather-observable pairwise reliability crossovers.
- V236 centers the five weather-conditioned exposure profiles around a fixed
  per-step mean while retaining the physical groups, fixed effective loads,
  budget, arbitrary-subset geometry, evaluator, and PPO configuration. Exact
  receding beats validation static in `5/5` scenes, with mean ordinary/macro
  margins `+0.026761/+0.074941`; behavior passes `4/5`. The online quality
  policy wins only `2/5` ordinary and `2/5` macro scenes, with mean margins
  `-0.006133/-0.017271`. V236 is closed without PPO; a forecastable lead
  signal must couple directly to instrument-specific value before training.

## 2026-08-30 - V237/V238 close the forecast-quality route before PPO

- V237 added legal future-quality features derived from noisy weather nowcasts.
  After fixing a policy-construction bug, the online quality policy won only
  `1/5` ordinary and `0/5` macro comparisons; the receding diagnostic passed
  `5/5` (`+0.026383/+0.064991`). Its first artifacts also exposed an
  independent normalization defect and are diagnostic only.
- V238 corrected forecast-quality normalization and reran fresh seeds
  `2901--2905`. The online policy still won only `1/5` ordinary and `1/5`
  macro comparisons, with mean margins `-0.006143/-0.011151`. The receding
  diagnostic passed `5/5` (`+0.043366/+0.109945`), but this is privileged
  latent opportunity, not a deployable policy result.
- The corrected eight-step forecast-quality correlations remained mixed:
  `0.394/-0.189/0.537/0.375/-0.525` for GMX500/LPS10/SI-111/Parsivel 2/FC4.
  The quality-only route is closed; no larger PPO run is justified by it.

## 2026-08-30 - V239 invalidated and balanced quality generation corrected

- A source audit found that the balanced actual-quality branch still fell
  through to the historical latent-event profile, so V239's actual and
  forecast quality columns described different processes. Its partial runs
  were stopped and are retained only as invalid diagnostics.
- The branch condition is corrected in `1e45d75`, with a regression test that
  identical nowcast and realized weather produce identical quality profiles.
  V240 must regenerate fresh scenes before this route can be evaluated.

## 2026-08-30 - V240 validates the corrected scene but not PPO learnability

- V240 regenerated the balanced weather-conditioned quality scene for fresh
  seeds `3101--3105` after the V239 source-consistency fix. The five channel
  quality correlations with their weather drivers were `0.79--0.91`, so the
  intended observable quality process is present and forecastable.
- The receding oracle passed the latent-opportunity screen in `5/5` scenes,
  with mean ordinary and macro margins `+0.037456/+0.101319` against the
  validation-selected static schedule. This is a privileged diagnostic, not
  deployable policy evidence.
- The online quality-only context policy failed the screen (`0/5` ordinary,
  `1/5` macro; mean margins `-0.012995/-0.028251`). The PPO rows also failed
  in all five scenes, but this run used only `1,024` training steps and one
  update; it is an under-training admission screen, not a valid conclusion
  about PPO capacity. All PPO rows had zero warm-up aborts and no always-on or
  always-off channels, with three to five mid-duty channels.
- V240 therefore closes scene validation successfully and defers the policy
  decision to a normally trained confirmation on the same corrected scene.

## 2026-08-30 - V241 rejects the stripped PPO configuration

- V241 reran the corrected balanced quality scene with fresh seeds `3201--3205`
  and `100,000` PPO timesteps per seed. The flexible arbitrary-subset action
  geometry and physical effective costs were unchanged.
- The stripped configuration lost to validation static in `0/5` ordinary and
  macro comparisons, and lost to the original dynamic baselines in `0/5`
  ordinary and macro comparisons. Mean losses were `0.392407` for PPO,
  `0.292391` for validation static, and `0.326913` for AoI; mean staticnorm
  macro losses were `1.037785`, `0.754716`, and `0.832621`, respectively.
- PPO beat random in `4/5` comparisons, but this is not the target evidence.
  Behavior gates passed in all five runs: zero warm-up aborts, zero always-on
  and always-off channels, and `4--5` mid-duty channels.
- V241 is therefore a configuration rejection, not a scene rejection. The
  corrected scene has demonstrated latent dynamic opportunity, but the
  complete PD-PPO training scaffold (online alert context, subtype-aware
  auxiliary supervision, and training-only action guidance) must be evaluated
  before concluding that the method cannot learn the flexible subset task.

## 2026-08-30 - V242 restores the complete training scaffold

- V242 evaluates the corrected balanced quality scene on fresh seeds
  `3301--3305` for `100,000` timesteps, restoring online alert context,
  subtype-aware auxiliary supervision, and training-only AWBC/action guidance.
  Final execution still uses no simulator event labels.
- The complete configuration beats AoI on the macro endpoint in `5/5` seeds
  and on the ordinary endpoint in `2/5`; it beats round-robin on macro in
  `3/5` and random in `4/5`. It beats validation-selected static in only
  `1/5` ordinary and `1/5` macro comparisons, with mean margins
  `-0.015147/-0.008286`. The mean PPO ordinary/macro losses are
  `0.475125/0.862288` (computed from the five raw per-seed files).
- Behavior is feasible (zero warm-up aborts), but not yet clean: mean
  switching rate is `0.028919`, mean always-on count `0.2`, mean always-off
  count `1.0`, and mean mid-duty count `3.8`. V242 therefore confirms that the
  training scaffold is required, but does not pass the static-shortcut and
  no-permanent-off gates. Further work must target action-value representation
  or the physical value/cost mapping, not claim completion.
## 2026-08-30 - V243 isolates the training-teacher effect

- Added a five-seed, 100,000-timestep development launcher for the corrected
  balanced quality scene. V243 retains the forecast-loss reward, online weather
  and alert context, subtype auxiliary prediction, arbitrary feasible subsets,
  and all execution constraints.
- V243 removes AWBC, behavior-cloning pretraining, and subtype teacher actions
  as a clean structural ablation of V242. It does not use bandit loss, labels,
  residual actions, or test-time event information.
- Results must be compared against V242 before any claim about action-value
  representation or permanent channel-off behavior is made.

## 2026-08-30 - V247 admits the six-channel dynamic scene

- V247 generated a fresh five-seed (`3701--3705`) scene with the physical
  six-channel configuration, arbitrary feasible subsets, budget `1.75`,
  startup budget `2.15`, minimum dwell `6`, and condition-dependent balanced
  channel quality. Final execution used online alert context only; simulator
  event labels were not supplied to the deployable policies.
- The latent dynamic-opportunity diagnostic was positive in all five scenes:
  the eight-step receding oracle beat the validation-selected static schedule
  on both ordinary loss and static-normalized event macro loss in `5/5` seeds,
  with mean margins `+0.043113` and `+0.127733`, respectively.
- The receding policy passed the behavior gate in `5/5`: six intermediate-duty
  channels, zero always-on channels, zero always-off channels, positive
  switching rate, and zero warm-up aborts. It covered `22/23` candidate masks
  in every seed.
- The online context-alert bandit was not uniformly stronger than static:
  it beat static in `2/5` ordinary and `2/5` macro comparisons, with mean
  margins `-0.000343` and `+0.000101`. This confirms that the scene has
  dynamic headroom without making every dynamic comparator trivially win.
- The gate aggregation exposed and fixed a legacy five-channel assumption in
  `scripts/107_v32_collect_physical_quality_gate.py`. The script now infers
  the expected intermediate-duty count from `action_geometry.json`, records
  switching rate, and requires positive switching for the behavior pass.
- Aggregate evidence is stored in
  `reports/aggregate/v247_sixch_admission_gate_20260830/`. The V247 scene is
  admitted for a complete six-channel PPO wave; V247 itself is a scene gate,
  not evidence that PPO has already learned the opportunity.

## 2026-08-30 - V248 rejects the first complete six-channel PPO configuration

- V248 ran the complete training scaffold on fresh seeds `3801--3805` using
  the V247-admitted six-channel configuration, B=`1.75`, startup budget
  `2.15`, minimum dwell `6`, and `100,000` PPO timesteps per seed.
- The run retained forecast-loss reward, online alert context, subtype
  auxiliary supervision, and training-only AWBC/teacher guidance. No event
  labels or action-value head were used at final execution.
- PD-PPO did not beat the validation-selected static schedule: ordinary wins
  `0/5` and macro wins `1/5`, with mean margins (static - PD-PPO)
  `-0.029549/-0.031311`. It did beat the best original dynamic heuristic
  family in `4/5` seeds on both endpoints, with mean margins
  `+0.013856/+0.010058`.
- The behavior gate failed structurally: `radiometer_basic` had zero duty in
  all five runs; always-off counts were `1,2,2,2,2`, always-on counts were
  `0,0,1,1,0`, and mid-duty counts were `5,4,3,3,4`. Warm-up aborts were zero.
- V248 is therefore a complete-policy rejection, not a scene rejection. The
  next bounded development test changes only the physically meaningful solar
  target weight to test whether the radiometer channel has sufficient forecast
  value under the current objective.
- Aggregate evidence is stored in
  `reports/aggregate/v248_full_pdppo_sixch_dev_20260830/`.
