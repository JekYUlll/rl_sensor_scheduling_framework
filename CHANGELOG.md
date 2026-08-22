# PD-PPO Scene Recalibration Changelog

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
