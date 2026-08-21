# Findings: ESWA Terminology and Narrative Rewrite

## 2026-07-03 Feature-Parity / CA-PD-PPO Direction
- The new algorithm task is not a paper-writing pass. It targets the strongest
  current challenge case: `context_alert_bandit_t0p5` beat PD-PPO on average in
  the framework supplement while PD-PPO still strongly beat the one-step forecast
  greedy diagnostic.
- The method boundary is strict: the primary method may use online alert/context
  observables and an internal context encoder, but it must not optimise against
  bandit margins, imitate the bandit, act as a residual over the bandit, or use a
  bandit-dependent actor prior.
- The first technical risk to resolve is fairness of information. If the bandit
  sees alert scores or alert-derived state that original PD-PPO lacks, then the
  current comparison is partly an observation-parity mismatch rather than a clean
  algorithmic defeat.
- Code audit result: the context-alert bandit is not using future forecast loss
  and is not using the simulator `event_subtype_id` directly. Its online
  observables are the three alert proxy scores in the truth CSV. Its advantage
  comes from hand-engineered thresholding plus validation-selected subtype masks.
- Original PD-PPO had partial context support via `agent_context_columns`, but
  did not expose derived features equivalent to the bandit's thresholded alert
  decision state: binary alert flags, argmax event-type one-hot, max alert
  confidence, alert onset age, rolling alert trend, previous specialist, and
  minimum-on-time remainder.
- The new primary-method implementation keeps the original forecast-loss reward
  and feasibility-masked categorical action distribution. It does not add
  bandit-margin reward, residual bandit actions, bandit imitation, or
  counterfactual improvement labels.

## ESWA Style Anchors
- ESWA scheduling papers commonly frame the work as a named scheduling problem
  plus a learning-based scheduling policy.
- The paper titles and abstracts use domain nouns directly:
  "task scheduling", "job-shop scheduling", "production plan", "distributed
  satellite system", "deep reinforcement learning".
- Method-specific names are introduced after the problem has been made clear.
- Baselines are described as scheduling policies or dispatching rules, not as
  internal experiment variants.
- The manuscript should therefore avoid version labels, run nicknames, and
  repository-specific shorthand in the main narrative.

## Current Manuscript Issues
- The abstract uses several internal terms before they are explained:
  frozen oracle, deployment constraints, compact static, deployable static,
  FW-MAE, static replay.
- The Introduction still reads partly like a posthoc experiment report: it
  explains why some baselines are strong before the general scheduling problem is
  stable.
- The Method section uses code-derived names (`candidate prior`, `AWBC`,
  `oracle-label`) without enough ESWA-style explanation.
- Results mix baseline labels (`compact static`, `deployable static`, `best duty`)
  with mechanism terms, making the table hard to read without project memory.

## Rewrite Direction
- Lead with the general problem:
  power-constrained sensing-system scheduling for time-series prediction.
- Define one stable vocabulary for the entire paper:
  sensing channel, schedule/action mask, fixed forecast evaluator,
  operational constraints, static subset baseline, static-priority baseline.
- Move Antarctic AWS/blowing snow into the benchmark description after the
  general method is introduced.

## Manuscript Terminology Audit
- The active English draft now uses "prediction-driven sensing-system
  scheduling" as the paper-level problem name.
- "Fixed forecast evaluator" replaces the earlier "frozen oracle" wording in
  reader-visible text.
- "Operational constraints" replaces deployment-specific phrasing and is defined
  in terms of power, minimum on-time, duty cycle, and switching limits.
- "Sensing channel" is used for the eight logical data streams, while "sensor"
  is reserved for instrument-level discussion where appropriate.

## Evidence Framing
- The main result remains positive against dynamic heuristic and
  duty-constrained non-PD-PPO baseline families.
- The static subset baseline is framed as a strong fixed-design reference, not
  hidden or treated as a weak comparator.

## Specialist-Bottleneck Theory Application
- The active manuscript can now present SCENEBAL-2 as an instance of a broader
  forecast-relevant specialist-bottleneck class rather than as an isolated
  calibrated scene.
- The theory is a sufficient-condition headroom argument: mandatory backbone,
  `r<K` specialist slots, positive-weight regimes, incompatible regime-best
  specialists, and positive mismatch loss imply that true fixed-static
  specialist allocation has strictly higher macro forecast loss than an ideal
  regime-aware dynamic policy.
- The theory must remain bounded. It supports the need for dynamic scheduling in
  SCENEBAL-2-like settings; it does not prove PPO global optimality, and it does
  not establish a universal result for arbitrary power-constrained sensing
  systems.
- The appendix now also explains why a simple cycle is not equivalent to
  state-dependent scheduling unless regimes are deterministic and phase-aligned
  with that cycle.

## Current Claim and Archive Boundary
- The strongest supported ESWA claim is tied to the final fixed-backbone,
  one-specialist 24-seed aggregate over seeds `117--140`, ordinary step
  improvement, true-static gates, behaviour gates, and static-normalised
  event-regime macro scoring.
- Unsupported or non-headline boundaries are not part of the active manuscript
  narrative. The raw unnormalised macro diagnostic and weaker-latent robustness
  failure remain internal audit evidence unless a later revision explicitly asks
  for a limitations appendix.
- The previous GitHub release is historical. Before submission, a new versioned
  archive should pin the exact final-benchmark code revision, aggregate tables,
  seed-level summaries, figure assets, and reproduction scripts.

## New-Claim Check
- The active manuscript and PDF are now on the new final-benchmark claim. The
  submission-facing path uses `24/24` seeds, true fixed-static replay,
  static-normalised event-regime macro scoring, behaviour-complexity auditing,
  and concise mechanism/robustness checks.
- The only residual old-claim issues found in this check were outside the main
  PDF text or in a stale sentence: active highlights still said `18 seeds`, and
  the problem-formulation section still called the stricter step-weighted
  true-static comparison a limitation. Both were corrected.
- Historical files may still exist in the repository, but the active manuscript
  path no longer has the old 18-seed / SCENEBAL-1 / V3.1 / metpair wording.

## Figure-Count Repair
- The older rewrite contained more figures, but several were tied to obsolete
  10-seed compact/deployable-static or fixed-budget evidence. Restoring them
  directly would weaken consistency with the current claim.
- The active repair is to keep only visuals that match the current final
  benchmark. The unsupported diagnostic figure was removed from the main
  Results narrative; the behaviour-audit and 24-seed evidence figures remain.

## Reader-Facing Terminology Boundary
- `SCENEBAL-2` should be treated as an internal experiment/archive identifier,
  not as a concept that readers must parse in the manuscript narrative.
- The paper body, highlights, captions, and visible figure text should describe
  the evidence as the final fixed-backbone, one-specialist benchmark or the
  regime-balanced benchmark.
- This keeps the claim intelligible to ESWA readers while preserving the exact
  internal identifier in file names, scripts, aggregate directories, labels, and
  the future reproduction archive.

## Supplementary Experiment Priority
- The current evidence gap is not primarily seed count. The final 24-seed
  benchmark already supports a bounded strong claim, while pure same-protocol
  seed expansion would not explain the mechanism or reduce the concern that the
  scene is over-specialised.
- The next required evidence is a final-protocol mechanism ablation. It should
  keep PPO, the met+one-specialist sensing geometry, strict no-duty-guard
  true-static replay, behaviour auditing, and static-normalised macro scoring,
  while removing or weakening current full-model components such as
  advantage-weighted imitation, regime/auxiliary routing paths, and the
  static-normalised training objective.
- A small robustness block should follow: perturb event mix, subtype latent
  separation, or specialist calibration/noise by modest amounts while
  preserving the one-specialist structure. This supports a broader ESWA
  reliability argument without inventing a new sensor system.
- Larger same-protocol fresh-seed expansion is now a secondary confirmation
  option, not the first next experiment.

## Mechanism Ablation Early Evidence
- The `no_imitation_guide` six-seed ablation passed all current operational,
  replay, behaviour, true-static, and static-normalised macro gates. This should
  be interpreted as robustness to removing the advantage-weighted imitation
  guide, not as evidence that the guide is required.
- The immediate implication is that the mechanism section should avoid a
  component-necessity claim for the imitation guide unless the full-vs-ablated
  margin comparison later shows a consistent benefit. The more defensible
  statement is currently that PD-PPO's positive final-benchmark result is not an
  artifact of a fixed-sensor shortcut and is also not solely dependent on
  imitation labels.
- The ongoing `no_regime_aux_path` and `no_staticnorm_train` ablations are more
  important for deciding whether the paper can claim that regime-aware routing
  support and the benchmark-aligned macro objective materially contribute to the
  final policy.
- The `no_regime_aux_path` ablation provides the first clear component-level
  evidence in this pilot. It keeps the coarse operational result positive
  (`6/6` step and macro gates, `6/6` behaviour, `6/6` true-static macro), but it
  loses the all-seed strict true-static step guarantee (`5/6`, seed `122`
  fails). The strongest mechanism phrasing should therefore be about reliability
  of the strict fixed-static comparison, not about regime support being required
  for every weaker aggregate.
- The `no_staticnorm_train` ablation did not degrade the current protocol:
  operational, replay, behaviour, true-static macro, and strict true-static
  step gates remained `6/6`. This rules out a strong necessity claim for the
  static-normalised training objective in the six-seed pilot. Its role should be
  described more cautiously as an objective/evaluation-alignment choice and a
  metric-boundary clarification.
- The paired full-reference re-audit over the same six seeds also passed all
  gates. Mechanism interpretation should therefore focus on `no_regime_aux_path`
  as the clear degradation signal, while treating `no_imitation_guide` and
  `no_staticnorm_train` as robustness-to-removal findings unless later expanded
  runs show consistent margin changes.
- Final threshold-sensitivity rows (`0.0`, `0.7`, and `0.9`) all passed the
  six-seed current protocol gates. The manuscript can say the pre-fixed
  deployment threshold is not an obvious tuning artifact in the pilot, but
  should not imply an exhaustive threshold sweep.
- The final mechanism story is therefore asymmetric: regime/subtype-aware
  policy support has clear evidence because removing it breaks the all-seed
  strict fixed-static step gate, while the imitation guide and static-normalised
  training objective do not show necessity in this pilot.

## Robustness Pilot Design
- The first robustness block should use fresh seeds `141--146` rather than
  extending the same 24-seed main benchmark. This reduces the risk that the
  robustness evidence is only a re-evaluation of known seeds.
- The first two perturbations should remain close to the current physical
  benchmark: an event-mix shift (`particle/flux/thermal = 0.35/0.30/0.35`) and a
  moderate subtype-latent weakening. These test whether the policy survives
  small changes in event prevalence and signal separability while preserving
  the one-specialist mechanism.
- Do not treat robustness perturbations as a new main claim. Their role is to
  support reliability of the one-specialist interpretation under mild simulator
  changes. They must still report true-static replay, behaviour complexity, and
  static-normalised macro scoring.

## Robustness Pilot Early Evidence
- The event-mix perturbation (`particle/flux/thermal = 0.35/0.30/0.35`) passed
  all six fresh seeds `141--146` under the current protocol: operational step,
  true-static macro, strict true-static step, and behaviour gates were all
  `6/6`.
- This is the first direct evidence that the final specialist-budget result is
  not tied only to the exact original event-type prevalence. It remains a
  mild robustness result, not a new universal generalisation claim.
- The weaker-latent perturbation is informative but not claim-upgrading:
  operational step remains `6/6`, but behaviour, true-static macro, and strict
  true-static step are only `5/6`. Seed `144` is the shared failure.
- Seed `144` is not a fixed sensor or simple cycle, but its top two masks cover
  almost all steps and switching is very sparse. The next useful layer is
  action-level subtype auxiliary support or router/memory architecture, not a
  third scenario-only perturbation.

## Manuscript Application on 2026-06-22
- The active paper now includes only supported positive evidence: the 24-seed
  final benchmark, a six-seed mechanism check, and a six-seed event-mixture
  robustness check.
- The paper does not include the raw unnormalised macro boundary, the
  weaker-latent failure, threshold-sensitivity rows, or the unfinished
  action-level auxiliary pilot.
- Reader-visible internal terms were removed from active source and generated
  PDF text: `SCENEBAL`, `metpair`, `router`, `specialist-budget`,
  `specialist-bottleneck`, `deployment threshold`, `subtype`, `weaker`, and
  `actionaux`.

## Figure and Table Readability Audit
- Active figures use mixed styles: TikZ method figures use pastel boxes and
  serif labels, generated result figures use small matplotlib fonts and long
  in-figure titles, and the generator-validation figure uses a dense six-panel
  raster layout.
- Figure 1 is structurally useful but too dense at final print size; the note
  line and some box text are small.
- Figure 2 is readable but visually inconsistent with Figure 1 and the
  matplotlib figures.
- Figure 4 is the least readable active figure: six panels plus table on one
  page make axis labels and annotations small.
- Figure 5 still had a stale in-figure title and crowded gate labels; the seed
  ticks and minimum-margin annotation also add clutter.
- Figure 6 has its legend over the plotted data and uses too much figure width
  for redundant 24/24 gate bars.
- Tables 1, 3, 4, and 5 are readable enough, but captions are too long and
  several labels read like experiment logs rather than manuscript tables.

## Figure and Table Style Decisions
- The key readability issue was not only font size inside Matplotlib. The old
  result figures were created at about `6.9` inches wide and then scaled down by
  LaTeX to the ESWA single-column text width, making final text around 7 pt or
  smaller. Active data figures should be generated near `5.35` inches wide so
  they are embedded close to 1:1 scale.
- Do not put manuscript-level titles inside figures. Captions already provide
  the narrative context, and in-image titles easily become stale when the claim
  changes.

- Gate figures should use short axis labels and rely on captions/tables for the
  full definitions. This is preferable to multi-line y-axis labels that become
  unreadable in the PDF.
- Avoid separate legends when they collide with data or panel titles. Direct
  line-end labels worked better for the behaviour audit figure.
- Generator validation should not combine a dense checklist and six small plots
  in one figure. The manuscript now uses a four-panel visual summary plus the
  validation table for detailed checks.
- Tables should avoid `resizebox` unless there is no alternative. Ragged
  fixed-width text columns keep the final font readable and avoid stretched word
  spacing.

## PPO-GPT / PPO-LEMMA Review Application Findings
- The original paper wording had a real method-risk: it described channel scores
  followed by deterministic projection while the PPO objective used a probability
  ratio as if the executed action had a tracked log probability. The current
  implementation and manuscript are now aligned by treating the action as the
  selected candidate-mask index under a masked categorical distribution.
- The final benchmark should continue to be written as one instantiation of a
  broader framework, not as a universal claim about all power-constrained
  sensing systems. The one-specialist setting is useful because incompatible
  event regimes make a constant specialist insufficient.
- The six-seed mechanism and event-mixture checks are supporting diagnostics.
  They should not be upgraded to headline robustness claims unless expanded.
- Event-type labels are not online final-test policy inputs. The paper should
  continue to distinguish station-side event-context proxies from simulator
  event labels used for auxiliary training, grouping, and diagnostics.
- The static-reference-normalised macro score remains benchmark-specific. The
  paper now reports continuous margins and seed-level distributions alongside
  win counts, which is more defensible for ESWA than gate-only reporting.

## Final Review Closure Findings
- The third wording/terminology audit initially found only two visible P2
  issues: duplicated appendix prefixes and the obsolete `FW-MAE` abbreviation.
  After correction and recompilation, the final subagent recheck reported no
  remaining P1/P2 issues.
- The active paper now consistently presents PD-PPO as masked categorical PPO
  over feasible candidate masks. The old projected/channel-score PPO wording
  should be treated as obsolete.
- The current manuscript claim remains intentionally bounded to the fixed
  evaluation-forecaster protocol and the one-specialist benchmark, with
  mechanism and event-mixture results framed as pilot/supporting evidence.

## 06-23-02 Strict-Pass Operating Finding
- The prior 06-23 pass is not assumed complete for `docs/06-23-02-PPO-LEMMA.md`.
  The new pass must build a fresh requirement matrix from that document, then
  verify each high-priority experiment or analysis item against concrete
  completed artifacts before editing the paper or claiming completion.
- Parsed `docs/06-23-02-PPO-LEMMA.md` into
  `reports/aggregate/ppo_lemma_062302_requirement_matrix_20260623.md`.
  The high-priority blockers are: 24-seed mechanism ablation with continuous
  margins, event-type by specialist expert-selection heatmap / distributional
  behaviour diagnostics, and explicit `L_event` formula with unused `L_guide`
  removed from the main loss equation.
- Existing 24-seed main benchmark rollouts are present for seeds `117--140`;
  these can support the expert-selection heatmap and other behaviour figures
  without new remote training.
- Existing mechanism ablation artifacts cover only seeds `117--122`; R1.1
  requires a new 24-seed remote run or a future completed artifact.

## 06-23-02 Strict-Pass Evidence Findings
- The local high-priority writing issues are now real fixes rather than prose
  assertions: the main PPO loss excludes the unused guide-prior term, the
  event-context auxiliary loss is explicitly defined, and the guide prior is
  only documented as an optional appendix regulariser with zero weight in the
  reported 24-seed protocol.
- The event-type behaviour figure is a better evidence match than the old scalar
  behaviour audit. It shows both seed-level complexity distributions and which
  specialist channels are selected under non-event, particle, flux, and thermal
  contexts.
- The fixed TCN forecaster validation benchmark is not positive evidence. Under
  ordinary validation nMAE, persistence beats the saved TCN diagnostic, so the
  paper must not use this item to strengthen the scheduler claim.
- The manuscript still contains six-seed mechanism text by design while the
  required 24-seed mechanism ablation is running. Any final claim upgrade must
  wait for the full `20260623b` collection and continuous macro-margin summary.
- The PPO training histories are useful for diagnostics but not for a validation
  convergence claim. They contain training loss, entropy, imitation label rate,
  and auxiliary accuracy, but no validation macro score or validation step loss
  over training steps.
- A threshold-rule or contextual-bandit baseline would be a useful future
  deployment comparison, but it needs station-specific thresholds or labels to
  avoid becoming another hand-tuned diagnostic. The current paper can discuss
  this omission without adding a weak or under-specified baseline.
- Figure 5 is more defensible when Panel A shows both ordinary step margin and
  macro margin by seed. This prevents the main figure from looking like a
  step-only claim while the text also relies on macro scoring.
- The updated `20260623b` first-batch no-event-context ablation does not
  reproduce the older six-seed 5/6 strict-step degradation. It passes 6/6 in the
  first batch. This weakens any mechanism-necessity wording and makes the
  continuous 24-seed margin comparison more important than pass-count phrasing.
- The completed 24-seed mechanism ablation resolves the instability of the old
  six-seed pilot. Full PD-PPO passes the strict fixed-mask step criterion in
  24/24 seeds. Removing the imitation guide lowers this to 23/24 and removing
  the event-context auxiliary signal lowers it to 21/24, while all variants
  retain 24/24 macro and behaviour gates. Removing the balanced training loss
  still passes strict step in 24/24 seeds.
- Continuous macro-margin deltas against Full PD-PPO are small and their
  bootstrap intervals overlap zero: no imitation `-0.0020 [-0.0103, 0.0061]`,
  no event-context auxiliary `-0.0045 [-0.0130, 0.0037]`, and no balanced
  training loss `-0.0007 [-0.0071, 0.0059]`. The correct manuscript reading is
  therefore reliability/stability support for the auxiliary training signals,
  especially event-context auxiliary training, not a claim that each component
  produces a large independent macro-margin gain.

## 06-23-02 Final Closure Findings
- The high-priority 06-23-02 requirements are now satisfied after subagent
  recheck: 24-seed mechanism ablation with continuous mean and median macro
  margins, 24-seed behaviour distribution plus event-type specialist heatmap,
  and explicit `L_event` with the unused guide-prior term moved out of the main
  loss equation.
- The final Table 11 must be kept in sync with
  `reports/aggregate/mechanism_ablation_continuous_margins_24seed_20260623b/`.
  A previous intermediate version missed the median macro-margin CI even though
  the aggregate contained it; the table generator now includes this field.
- `paper/figures/gen_fig_framework_and_support.py` should remain scoped to
  Figure 1. Historical six-seed mechanism plotting logic was removed because it
  could overwrite the current 24-seed mechanism evidence if the script were run
  during future figure refreshes.
- Figure 1 is acceptable for the current closure after spacing fixes, but its
  arrows are still dense. Further visual refinement is optional polish, not an
  evidence or claim blocker.
- Table 11 wraps the long phrase "No event-context auxiliary signal" in the
  rendered PDF. This is a narrow-table formatting compromise and not a content
  blocker.
- The stronger event-mix variants requested as low priority remain outside the
  current claim. The manuscript must not turn the six-seed changed-mixture check
  into a broad robustness claim.

## 06-23-03 Review-Pass Findings
- Table 11's content is correct but the single-row nested-cell layout is a real
  readability risk. The next version should use two physical rows per policy
  variant: one for point estimates and one for 95% intervals.
- The active behaviour figure already contains an event-type by specialist
  heatmap, but the title and caption say only "specialist use by event type."
  For ESWA readability, it should explicitly say selection frequency / duty
  fraction averaged over the 24 final-test seeds and connect this panel to the
  scalar behaviour diagnostics.
- The fixed TCN validation diagnostic should not be added as positive evidence
  in its current form. The saved diagnostic reports TCN nMAE values orders of
  magnitude worse than persistence, which is either a severe forecaster-quality
  problem or a scale/evaluation mismatch. In either case, it cannot support the
  reward-signal-reliability claim.
- Figure 1 and Figure 2 do not need another structural rewrite for this pass:
  the current framework figure has three stages and the split timeline already
  reports the actual step counts. They still need post-compile visual
  verification.

## 06-23-03 Strict Closure Findings
- The high-priority 06-23-03 items are now resolved in the rendered PDF:
  Table 11 uses a readable two-row layout, and Figure 7 Panel B is an explicit
  specialist-selection frequency heatmap with matching caption and Results text.
- The fixed TCN validation request cannot strengthen the paper. After correcting
  the warm-up padding artifact, the 24-seed TCN validation remains worse than
  persistence. The correct handling is to keep the audit artifact for internal
  traceability and not include the diagnostic table in the manuscript.
- The current manuscript should continue to call the completed event-mixture
  evidence a mild six-seed sensitivity check. A stricter rare-flux event-mix run
  is queued on `remote-gpu`, but until it finishes it is not a claim boundary
  and not a manuscript result.
- Figure labels should use "fixed evaluation forecaster" rather than "Fixed
  TCN" unless a future paper section explicitly validates the TCN forecaster
  against simple forecasting references.
- The appendix training curves are acceptable as optimisation diagnostics only.
  They should not be cited as validation convergence or sample-efficiency
  evidence unless future logs include validation macro/step metrics over
  training updates.

## Rare-Flux Event-Mix Completion Analysis
- The queued rare-flux event-mixture robustness run completed on `remote-gpu`
  and was synced locally. The analysis report is
  `reports/aggregate/rare_flux_event_mix_result_analysis_20260625.md`.
- The rare-flux six-seed block over seeds `147--152` passed all headline gates:
  operational step `6/6`, true-static macro `6/6`, strict true-static step
  `6/6`, and behaviour complexity `6/6`. Mean operational step margin is
  `0.092293`, mean macro margin is `0.087765`, and mean macro margin vs the
  true-static reference is `0.056582`.
- The result is positive but tighter than the main 24-seed benchmark. The
  weakest rare-flux seed is `147`, with operational step margin `0.010525` and
  strict true-static step margin `0.002898`. This supports robustness to a
  lower-flux changed event mixture, but not a universal rare-event claim.
- Together with the earlier higher-flux changed-mixture block over seeds
  `141--146`, the robustness evidence can be described as two fresh six-seed
  changed-mixture sensitivity checks. Both event-mixture checks pass all gates;
  the weaker-latent perturbation remains internal boundary evidence because it
  passes only `5/6` on behaviour, true-static macro, and strict-step gates.
- The raw unnormalised macro diagnostic remains unsupported as a headline
  criterion: main `0/24`, higher-flux changed mixture `0/6`, and rare-flux
  changed mixture `0/6`. The manuscript should keep the claim tied to ordinary
  step loss and static-reference-normalised event-regime macro scoring.

## Rare-Flux Manuscript Migration
- The active manuscript now includes the rare-flux result as a second
  changed-mixture sensitivity row, not as a broader robustness claim.
- `paper/tables/event_mix_robustness_summary.tex` now reports the main
  benchmark, the higher-flux mixture `0.35/0.30/0.35`, and the lower-flux
  mixture `0.45/0.10/0.45`, with step/macro mean--median margins and strict
  fixed-mask / behaviour gate counts.
- The Results text states that both changed-mixture settings preserve the
  ordinary step gate, static-reference-normalised macro gate, strict fixed-mask
  step gate, and behaviour gate, while noting that the lower-flux step margin is
  tighter than the main benchmark.
- The Discussion now explicitly limits the inference: the checks do not
  establish robustness to arbitrary event prevalence, different event detectors,
  or uniformly mixed regimes.

## Post-Codex Manuscript Polish Findings
- After Codex promoted the ESWA manuscript to `paper/main.tex` and refreshed the
  active section set, the remaining high-confidence polish targets were prose
  framing issues, not evidence or algorithm changes.
- The safest polish pattern is to replace reader-steering constructions with
  direct technical statements: e.g. "This paper presents" -> method noun first;
  "This pattern explains" -> the mechanism noun first; "should be read as" ->
  direct bounded interpretation.
- The strongest current claim remains unchanged: fixed-forecaster protocol,
  fixed-backbone one-specialist benchmark, 24 independent final-test seeds,
  ordinary step loss plus static-reference-normalised event-regime macro score,
  fixed-mask replay, and behavioural diagnostics. The changed-mixture checks
  remain benchmark-local sensitivity evidence.
- Manuscript terminology is now clean at both source and PDF levels for the old
  internal-code and method-term set. The next submission-facing risks are not
  prose residuals but packaging: pin the final code revision, archive evidence
  artifacts, and run the final submission-package audit.
- Full `git diff --check` currently reports trailing whitespace in generated
  SVG files from the broader Codex figure refresh. The current prose polish did
  not introduce those warnings; the touched LaTeX source files pass targeted
  `git diff --check`.

## 06-25-01 PPD Deep-Rewrite Findings
- The 06-25-01 review correctly identifies the remaining high-level problem:
  the old manuscript still over-weighted the algorithm label and repeated
  protocol/result caveats across abstract, introduction, methods, results,
  discussion, and conclusion.
- The stable paper frame after this pass is: forecast-oriented specialist
  scheduling under a mandatory backbone and scarce specialist slot, evaluated by
  a fixed-forecaster replay protocol. PD-PPO remains the implementation, not the
  whole novelty claim.
- The title and abstract now lead with the problem rather than the algorithm:
  `Adaptive Specialist Sensor Scheduling for Forecast-Oriented Sensing under
  Power Constraints`.
- The Introduction now exposes the paper's logic as H1--H3: dynamic specialist
  value under incompatible event regimes, fixed-mask replay as the constant-mask
  test, and behavioural non-degeneracy as the state-dependent scheduling test.
- Related work needed a compact positioning object more than additional prose.
  `tables/related_work_positioning.tex` now contrasts the work with
  AoI/estimation scheduling, active perception/POMDP sensing, DRL adaptive
  sensing, and forecasting-model evaluation without adding unverified citations.
- The formulation/method boundary is cleaner: Section 3 defines feasible masks,
  specialist-bottleneck rationale, and metrics; Section 4 contains the policy,
  training reward, PPO objective, auxiliary training signals, chronological
  separation, and references.
- Main-text theory should remain design rationale. The sufficient-condition
  propositions stay in the paper, but proof density is pushed to the appendix.
- The rebuilt PDF is 47 pages, down from the backed-up 53-page version, while
  preserving the supported numerical claims and benchmark-local boundaries.

## 06-15 Comparison Style Repair Findings
- The closest available around-06-15 compiled reference is the 2026-06-12
  `paper/pdppo_crst_rewrite.pdf`. It was less overloaded with project labels
  than the active 06-25 draft, even though its evidence protocol is older.
- The renewed AI-style problem was not just individual phrases. The active draft
  had turned many descriptive ideas into repeated labels, including fixed-mask,
  one-specialist, static-reference-normalised, regime-balanced, event-context,
  step-margin, benchmark-local, and fixed-forecaster. These terms made the paper
  read like an internal experiment log.
- The safer manuscript style is to keep a small number of true technical names
  (`PD-PPO`, `masked PPO`, `fixed forecasting model`, `fixed mask`, `single
  specialist`, and `macro score normalised by static references`) and write the
  rest as ordinary sentences.
- Visible colon removal needs PDF-level checks, not only source checks. LaTeX
  caption labels, algorithm line numbers, and generated figure PDFs can reinsert
  colons even after the TeX prose is clean. This pass set caption separators to
  periods, removed algorithmic line numbers, and regenerated affected figure
  assets.
- The final active manuscript source and PDF now scan clean for first-person
  pronouns, the stale style/term list, and source-visible prose colons. The only
  remaining PDF colons are front-matter labels generated by `elsarticle`.

## 2026-07-01 LEMMA Fourth-Round Findings
- The key theoretical risk was not the algebra of Proposition 1; it was the
  interpretation of the constructed policy. The dynamic comparator in the proof
  needs event-label access and should be described as an oracle/event-label replay
  upper bound, not as deployable PD-PPO.
- The best bridge between theory and experiment is already in the manuscript:
  event label replay has a mean macro margin of 0.0773 against fixed mask replay,
  while learned PD-PPO has 0.0710. This supports the claim that observable event
  proxies recover most of the oracle opportunity, without claiming that the proof
  establishes learnability.
- LEMMA's suggested phrase that event label replay improves all three event types
  in all 24 seeds would overstate Table 11 because per-event positive seed counts
  are 18/24, 12/24, and 16/24. The safe wording is that all three mean event rows
  improve and the macro score improves in all 24 seeds.
- Section 5.3 should remain protocol-facing. Forecaster architecture and PD-PPO
  optimisation hyperparameters belong with the method section, while seeds,
  partitions, candidate masks, references, and statistical summaries remain in
  the setup section.
- Generator validation checks are useful domain-boundary evidence but interrupt
  the main setup flow. Moving the table and figure to the appendix keeps the
  main text focused while retaining recoverable evidence.
- PD-PPO should be framed as the masked PPO implementation used for this forecast
  scheduling protocol. The current paper establishes learned adaptive scheduling
  against fixed and rule-based references, not superiority over other RL
  algorithms; comparing DQN/SAC-style alternatives under the same replay protocol
  remains future work.

## 2026-07-01 N-Series Follow-Up Findings
- After moving large validation floats into the appendix, section ownership must
  be verified in the rendered PDF, not only in source order. Without explicit
  page barriers, floats can appear after the next appendix section heading and be
  numbered under the wrong appendix.
- In this document class/reference setup, `\ref{apx:...}` for appendix sections
  already expands to strings such as `Appendix A.2`. Writing `Appendix~\ref{...}`
  therefore renders as `Appendix Appendix A.2`. For appendix section references,
  use the bare section reference output or switch the label/reference convention
  deliberately.
- Table 6 remains the correct reference for the event-label replay diagnostic:
  it is the comparison-reference taxonomy and its event-aware diagnostic replay
  row states that the diagnostic uses privileged event type information and is
  not the learned controller.

## 2026-07-02 Reference-Audit Findings
- Reference audits should be applied to the active included manuscript only.
  `raw.tex`, `rewrite_sections/`, and `_archive/` may retain historical citation
  keys such as `Liang2024` or `Bajcsy_2017`; these are not current `main.tex`
  sources and should not be used to judge final citation residue.
- DOI/Crossref metadata can differ from shorthand audit notes. For this pass,
  DOI metadata supported Aloni as Environmental Modelling & Software 185, 106283
  with formal publication year 2025, and Jonah as IEEE Access 14, 40042--40059.
- The current reference list now includes foundational PPO and TCN citations at
  the first method/forecaster mentions, while the indirect VLDB graph-coverage
  citation was removed to reduce Related Work citation padding.

## 2026-07-02 Framework-First Route Correction
- The user explicitly rejected the route that would package the paper primarily
  as a benchmark/protocol study. That diagnosis may remain useful as an external
  review risk audit, but it is not the active manuscript positioning.
- The active positioning is: PD-PPO is a prediction-driven constrained
  reinforcement-learning framework for sensing-system scheduling under a power
  budget. The benchmark is the evaluation environment, not the main novelty.
- The paper should not claim a novel PPO algorithm. PPO itself is standard; the
  contribution is the framework combination of downstream forecast-loss reward,
  executable candidate masks, online feasibility masking for hardware operating
  rules, fixed forecast evaluator training/evaluation separation, and replay
  diagnostics that rule out fixed or cyclic shortcuts.
- Future manuscript edits should therefore move method and contribution wording
  toward "prediction-driven constrained scheduling framework", while keeping
  synthetic-data, fixed-evaluator, and non-field-deployment limits in Discussion
  rather than letting them dominate the abstract or introduction.
- The highest-value supplementary experiments for this framing are not more
  same-protocol seeds. They are:
  1. a forecast-greedy / one-step lookahead feasible-mask baseline;
  2. a contextual-bandit baseline using the same observations and forecast
     reward without sequential credit assignment;
  3. a reward ablation holding PPO and masks fixed while comparing forecast-loss
     reward against AoI and uncertainty/covariance rewards;
  4. a lightweight forecaster-sensitivity check if time permits;
  5. event-proxy quality and target-wise raw-metric audits for interpretability.
- SAC is low priority for the current action geometry because the scheduler uses
  a small feasible-mask action set. Masked DQN/Double DQN is a useful RL
  comparator, but it is lower ROI than greedy, bandit, and reward-ablation
  checks for defending the framework claim.

## 2026-07-02 Framework Supplement Findings
- The one-step forecast-greedy diagnostic strongly supports a sequential policy
  claim: even with privileged final-test future-loss access, one-step greedy is
  worse than PD-PPO in all 24 seeds and has much higher average forecast loss.
  This argues that the learned schedule is not merely reproducing a myopic
  forecast-loss selector.
- The context-alert bandit is the most important new challenge. It uses
  station-side context proxy columns and validation-selected subtype masks, not
  final-test tuning, and it beats PD-PPO on the majority of seeds. The manuscript
  should not claim unconditional superiority over all context-aware non-RL
  policies unless PD-PPO is improved or this bandit is framed as a
  hand-engineered diagnostic reference.
- Reward-ablation evidence is mixed. Forecast reward is better than AoI on the
  two-seed pilot, but the coverage reward proxy is competitive and slightly
  better on the two-seed step-loss average. This suggests that the framework
  gain is not explained by forecast reward alone; candidate masks, auxiliary
  context losses, and the calibrated action geometry also matter.
- The most defensible framework claim after these supplements is:
  PD-PPO implements a prediction-driven constrained RL scheduling framework that
  outperforms myopic forecast-greedy and static/cyclic references in the main
  evidence package, while context-aware hand-coded policies remain a meaningful
  comparator and should be reported or reserved as a future-strengthening
  baseline.

## 2026-07-03 Feature-Parity and CA-PD-PPO Findings
- `context_alert_bandit_t0p5` uses online station-context proxy scores rather
  than final-test forecast loss or simulator event labels. The relevant
  observables are the particle, flux, and thermal alert columns.
- Original PD-PPO already had partial access to raw context columns in this
  experiment family, but it did not expose the same derived decision state used
  naturally by the bandit-style rule: alert flags, max confidence, alert onset
  age, alert trend, previous specialist, and remaining minimum-on time.
- The implemented feature-parity tail is online-observable by construction:
  alert scores, binary alert flags, argmax one-hot over calm/particle/flux/
  thermal, confidence, onset age, rolling alert trend, previous specialist, and
  remaining minimum-on time are derived from current/past alert columns,
  previous action state, and the operating-rule state. It deliberately removes
  the simulator `event_flag` from the policy state for the new dev comparison.
- CA-PD-PPO is implemented as a small context encoder fused into the masked PPO
  actor path, not as a residual or prior over the context-alert bandit. It keeps
  the same forecast-loss reward and the same feasibility-masked categorical
  action distribution.
- The new dev decision table uses the sign convention `baseline - PD-PPO`:
  positive step or macro margin means the PD-PPO variant is better than the
  compared baseline.
- Final dev comparison over seeds 201--224:
  - `original_clean` remained behind `context_alert_bandit_t0p5`: 6/24 macro
    wins, mean macro margin `-0.007329`.
  - `feature_parity` improved the step-loss gap but did not solve the bandit
    challenge: 8/24 macro wins, mean macro margin `-0.007684`.
  - `ca_pdppo` produced the only positive context-bandit macro mean:
    13/24 macro wins, mean macro margin `0.008257`, bootstrap 95% CI
    `[0.000011, 0.019656]`.
  - All three variants still beat `forecast_greedy_one_step` strongly; CA-PD-PPO
    has 24/24 macro wins and mean macro margin `0.176646` against it.
- Decision-rule outcome: CA-PD-PPO is a real improvement and is competitive with
  the strong context-alert bandit, but it does not pass the user's fresh-final
  gate because 13/24 macro wins is below the required 15/24. Do not launch a
  final 24-seed evaluation from this result alone, and do not add bandit-dependent
  patchwork modules to the main method.

## 2026-07-03 CA-PD-PPO Failure-Structure Findings
- The 11/24 CA-PD-PPO losses against `context_alert_bandit_t0p5` are not a
  broad failure of context-aware masked PPO. They are concentrated in flux
  subtype windows and in lower-confidence/no-alert or alert-boundary regions.
- High-confidence context windows are already favourable to CA-PD-PPO. This
  argues against adding bandit imitation, residual bandit actions, or
  bandit-margin rewards; the clean next step is better context calibration and
  context-to-action fusion inside the masked PPO actor.
- Alert-lag analysis suggests the remaining gap is strongest near late-event,
  post-offset, and outside-alert intervals. Mid-event behaviour is not the main
  weakness.
- PPO training proxies do not show an obvious value-loss/advantage-instability
  explanation. The bounded dev2 wave should therefore stay small and test only
  method-consistent changes: context capacity, gated fusion, optional context
  LayerNorm, and rollout length.
- Fresh final evaluation remains blocked by the predeclared gate. Only launch
  final seeds after a clean dev2 variant reaches positive mean macro margin
  above `0.010`, at least 15/24 macro wins versus the context-alert bandit,
  nonnegative bootstrap lower bound, no static/greedy regression, no aborts,
  and acceptable switching rate.
- The first bounded dev2 variant, `ctx128`, confirms that context capacity alone
  is not the missing ingredient. It improves neither the win-count gate nor the
  mean macro-margin gate: 14/24 macro wins and mean macro margin `0.004083`
  versus the context-alert bandit. Continue to `gated` and `gated_ctx128`; do
  not launch fresh final from `ctx128`.
- The second bounded dev2 variant, `gated`, confirms that a small gated-add
  context fusion alone is also insufficient. It reaches only 13/24 macro wins
  and mean macro margin `0.002763` versus the context-alert bandit, despite
  preserving 24/24 macro wins versus the forecast-greedy diagnostic. Continue to
  `gated_ctx128`; do not launch fresh final from `gated`.
- The third bounded dev2 variant, `gated_ctx128`, improves mean macro margin
  over `gated` but still misses both final-launch gates: 13/24 macro wins and
  mean macro margin `0.006706` versus the context-alert bandit, with slightly
  negative mean step margin. Larger gated context fusion is therefore not enough
  by itself. Continue to `nsteps2048`; do not launch fresh final from
  `gated_ctx128`.
- The fourth bounded dev2 variant, `nsteps2048`, weakens the context-bandit
  comparison rather than solving it: 10/24 macro wins, mean macro margin
  `0.002962`, 10/24 step wins, and mean step margin `-0.000719`. Longer PPO
  rollout alone is therefore not a clean path to stable dominance over the
  context-alert bandit.
- Final dev2 decision: none of `ctx128`, `gated`, `gated_ctx128`, or
  `nsteps2048` passes the fresh-final gate. The method-consistent result is
  still that CA-PD-PPO is competitive with a strong context-aware hand-coded
  baseline and remains strong against forecast-greedy, but the current evidence
  does not justify a confirmatory fresh final run or a claim of stable
  superiority over `context_alert_bandit_t0p5`.
## 2026-07-10 Post-Hermes Audit: Initial Scope
- This audit treats the final fixed-backbone, one-specialist SCENEBAL-2
  aggregate over seeds 117--140 as the active main-text evidence package.
- The immediate risk is not necessarily a numerical error: the plan explicitly
  distinguishes the reported static-normalised regime macro score from a raw
  unnormalised subtype-macro sensitivity diagnostic. The audit must establish
  that the manuscript names the former consistently and does not imply
  aggregation-invariant superiority.
- CA-PD-PPO/context-bandit development results are a separate supplementary
  framework stress test. They must not be mixed into, or used to invalidate,
  the completed SCENEBAL-2 primary aggregate without an explicit paper claim.

## 2026-07-10 Post-Hermes Audit: Confirmed Evidence Boundaries
- The local and `remote-gpu` copies of the primary SCENEBAL-2 aggregate agree
  byte-for-byte for both `metpair_seed_summary.csv` and
  `metpair_claim_summary.json`. The 24 truth CSVs also have 24 distinct SHA-256
  hashes, so the seed dimension represents distinct generated scenarios.
- Ordinary step-loss results are internally consistent: the old-claim collector
  reproduces the active table's best-operational step margin of `0.149379`, with
  `24/24` paired wins and a seed-level bootstrap interval approximately
  `[0.1124, 0.1895]`. The primary step claim is therefore not affected by the
  macro normalizer issue below.
- There is a real macro-normalizer protocol mismatch. Training and ordinary
  policy evaluation save validation-derived subtype normalizers, but
  `70_v31_split_replay_gate.py` derives a second set from final-test static
  candidates and `72_v31_collect_metpair_strongclaim.py` reuses those final-test
  denominators for replay-static macro summaries. This conflicts with the paper
  formula and prose saying that macro normalizers are fitted on validation.
- A read-only recomputation from saved final rollouts and validation candidate
  tables fixes the normalizer without retraining. It preserves all signs:
  validation-frozen macro margins are `24/24` positive versus validation static
  (mean `0.07782`), best operational reference (mean `0.07711`), and true fixed
  replay static reference (mean `0.07047`, 95% bootstrap interval
  `[0.06021, 0.08071]`). The existing `0.0710` replay-static number uses the
  final-test static normalizers, not the validation-frozen metric described in
  the manuscript.
- The active main table mixes comparator scopes: `0.1494` and `0.0841` are
  best-operational-reference margins, while the caption and first two rows refer
  to the validation-selected fixed schedule. The values are valid but need
  explicit labels or separate rows.
- Seed 117 and seed 122 were used in the two-seed SCENEBAL-2 pivot pilot before
  expansion. The run configuration then remained fixed for the expansion. The
  post-pilot 22-seed subset (118--121, 123--140) still gives `22/22` positive
  step and frozen-normalizer replay-static macro margins, with means `0.14965`
  and `0.07063`, respectively. The manuscript should distinguish independent
  scenario seeds from strict post-pilot confirmatory seeds.
- The privileged event-label replay artifacts are retained on `remote-gpu`, but
  only 12/24 replay metric files are present locally. This is a reproducibility
  packaging gap, not a changed result. The submission archive must sync each
  replay summary, metric table, and rollout artifact from the remote result
  directories before claiming self-contained reproduction.

## 2026-07-10 Evidence Repair: Resolved Findings
- `scripts/86_v31_collect_validation_frozen_macro.py` now recomputes the macro
  score solely from final rollouts and the per-run
  `validation_static_candidates.csv` normalizers. Its CSV-input mode now also
  honors `--seeds`, allowing the same collector to produce a post-pilot subset.
- The corrected primary comparison is PD-PPO versus the validation-selected
  static schedule: 24/24 macro wins, mean margin `0.0778198`, 95% percentile
  bootstrap interval `[0.0664682, 0.0896082]`; the paired ordinary-step result
  is also 24/24, mean `0.1530815`.
- Excluding configuration-pivot seeds 117 and 122 leaves a post-pilot 22-seed
  replication: 22/22 validation-frozen macro wins, mean `0.0779690`, interval
  `[0.0669584, 0.0895956]`. The conclusion is therefore not driven by the
  pilot seeds.
- Validation-frozen ablations preserve 24/24 macro wins for all variants.
  The no-event-context auxiliary has 22/24 ordinary-step wins; the other two
  variants retain 24/24. Their paired macro-margin intervals overlap zero, so
  auxiliary terms are not presented as large isolated macro effects.
- The fresh higher-flux setting retains 6/6 validation-frozen macro wins
  (mean `0.0763879`, interval `[0.0568861, 0.1013251]`). The lower-flux setting
  is omitted from the same-metric robustness table because seed 147 has zero
  validation flux windows, making a three-regime macro undefined.
- The main policy has zero warmup aborts in every seed. The required weather
  backbone is always active and the radiometer has zero duty; the other four
  specialists have intermediate duty in every main seed. This is a calibrated
  action-space boundary, now stated explicitly in Results.
- The archive is provenance-complete rather than a second full raw-data copy:
  it contains all extracted rows required for deterministic reaggregation, the
  collector code, a 530-file remote path/size/SHA-256 manifest, and source
  snapshots. The remote filesystem was above quota, so the immutable manifest
  is the reliable raw-artifact pin.

## 2026-07-17 Whole-Paper Structure Audit: Preliminary Findings
- The canonical source remains `paper/main.tex`; active prose comes from
  `paper/sections/*.tex`. The current compiled manuscript is 56 pages.
- The main paper has a recognizable contribution chain: general scheduling
  motivation, related work, formulation, masked-PPO method/protocol, controlled
  benchmark, held-out results, discussion, and conclusion. The abstract and
  Introduction consistently frame PD-PPO as a prediction-driven constrained
  scheduling framework rather than a novel PPO optimizer.
- The newly added `tables/development_contextaware_summary.tex` is not promoted
  into the confirmatory main result. It appears in a clearly labelled
  development-only appendix, reports only step-margin/gate fields, states that
  historical macro values used a superseded normalizer, and records that no
  variant passed the fresh-expansion gate.
- The main structural risk is hierarchy and volume, not a missing section: the
  eight-section paper is followed by nine appendix sections in the same PDF.
  Theory proofs, platform rendering, protocol details, generator validation,
  ablation distributions, training diagnostics, development variants, the
  static-selection ledger, and event-mixture sensitivity compete for attention
  and may obscure the single main contribution.
- The front matter is complete (authors, correspondence, abstract, keywords,
  abbreviations, CRediT, funding, conflict, data availability, AI declaration,
  acknowledgements), but completeness of submission-facing wording and visual
  placement still requires rendered-PDF inspection.

## 2026-07-17 Whole-Paper Structure Audit: Argument-Level Findings
- The primary Results hierarchy is substantially improved: Table 7 places the
  validation-selected fixed schedule first, labels the strongest per-seed rule
  as post hoc, and keeps forecast-greedy/context-alert comparisons as claim
  boundaries. The Results prose follows that hierarchy rather than treating all
  rows as equally confirmatory.
- The main source contains approximately 9 figures, 15 tables, 18 displayed
  equations, and 2 propositions. Problem Formulation is about 1,362 words,
  longer than the roughly 969-word method section and 969-word Results section.
  This makes the paper read more theoretically elaborate than the empirical
  contribution warrants; the specialist-bottleneck proposition is especially
  close to its own incompatibility assumptions and should not carry the novelty
  claim by itself.
- The Discussion is only about 579 words and the Conclusion about 151 words.
  Relative to the 56-page review manuscript and extensive appendix, interpretation
  is underdeveloped: the paper spends more space documenting diagnostics than
  explaining when prediction-driven sequential scheduling is useful, what the
  context-alert result means for method design, and how the one-specialist result
  transfers to broader sensing systems.
- Two possible reproducibility gaps require focused verification: (1) the main
  method repeatedly uses estimated state, covariance/uncertainty, and freshness,
  but no obvious estimator/partial-observation update subsection appears in the
  section map; (2) comparison roles are defined, but the executable definitions
  of AoI, round robin, random, forecast-greedy, and context-alert policies may be
  absent or too brief. These would be substantive method omissions if not covered
  by active tables/appendix.
- The Related Work section is concise (about 453 words across four subsections).
  It establishes the broad areas but may not compare PD-PPO closely enough with
  the nearest RL sensor-scheduling and task-oriented active-sensing methods to
  make the technical gap immediately defensible for ESWA.

## 2026-07-17 Whole-Paper Structure Audit: Confirmed Completeness Gaps
- The estimator/partial-observation path is not reproducibly specified. The
  manuscript says the policy and fixed forecaster receive an estimated state,
  uncertainty, last observations, masks, and freshness, but it does not define
  the estimator update, process/measurement models, treatment of inactive
  channels, uncertainty propagation, or observation-noise parameters. The
  observability table names sources rather than supplying this missing method.
- Baseline implementation detail is insufficient. Table 6 states that the
  rule-based family uses cycling, stale-channel priority, or random rules, and
  describes forecast-greedy/context-alert roles, but gives no executable
  definitions: cycle order/timing, AoI tie-breaking, random distribution,
  forecast-greedy evaluation horizon, alert construction/threshold, or how
  minimum on-time and feasibility are applied. These details are central because
  the paper's claim is comparative.
- The synthetic generator description is too compact for standalone
  reproduction. It names a semi-Markov event process and literature anchors but
  omits transition/duration distributions, coupling equations, channel noise,
  proxy-lead generation, target weights, and most acceptance thresholds. The
  appendix reports check outcomes but not a complete generator parameterization.
- The abstract, Introduction, Discussion, and Conclusion say PD-PPO improves
  over “rule-based references/schedules,” while the main table separately shows
  that a handcrafted context-alert rule is competitive (PD-PPO macro wins 7/24,
  mean margin -0.0011). Because context-alert is itself rule based, the generic
  phrase is overbroad. It should consistently say conventional AoI/round-robin/
  random rules, or explicitly define the excluded stress-test family.
- The front matter calls all 24 seeds “held-out,” although seeds 117 and 122
  were used during the configuration pivot. Their final windows are temporally
  held out, but they are not fully post-pilot confirmation seeds. The Setup is
  honest about this; the Abstract, Introduction, figure caption, and Conclusion
  should distinguish the 24-seed full aggregate from the 22-seed post-pilot
  replication.
- The paper presents an algorithm/framework contribution but its confirmatory
  table has no alternative learned-policy baseline under the same action and
  feasibility interface, and no same-PPO reward comparison against AoI or
  uncertainty. Forecast-greedy and context-alert are valuable stress tests, but
  they do not isolate whether PPO or the forecast-loss reward provides the gain.
  This is the largest remaining experimental-completeness risk for an ESWA
  method paper.

## 2026-07-17 Whole-Paper Structure Audit: Final Layout and Priority Findings
- The current `paper/main.pdf` is synchronized with the active sources: no
  manuscript source is newer than the PDF. It contains 56 letter-size review
  pages. The compile log has no undefined citation/reference or fatal error;
  only one 1.8-pt overfull box and several underfull table cells remain.
- Main-text visual continuity is acceptable. The framework figure appears near
  the start of the paper, result figures and tables are readable at page scale,
  and no sampled main-text page shows clipping or overlap.
- Appendix pagination is visibly defective. Appendix H has a heading-only page
  (PDF page 49), while its fixed-schedule table moves to page 50. Appendix G is
  also spread across two sparse pages. Consecutive `\\clearpage` commands around
  the appendix inputs cause this and should be removed or replaced by controlled
  float barriers in the submission layout.
- The Discussion is structurally underweight relative to the formulation,
  diagnostics, and nine appendices. It should synthesize the static-versus-
  adaptive design rule, the context-alert result, online computational cost,
  transfer beyond one specialist slot, and the practical meaning of the frozen
  forecaster result rather than adding more limitations.
- The `epsilon_s=max(0.001,0.002L_ref)` superiority threshold is labelled
  prespecified but has no engineering or statistical rationale. It should be
  justified, demoted to a diagnostic, or removed because paired margins and
  confidence intervals already carry the primary inference.
- Generality remains empirical rather than demonstrated: the main evidence uses
  one power budget, one active specialist slot, one simulator family, and one
  fixed TCN evaluator. A capacity/budget sensitivity and a second frozen
  forecaster replay are higher-value additions than more development variants.
- Submission-facing data availability is not yet complete: the manuscript says
  code and aggregate evidence will be archived, but gives no persistent URL,
  DOI, commit, or anonymized package identifier. This is acceptable for the
  supervisor draft but not the final submission package.

## 2026-07-18 Method and Experimental Closure: Initial Findings
- `docs/07-18-01.md` preserves the correct paper identity: forecast-loss-driven
  constrained scheduling is the primary contribution, masked PPO is the
  executable implementation, and the replay protocol is supporting evidence.
  It explicitly rejects turning the manuscript into a benchmark catalogue or
  importing the separate no-warmup/deployment-constrained paper line.
- The decisive experimental gap is a matched reward comparison. The prior
  `reward_proxy_mode` implementation already supports `forecast`, `aoi`,
  `coverage`, and `instant_error`, but only AoI and coverage were piloted on
  seeds 117--118. Those two-seed results are exploratory and coverage is not a
  direct uncertainty/covariance control.
- The old AoI pilot appears to preserve the custom masked-PPO training path and
  final fixed-forecaster evaluation, but its raw run directories are not local;
  remote manifests and commands must be checked before treating it as a valid
  same-architecture control or expanding it.
- The existing comparator package contains fixed schedules, conventional
  dynamic rules, one-step forecast-greedy, and context-alert references. None is
  a matched learned-policy baseline under the final candidate-mask and hard
  feasibility interface. Existing historical DQN code cannot be cited until its
  current-interface parity is demonstrated.
- The final experiment matrix must be frozen before expansion. It may compare
  forecast reward with AoI and uncertainty rewards, and PD-PPO with one matched
  learned baseline, but it must not alter the SCENEBAL-2 simulator or tune reward
  definitions against final-test outcomes.
- Remote audit confirms that the seed-117 mainline, AoI, and coverage runs use
  byte-identical truth CSVs and manifests that differ only in
  `ppo_controls.reward_proxy_mode`. The AoI and coverage mode is also recorded in
  `v2_ppo_metadata.json`; the mainline predates explicit recording and therefore
  implies the default forecast mode.
- The old reward pilots are nevertheless not strict single-factor controls. Each
  variant retrained its own TCN evaluator and recomputed validation static
  candidates/normalizers. For seed 117, the evaluator-derived normalizers and
  selected-static candidate scores differ slightly across forecast, AoI, and
  coverage directories despite identical truth. A formal reward comparison must
  reuse the mainline per-seed truth, fixed evaluator checkpoint, validation
  candidates, normalizers, start indices, and candidate masks, and retrain only
  the scheduler.
- The existing AoI proxy is explicit and interpretable: it averages normalized
  time since the last valid target observation over the lookback window. The
  coverage proxy instead measures the fraction of target entries observed in the
  lookback; it is not an uncertainty/covariance objective and cannot satisfy the
  requested uncertainty reward control.
- The current `CustomPPO._make_env` and `evaluate_custom_ppo` reconstruct
  `WarmupEnvConfig` field by field. They copy the forecast/AoI proxy selector but
  omit the event-subtype reward multipliers, static-normalizer tuple/default,
  and common-random-number switch. Consequently, the existing SCENEBAL-2 policy
  was trained with forecast loss, but the subtype-balanced reward shaping
  recorded in metadata was not fully propagated to the episode environments.
  This is an implementation/provenance defect, not evidence that the policy used
  an unrelated objective.
- `src/v2/dqn.py` uses the same candidate masks and online feasibility mask, but
  it has the same field-by-field configuration problem and omits even more of the
  current hard-constraint/context fields. It is not a matched learned baseline
  until environment construction is replaced by a complete dataclass copy and
  a current-protocol launcher is added.
- The corrective experiment must not compare newly fixed controls against the
  old policy as though they were single-factor variants. The bounded pilot will
  retrain forecast-, AoI-, and uncertainty-reward PPO policies after the common
  environment-copy fix, while reusing each seed's frozen truth, TCN evaluator,
  candidate set, validation normalizers, static selection, and final windows.
  Only if the fixed forecast policy passes the existing behavioural/performance
  gates will the triplet be expanded.

## 2026-07-18 Method and Experimental Closure: Implementation Audit
- The old SCENEBAL-2 actor and event-aware critic consumed the exact simulator
  `event_flag` through both the observation tail and direct event-gate inputs.
  This conflicts with the manuscript's online-observability claim. Corrective
  runs disable that feature and route actor/critic event context through the
  saved station-side `agent_context_event_alert` proxy. Exact subtype labels
  remain restricted to the RL-training partition for the existing auxiliary
  losses and event-conditioned episode sampling.
- `WarmupSchedulingEnv` is a normalized sample-and-hold partial-observation
  buffer, not the Kalman-style estimator implied by current manuscript prose.
  It fuses simultaneous active-sensor measurements by inverse noise variance,
  carries inactive variables forward, and tracks masks, sensor mode, warm-up,
  freshness, previous action, power/time, and online alert context. The policy
  does not currently receive an explicit posterior covariance vector. The
  manuscript must describe this implementation directly rather than inventing
  an estimator state or uncertainty input.
- Added a bounded diagonal predict-update uncertainty proxy for the matched
  reward control. Process variance is estimated from normalized first
  differences on the RL-training normalization partition; active measurements
  use sensor noise variances, and inactive variables accumulate process
  variance. This proxy changes only the training reward in the uncertainty
  control; final scoring remains the frozen TCN forecast loss.
- Replaced field-by-field environment reconstruction in PPO and DQN with full
  dataclass copies. This preserves subtype reward multipliers, static
  normalizers, dwell/energy settings, and future configuration fields across
  training episodes and final evaluation.
- Added a strict control-source reuse path. It verifies source truth hashes,
  seed, sensor configuration, constraints, lookback/horizon, candidate masks,
  validation starts, final starts, frozen evaluator, static normalizers, and
  selected fixed-schedule assets. Seed-117 and seed-118 preflights passed with
  six identical executable masks and the source-specific final/validation
  windows recorded in `control_source_preflight.json`.
- The frozen bounded matrix is: corrected forecast-reward PPO, AoI-reward PPO,
  and diagonal-uncertainty-reward PPO on seeds 117 and 118, all at 200k steps;
  each reuses the same seed-specific source assets and excludes exact event
  labels at policy execution. Expansion is conditional on corrected forecast
  performance and behavior, not on choosing the best final-test reward mode.
- A matched Double-DQN launcher has been added for the second learning baseline.
  It uses the same six candidate masks, hard feasibility checks, online
  observation contract, frozen forecast reward, training range, and final
  windows; it has no teacher imitation, bandit prior, or final-test event label.
  It still requires a remote smoke run before formal pilot launch.
- The four online context columns require more precise wording than the current
  manuscript uses. They are not the exact `event_flag` at execution, but the
  generator constructs each subtype score from the corresponding synthetic
  subtype interval, a 16-step leading ramp, and Gaussian noise. They therefore
  represent supplied synthetic warning signals, not the output of a validated
  station-side event detector. This assumption belongs in the simulator
  appendix and must not be blurred into an observed physical event label.
- Candidate-mask reconstruction is now explicit: the action surface contains
  six masks (meteorological core alone, or the core paired with one of the five
  optional channels). Online feasibility masking and the projector enforce
  steady/startup power and maximum-active constraints; the environment then
  applies the six-epoch global mask-hold rule.
- The corrected forecast pilot preserves the positive main mechanism after
  removing exact event labels from online execution. Seed 117 has macro score
  `0.785992` versus static `0.817379`; seed 118 has `0.653137` versus
  `0.708560`. Both ordinary forecast losses also improve over static.
- The pilot is genuinely adaptive rather than merely nonconstant. Seed 117/118
  action entropy is `1.857/1.882` bits, subtype-mask mutual information is
  `0.615/0.641` bits, and both traces pass the fixed/cycle/complexity gates.
  Specialist duties are distributed over thermo-hygrometer, surface infrared,
  laser, and FC4; the weather backbone is mandatory and the radiometer remains
  unused.
- The corrected seed-118 policy also beats the validation-frozen context-alert
  stress test in the pilot replay by `0.011685` macro units. This is an early
  implementation check, not a seed-expanded claim.
- The old 24-seed policy, mechanism-ablation table, event-type figures, and
  changed-mixture claims were generated before the online event-label defect was
  corrected. They cannot remain confirmatory evidence. The corrected forecast
  expansion must replace the main table and figures; old component ablations
  must either be rerun under the corrected observation contract or removed in
  favor of the matched reward-objective evidence.
- A second information-boundary audit found two remaining direct event-gate
  inputs in PPO: the rollout bootstrap value and the behavior-cloning warm-start
  batch. Both now use `online_event_context()`. Exact subtype labels remain only
  as training targets for the subtype guide and auxiliary classifier. A focused
  regression test verifies that rollout and warm-start actor inputs follow the
  warning scores even when exact simulator labels disagree.
- The complete two-seed matched reward pilot is inconclusive on reward ranking.
  Forecast reward beats AoI reward in `1/2` seeds with mean loss improvement
  `+0.004340` and beats uncertainty reward in `1/2` seeds with mean improvement
  `+0.000139`; both bootstrap ranges cross zero. All three variants beat their
  validation-selected fixed schedules in `2/2` seeds and have zero warm-up
  aborts. This supports expansion but not a forecast-reward superiority claim.
- The first formal expansion was stopped before completion after the residual
  event-gate input was discovered. The authoritative restart is tagged
  `20260718corrected24r2`; all 24 forecast seeds use the same corrected actor and
  critic input contract, and the matched AoI/uncertainty sweep is queued behind
  it. Earlier incomplete `corrected24` and `corrected24r1` directories are not
  evidence sources.
- The manuscript previously misdescribed the formal `subtype_auto` guide as a
  per-state forecaster-greedy label. The code instead maps training-partition
  calm/particle/flux/thermal labels to predefined feasible guide masks; the
  frozen forecaster supplies PPO reward. The method section now separates these
  channels and records the 10,000-step supervised warm start.
- A deeper inference-path audit found a separate and more consequential issue:
  when `subtype_router_enabled` was true, deterministic evaluation could map the
  auxiliary subtype classifier directly to a predefined action and bypass the
  PPO actor. The earlier positive two-seed pilot and all interrupted
  `corrected24*` expansions therefore represent a hybrid classifier-router/PPO
  policy, not the intended masked-PPO method. They are retained only as
  exploratory implementation history and cannot support the primary claim.
- The confirmatory restart now disables the hard router. Two bounded candidates
  are running on seeds 117 and 118: plain masked PD-PPO with no separate context
  encoder (`20260718cleanpilot`) and a clean context-aware variant that encodes
  the four online warning features but still executes only actor logits
  (`20260718capilot`). Process arguments were inspected directly and contain
  `--no-subtype-router`; the plain variant contains `--no-context-encoder`, and
  the context-aware variant contains `--context-encoder --context-feature-dim
  4 --context-layer-norm`.
- Exact subtype labels remain permissible only on the policy-learning partition
  as guide/auxiliary targets and reward stratification. The selected final
  policy must consume online warning proxies at rollout, bootstrap, warm start,
  and final inference, and must not use auxiliary-classifier routing.
- The plain actor-only pilot has now completed and independently clears the
  frozen two-seed gate. Seed 117/118 ordinary margins over the
  validation-selected static schedule are `+0.043472/+0.070925`; their
  validation-normalized subtype-macro margins are `+0.031334/+0.048653`.
  Both have zero aborts and pass the same action-trace complexity audit, with
  mask entropy `1.858/1.883` bits and subtype-mask mutual information
  `0.613/0.524` bits. This establishes a valid fallback primary method while
  the paired CA actor results remain pending; it does not pre-empt the frozen
  architecture-selection rule.
- The compact robustness addition is now specified in executable form. A
  multi-output ridge forecaster is fitted on exactly the forecaster-fitting
  partition and rollout mixture, a corresponding static mask is selected only
  on calibration/validation windows, and already frozen final trajectories are
  rescored. Exact subtypes enter only post-prediction loss grouping. This tests
  evaluator-family sensitivity without retraining the policy or selecting on
  final outcomes.

## 2026-07-18 Clean 24-Seed Main Evidence
- The actor-only forecast-reward policy completes seeds `117--140` and preserves
  the frozen information contract: no hard subtype router and no exact online
  simulator event flag.
- Validation-frozen macro evidence is uniformly positive against the
  validation-selected static schedule: `24/24` wins, mean margin `+0.080126`,
  95% bootstrap CI `[+0.067398,+0.093035]`, median `+0.083573`, and minimum
  `+0.019128`. Ordinary forecast loss also wins `24/24`, with mean margin
  `+0.157971` and CI `[+0.116100,+0.205185]`.
- The unchanged seeds `119--140` replicate this direction in `22/22` cases;
  their mean macro margin is `+0.083774` with CI
  `[+0.070685,+0.096705]`. The main result therefore does not rely on the two
  bounded architecture-selection seeds.
- The macro comparison is `24/24` against AoI, round robin, and random, and
  remains `24/24` against the post-hoc strongest of those three in each seed.
  This supports the primary adaptive-scheduling claim independently of the
  still-pending strong context, myopic, and privileged references.
- Behavior-complexity gates pass in all 24 clean traces. Each trace uses four
  specialist masks and is subtype-dependent; the weather backbone is mandatory
  and therefore always on, while the radiometer is unused. The meaningful
  learned decision is allocation among the thermo-hygrometer, surface IR,
  laser, and FC4 specialists.
- `scripts/86_v31_collect_validation_frozen_macro.py` previously assumed the
  truth CSV was copied into every run. Clean source-reuse runs intentionally do
  not duplicate it. The collector now resolves the metadata/control-source
  truth and verifies its SHA-256 before aggregation.
- Offline action grouping gives a clean mechanism match without hard routing:
  particle windows select the laser with mean duty `0.9928`, flux windows
  select FC4 with `0.9783`, and thermal windows select surface IR with `0.9914`.
  Calm windows distribute the specialist slot among thermo-hygrometer, surface
  IR, laser, and FC4. The validation-selected fixed policy is FC4 in 13 seeds,
  surface IR in 9, and laser in 2, so no single selected static specialist can
  cover all three event regimes.
- These near-deterministic event-window mappings must be interpreted together
  with the supplied synthetic warning scores and training-partition subtype
  guide. They establish correct actor execution and regime-dependent allocation,
  not robustness to an independently estimated field event detector.
## 2026-07-18 Clean strong-reference boundary

- The complete 24-seed strong-reference aggregate is
  `reports/aggregate/pdppo_framework_baselines_clean_24seed_20260718/`.
- One-step forecast-greedy is decisively weaker than PD-PPO: `24/24` wins and
  mean margins `+0.269041` (ordinary) and `+0.178989` (macro).
- Context-alert and exact-label policies are not beaten consistently. PD-PPO
  ordinary wins are `10/24` against each; macro wins are `11/24` and `12/24`.
  The mean macro margins (`+0.001222`, `+0.001768`) are negligible compared
  with the `+0.080126` primary margin against validation-selected static.
- The final paper must distinguish these information-advantaged references
  from fair deployable baselines. It may claim sequential value over myopic
  forecast-greedy selection, but not superiority over direct context routing.

## 2026-07-18 Double-DQN execution profiling

- The initial formal DQN expansion was compute-bound by sequential frozen-TCN
  inference on CPU, not by the value-learning updates. At the observed rate it
  would have delayed manuscript closure by several hours.
- Moving only the frozen training evaluator to CUDA reduced a 10,000-step run
  plus complete evaluation to 2 minutes 22 seconds. Candidate masks, reward
  weights, observations, seeds, transitions, optimizer settings, and total
  timesteps are unchanged.
- To prevent backend roundoff from entering paired final metrics, the formal
  implementation reloads the checksummed evaluator on CPU after training and
  uses that CPU instance for every final DQN/static/rule rollout. The collector
  rejects runs that do not record `cuda` for training scoring and `cpu` for
  final scoring.
- A full 4,096-window seed-117 backend audit measures the actual reward-level
  difference: mean absolute CPU/CUDA loss difference `0.000165`, or `0.0236%`
  of the CPU mean; the signed mean difference is `+0.000062`. The raw physical
  prediction maximum is dominated by the unscaled flux output and is not used
  as the equivalence criterion. The durable audit is
  `reports/analysis/pdppo_dqn_oracle_backend_audit_20260718.md`.
- The matched-reward collector now hashes normalized complete metadata after
  removing only output paths and `reward_proxy_mode`. A seed-117 three-mode
  remote smoke test passes with one shared protocol SHA-256, confirming that
  forecast, AoI, and uncertainty runs differ only in the intended reward proxy.

## 2026-07-18 Independent forecaster sensitivity

- The 24-seed ridge-forecaster rescore completed without changing policy
  checkpoints or trajectories. Each seed fits the ridge model on indices
  `0--24500`, selects its own fixed mask on four validation windows, and scores
  the original frozen final windows.
- PD-PPO beats the original fixed trajectory in `24/24` seeds under both
  ordinary and macro ridge losses. Against the ridge-selected fixed schedule,
  ordinary wins remain `24/24`; macro wins are `23/24`, with mean margin
  `+0.133435` and 95% CI `[+0.111065,+0.154450]`.
- Seed `129` is the only negative ridge-selected-static macro case
  (`-0.022431`). The post-pilot subset remains `21/22` with mean
  `+0.133948`. This is a strong evaluator-family sensitivity result, not a
  claim of joint scheduler/forecaster training or universal invariance.

## 2026-07-18 Matched Double-DQN control

- The complete protocol-matched Double-DQN comparison is
  `reports/aggregate/pdppo_matched_dqn_clean_24seed_20260718/`; its audit status
  is `passed` for all seeds `117--140`.
- PD-PPO has lower macro loss in `24/24` seeds, with mean DQN-minus-PD-PPO
  margin `+0.069719` and 95% CI `[+0.053916,+0.085406]`. Ordinary-loss wins
  are `23/24`, with mean margin `+0.140775` and CI
  `[+0.104129,+0.178748]`.
- Seed `121` is the sole ordinary-loss exception (`-0.003950`) but retains a
  positive macro difference (`+0.002521`). Double-DQN itself beats the fixed
  schedule in only `12/24` macro comparisons.
- Double-DQN passes the behavior gate in `21/24` seeds and has zero aborts;
  seeds `122`, `126`, and `131` fail the full action-trace criterion. PD-PPO
  passes in all 24. The comparison therefore supports the complete PD-PPO
  training package on the shared decision surface, not an isolated claim about
  the PPO clipping operator.

## 2026-07-18 Matched reward controls

- The formal same-architecture reward aggregate is
  `reports/aggregate/pdppo_clean_matched_reward_24seed_20260718/`. The strict
  protocol audit passes all 24 seeds and all three reward modes.
- Forecast reward does not materially separate from AoI reward: ordinary wins
  are `10/24` with mean AoI-minus-forecast difference `-0.000874` and macro
  wins are `13/24` with mean `+0.001005`; both confidence intervals contain
  zero.
- Forecast reward also does not materially separate from diagonal uncertainty:
  ordinary wins are `11/24` with mean `+0.000105`, and macro wins are `12/24`
  with mean `+0.000807`; both confidence intervals contain zero.
- Every reward mode beats its validation-selected fixed schedule in all 24
  seeds and passes the basic action-validity checks. The forecast objective
  remains the direct task definition, but the labelled guide, context auxiliary
  loss, warning scores, and small feasible action surface make the final policy
  insensitive to which of the three scalar proxies supplies the PPO reward.

## 2026-07-18 Complete final-partition sensitivity

- The primary 24-seed evidence is intentionally event-stratified: eight
  prespecified 512-epoch windows balance particle, flux, and thermal regimes.
  Window selection uses generated truth event coverage and transport intensity,
  never a policy loss. This is a valid primary estimand but was previously
  under-specified in the manuscript.
- Continuous replay over every epoch with a complete horizon provides a direct
  coverage check without retraining. The scoreable final interval is
  `[64750,69992)`, or 5,242 epochs; the last eight epochs have no complete
  eight-step future target.
- All 24 source metadata files record CPU oracle evaluation, the same final
  interval and start, the same frozen policies and selected fixed masks, and
  validation-fitted subtype normalizers. The aggregate rejects any different
  device, scope, step count, or seed set.
- PD-PPO remains ahead of validation-selected static in every seed: ordinary
  margin `+0.124728` on average (`24/24`, minimum `+0.009150`) and macro margin
  `+0.079260` (`24/24`, minimum `+0.013825`). The result direction therefore is
  not an artifact of selecting transport-rich primary windows.
- The continuous behavior fields have the expected structural endpoints: the
  mandatory weather backbone is always on and the unavailable/basic radiometer
  is always off. Useful specialist use remains dynamic: all four useful
  specialists have intermediate duty in 23 seeds and three do so in one seed;
  aborts are zero.

## 2026-07-18 Final manuscript interpretation

- The strongest supported claim is the complete framework result: clean
  actor-only PD-PPO learns a feasible, sparse, regime-dependent schedule that
  consistently beats validation-selected static and conventional/myopic
  references under the reported six-channel decision geometry.
- The reward controls do not isolate forecast loss as the sole empirical source
  of the gain. They instead show that the framework remains effective under
  three scalar objectives when the labelled guide, auxiliary context task, and
  warning inputs are shared. Proposition 2 continues to establish objective
  non-equivalence without claiming empirical reward superiority.
- Context-alert and exact-label mappings remain information-advantaged stress
  tests, not conventional rule baselines. Their near parity with PD-PPO is
  reported in Results and interpreted in Discussion.
- Full-partition replay and ridge rescoring address two separate robustness
  questions: evaluation-window coverage and evaluator-family sensitivity.
  Neither changes a policy checkpoint or permits final-test selection.
- The final active source contains no V3.1, SCENEBAL, no-warmup, env-dwell,
  hard-router, or separate-v1 terminology. The method, evidence, limitations,
  and conclusion now follow one stable narrative.

## 2026-07-19 Pilot-seed provenance reconciliation

- The apparent `117/122` versus `117/118` conflict comes from two different
  evidence contracts. Seeds 117 and 122 were inspected in the superseded July
  10 SCENEBAL-2 configuration-pivot line, which was later invalidated for the
  clean online-observability claim after the exact-event and hard-router audits.
- The current manuscript is based on the corrected actor-only contract. Its
  predeclared architecture gate compared plain and context-encoder actors on
  seeds 117 and 118. The frozen decision file
  `reports/aggregate/pdppo_clean_method_gate_20260718/clean_candidate_decision.json`
  selects `plain_pdppo` because the context encoder improved mean macro score by
  only 0.002891, below the frozen 0.005 materiality threshold; both candidates
  passed both seed gates.
- The selected plain actor was then expanded unchanged on seeds 119--140 under
  tmux tag `pdppo_clean_reward_formal22_20260718`. These 22 seeds are the current
  post-pilot replication. Historical archives must retain their original
  `117/122` wording, but all active submission materials must use the corrected
  `117/118` and `119--140` boundary.

## 2026-07-19 Submission compression audit

- The active Figure 1 is included as
  `paper/figures/figure_pdppo_framework_image2.pdf` and is generated by the
  tracked Matplotlib script `paper/figures/gen_fig_framework_and_support.py`.
  It is a programmatic vector/raster research figure, not an image-generation
  model output. The older standalone PNG is not the active included asset.
- The current 67-page named manuscript places Sections 1--8 on approximately
  pages 1--44. The highest-value low-risk moves are the physical-platform
  rendering, the complete fixed-schedule ledger, the generator-validation
  figure, the full-partition sensitivity table, and the ridge sensitivity
  table. Their numerical summaries must remain in the main text.
- The current Data Availability statement promises a future release. It must
  not be replaced with an anonymous-repository claim until a real accessible
  archive exists. A local anonymized submission/evidence package can be built
  now; public hosting remains a separate release action.

## 2026-07-19 Final package findings

- Moving audit material rather than shrinking typography reduced the named PDF
  from 67 to 50 pages and the anonymous PDF from 66 to 49 pages. The Conclusion
  ends before the theoretical appendix begins on page 42; the scientific body
  therefore remains readable while the review package is materially shorter.
- The 9-page supplement is self-contained and carries the exact history-feature
  equations, actor fusion, protocol quantities, reward-control recursions,
  platform rendering, simulator/observation specification, executable baseline
  definitions, continuous replay, ridge rescoring, generator diagnostics,
  fixed-mask ledger, and information-access audit.
- Anonymous isolation holds at four levels: empty PDF author metadata, no author
  strings in extracted PDF text, no author source files in the anonymous source
  archive, and no author identity or local home paths in the evidence archive.
- The active framework figure is programmatically generated by Matplotlib and
  `pdfimages -list` reports no embedded raster image objects. It is suitable as
  a non-generative, source-controlled research figure.
- The evidence archive is reproducible at the claimed level: aggregate checks
  recover the 24/24 fixed-schedule result, 5,242-epoch continuous replay,
  24/24 macro and 23/24 ordinary Double-DQN comparisons, and 23/24 ridge result;
  focused tests pass. It deliberately does not claim that omitted external
  meteorological anchors are redistributable.
- Data Availability remains the sole open gate. A local zip is not an accessible
  repository, so the manuscript correctly retains provisional release wording
  until an actual anonymous URL exists.

## 2026-07-19 Language-consolidation findings

- `docs/07-19-01.md` is directionally sound for terminology and tone, but its
  proposed title conflicts with the supervisor-approved title and was not
  adopted. Its future-tense Data Availability warning remains valid.
- Article-level comparator names are now `warning-score rule` and `exact-label
  reference`; `replay` is reserved for implementation descriptions. The metric
  is defined once as the validation-normalized equal-regime macro score and is
  shortened to `macro score` thereafter.
- The manuscript previously blurred a five-specialist action surface with the
  four specialist actions used in final traces. The revised Results text and
  Figure 3 caption state explicitly that the basic radiometer has zero final
  duty and that the remaining four specialist choices receive nonzero duty.
- Active-source residual scans found none of the targeted stale terms. Two old
  labels remain only in unreferenced historical table files and do not enter
  the canonical or anonymous build; they were left untouched to preserve
  provenance.
- The language pass does not require new experiments. Its persistent outputs
  are the canonical 51-page named manuscript, the 50-page anonymous manuscript,
  the 9-page supplement, and the checksum-verified ESWA submission package.

## 2026-07-19 Figure-system findings

- The main weakness identified by `docs/07-19-02.md` was presentation rather
  than missing evidence. The scientific experiment chain remains frozen.
- Connecting margins across an arbitrary seed order implied continuity that the
  experiment does not possess. Figure 2 now uses independent markers and keeps
  the ranked order only as a display aid.
- The evidence hierarchy is clearer when conventional/myopic comparators and
  stronger information/learner controls occupy separate panels. Double-DQN was
  therefore moved into Figure 2 and removed from Figure 4; no result was added
  or omitted.
- Figure 3 is the only main asset containing raster objects because its heatmap
  is rendered at 300 dpi. Its text, axes, and other geometry remain embedded in
  the PDF; Figures 1, 2, and 4 contain no raster image objects.
- The full macro metric name is retained only where defined or where claim
  scope requires it. `macro score` is the stable short form elsewhere.
- The canonical seed wording now appears in Setup: 24 evaluation seeds,
  including a 22-seed post-pilot replication after architecture selection on
  pilot seeds 117 and 118. Results retain the exact 119--140 replication detail.
- The supervisor-approved title was retained despite the report's alternative.
  The provisional Data Availability statement was also retained because a
  local archive is not an accessible anonymous repository.

## 2026-07-19 Proposition matrix and Figure 1 findings

- Figure 1's earlier weakness was structural rather than decorative: timeline,
  online execution, reward construction, and training supervision occupied one
  visual layer, so arrows crossed and the frozen forecaster could be mistaken
  for a deployment-time component.
- The accepted figure separates those responsibilities into three panels and
  uses solid paths for runtime execution and dashed paths for training-only
  updates. This makes the method boundary readable without changing the method
  or its claims.
- The Proposition 1 matrix now encodes compatibility through both luminance and
  explicit text (`matched`, `partial`, and `mismatch`). Its interpretation does
  not depend on color, and the grayscale page remains legible.
- No experiment, numerical result, proposition, or claim boundary changed in
  this phase. All modifications are source-controlled TikZ/Matplotlib artwork
  and matching caption/layout refinements.

## 2026-07-19 Typography and prose findings

- Visual similarity was insufficient for font consistency. The earlier figure
  set mixed DejaVu Sans and TeX Gyre Heros and used multiple plotting base
  sizes. The accepted assets now share one embedded family and size hierarchy.
- Scaling the Proposition matrix to line width enlarged its typography and hid
  the inconsistency. Fixed-size text and explicit cell geometry give a stable
  result at manuscript scale and preserve grayscale readability.
- The main first-person usage was confined to two abstract sentences, but
  colon-led explanations and contrast templates occurred throughout Methods,
  Results, and Discussion. Contextual sentence reconstruction produced cleaner
  transitions than global phrase replacement.
- Several factual contrasts were clearer as positive definitions. For example,
  the evaluator comparison now states that forecaster parameters remain fixed,
  and the Double-DQN paragraph identifies a package-level learner comparison
  directly.
- Longer natural-language table headers required visual inspection even though
  LaTeX reported no overflow. Explicit line breaks and a smaller five-column
  control table eliminated column crowding.
- This pass changes presentation only. Experimental values, seed accounting,
  comparator information access, proposition statements, and claim boundaries
  remain unchanged.

## 2026-07-19 Draw.io MCP and Figure 1 findings

- The maintained official implementation is `@drawio/mcp`, published by the
  draw.io project. It was registered at user scope with Codex as a stdio server
  using `npx -y @drawio/mcp`; the existing GitHub MCP entry was not changed.
- A direct MCP handshake reported server version 1.4.1. The native
  `list_pages`, `get_page`, and `open_drawio_xml` calls successfully recognized
  the new one-page source and produced an editable diagrams.net URL.
- The old Figure 1 represented the correct three-layer distinction but reduced
  the online policy to generic score boxes. The accepted draw.io version shows
  all six logical channels, the sample-and-hold update, the full scheduler
  observation, parallel state/history and warning-context encoders, gated
  fusion, candidate-mask logits, hard feasibility checks, the masked action
  distribution, and the executable subset feedback loop.
- Training-only information is confined to a separate dashed panel containing
  the frozen forecaster, forecast-loss reward, PPO objective, and supervised
  guide/context terms. The final-execution boundary is therefore visible from
  geometry and line style as well as color.
- Four rendered revisions were inspected. Rejected versions contained clipped
  text, vertically stacked encoders, or training connectors crossing labels.
  The accepted version has no overlaps at manuscript scale and remains
  interpretable in grayscale.
- The active PDF is pure vector artwork with no embedded raster objects. Its
  only embedded font family is DejaVu Sans, and the editable `.drawio` source is
  included in the anonymous source bundle.

## 2026-07-19 Figure 1 iconography review

- Figures 2--4 use DejaVu Sans, the Okabe--Ito blue/teal/orange palette, dark
  charcoal headings, slate body text, flat fills, and no shadows. Figure 1 now
  follows the same visual vocabulary instead of introducing a separate icon or
  font style.
- Draw.io-native vector symbols were assigned by function. Line-chart,
  network, control, target, antenna, waveform, document, shield, filter,
  checked-document, and update-cycle symbols identify the protocol, sensing,
  policy, feasibility, execution, reward, and learning modules. No raster art,
  emoji, or generated imagery is present.
- The first iconized rendering established that the symbols remained legible,
  but several labels competed with their icon columns. The second rendering
  shortened descriptions and restored the type hierarchy. The third rendering
  removed the sensing-title wrap and isolated one remaining wrap in the masked
  action box for final correction.
- The main figure now exposes five algorithmic ideas at a glance: chronological
  separation, partial-observation construction, context-aware policy encoding,
  hard feasibility masking, and forecast-loss PPO learning. The iconography
  supports those ideas without adding modules or changing the method claim.
- The final source was persisted through Draw.io MCP `set_page` and verified by
  MCP `get_page`. The accepted masked-action wording is `Filter infeasible
  masks`, `Sample in training`, and `Top mask at test`.
- At the manuscript's full-width placement on page 4, the diagram is legible in
  colour and grayscale. Panel boundaries, solid online edges, dashed training
  edges, and icon silhouettes preserve the hierarchy without relying on hue.
- The exported PDF contains no raster image objects and embeds only DejaVu Sans
  variants. The icon additions therefore retain the paper's vector-artwork and
  typography requirements.

## 2026-07-19 Figure 1 spacing findings

- The apparent inconsistency came from icon-to-label geometry in five online
  modules, not from the paper's caption settings. The previous 42-unit icon
  cards ended only 1--2 source units before the text inset in the sample update,
  scheduler observation, feasibility mask, masked policy, and executable subset.
- Expanding the inset alone created unnecessary title and body wrapping. The
  accepted geometry instead reduces those icon cards to 34 units and sets the
  text inset to 54 units. The resulting 6--7-unit gap remains visible at page
  scale while preserving usable label width.
- `Exclude invalid masks` replaces the longer masked-policy phrase so the label
  hierarchy remains compact after the spacing correction. This is presentation
  wording only and does not change action selection.
- Manuscript pages containing Figures 1--4 were reviewed together. Figures 2--4
  have consistent plot-to-caption spacing and no comparable internal crowding,
  so a global `textfloatsep` or caption-skip change would have degraded otherwise
  acceptable pages.

## 2026-07-19 ESWA-reference Figure 1 hierarchy findings

- Three primary ESWA examples were inspected: an engineering-optimization
  workflow, a short-time-series prediction framework, and an RL intrusion-
  response architecture. Their recurring design pattern is a single dominant
  process, compact supporting material, large labels, and sparse semantic
  connectors. The old Figure 1 used consistent colours but assigned nearly
  equal visual weight to protocol, execution, and training.
- The accepted redesign uses a 1600 x 900 canvas. The chronological protocol is
  a compact top strip, the online constrained scheduling loop occupies the
  dominant central area, and training-only updates form a shallow lower strip.
  This improves the manuscript-scale type size without deleting method content.
- The first hierarchy rendering retained obsolete connector waypoints and made
  the masked-policy description overflow. It was rejected. The second rendering
  fixed those defects but added three edge labels in gaps too narrow to support
  them; the labels and a redundant flow legend were rejected as visual clutter.
- The third rendering confines the black execution feedback loop to the online
  panel and gives the two dashed training feedback edges a separate route above
  the training modules. No connector crosses a module or label.
- Colour and grayscale standalone reviews show that panel boundaries, solid and
  dashed paths, line weight, and typographic hierarchy remain distinguishable
  without hue. The exported PDF contains no raster image objects and embeds
  only DejaVu Sans variants.
- Manuscript-scale review identified one local layout issue outside the artwork:
  the top float retained excess whitespace after the figure became shorter. A
  Figure-1-only `-1.2em` vertical adjustment removes approximately one text line
  without changing global float spacing or the placement of Figures 2--4.
- Both named and anonymous manuscripts remain 50 pages. The anonymous PDF has
  an empty Author field, and the staged manuscript text and standalone Figure 1
  match their canonical counterparts by hash.

## 2026-07-19 Proposition 1 figure reconstruction findings

- The previous unnumbered matrix had a scientific-communication defect, not
  merely a style defect. Its `matched`, `partial`, and `mismatch` cells looked
  like measured compatibility data, although `partial` had no mathematical or
  empirical definition. It also mixed simulator channel names with a general
  sufficient condition and did not show why fixed allocation must fail.
- The replacement is a numbered conceptual schematic. Panel A defines an
  illustrative three-regime instance with incompatible optima. Panel B follows
  one fixed specialist across regimes and exposes positive excess losses. Panel
  C follows the regime-specific optima when temporal rules permit. The lower
  strip reproduces the macro-loss inequality proved in Appendix A.1.
- The caption explicitly identifies the diagram as conceptual and defers the
  sensor-specific mapping to Section 5. The proposition, proof, and experiment
  are unchanged; the figure no longer implies unsupported compatibility data.
- Five rendered revisions were required. Rejected versions had panel overflow,
  title/subtitle collisions, overlapping matrix cells, an 87.2-pt manuscript
  overrun, or a float that appeared before the proposition had finished. The
  accepted source is natively 13.05 cm wide and uses `[H]` only for this figure
  so that the proposition always precedes its illustration.
- The standalone and embedded versions remain legible in colour and grayscale.
  The PDF is vector-only. Adding the numbered schematic shifts the former
  Figures 2--4 to Figures 3--5; label-based references compile without errors.

## 2026-07-19 Figure 2 manuscript-spacing correction findings

- A zoomed manuscript-page audit exposed defects that were less apparent in
  the standalone view: panel titles touched two-line subtitles, the formula
  frame sat too close to the panels, and automatic hyphenation broke several
  short labels.
- Increasing panel height from 4.35 cm to 5.00 cm created independent vertical
  bands for titles, explanatory text, matrices, and conclusions. Explicit line
  breaks removed the unwanted label hyphenation.
- The lower formula frame was moved from y=-3.10 cm to y=-3.25 cm after the
  final visual review. Its clearance from the upper panels increased from
  approximately 0.21 cm to 0.36 cm without changing the caption or page break.
- The final embedded Figure 2 was inspected on manuscript page 12 after the
  rebuild. No panel, formula, caption, or surrounding-text overlap remains.

## 2026-07-19 07-19-03 refinement findings

- The current clean-evidence source already preserves the required 24-seed and
  22-seed post-pilot boundaries; the refinement is editorial and graphical.
- `scripts/95_v31_build_clean_paper_assets.py` is the authoritative generator
  for Figures 3--5 and is normally invoked through the `darts` conda environment.
- Its default manifest target is under `reports/aggregate`; a new optional
  `--manifest-output` path permits regeneration without modifying frozen reports.
- Figure 1 now has a tracked Matplotlib generator and a compact chronology,
  online loop, and dashed training-only path.
- The final source-to-backup comparison confirms that editorial changes did not
  alter numerical tokens, citation keys, displayed equations, or Proposition 1.
- All four build logs are free of undefined citations, undefined references,
  and rerun warnings. Each retains the same 1.79993-pt output-routine overfull
  box in front matter; no table, figure, or body-text overflow remains.
- The active Data Availability statement still avoids a public-URL claim. A
  public anonymous repository URL remains an external submission-stage gate.
- The submission package was intentionally not rebuilt or edited; the
  orchestrator retains responsibility for its post-verification rebuild.

## 2026-07-19 Main-figure insertion audit findings

- The figure counters and labels were valid, but the reader-facing insertion
  chain was not. Figure 1, Figure 3, and Figure 5 had no explicit textual anchor,
  while the explanatory sentences for Figure 2 and Figure 4 followed the
  artwork.
- The latest copy-editing pass had also replaced the maintained 69 kB Draw.io
  architecture with a 13 kB workflow sketch. The sketch duplicated much of the
  chronological information in Supplementary Fig. S1 and omitted the sensing
  channels, observation construction, dual encoders, gated fusion, candidate
  masks, runtime checks, and training-supervision boundary.
- The maintained `figure_pdppo_framework_drawio.pdf` was restored as Figure 1.
  The compact `figure_pdppo_workflow.pdf` remains a historical local asset but
  is no longer referenced or included in the anonymous source bundle.
- Compiled-page review confirms the corrected order: Figure 1 follows its
  Introduction anchor on page 4, Figure 2 follows the proposition discussion on
  page 12, and Figures 3--5 follow their Results anchors on pages 30, 32, and
  35. No figure crosses a section boundary or creates a blank page.

## 2026-07-19 Figure 1 online-execution boundary clarification

- `NOT USED IN FINAL EXECUTION` was intended to state that the lower feedback
  path does not participate in online action selection. It was ambiguous because
  the frozen forecaster is still used to score final trajectories offline.
- The Draw.io source now labels Panel C as `Offline scoring and training
  updates`. The first replacement badge, `NOT USED FOR ONLINE ACTION SELECTION`,
  was accurate but too crowded at manuscript scale; the accepted badge is the
  concise `OFFLINE ONLY`. The caption retains the full semantic explanation.

## 2026-07-22 strict theory and implementation-consistency findings

- The review memo correctly identified that the former action-set equation
  omitted temporal state. Source inspection exposed a more specific issue: the
  implemented categorical mask filters power and startup feasibility, while
  `WarmupSchedulingEnv._apply_min_dwell_guard` enforces the minimum duration by
  retaining the complete previously executed subset. The paper now distinguishes
  the actor proposal $\bar a_t$ from the executed mask $a_t=G_t(\bar a_t)$.
- The candidate-mask builder was executed with the reported sensor file and
  constraints. It generates exactly six masks, including the weather-backbone-only
  mask. That mask costs 0.25 under a 0.75 steady-power budget and has 0.30 startup
  peak under a 0.95 peak budget, so the masked action index set is nonempty.
- Proposition 1 previously mixed a general regime-weight vector with the paper's
  equal-regime macro score. It now uses a defined weighted normalized objective
  $\mathcal{M}_{\rho}$ and identifies the reported score as its equal-weight
  three-event special case.
- Proposition 2 previously invoked a linear-Gaussian setting without specifying
  a covariance recursion and simultaneously used a binary event variable. The
  proof now uses two independent Gaussian random walks, exact age-dependent
  posterior variances, and an explicit parameter interval in which total AoI and
  covariance trace prefer one schedule while forecast MAE prefers the other.
- The former reward equation included a residual violation penalty that is absent
  from the reported implementation. The revised equation contains normalized
  forecast loss, switching cost, and warm-up-abort cost; hard power/startup rules
  and the deterministic duration guard are described outside the reward.
- The abstract count was 253 under a conservative tokenization. Removing the
  pilot/replication aside reduced it below the 250-word journal limit while the
  seed provenance remains explicit in the protocol and Results sections.
- Draw.io exported Figure 1 as PDF 1.7. Ghostscript rewrote the generated vector
  asset as PDF 1.5, removing the `pdflatex` inclusion-version warning without
  changing the editable Draw.io source.

## 2026-07-22 citation and source-alignment findings

- The citation report's main conclusion was correct: the active 23-reference
  bibliography contains real sources, but one DOI was wrong and several records
  lacked type or proceedings metadata. The Fernández-Bes DOI is now corrected;
  Tran is explicitly a preprint; Monrad-Krohn is explicitly a PANGAEA dataset;
  and the Wei, Pendyala, and Murad proceedings records are complete.
- Crossref confirms the Fernández-Bes author as Antonio G. Marques. The report's
  proposed `García-Marques` spelling was therefore not applied.
- The former objective survey used one long citation cluster. Splitting it into
  estimation, censoring, freshness, tracking, and delay-aware remote-estimation
  sentences removes the implication that every cited paper supports every
  objective class.
- The forecasting citations support architecture families, not a published
  method for scoring schedule-induced observations. The revised text states
  that they provide candidate backbones and identifies the common frozen
  forecaster as this paper's protocol choice.
- Monrad-Krohn et al. report cold-region particle measurements from
  Ny-Ålesund, not Antarctic measurements. The simulator text now separates the
  Antarctic AWS and drifting-snow anchors from numerical modelling and
  cold-region particle measurements.
- The main bibliography still contains exactly 23 entries. Five equipment
  manual keys seen in an initial repository-wide table scan do not enter
  `main.bbl`; the report was therefore aligned with the compiled main paper.
- Full validation evidence is recorded in
  `reports/analysis/pdppo_citation_revision_20260722.md`.
