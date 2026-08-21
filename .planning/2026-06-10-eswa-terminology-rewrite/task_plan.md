# Task Plan: ESWA Terminology and Narrative Rewrite

## Goal
Rewrite the active English PD-PPO manuscript for an Expert Systems with
Applications style: clear problem setting, stable terminology, fewer internal
project names, and a readable abstract/introduction. The rewrite must be manual
and argument-level, not a script-only word replacement pass.

## Scope
- Active source:
  `paper/main.tex` and `paper/sections/*.tex`.
  `paper/pdppo_crst_rewrite.tex` and `paper/rewrite_sections/*.tex` are
  historical rewrite artifacts and are no longer the canonical editing source.
- Do not edit `paper/raw.tex`.
- Keep numerical results unchanged unless a later completed evidence block has
  already been migrated into the canonical manuscript. The current main evidence
  is the final fixed-backbone, one-specialist 24-seed aggregate over seeds
  `117--140`.

## Reference Style
Use ESWA scheduling articles as style anchors:
- problem-first title and abstract;
- terms such as "scheduling problem", "scheduling policy", "operational
  constraints", "state/action/reward", "baseline policies";
- avoid internal route names, version IDs, and posthoc diagnostic labels.

## Phases
- [x] Back up the current active manuscript source.
- [x] Inspect ESWA article positioning and current manuscript terminology.
- [x] Rewrite abstract, keywords, and abbreviation block.
- [x] Rewrite Introduction around the generic power-constrained sensing-system
  scheduling problem before the Antarctic case study.
- [x] Rewrite Method/Formulation sections with a stable terminology set.
- [x] Rewrite Results/Discussion to use baseline-family labels consistently.
- [x] Compile and run one terminology/style audit.
- [x] Apply the approved specialist-bottleneck theory extension to the canonical
  manuscript source.
- [x] Rebuild `paper/main.pdf` and check references/citations after the theory
  application.
- [x] Add the final design-balance explanation and current data-availability
  statement to the canonical manuscript.
- [x] Rebuild `paper/main.pdf` and verify the new evidence statements appear in
  rendered PDF text.
- [x] Research and decide the next supplementary experiment package for the
  current final fixed-backbone, one-specialist benchmark.
- [x] Implement a current-protocol mechanism-ablation runner and collector.
- [x] Run a six-seed mechanism-ablation pilot before any larger same-protocol
  seed expansion. Completed on `remote-gpu`; aggregate summary:
  `reports/aggregate/mechanism_ablation_pilot_117_118_119_120_121_122_20260621/mechanism_ablation_summary.md`.
- [x] Run small benchmark-perturbation robustness pilots if the mechanism pilot
  confirms a clear component-level story. Completed on `remote-gpu`; aggregate
  summary:
  `reports/aggregate/robustness_pilot_141_142_143_144_145_146_20260622/robustness_summary.md`.
- [x] Defer the bounded PPO action-level subtype auxiliary pilot for the
  `weaker_latent10` failure mode after the user decided that no further
  exploration is needed for the current relative-strong claim. The remote tmux
  session `actionaux_20260622` was stopped before completion.
- [x] Apply the completed mechanism and event-mixture robustness evidence to
  `paper/main.tex` and active section/table inputs.
- [x] Remove unsupported or non-headline boundaries from the manuscript narrative:
  raw unnormalised macro sensitivity, weaker-latent failure, and unfinished
  action-auxiliary pilot results remain internal evidence, not paper claims.
- [x] Rebuild `paper/main.pdf` and run source/PDF residual scans for internal
  terms and unsupported-boundary wording.
- [x] Audit active manuscript figures and tables for readability.
- [x] Apply one unified publication style to active result figures and tables.
- [x] Rebuild `paper/main.pdf`, render the affected pages, and perform an
  iterative readability check.
- [x] Record the figure/table style decisions and final verification.
- [x] Back up the manuscript before applying the 2026-06-23 PPO-GPT and
  PPO-LEMMA review recommendations.
- [x] Apply the PPO action/log-probability correction to the canonical
  manuscript: PD-PPO is now described as masked categorical PPO over executed
  candidate masks, not as channel scores followed by an unaccounted projection.
- [x] Add the required supplementary evidence objects from the 2026-06-23 review
  docs: action-space instantiation, online observability audit, protocol and
  hyperparameter table, event-type loss decomposition, event-type diagnostic
  figure, fixed-mask selection summary, and per-seed fixed-mask selection
  ledger.
- [x] Replace count-only result visualisation with a seed-level paired-margin
  distribution panel for the main 24-seed result.
- [x] Complete the required subagent review loop: one completeness review over
  the two 2026-06-23 docs and three wording/terminology audit rounds.
- [x] Back up the active paper before the stricter 2026-06-23-02 PPO-LEMMA
  implementation pass:
  `paper_archives/paper_pre_ppo_lemma_062302_20260623_025244.tar.gz`.
- [x] Parse `docs/06-23-02-PPO-LEMMA.md` into a strict, checkable
  implementation matrix, with priority levels and explicit evidence artifacts.
- [x] Audit existing reports, tables, figures, and manuscript text against the
  06-23-02 matrix. Mark every missing high-priority experiment or analysis item
  as `missing`, not as implicitly satisfied.
- [x] Complete the high-priority experiment/analysis supplements required by
  06-23-02. Long-running experiment work must use `remote-gpu`; no paper claim
  may cite unfinished runs.
- [x] Generate or update the required report files, tables, and figures from
  completed evidence only.
- [x] Apply manuscript changes for completed evidence, keeping unsupported
  boundaries out of the paper body.
- [x] Rebuild `paper/main.pdf` and run claim, terminology, and stale-artifact
  scans.
- [x] Use a subagent to audit whether all 06-23-02 requirements, especially
  high-priority experiment supplements, have been fully implemented.
- [x] Close the stricter 06-23-02 pass only after the implementation matrix,
  experiment artifacts, manuscript, PDF checks, subagent audit, and planning
  files all agree.
- [x] Run the 06-23-03 ESWA review pass from `docs/06-23-03.md`: build a fresh
  requirement matrix, back up the manuscript, and close the high-priority
  table/figure blockers without adding unsupported evidence.
- [x] Redesign Table 11 so point estimates and bootstrap confidence intervals
  are visually aligned by policy variant.
- [x] Make the event-type specialist-selection panel explicitly read as a
  cross-seed expert-selection frequency heatmap, with matching caption and
  results text.
- [x] Apply medium/low-priority narrative fixes that do not require new remote
  experiments: Section 6 evaluation-map bridge, Discussion bias/practicality
  paragraphs, contribution-boundary tightening, event-mix limitation, and
  Table 11 delta interpretation.
- [x] Rebuild `paper/main.pdf`, inspect affected pages, and run equation,
  terminology, and unsupported-evidence scans.
- [x] Audit the fixed TCN validation request from 06-23-03. The corrected
  24-seed diagnostic is negative against persistence and is recorded as
  internal evidence only, not migrated into the manuscript.
- [x] Monitor the stricter rare-flux event-mixture run queued on `remote-gpu`
  in tmux `eswa062303_flux10`; sync and analyse it after completion. The run
  passed all evidence gates and is analysed in
  `reports/aggregate/rare_flux_event_mix_result_analysis_20260625.md`.
- [x] Apply the completed rare-flux robustness result to the manuscript
  robustness table and discussion wording using cautious, benchmark-local
  language. The paper now presents two fresh six-seed changed-mixture
  sensitivity checks, not a universal rare-event robustness claim.
- [x] Run a post-Codex manuscript polish pass on `paper/main.tex` and active
  included sections. This pass preserved all numerical claims, removed residual
  reader-steering / AI-style phrasing, kept the fixed-forecaster and
  benchmark-local boundaries explicit, rebuilt `paper/main.pdf`, and verified
  source/PDF terminology scans.
- [x] Apply the deeper 2026-06-25-01 PPD rewrite from
  `docs/06-25-01-PPD.md` after backing up the manuscript. This pass changed the
  title and abstract to a problem-first specialist-scheduling frame, rewrote the
  Introduction around explicit H1--H3 hypotheses, added a related-work
  positioning table, moved training-reward details into the method section,
  lowered main-text proof density, compressed repetitive results/discussion
  wording, rebuilt `paper/main.pdf`, and verified source/PDF residual scans.
- [x] Apply the 2026-06-25 06-15-comparison style repair after the user flagged
  renewed AI-style prose, excessive hyphenated labels, and inconsistent coined
  terminology. This pass compared the active manuscript against the closest
  around-06-15 compiled/source reference (`paper/pdppo_crst_rewrite.pdf`,
  2026-06-12), restored more natural wording, removed reader-visible
  colon-style exposition, removed first-person residue, converted over-coined
  labels such as `fixed-mask`, `one-specialist`,
  `static-reference-normalised`, `regime-balanced`, `event-context`, and
  `step-margin` back into plain technical language, regenerated affected
  figures, rebuilt `paper/main.pdf`, and verified source/PDF/log/diff checks.
- [x] Apply the 2026-07-01 LEMMA fourth-round review report from
  `docs/07-01-01-LEMMA.md` after backing up the active manuscript. This pass
  clarified Proposition 1 as an oracle/event-label replay upper-bound result,
  added benchmark support for the incompatible-specialist premise, expanded the
  non-equivalence proof scope, moved forecaster/training details into the method
  section, moved generator validation checks to the appendix, split mechanism
  ablation from robustness checks, clarified duplicate heatmap usage, narrowed
  the PD-PPO claim against non-RL references, rebuilt `paper/main.pdf`, and
  verified source/PDF/log/diff checks.
- [x] Apply the 2026-07-01 N-series follow-up fixes. This pass corrected appendix
  float ownership after generator validation moved to the appendix, fixed the
  rendered PD-PPO loss label in Eq. (11), removed duplicated Appendix references
  and a doubled period in Section 3.2, verified that Table 6 is the comparison
  reference taxonomy, rebuilt `paper/main.pdf`, and verified source/PDF/log/diff
  checks.
- [x] Apply the 2026-07-01 reference-audit corrections. This pass fixed the
  Golovin--Krause and AntAWS severe metadata errors, updated formal venue/year
  and volume/page/article-number data for audited references, added foundational
  PPO and TCN citations, removed the weak Liang 2024 graph-coverage citation,
  rebuilt `paper/main.pdf`, and verified `.bbl`/PDF/log/diff checks.
- [x] Pin the final code revision and prepare a versioned evidence archive for
  submission. The archive should contain the exact code revision, final aggregate
  tables, seed-level summaries, figure assets, manuscript source, and
  reproduction scripts for the current ESWA evidence package.
- [x] Run the framework-first supplementary experiment package requested on
  2026-07-02. This package should support the PD-PPO framework framing, not a
  benchmark-only framing. Minimum evidence: forecast-greedy / one-step
  lookahead diagnostic baseline, context-alert bandit baseline, and a
  PPO reward-ablation plan or run comparing forecast-loss reward against
  AoI/uncertainty-style rewards. Long-running work must use `remote-gpu`.
- [x] Aggregate the framework supplementary results into a durable report under
  `reports/aggregate/`, including seed-level CSV, summary JSON/CSV/Markdown,
  and explicit interpretation of whether each supplement supports or challenges
  the PD-PPO framework claim.
- [x] Improve PD-PPO against the `context_alert_bandit_t0p5` challenge without
  changing the main method into a bandit-dependent patchwork. Primary candidates:
  feature-parity PD-PPO and CA-PD-PPO, both preserving forecast-loss reward and
  feasibility-masked PPO. Forbidden for the primary method: residual action over
  bandit, bandit-margin reward, counterfactual improvement labels, and
  bandit-dependent actor priors.
- [x] Complete the feature-parity audit: identify online observables used by the
  context-alert bandit, compare them with original PD-PPO observations, and add
  missing online-only alert/context features without using simulator event labels
  at final evaluation.
- [x] Implement CA-PD-PPO as a simple context-aware extension of masked PPO:
  shared observation encoder plus alert/context encoder fused before the actor
  head, with the original forecast-loss reward and feasibility-masked action
  distribution unchanged.
- [x] Run fresh development-seed comparisons among original PD-PPO,
  feature-parity PD-PPO, CA-PD-PPO, fixed mask, forecast greedy, and
  context-alert bandit. Report step and macro margins against bandit, win
  counts, bootstrap 95% CIs, behaviour diagnostics, and whether forecast greedy
  remains beaten.
- [x] Apply the decision rule: continue to a fresh final 24-seed evaluation only
  if CA-PD-PPO has positive mean macro margin against bandit and at least 15/24
  macro wins on development seeds; if it only matches, frame it as competitive
  with a strong context-aware rule; if it loses, keep the bandit as a diagnostic
  baseline and do not promote patchwork modules.
- [x] Run a final submission-package audit after the archive is pinned: verify
  `paper/main.pdf`, source/PDF terminology scans, table/figure asset presence,
  author metadata, data/code availability wording, and that no historical release
  is cited as the current evidence package.

## Terminology Decisions
| Old / risky term | Replacement / rule |
|---|---|
| power-limited / power-constrained sensing system | avoid as a system label; use "sensing system scheduling under a power budget" |
| prediction-driven scheduling as the general problem | introduce general scheduling first; prediction-driven is the proposed formulation/method |
| frozen oracle | fixed forecast evaluator; use "forecast oracle" only once if needed |
| deployment constraints | avoid; use "power budget and operating rules" or name the specific rule |
| operational constraints | use sparingly; prefer "operating rules" for minimum activation time, duty cycle, and switching |
| dwell | minimum activation time / minimum on-time |
| duty | duty cycle; define as the fraction of epochs in which a channel is active |
| compact static | static subset baseline |
| deployable static | static-priority baseline or static-priority duty replay |
| candidate prior | candidate-policy guide |
| AWBC | expand as advantage-weighted imitation loss in reader-facing prose and tables |
| FW-MAE | avoid in the active manuscript; use the current step-weighted forecast loss terminology |
| freshness | avoid alone; define as AoI / time since the last valid observation |
| sensor channels | use "channels of the sensing system" or "logical sensing channels of the system" |
| heuristics | prefer "rule-based scheduling baselines"; use heuristic only after examples are clear |
| duty-constrained baseline | use "baseline under the same duty-cycle rule/limit" |

## Notes
- The paper can still use PD-PPO as the method name.
- The Antarctic AWS benchmark should be introduced as the evaluation case, not
  as the first sentence of the paper.
- Source-level residual "oracle" strings are internal LaTeX labels or file names,
  not reader-visible terminology. A PDF text scan found no visible hits for the
  old terminology set.
- The current supported claim is attached to the final fixed-backbone,
  one-specialist 24-seed aggregate, ordinary step gates, behaviour gates,
  true-static gates, and static-normalised event-regime macro scoring. Raw
  unnormalised macro sensitivity and weaker-latent failure evidence should not
  be written into the paper unless a later revision explicitly needs a
  limitations appendix.
- The mechanism-ablation and event-mixture robustness pilots are complete and
  have been migrated into the active manuscript as concise tables and narrative.
- `SCENEBAL-2` is an internal experiment/archive identifier, not a reader-facing
  manuscript concept. In the paper body, highlights, captions, and visible
  figure text, use problem-level language such as "the final fixed-backbone,
  one-specialist benchmark" or "the regime-balanced benchmark". Internal file
  names, labels, scripts, and archive manifests may keep the identifier for
  reproducibility.
- The previous public release link is historical. The current paper needs a new
  versioned archive containing the final aggregate tables, seed-level summaries,
  figure assets, and reproduction scripts before submission.
- The first robustness block is mixed: event-mix shift passes `6/6`, while
  weaker latent separation passes only `5/6` because seed `144` fails behaviour
  complexity and macro/strict gates. This should trigger a bounded algorithmic
  pilot only if future work pursues a stronger robustness claim. It is not
  required for the current relative-strong claim.
- Active result figures now use a shared manuscript plotting style implemented
  in `paper/figures/paper_plot_style.py`: colorblind-safe palette, serif font
  matching the paper, vector PDF outputs, no in-image global titles, compact
  gate labels, and natural figure widths close to the ESWA single-column text
  width so LaTeX does not shrink labels below readable size.
- After the 2026-06-23 PPO review application, reader-visible method wording
  uses "masked PPO", "candidate masks", and "online feasibility masking". Avoid
  reintroducing "projected/channel-score PPO" unless the actual probability
  model is also changed.
- The required subagent loop for the 2026-06-23 PPO-GPT / PPO-LEMMA application
  is complete: one document-coverage review plus three wording/terminology
  audit rounds. The final read-only recheck found no remaining P1/P2 issues.
- The stricter 2026-06-23-02 PPO-LEMMA pass is complete for all high-priority
  requirements. The 24-seed mechanism ablation now appears in Table 11 with
  mean and median macro-margin 95% bootstrap intervals and in Figure 8 as
  continuous seed-level margins. The final subagent recheck found no P0/P1
  blockers; remaining P2 notes are visual polish only.
- The 06-23-03 ESWA review pass starts from backup
  `paper_archives/paper_pre_062303_20260624_000251.tar.gz` and requirement
  matrix
  `reports/aggregate/ppo_lemma_062303_requirement_matrix_20260624.md`.
- The 06-23-03 paper-side pass is complete for high-priority items and local
  medium/low-priority items. The stricter rare-flux event-mixture run has now
  completed, has been analysed locally, and has been migrated into the
  robustness table and discussion wording with cautious benchmark-local
  language.
- The stricter rare-flux event-mixture run completed and passed all gates:
  operational step `6/6`, true-static macro `6/6`, strict true-static step
  `6/6`, and behaviour `6/6`. It strengthens the event-mixture sensitivity
  evidence but should be migrated carefully as a second six-seed changed-mixture
  check, not as a universal robustness claim.
- The rare-flux result has now been migrated into the active manuscript. The
  robustness table includes higher-flux and lower-flux changed-mixture rows, the
  Results text says the claim is local to changed mixtures within the same
  benchmark family, and the Discussion explicitly rejects arbitrary event
  prevalence or detector robustness.
- The 2026-06-25 06-15-comparison style repair changes the preferred visible
  wording again. In reader-facing prose, captions, tables, and figure text,
  avoid turning descriptive phrases into project labels. Prefer `fixed mask`,
  `single specialist`, `macro score normalised by static references`, `event
  context`, `minimum duration`, and `final test` over hyphenated labels such as
  `fixed-mask`, `one-specialist`, `static-reference-normalised`,
  `event-context`, and `final-test`.

## 2026-07-03 CA-PD-PPO Dev2 Checklist
- [x] Analyse CA-PD-PPO losing seeds against `context_alert_bandit_t0p5`.
- [x] Write diagnostic artifacts under
  `reports/analysis/ca_pdppo_failure_20260703/`.
- [x] Implement method-consistent bounded dev2 hooks without bandit-dependent
  modules.
- [x] Validate local and remote syntax/tests.
- [x] Launch bounded dev2 on `remote-gpu` tmux session
  `ca_pdppo_dev2_20260703`.
- [x] Monitor completion of `ctx128`.
- [x] Monitor completion of `gated`.
- [x] Monitor completion of `gated_ctx128`.
- [x] Monitor completion of `nsteps2048`.
- [x] Aggregate `reports/aggregate/ca_pdppo_bounded_dev2_20260703capdppodev2/`.
- [x] Apply the predeclared final-launch gate and either start fresh final
  seeds 301--324 or stop with CA-PD-PPO as competitive/exploratory evidence.

## 2026-07-10 Post-Hermes Experiment-Evidence Audit
- [x] Trace every active main-text numerical claim to the SCENEBAL-2 24-seed
  aggregate and its seed-level source files.
- [x] Check protocol integrity: chronological split, final-test isolation,
  fixed-schedule selection/replay, baseline feasibility parity, and the
  independence/meaning of the seed dimension.
- [x] Recompute or independently cross-check reported win counts, margins,
  confidence intervals, sign tests, and action-trace gates from the durable
  result artifacts.
- [x] Audit the paper for metric-boundary errors, especially ordinary loss
  versus static-normalised regime macro scoring and learned-policy versus
  privileged event-label replay.
- [x] Classify findings as correction-required, supplementary-evidence-needed,
  or no-action; manuscript edits are deferred until the frozen-normalizer
  reaggregation is generated.
- [x] Repair the macro collector and regenerate all active macro-derived
  tables/figures from validation-frozen normalizers before any submission build.
- [x] Sync and freeze a versioned per-seed evidence archive, then update the
  manuscript comparator labels and post-pilot seed-scope statement.

## 2026-07-10 Evidence Repair Goal

Goal: complete the audit repairs without retraining PPO. The binding primary
claim must use validation-frozen macro normalizers; final-test-selected static
and event-label schedules remain explicitly privileged diagnostics.

- [x] Freeze and checksum the remote authoritative per-seed evidence subset.
- [x] Implement a reusable validation-frozen macro collector with tests or
  deterministic cross-checks against saved rollout data.
- [x] Regenerate the main 24-seed macro summary and a post-pilot 22-seed
  replication summary from frozen normalizers.
- [x] Reaggregate the macro-derived ablation and changed-mixture robustness
  artifacts, or remove any artifact whose saved inputs are incomplete.
- [x] Rebuild all affected tables/figures, then revise the canonical manuscript
  to separate direct validation baselines from privileged diagnostics.
- [x] Create a versioned evidence archive with manifest, checksums, source
  revision/bundle, and reproduction entry point.
- [x] Compile the canonical PDF and run source/PDF/evidence consistency audit.

## 2026-07-10 Evidence Repair Goal: Completion Checklist

- [x] Freeze and checksum the remote authoritative per-seed evidence subset.
- [x] Implement a reusable validation-frozen macro collector with deterministic
  reaggregation from archived seed rows.
- [x] Regenerate the main 24-seed macro summary and a post-pilot 22-seed
  replication summary from frozen normalizers.
- [x] Reaggregate macro-derived ablation and higher-flux sensitivity artifacts;
  retain lower-flux evidence as unavailable under the three-regime metric.
- [x] Rebuild affected tables and figures, and separate direct validation
  baselines from privileged diagnostics in the canonical manuscript.
- [x] Create a versioned provenance archive with remote manifest, checksums,
  source snapshots, reproduction commands, and extracted per-seed rows.
- [x] Compile the canonical PDF and complete final source/PDF/evidence checks.

## 2026-07-17 Manuscript Completeness and Structure Audit

- [x] Map the active `main.tex` section, figure, table, and appendix structure.
- [x] Check whether the abstract, introduction, method, results, discussion, and
  conclusion express one consistent contribution and evidence hierarchy.
- [x] Audit recent supplementary/development evidence for accidental promotion
  into confirmatory claims.
- [x] Inspect the rendered PDF for section balance, float placement, visual
  continuity, and appendix overload.
- [x] Produce a severity-ranked review with exact source/PDF locations and a
  bounded revision order; do not edit the manuscript during this review.

## 2026-07-18 Method and Experimental Closure Goal

Goal: close the reproducibility and experimental-disentanglement gaps identified
in `docs/07-18-01.md` while preserving the paper as a prediction-driven,
feasibility-masked PPO framework. The canonical paper remains `paper/main.tex`
with active `paper/sections/*.tex`; `raw.tex`, v1, and the separate no-warmup
paper line are out of scope.

- [x] Recover the complete SCENEBAL-2 implementation and evidence context,
  including the prior two-seed reward-proxy pilots, current validation-frozen
  aggregation, remote raw runs, and exact scheduler observation/action/reward
  interfaces.
- [x] Define and freeze an experiment matrix before final runs: same-architecture
  forecast/AoI/uncertainty reward controls, one matched learned-policy baseline,
  paired seed protocol, metrics, behavior gates, and keep/stop rules.
- [x] Implement and validate any missing explicit reward-mode and matched-baseline interfaces
  without changing the primary PD-PPO architecture, action candidates, hard
  feasibility masking, chronological partitions, or final-test evaluator.
- [x] Add focused tests and run local CPU/syntax smoke checks only; all formal
  training and evaluation must run through `ssh remote-gpu` in tmux.
- [x] Run a bounded pilot, diagnose failures once, and freeze the final
  configuration without tuning on final-test outcomes.
  - [x] Complete and aggregate the matched forecast/AoI/uncertainty PPO pilot.
    This first aggregate is exploratory only: a later audit found that its
    deterministic evaluator could invoke a hard auxiliary-classifier router.
  - [x] Complete and compare the clean no-router PD-PPO and no-router
    context-encoder pilots on seeds 117 and 118; freeze one primary actor-only
    policy before expanding any PPO evidence.
  - [x] Complete and aggregate the same-mask Double-DQN pilot.
  - [x] Freeze the corrected actor/critic online-input contract before the
    formal 24-seed restart.
- [x] Run the approved seed expansion on `remote-gpu`, monitor it to completion,
  sync lightweight artifacts, and produce seed-level CSV, aggregate statistics,
  confidence intervals, behavior diagnostics, and an honest keep/pivot decision.
  - [x] Complete the 24-seed clean actor-only primary comparison, post-pilot
    replication, behavior audit, and offline mechanism decomposition.
  - [x] Complete the 24-seed context-alert, exact-label, and one-step
    forecast-greedy reference replays.
  - [x] Complete and aggregate all 24 matched AoI/uncertainty reward controls.
  - [x] Complete and aggregate the same-mask Double-DQN comparator over all 24
    seeds.
- [x] Complete one compact robustness addition only after the mandatory reward
  and learned-policy controls are sound; prefer a second frozen-forecaster replay
  or budget/capacity sensitivity based on implementation cost and validity.
- [x] Specify the estimator/partial-observation update, executable baseline
  definitions, and simulator parameters directly from code and saved configs.
- [x] Revise the manuscript globally after evidence is frozen: correct seed and
  comparator scope, integrate new controls, rebalance Discussion, and avoid
  importing terminology or claims from other paper lines.
- [x] Rebuild all affected tables/figures and `paper/main.pdf`; verify numerical
  traceability, terminology consistency, citations/references, pagination, and
  source/PDF agreement before marking the goal complete.

### Clean-policy decision rule (frozen 2026-07-18)

- The primary method must execute the feasibility-masked PPO actor at final
  evaluation. The optional auxiliary subtype classifier may remain a
  training-only loss but must not route actions.
- Compare the plain actor and the context-encoder actor only on the bounded
  development seeds 117 and 118. Both must use online warning scores, the same
  six masks, the same frozen evaluator, and the same final windows.
- Prefer the plain actor unless the context encoder gives a material and
  behavior-valid improvement. Do not select on a single favorable final-test
  scalar or add bandit-dependent residuals, priors, rewards, or labels.
- After selection, freeze the architecture and run the matched
  forecast/AoI/uncertainty expansion and same-mask Double-DQN comparator.
  Incomplete `corrected24`, `corrected24r1`, and `corrected24r2` directories are
  excluded from all evidence aggregation.

## 2026-07-19 Submission Consistency and Anonymization Pass

- [x] Resolve the apparent pilot-seed conflict from experiment provenance rather
  than by string replacement: the superseded July 10 evidence used seeds 117
  and 122, whereas the corrected no-router actor-only method used seeds 117 and
  118 for its bounded architecture choice and froze seeds 119--140 for expansion.
- [x] Make the training-only subtype guide, behaviour-cloning warm start, and
  auxiliary-label boundary explicit in the matched-control interpretation.
- [x] Replace the broad Figure 2 `rule-based` label with the exact conventional
  comparator family and keep warning-score/exact-label references separate.
- [x] State the aggregation boundary of the reported ordinary and
  validation-normalized equal-regime metrics in Limitations.
- [x] Add the concise estimator update order requested by the final manuscript
  audit without duplicating the existing estimator specification.
- [x] Add source-controlled ESWA title-page and anonymized-manuscript build
  targets; remove author metadata, CRediT, and acknowledgements from the
  anonymized output without duplicating the scientific manuscript.
- [x] Compile named and anonymous manuscript PDFs and title page; verify author
  leakage, abstract length, citations/references, page count, and diff hygiene.
- [x] Record a bounded page-reduction plan from the rendered section map. Do not
  mechanically shorten Discussion or move material until cross-reference and
  review-PDF consequences are checked.

## 2026-07-19 Submission Compression and Package Audit

- [x] Confirm that no new training, baseline, reward, or budget experiment is
  needed for the frozen scientific claim.
- [x] Audit Figure 1 provenance. The active asset is produced by the tracked
  Matplotlib script `paper/figures/gen_fig_framework_and_support.py`; it is not
  a generative-image asset.
- [x] Create a self-contained supplementary manuscript and move audit-oriented
  platform, fixed-selection-ledger, generator-figure, complete-partition, and
  ridge-table material into it with stable cross-references.
- [x] Compress Section 4 by moving history-indicator, encoder,
  forecaster-mixture, and reward-control implementation detail into a methods
  appendix while retaining the reconstructable algorithm chain in the main
  manuscript.
- [x] Rebuild named, anonymous, title-page, and supplementary PDFs; visually
  inspect affected pages and audit references, citations, identity leakage,
  asset provenance, abstract/highlight limits, and page counts.
- [x] Prepare an anonymous submission directory and source bundle containing
  the manuscript, title page, highlights, supplement, declarations, and
  individual figures without exposing author identity in review files.
- [ ] Replace provisional Data Availability wording only when an actually
  accessible anonymous repository or archive exists; do not fabricate a link.

### Errors encountered

- The first attempt to append this phase combined an empty `findings.md` patch
  hunk with the plan update, so `apply_patch` rejected the entire patch. No file
  was changed. The update was split into valid file-specific patches.
- The first combined manuscript-migration patch matched one stale line break in
  Section 4 and was rejected atomically. No manuscript file changed. The move
  was split into smaller section-level patches before compilation.
- A later multi-file reference patch also missed one theoretical-appendix line
  break and was rejected atomically. File-level patches then updated the main
  references and supplementary numbering consistently.
- The first submission-copy command ran from the package directory while using
  framework-root-relative paths. It failed before copying and created only an
  empty nested staging tree, which was removed before rebuilding from the
  correct root.
- The first package verification command looked for anonymous LaTeX sources in
  the package root instead of `anonymous_source/`. Evidence checks and the cover
  letter had passed; archive creation had not started. The source build was
  rerun from its actual staging directory before checksums were generated.
- A later artifact-copy command was launched from `anonymous_source/` while
  still prefixing paths with `anonymous_source/` and `title_page_source/`.
  Those three copies failed harmlessly; the PDFs were then copied from the
  package root and verified byte-for-byte against the independently compiled
  staging outputs.
- The first repository verifier invocation omitted the
  `anonymous_repository/` prefix. Re-running the correct path passed all four
  aggregate checks.
- The first supplement contact-sheet/test command used a package-root-relative
  PDF path from inside `anonymous_repository/`, and the default Python lacked
  pytest. The corrected PDF path rendered successfully and the existing
  `darts` environment ran the test suite.
- That first archive test exposed four omitted audit/aggregation scripts and
  the generated-truth source module. They were added to the anonymous evidence
  archive; the complete focused test suite then passed with one existing skip.

## 2026-07-19 Journal-language consolidation pass

- [x] Read `docs/07-19-01.md` against the current 50-page manuscript and reject
  stale or conflicting advice, especially the proposed title change that
  conflicts with the supervisor-approved title.
- [x] Standardize reviewer-facing terminology for evaluation seeds, macro
  score, evaluable epochs, warning-score rule, exact-label reference,
  training-only supervision, action-trace diagnostics, and secondary
  forecaster without changing evidence or claim boundaries.
- [x] Rewrite the flagged abstract, introduction, related-work, method-control,
  results, discussion, conclusion, table-caption, and active-figure wording by
  hand rather than by global string replacement.
- [x] Rebuild the named, anonymous, and supplementary manuscripts; inspect the
  affected rendered pages and run terminology, numerical, anonymous-source,
  citation/reference, and diff-hygiene checks.
- [x] Synchronize the independently compilable anonymous source and upload
  package, regenerate checksums, and record the final editorial decisions.

### Errors encountered

- The contact-sheet `montage` command emitted an ImageMagick font warning and
  returned status 1, although it produced the requested contact sheets. The
  generated sheets and the affected PDF pages were inspected directly; no
  manuscript rendering fault was found.
- The first multi-file patch for the final four-specialist clarification missed
  a line break in Discussion and was rejected atomically. The exact source
  context was then inspected and the correction was applied without partial
  edits.
- One final verification command invoked the package checksum manifest and PDF
  renderer from the framework root rather than the package directory. It found
  no listed files and changed nothing. Re-running both commands from
  `submission/eswa_pdppo_20260719/` verified all 12 checksums and rendered the
  anonymous first page successfully.

## 2026-07-19 Figure-system and final editorial refinement

- [x] Read `docs/07-19-02.md` against the current manuscript and freeze the
  applicable scope: retain the supervisor-approved title, preserve the
  scientific claims, do not add experiments, and leave Data Availability open
  until a real anonymous URL exists.
- [x] Apply the remaining high-value sentence and caption edits by hand, using
  one canonical seed sentence and plain names for controls, runtime states, and
  comparator families.
- [x] Redesign Figure 1 as a readable protocol diagram that separates online
  inputs, feasibility masking, policy learning, frozen-forecaster scoring,
  validation-only selection, final evaluation, and training-only supervision.
- [x] Redesign Figure 2 without connected seed lines and expose the evidence
  hierarchy across fixed/conventional/myopic and strong contextual/learned
  comparators without duplicating ridge sensitivity.
- [x] Harmonize Figures 3 and 4 around one font, panel-label style, sign
  convention, palette, and line-weight system; remove the duplicated DQN panel
  from Figure 4 after moving it to Figure 2.
- [x] Add a compact supplementary terminology table and cite it from the main
  manuscript without expanding the scientific scope.
- [x] Rebuild and visually audit all manuscript targets, synchronize the
  anonymous source and separate figures, regenerate archives/checksums, and run
  terminology, numerical, anonymity, citation/reference, font, and diff checks.

### Errors encountered

- The first broad prose patch missed a source line break and was rejected
  atomically; no manuscript file changed. The edits were reapplied in small,
  section-specific patches.
- A combined final terminology patch missed the current Discussion wording and
  was rejected atomically. Exact source context was inspected before retrying.
- The first submission-package sync was launched from `anonymous_source/` while
  using framework-root-relative paths. It stopped on the first copy and changed
  no package file; the sync was rerun from the framework root.
- The first staged-build log scan included transient undefined-reference
  warnings from LaTeX's initial pass. Final `.log` files after `latexmk`
  convergence contain no undefined references or citations.

## 2026-07-19 Proposition matrix and framework-figure redesign

- [x] Audit the Proposition 1 specialist--regime matrix and Figure 1 against
  the active method, paper palette, rendered manuscript, and ESWA artwork
  constraints.
- [x] Restyle the Proposition 1 matrix with the manuscript's sans-serif visual
  language and colorblind-safe teal/orange/slate semantics, including a compact
  legend that remains legible in grayscale.
- [x] Redesign Figure 1 as two coordinated layers: chronological offline
  development/evaluation and the online feasibility-masked PD-PPO loop.
- [x] Distinguish runtime data flow, hard feasibility control, training-only
  reward/update paths, and validation/final-test boundaries through consistent
  line styles and labels rather than crossed arrows.
- [x] Render and inspect multiple Figure 1 revisions at actual manuscript size;
  correct text density, alignment, arrow routing, and caption consistency before
  accepting the asset.
- [x] Rebuild named, anonymous, and supplementary targets; audit the affected
  pages, vector/font properties, references, anonymity, source bundles, separate
  figures, and package checksums.

### Errors encountered

- The first package-cleanup command was run from `anonymous_source/` while also
  prefixing its input with `anonymous_source/`. It stopped at the initial
  `cp` with no file changes. The command was rerun from the package root, after
  which all 12 checksums passed.
- An early aggregate LaTeX-output scan included first-pass undefined-reference
  warnings. A separate converged build and direct scan of the final staged
  `.log` confirmed that no undefined references or citations remain.

## 2026-07-19 Figure typography and objective-prose convergence

- [x] Audit the embedded font families, base sizes, and rendered hierarchy of
  all four active manuscript figures and the Proposition 1 matrix.
- [x] Standardize active figure text on DejaVu Sans and a common 8.5-point
  hierarchy; remove whole-object scaling from the Proposition 1 matrix.
- [x] Regenerate all four main figures and iteratively correct the matrix
  geometry until headings, cells, and legend labels render without clipping.
- [x] Manually revise the active English manuscript to remove first-person
  authorial subjects while preserving precise attribution and claim strength.
- [x] Rewrite prose that uses colon-led list templates or `rather than`
  contrasts; retain colons only where required by mathematics or formal table
  syntax, and avoid mechanical phrase substitution.
- [x] Re-read the revised paragraphs in sequence for transitions, terminology,
  comparator scope, and evidence consistency.
- [x] Rebuild named and anonymous manuscripts, visually inspect affected pages,
  synchronize the anonymous source and separate figures, regenerate archives
  and checksums, and verify font, reference, anonymity, and text-hash parity.

### Errors encountered

- The first font-update package sync was invoked from `anonymous_source/` with
  framework-root-relative paths. It failed on the first copy and changed no
  files. The sync was rerun from the framework root.
- The first synchronized package differed from the canonical anonymous PDF by
  one regenerated table note. `clean_main_comparisons.tex` was synchronized,
  after which the staged manuscript and package were rebuilt.
- The first table-stability checksum used the pre-regeneration main-table note
  and therefore reported one mismatch. The generated source was inspected,
  the maintained template was aligned with the edited output, and a second
  regeneration reproduced all three tables byte for byte.
- A page-location shell loop returned status 1 because its final search found
  no match, although all three LaTeX builds had already completed. Direct page
  rendering identified and verified the target tables.

## 2026-07-19 Draw.io MCP installation and Figure 1 redesign

- [x] Verify the maintained draw.io MCP implementation, the local Codex MCP
  configuration format, and the installed draw.io export path.
- [x] Install the official draw.io MCP at user scope and verify its Codex
  registration without changing unrelated MCP servers.
- [x] Translate the active PD-PPO method into an editable native `.drawio`
  architecture with a clear protocol layer, online scheduling loop, and
  training-only update path.
- [x] Export a vector PDF and complete at least three rendered revisions for
  hierarchy, alignment, arrow routing, typography, palette, and grayscale
  readability at the manuscript's actual column width.
- [x] Replace the active Figure 1 asset and align its caption and surrounding
  text without changing experimental claims.
- [x] Rebuild the named and anonymous manuscripts, inspect the affected page,
  verify vector/font properties, and synchronize the submission package and
  checksums.

### Errors encountered

- An initial broad configuration search traversed unrelated JetBrains metadata
  and produced noisy truncated output. It changed no files; subsequent checks
  were restricted to the Codex MCP configuration and relevant paper assets.

## 2026-07-19 Figure 1 iconography and visual-language refinement

- [x] Audit the accepted draw.io Figure 1 against the typography, palette, and
  final-size hierarchy used by Figures 2--4.
- [x] Add restrained draw.io-native line icons to the protocol stages and the
  major online/training modules without introducing clip art or raster assets.
- [x] Shorten module descriptions and increase the effective body-text size so
  the final manuscript rendering matches the 8.5-point figure system more
  closely.
- [x] Complete at least three rendered revisions for icon consistency, label
  alignment, connector clearance, and grayscale readability.
- [x] Rebuild named and anonymous manuscripts, synchronize the editable source
  and Figure 1 in the submission package, and regenerate all checksums.

### Errors encountered

- The first iconography edits were applied directly to the native draw.io XML,
  although the user had requested the installed Draw.io MCP workflow. This was
  corrected before finalization. The active page was read with MCP
  `list_pages`/`get_page`, the final label refinements were written with MCP
  `set_page`, and the page was read back through MCP before export.
- The first post-compile diagnostic command used an invalid ripgrep escape for
  `\hbox`. Both PDFs had already compiled successfully. A corrected diagnostic
  scan subsequently found no undefined citation, reference, or PDF-version
  warning.

## 2026-07-19 Figure 1 icon-to-text spacing correction

- [x] Inspect all active main-figure pages at manuscript scale and distinguish
  source-art spacing from LaTeX float and caption spacing.
- [x] Use Draw.io MCP to regularize the icon-card width, icon centering, and
  text inset in the crowded online scheduling modules.
- [x] Re-export the pure-vector asset and inspect the standalone diagram and
  page-4 manuscript rendering in colour and grayscale.
- [x] Rebuild both manuscripts and synchronize the verified Figure 1 source,
  standalone asset, source bundle, and checksum manifest.

### Review decision

- The first spacing revision increased the icon-text gap but narrowed several
  text columns enough to create excessive line wrapping. It was rejected before
  export to the manuscript. The accepted second revision uses smaller 34-unit
  icon cards, 54-unit text insets, and approximately 6--7 units of clear space
  between icon cards and labels.
- Figures 2--4 use consistent LaTeX caption spacing and do not show the same
  icon-label crowding. No global float or caption spacing override was added.

## 2026-07-19 ESWA-reference Figure 1 hierarchy refinement

- [x] Inspect representative ESWA prediction, engineering-optimization, and
  reinforcement-learning framework figures from primary article sources.
- [x] Preserve the accepted Figure 1 source and export before the hierarchy
  redesign.
- [x] Use Draw.io MCP to make the online scheduling loop visually dominant,
  compress protocol/training support bands, shorten module copy, and clarify
  the principal data flows.
- [x] Complete repeated full-size and manuscript-scale colour/grayscale reviews
  for hierarchy, legibility, line routing, and caption fit.
- [x] Rebuild named and anonymous manuscripts, synchronize the submission
  package, regenerate checksums, and record the reference-derived decisions.

## 2026-07-19 Proposition 1 figure reconstruction

- [x] Audit the current matrix against the proposition, proof, and simulator
  mapping, including the risk that qualitative cells appear empirical.
- [x] Replace the matrix with a numbered schematic that directly distinguishes
  incompatible regime optima, fixed-policy mismatch, and adaptive allocation.
- [x] Export and inspect the figure at standalone, manuscript, and grayscale
  scales; revise until labels, notation, and reading order are unambiguous.
- [x] Rebuild named and anonymous manuscripts and renumber/synchronize all
  standalone submission figures and checksums.

## 2026-07-19 Figure 2 manuscript-spacing correction

- [x] Audit the embedded manuscript-page rendering instead of relying on the
  standalone artwork alone.
- [x] Separate panel titles, subtitles, content, conclusions, and the formula
  into fixed vertical bands.
- [x] Remove automatic label hyphenation and enlarge the gap between the three
  upper panels and the lower formula frame.
- [x] Rebuild both manuscripts and synchronize the corrected Figure 2 source,
  standalone asset, source bundle, and checksum manifest.

## 2026-07-19 Main-figure insertion audit and repair

- [x] Audit the compiled PDF for figure numbering, source asset, section
  placement, first textual reference, and submission-package consistency.
- [x] Restore the maintained full PD-PPO architecture as Figure 1 after a later
  copy-editing pass had inserted an oversimplified workflow asset.
- [x] Place explicit textual anchors before Figures 1--5 while preserving their
  Introduction, formulation, and Results subsection roles.
- [x] Rebuild named and anonymous manuscripts, inspect all affected pages, and
  synchronize the anonymous source and standalone figure package.

## 2026-07-19 07-19-03 refinement implementation

- [x] Reconcile the instruction memo, clean-evidence contract, and current source.
- [x] Apply the frozen title, comparator, metric, action, and forecaster vocabulary.
- [x] Add a tracked Matplotlib generator for the compact workflow figure.
- [x] Regenerate Figures 3--5 from the authoritative clean-asset script.
- [x] Compile the named manuscript, anonymous manuscript, supplement, and title page.
- [x] Run final terminology, line-length, artifact, and diff checks.

## 2026-07-22 strict theory revision

- [x] Read and triage `docs/07-22-01.md` against the current 50-page source.
- [x] Record the dirty worktree before editing and archive the current paper and
  ESWA submission package with SHA-256 checksums.
- [x] Formalize structural and epoch-dependent feasibility, including the
  minimum-duration state and nonempty feasible-action condition.
- [x] Align Proposition 1 and its figure with a defined weighted normalized
  objective whose equal-weight special case is the reported macro score.
- [x] Replace the underspecified binary-event ``linear-Gaussian'' argument for
  Proposition 2 with a complete Gaussian random-walk counterexample.
- [x] Correct reward notation to match the implemented hard-mask training path
  and add missing assumptions and symbol definitions.
- [x] Remove implementation-specific prose from the theory appendix and soften
  the fixed-cycle statement.
- [x] Rebuild named, anonymous, supplementary, and title-page PDFs; inspect the
  affected theory pages and synchronize the verified submission package.

## 2026-07-22 strict citation and source-alignment revision

- [x] Read `docs/07-22-02-citation.md`, record the dirty worktree, and archive
  the paper and submission package with SHA-256 checksums before editing.
- [x] Verify the high-risk DOI, publication-type, page-range, and proceedings
  metadata against publisher or Crossref records.
- [x] Correct the active BibTeX records without changing historical citation
  keys that are already used throughout the manuscript.
- [x] Rewrite the active related-work claims so each citation supports the
  nearby objective, application, or architecture statement.
- [x] Distinguish external architecture precedents from manuscript-specific
  forecaster hyperparameters and generator acceptance checks.
- [x] Rebuild and inspect the named and anonymous manuscripts, including the
  rendered bibliography and all edited citation contexts.
- [x] Synchronize the submission source bundle, standalone PDFs, and checksum
  manifest, then record the final citation audit in the planning files.

### Errors encountered

- The first workflow-figure run warned that the default Matplotlib cache was
  unwritable. The final figure commands set `MPLCONFIGDIR` to a task-specific
  directory under `/tmp`.
- Several broad prose patches missed their evolving context and were rejected
  atomically. Smaller targeted patches were then applied and verified.
- The first log scan used an invalid ripgrep escape for `\\hbox`; a simpler
  pattern completed the scan.
- The first abstract-count regular expression did not match the environment.
  A `sed`/`detex`/`awk` pipeline produced the final count.
- Listing the backup archive through `head` caused a harmless broken-pipe
  message. Targeted extraction to `/tmp` completed the evidence comparison.
- A final parallel named/anonymous rebuild produced an auxiliary-file race
  notice despite successful PDFs. A subsequent standalone anonymous `latexmk`
  run reported the target fully up to date.
- The first one-expansion audit did not match a phrase split across TeX lines
  and stopped the combined check early. A whitespace-normalized audit confirmed
  exactly one active expansion before the remaining checks were rerun.
## 2026-07-27 Figure 1 text-overflow repair

- [x] Archive the current Draw.io source, exported Figure 1, and compiled manuscript.
- [x] Repair overflowing labels and typography through the Draw.io MCP.
- [x] Export and visually inspect the standalone figure and its manuscript page.
- [x] Recompile the canonical and anonymous manuscripts and synchronize Figure 1 in the submission package.
