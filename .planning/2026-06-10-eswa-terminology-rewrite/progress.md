# Progress: ESWA Terminology and Narrative Rewrite

## 2026-07-03 00:24 CST
- Started a new algorithm/evidence GOAL: improve PD-PPO against the
  `context_alert_bandit_t0p5` challenge while preserving the primary method
  identity as prediction-driven masked PPO under hard feasibility constraints.
- Updated the active planning checklist with the user-specified primary
  candidates: feature-parity PD-PPO and CA-PD-PPO.
- Recorded the forbidden primary-method components: residual action over bandit,
  bandit-margin reward, counterfactual improvement labels, and bandit-dependent
  actor priors. These remain appendix-only exploratory variants if ever used.
- Current first action: audit exact online observables used by the context-alert
  bandit, compare them with PD-PPO's current state vector, then implement missing
  online-only alert/context features without simulator event labels at final
  evaluation.
- Feature-parity audit found that `context_alert_bandit_t0p5` reads only three
  station-side proxy columns at the current time step:
  `agent_context_particle_alert`, `agent_context_flux_alert`, and
  `agent_context_thermal_alert`. It thresholds the largest score at `0.5` and
  maps calm/particle/flux/thermal to validation-selected fixed candidate masks.
- The original PD-PPO environment already supported raw `agent_context_columns`,
  but `_state()` always appended the simulator `event_flag`. The custom PPO
  actor/critic path also passed `_env_is_event(env)` into event-aware modules.
  This created a real observation-policy ambiguity for final evaluation.
- Implemented first-pass feature parity and CA-PD-PPO code locally:
  `src/v2/env.py` now has optional alert-derived online features and an option
  to remove `event_flag` from the policy state; `src/v2/custom_ppo.py` now has
  an optional actor-side context encoder while preserving masked categorical PPO;
  `scripts/25_v2_train_custom_ppo.py`, `scripts/58_v31_split_protocol_run.py`,
  and `scripts/64_v31_eval_saved_run_operational_baselines.py` now pass and
  restore the new settings; `tests/v2/test_custom_ppo.py` has shape/mask tests.

## 2026-06-10
- Created a dedicated planning directory for the ESWA terminology rewrite.
- Backed up current active source to:
  `paper/_archive/eswa_terminology_rewrite_20260610/`.
- Inspected the active manuscript:
  `paper/pdppo_crst_rewrite.tex` and `paper/rewrite_sections/*.tex`.
- Confirmed that `raw.tex` and legacy `sections/*.tex` are outside scope.
- Rewrote the active English draft around the generic
  power-constrained sensing-system scheduling problem before introducing the
  Antarctic blowing-snow benchmark.
- Replaced reader-visible internal terminology across the active manuscript and
  currently included tables:
  "frozen oracle" -> "fixed forecast evaluator";
  "deployment constraints" -> "operational constraints";
  "dwell" -> "minimum on-time";
  "deployable/compact static" -> "static subset" or "static-priority" baseline;
  "candidate prior" -> "candidate-policy guide".
- Abstract audit: 227 words, below the 250-word limit.
- Corrected residual old terminology in visible TikZ figure text and included
  appendix tables.
- Compile command:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
- Compile result: success; output `paper/pdppo_crst_rewrite.pdf`, 35 pages,
  about 2.8 MB. BibTeX still reports four empty-page warnings in existing
  bibliography entries.
- PDF text audit: no visible hits for the old terminology set
  (`frozen oracle`, `candidate prior`, `deployment constraints`,
  `deployable static`, `compact static`, `V3.1`, `env-dwell`, `h75`,
  `negative result`, `only a protocol`, `we cannot claim`).
- Fine-proofread the abstract for journal-facing clarity. Main edits: replaced
  awkward passive phrasing, clarified "sensing channels", changed "held-out
  seeds" to "independent final-test seeds", and simplified the operational
  behavior sentence. Abstract length is now 221 words.
- Recompiled successfully after the abstract edit; output remains
  `paper/pdppo_crst_rewrite.pdf` at 35 pages.
- Applied supervisor terminology corrections after manual review, not by bulk
  word replacement:
  - main problem wording is now "sensing system scheduling under a power budget";
  - prediction-driven wording is introduced as the proposed forecasting-oriented
    formulation, not as every constrained scheduling problem;
  - "freshness" is replaced by AoI / age of information;
  - duration/duty/switching are described as operating rules rather than
    deployment constraints;
  - "duty-constrained" and "dynamic heuristic" are replaced by same-duty
    baselines and rule-based dynamic baselines;
  - "sensor/sensing channels" wording is standardised around channels of the
    sensing system.
- Updated visible figure/table wording and PDF metadata keywords accordingly.
- Abstract length after this pass: 248 words.
- Compile result after this pass: success; output `paper/pdppo_crst_rewrite.pdf`,
  36 pages. The only remaining PDF-text hit for `multi-horizon` is in a cited
  paper title, not manuscript prose.

## 2026-06-21
- Reconfirmed the active target journal as *Expert Systems with Applications*
  (ESWA), superseding the older CRST rewrite target.
- Updated `.planning/.active_plan` to this ESWA planning directory.
- Added an explicit note to `task_plan.md` that
  `paper/pdppo_crst_rewrite.tex` is a historical filename only; it does not
  indicate the current target journal.
- Updated the historical 05-25 rewrite/evidence files that still described CRST
  as the active target:
  - `docs/05-25-crst-rewrite-strategy.md`
  - `docs/05-25-full-rewrite-evidence-ledger.md`
- Their filenames remain historical, but their current content now states
  *Expert Systems with Applications* (ESWA) as the target and frames the paper as
  an intelligent sensing-system scheduling study with Antarctic blowing-snow as
  the benchmark application.

## 2026-06-21 20:38 CST
- Reconciled this plan with the promoted canonical source:
  `paper/main.tex` and `paper/sections/*.tex`; the older
  `paper/pdppo_crst_rewrite.tex` / `paper/rewrite_sections/*.tex` files are
  now historical rewrite artifacts only.
- Applied the approved specialist-bottleneck theory extension to the canonical
  manuscript source:
  `paper/sections/01_introduction.tex`,
  `paper/sections/02_related_work.tex`,
  `paper/sections/03_problem_formulation.tex`,
  `paper/sections/07_discussion_future_work.tex`,
  `paper/sections/appendix_theory.tex`,
  `paper/main.tex`, and `paper/references.bib`.
- Rebuilt `paper/main.pdf` with XeLaTeX. The build succeeds with no undefined
  references/citations; only the pre-existing BibTeX empty-page warnings remain.

## 2026-06-21 21:29 CST
- Added a final consistency pass to the canonical manuscript:
  SCENEBAL-2 subtype-balance/design wording in
  `paper/sections/05_simulation_setup.tex`, raw macro sensitivity wording in
  `paper/sections/06_results.tex`, and a forward-looking SCENEBAL-2 archive
  statement in `paper/main.tex`.
- Verified the old public release is historical and not a current SCENEBAL-2
  archive (`v0.1.0` release URL returns `404`; no current tags from
  `git ls-remote --tags`).
- Rebuilt `paper/main.pdf` successfully. The rendered PDF is `37` pages, has no
  undefined citations or references, and contains the new design, metric
  boundary, and versioned-archive statements.
- Renamed the active SCENEBAL-2 evidence table label from the historical
  `metpair` label to `tab:scenebal2_staticnorm_macro_summary` and updated the
  Results cross-reference.

## 2026-06-21 21:45 CST
- Audited whether the active paper is fully on the new SCENEBAL-2 claim.
- Fixed two residual old-claim items:
  `paper/highlights.txt` / `paper/pdppo_crst_rewrite_highlights.txt` now use
  24-seed SCENEBAL-2 language, and
  `paper/sections/03_problem_formulation.tex` now treats the ordinary
  step-weighted true-static comparison as a separate strict gate rather than an
  old limitation.
- Rebuilt `paper/main.pdf` successfully and rescanned source/PDF text. No old
  `18 seeds`, `SCENEBAL-1`, `V3.1`, `metpair`, `seed45`, `h075`, `CRST`, or
  `pdppo_crst` residue remains in the active submission-facing source set.

## 2026-06-21 22:28 CST
- Audited the reduced figure count in the current mainline manuscript.
- Rejected direct reinsertion of stale old-protocol figures
  (`figure_operational_summary`, `figure_operational_behavior`, and
  `figure_fixed_budget_power_error`) because they do not represent the current
  SCENEBAL-2 evidence protocol.
- Added `paper/figures/gen_fig_scenebal2_diagnostics.py` and generated two new
  SCENEBAL-2 figures:
  `figure_scenebal2_metric_boundary.{pdf,png}` and
  `figure_scenebal2_behavior_audit.{pdf,png}`.
- Inserted both figures into `paper/sections/06_results.tex` and rebuilt
  `paper/main.pdf`. The active manuscript now has seven figures, `38` pages, no
  undefined references/citations, and only the existing empty-page BibTeX
  warnings.

## 2026-06-21 Reader-Facing Terminology Cleanup
- Applied the user-facing terminology constraint that internal experiment codes
  should not carry the manuscript narrative.
- Replaced reader-visible `SCENEBAL-2` wording in the active manuscript,
  highlights, table caption, and generated figure titles with problem-level
  language: final specialist-budget benchmark, regime-balanced
  backbone-plus-specialist benchmark, and specialist-bottleneck benchmark.
- Replaced reader-visible `router-confidence deployment threshold` wording with
  "pre-fixed deployment threshold on policy scores".
- Regenerated the three active Results figures:
  `figure_scenebal2_24seed_evidence.{pdf,png}`,
  `figure_scenebal2_metric_boundary.{pdf,png}`, and
  `figure_scenebal2_behavior_audit.{pdf,png}`.
- Rebuilt `paper/main.pdf` successfully. `pdftotext` found no reader-visible
  hits for `SCENEBAL`, `router-confidence`, `V3.1`, `metpair`, `h075`,
  `seed45`, or `old-claim`; `main.log` / `main.blg` had no undefined
  citations or references.

## 2026-06-21 Supplementary Experiment Decision
- Researched the current evidence gap before launching new runs. ESWA's scope
  emphasises applied intelligent-system design, testing, implementation, and
  practical guidance; DRL scheduling literature also commonly uses multi-run
  reliability, ablation, and robustness checks rather than only a final score.
- Audited the active Results section, final 24-seed table, decision audit JSON,
  and current seed-sweep runners. The current final benchmark already has
  24/24 operational, replay, behaviour, true-static macro, and strict
  true-static step gates, but lacks a final-protocol mechanism ablation and a
  small benchmark-perturbation robustness block.
- Wrote the supplementary experiment decision report:
  `reports/aggregate/final_benchmark_additional_experiment_decision_20260621.md`.
- Decision: prioritise a six-seed mechanism-ablation pilot aligned with the
  current true-static/behaviour/static-normalised macro protocol, followed by
  small perturbation robustness pilots. Do not prioritise 48/72-seed
  same-protocol expansion until those mechanism and robustness gaps are closed.

## 2026-06-21 Mechanism Ablation Pilot Launch
- Set the active goal to supplement the final specialist-budget benchmark with
  current-protocol experiments while keeping PPO, the met+one-specialist sensing
  geometry, and the remote-gpu-only execution boundary.
- Added and syntax-checked:
  - `scripts/run_v31_finalbenchmark_mechanism_ablation_pilot_20260621.sh`
  - `scripts/watch_v31_finalbenchmark_mechanism_ablation_20260621.sh`
  - `scripts/78_v31_collect_mechanism_ablation.py`
- The runner launches three train-time ablations over seeds `117--122`:
  `no_imitation_guide`, `no_regime_aux_path`, and `no_staticnorm_train`; then it
  collects each with the current operational, strict true-static, replay,
  behaviour, static-normalised macro, and raw macro boundary protocol. It also
  runs full-policy threshold-sensitivity re-evaluations at confidence values
  `0.0`, `0.7`, and `0.9`.
- Synced only those new scripts to `remote-gpu` under
  `~/_code/microclimate_demo/rl_sensor_scheduling_framework/scripts/`.
- Verified on `remote-gpu`: project scripts exist, conda env `darts` provides
  Python 3.12.12, bash/Python syntax checks pass, six 4090 GPUs were initially
  idle, and disk space is sufficient.
- Started remote tmux session `mech_ablation_20260621` with command:
  `bash scripts/run_v31_finalbenchmark_mechanism_ablation_pilot_20260621.sh`.
  Initial health check shows six `no_imitation_guide` seed jobs running through
  `scripts/25_v2_train_custom_ppo.py`.
- Follow-up health check shows all six GPUs engaged with about 1.5 GB memory per
  process and `custom_ppo_update` logs advancing past roughly 20k/200k
  timesteps for the first ablation wave. The `awbc_label_rate=0.000` log value
  confirms that the no-imitation ablation switch is active.
- A later remote watch shows the first `no_imitation_guide` wave advancing
  through roughly 100k/200k timesteps across all six seeds with no early
  parameter or runtime error.
- The first completed ablation, `no_imitation_guide`, passed all current
  protocol gates over seeds `117--122`: operational step/macro, replay
  step/macro, behaviour-complexity, true-static macro, and strict true-static
  step gates were all `6/6`. Its aggregate decision file is
  `reports/aggregate/mechanism_ablation_no_imitation_guide_117_118_119_120_121_122_decision_audit_20260621.json`.
- Interpretation boundary: this is not evidence that the imitation guide is
  necessary. It shows the final protocol remains positive even when the
  advantage-weighted imitation guide is disabled. The mechanism story therefore
  cannot rely on claiming that the imitation guide is essential; it should be
  treated as a training stabiliser or optional accelerator unless later
  comparisons show a consistent margin advantage for the full model.
- The second ablation wave, `no_regime_aux_path`, is now running on all six GPUs
  in the same tmux session. Logs confirm the auxiliary regime/subtype path is
  disabled (`subtype_aux=0`, `subtype_acc=nan`) while the imitation guide remains
  active (`awbc_label_rate=1.000`).
- The `no_regime_aux_path` ablation has completed training and current-protocol
  collection. It still passes operational step/macro, replay step/macro,
  behaviour, and true-static macro gates at `6/6`, but strict true-static step
  falls to `5/6`: seed `122` loses in sign to the true fixed-static reference.
  The decision file is
  `reports/aggregate/mechanism_ablation_no_regime_aux_path_117_118_119_120_121_122_decision_audit_20260621.json`,
  with decision `pivot_true_static_step_sign_failure`.
- Interpretation boundary: unlike `no_imitation_guide`, this is positive
  mechanism evidence. Removing the regime/subtype auxiliary route does not
  destroy the coarse operational claim, but it breaks the all-seed strict
  true-static step gate that the full model satisfies. This supports presenting
  regime-aware policy support as important for the strongest claim.
- The third ablation wave, `no_staticnorm_train`, started automatically at
  `2026-06-22T00:27:13+08:00` on `remote-gpu`.
- Follow-up at `2026-06-22T00:58:05+08:00`: `no_staticnorm_train` is still
  running normally over seeds `117--122`, with the six PPO jobs around
  `99k--104k / 200k` timesteps. The subtype auxiliary path is active in this
  branch (`subtype_aux` nonzero and high subtype accuracy), so this is a
  targeted training-objective normalization ablation rather than a duplicate of
  `no_regime_aux_path`.
- While the third ablation runs, the next robustness pilot is being prepared.
  The first robustness unit should keep PPO and the met-plus-one-specialist
  sensing geometry unchanged and perturb only natural simulator parameters:
  event subtype mix and/or subtype latent separation. This follows the
  supplementary decision report's instruction to test scenario robustness
  without replacing the sensing system.
- Added, syntax-checked locally and remotely, and synced to `remote-gpu`:
  - `scripts/run_v31_finalbenchmark_robustness_pilot_20260622.sh`
  - `scripts/watch_v31_finalbenchmark_robustness_20260622.sh`
  - `scripts/79_v31_collect_robustness.py`
- The robustness runner is prepared but not yet launched. Default seeds are
  fresh seeds `141--146`. It contains two bounded perturbation units:
  `event_mix_flux30` changes the event subtype mix from `0.40/0.20/0.40` to
  `0.35/0.30/0.35`; `weaker_latent10` weakens subtype latent cues by roughly
  ten percent. Both keep PPO, the met-plus-one-specialist geometry, budget,
  true-static replay, behaviour auditing, and static-normalised macro
  collection unchanged.
- The third ablation, `no_staticnorm_train`, completed training and collection.
  It passed all current gates over seeds `117--122`: operational step/macro,
  replay step/macro, behaviour, true-static macro, and strict true-static step
  are all `6/6`. Mean step margin is `0.1311` and mean macro margin is
  `0.0870`. Its decision file is
  `reports/aggregate/mechanism_ablation_no_staticnorm_train_117_118_119_120_121_122_decision_audit_20260621.json`.
- Interpretation boundary: `no_staticnorm_train` does not support a necessity
  claim for static-normalised training. It should be framed as an objective and
  evaluation-alignment design choice, not as a component whose removal breaks
  the final result. The only clear mechanism-degradation result in this pilot
  remains `no_regime_aux_path`, which breaks the all-seed strict true-static
  step gate (`5/6`).
- A paired full-reference re-audit over seeds `117--122` completed after the
  ablations and passed all current gates at `6/6`; mean step margin is `0.1288`
  and mean macro margin is `0.0856`.
- The mechanism runner is now executing threshold-sensitivity re-evaluations on
  the full checkpoints, starting with confidence `0.0`.
- The full mechanism-ablation pilot has completed and was synced locally. The
  aggregate report is
  `reports/aggregate/mechanism_ablation_pilot_117_118_119_120_121_122_20260621/mechanism_ablation_summary.md`.
- Threshold-sensitivity re-evaluations at confidence values `0.0`, `0.7`, and
  `0.9` all passed the current six-seed gates (`6/6` operational step,
  true-static macro, strict true-static step, and behaviour). This means the
  fixed deployment threshold is not brittle in the six-seed pilot.
- Final mechanism interpretation: the only ablation that clearly weakens the
  strongest current claim is `no_regime_aux_path`, which falls to `5/6` on the
  strict true-static step gate. `no_imitation_guide` and `no_staticnorm_train`
  both remain positive under all current gates, so they should be reported as
  robustness-to-removal findings rather than component-necessity evidence.
- Added the Chinese interpretation report:
  `reports/aggregate/final_benchmark_mechanism_ablation_interpretation_20260622.md`.

## 2026-06-22 Robustness Pilot Launch
- Started remote tmux session `robustness_20260622` at
  `2026-06-22T01:44:41+08:00` with fresh seeds `141--146`.
- The runner is executing the first perturbation unit, `event_mix_flux30`, with
  event subtype probabilities changed to particle/flux/thermal =
  `0.35/0.30/0.35`.
- Initial health checks confirm:
  - top-level runner log is active;
  - six seed logs exist;
  - six `25_v2_train_custom_ppo.py` processes are running on `remote-gpu`;
  - all six GPUs are bound by the seed processes at about 1.5 GB memory each;
  - command-line arguments preserve PPO, the met-plus-one-specialist sensing
    geometry, the `0.75` per-step budget, strict current-protocol evaluation
    settings, static-normalised scoring, and behaviour-audit compatibility.
- Estimated completion for the two prepared robustness units, if no failure
  occurs: roughly `2026-06-22 04:30--05:30 CST`. If a perturbation fails and
  requires an immediate direction pivot, the first actionable conclusion is more
  likely `2026-06-22 08:00--12:00 CST`.
- Follow-up at `2026-06-22T01:52:14+08:00`: all six `event_mix_flux30` seeds
  are actively training, around `79k--84k / 200k` timesteps. No Traceback/Error
  lines were observed. `awbc_label_rate=1.000`, and subtype auxiliary accuracy
  is roughly `0.97--0.996`, confirming that the full regime-aware route is active
  under the event-mix perturbation.
- `event_mix_flux30` completed at `2026-06-22T02:07:37+08:00` and passed all
  current gates on fresh seeds `141--146`: operational step `6/6`,
  true-static macro `6/6`, strict true-static step `6/6`, and behaviour `6/6`.
  The decision audit is
  `reports/aggregate/robustness_event_mix_flux30_141_142_143_144_145_146_decision_audit_20260622.json`.
- The same runner then started the second perturbation unit, `weaker_latent10`,
  at `2026-06-22T02:07:37+08:00`.
- During the first collect, a stale internal `SCENEBAL-1` recommendation string
  was observed in the generic decision script. The text was patched locally and
  synced to `remote-gpu` before the second unit's collect phase:
  `scripts/75_v31_decide_scenebal1_stress_claim.py` now says "final
  specialist-budget claim" in recommendation text, and
  `scripts/run_v31_scenebal1_router_conf05_reaudit_20260621.sh` now defaults to
  a generic final-specialist-budget label. This is a report-text cleanup only;
  metric logic and artifacts were not changed.
- Follow-up at `2026-06-22T02:14:30+08:00`: all six `weaker_latent10` seeds are
  actively training, around `68k--74k / 200k` timesteps, with no Traceback/Error
  lines. The run is still using the full PPO route with `awbc_label_rate=1.000`
  and active subtype auxiliary learning under the weaker-latent perturbation.
- `weaker_latent10` completed training at `2026-06-22T02:29:08+08:00` and
  collect at `2026-06-22T02:30:43+08:00`. The robustness pilot finished at
  `2026-06-22T02:30:43+08:00`, earlier than the conservative estimate.
- Final robustness aggregate:
  `reports/aggregate/robustness_pilot_141_142_143_144_145_146_20260622/robustness_summary.md`.
- Result: `event_mix_flux30` passed all gates at `6/6`. `weaker_latent10`
  produced a mixed result: operational step `6/6`, but true-static macro,
  strict true-static step, and behaviour complexity all fell to `5/6`, with
  seed `144` as the common failure.
- Seed `144` failure diagnosis: the learned policy is not fixed-like or
  simple-cycle-like (`fixed_like=false`, `simple_cycle_like=false`,
  `state_dependent=true`), but it is low complexity. In the behaviour audit,
  the top two masks account for `94.97%` of steps, transition count is only
  `12`, and `behavior_complexity_gate_pass=false`. This indicates an
  action-level routing collapse under weaker latent cues, not a total loss of
  state dependence.
- Added the Chinese robustness interpretation report:
  `reports/aggregate/final_benchmark_robustness_interpretation_20260622.md`.

## 2026-06-22 Action-Level Subtype Auxiliary Pilot
- Prepared, syntax-checked locally/remotely, and synced:
  - `scripts/run_v31_finalbenchmark_actionaux_pilot_20260622.sh`
  - `scripts/watch_v31_finalbenchmark_actionaux_20260622.sh`
- Purpose: test whether a small action-level subtype auxiliary objective can
  recover the `weaker_latent10` failure mode without changing PPO, sensor
  geometry, power budget, or the true-static/behaviour evaluation protocol.
- First bounded unit:
  - perturbation remains `weaker_latent10`;
  - seeds remain `141--146`;
  - `SUBTYPE_ACTION_CE_COEF=0.5`;
  - `SUBTYPE_ACTION_MARGIN_COEF=0.05`;
  - `SUBTYPE_ACTION_MARGIN=0.25`.
- Started remote tmux session `actionaux_20260622` on `remote-gpu`.
- Initial health check at `2026-06-22T02:42:26+08:00`: tmux session is active,
  six PPO processes are running, and command-line arguments confirm
  `subtype-action-ce-coef=0.5`, `subtype-action-margin-coef=0.05`, and
  `subtype-action-margin=0.25` are active. GPU memory is allocated on the six
  devices.
- Follow-up at `2026-06-22T02:47:45+08:00`: all six seeds are actively training
  around `58k--60k / 200k` timesteps. The new action-level auxiliary terms are
  active in logs (`subtype_action_ce` nonzero and `subtype_action_margin`
  nonzero on most updates), so this is a real algorithm-layer pilot rather than
  a no-op rerun.
- The user then decided that no further exploration is needed for the current
  relative-strong claim. Stopped remote tmux session `actionaux_20260622` before
  completion and verified no action-aux processes remained; all six GPUs were
  released. This pilot is deferred and should not be treated as current evidence.

## 2026-06-22 Manuscript Evidence Application
- Applied the completed evidence to the canonical ESWA manuscript source:
  `paper/main.tex`, active `paper/sections/*.tex`, submission highlights, and
  active tables.
- Added two concise Results tables:
  - `paper/tables/mechanism_ablation_summary.tex`
  - `paper/tables/event_mix_robustness_summary.tex`
- Updated the Results, Discussion, and Conclusion narrative to state only the
  supported evidence: the 24-seed final benchmark, the six-seed mechanism check,
  and the six-seed event-mixture robustness check.
- Removed the unsupported metric-boundary subsection and figure from the active
  paper narrative. The raw unnormalised macro diagnostic, weaker-latent failure,
  and unfinished action-level auxiliary pilot remain internal evidence and are
  not paper claims.
- Replaced reader-facing internal terms with plain manuscript language:
  `SCENEBAL-2` -> final fixed-backbone, one-specialist benchmark; `router` /
  deployment threshold -> fixed score threshold; `subtype` -> event type.
- Rebuilt `paper/main.pdf` successfully with:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
- Final verification:
  - `main.pdf` is 38 pages.
  - Source and PDF text scans found no visible hits for `SCENEBAL`, `metpair`,
    `seed45`, `h075`, `router`, `specialist-budget`,
    `specialist-bottleneck`, `deployment threshold`, `raw unnormalised`,
    `weaker`, `actionaux`, `strong-claim`, `subtype`, `bounded`, or `boundary`.
  - `main.log` has no undefined references/citations; only a minor 1.8 pt
    overfull box remains.

## 2026-06-22 Figure/Table Readability Pass
- Started the requested looped figure/table readability correction.
- Read the active ESWA plan and academic plotting guidance. Adopted these
  constraints for the pass: vector PDF output for data figures, no decorative
  elements, colorblind-safe palette, no in-figure global titles when captions
  already provide context, no text below about 7 pt at final size, and shared
  style across result figures.
- Rendered active manuscript pages 12--27 from `paper/main.pdf` and inspected
  the method figures, timeline, sensor table, generator-validation table/figure,
  24-seed evidence figure, main evidence table, behaviour audit figure, and
  mechanism/robustness tables.
- First audit findings: result figures have small fonts and crowded labels;
  Figure 5 has stale internal wording in the image itself; Figure 6 legend
  overlaps data; Figure 4 is over-dense; table captions are too long.
- Added a shared plotting module:
  `paper/figures/paper_plot_style.py`.
- Regenerated and normalised the active data/result figures:
  - `paper/figures/figure3_synthetic_statistics.{pdf,png,svg}`
  - `paper/figures/figure_scenebal2_24seed_evidence.{pdf,png}`
  - `paper/figures/figure_scenebal2_behavior_audit.{pdf,png}`
- Reworked the generator validation figure from a dense six-panel raster into a
  four-panel vector figure covering scalar distributions, wind autocorrelation,
  and event-window heterogeneity. Updated the manuscript to include the PDF
  version rather than the PNG.
- Removed old/stale in-figure titles from the result figures and moved context
  into captions. The 24-seed evidence figure now uses compact gate labels, and
  the behaviour audit figure uses end-of-line labels instead of a legend that
  collided with the panel title.
- Reduced natural figure widths from roughly 6.9 inches to roughly 5.35 inches,
  matching the ESWA single-column text width. This prevents LaTeX from shrinking
  the plotted text below a readable final size.
- Reworked tables for readability:
  - `paper/tables/sensor_specs.tex`: removed `resizebox`, shortened caption,
    used ragged fixed-width text columns.
  - `paper/tables/g1_generator_validation.tex`: removed `resizebox`, used
    ragged fixed-width text columns.
  - `paper/tables/scenebal2_24seed_staticnorm_macro_summary.tex`,
    `paper/tables/mechanism_ablation_summary.tex`, and
    `paper/tables/event_mix_robustness_summary.tex`: shortened captions,
    widened layouts with `tabular*`, and moved explanatory detail into table
    notes.
- Rebuilt `paper/main.pdf` successfully with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

## 2026-06-22 Framework and Evidence Figure Supplement
- Supplemented the manuscript with the figure/table set decided after the
  baseline-count and old-figure-style discussion.
- Called the built-in image generation path for a modern-minimal main framework
  redraw. The generated preview established the wide runtime/training/evidence
  layout, but the tool did not expose a project-local file path; therefore the
  submission asset was reproduced as a deterministic matplotlib figure so the
  paper has tracked, reproducible PDF/PNG outputs.
- Added `paper/figures/gen_fig_framework_and_support.py`, which generates:
  - `paper/figures/figure_pdppo_framework_image2.{pdf,png}`;
  - `paper/figures/figure_mechanism_robustness.{pdf,png}`.
- Replaced the older TikZ framework inclusion in
  `paper/sections/04_framework_protocol.tex` with the new framework figure.
- Added `paper/tables/baseline_reference_taxonomy.tex` and inserted it into the
  Baselines and gates subsection so readers can distinguish static references,
  rule-based dynamic baselines, true fixed-static replay, explicit event-type
  replay diagnostics, behaviour audit, and method ablations.
- Added the mechanism/robustness figure to `paper/sections/06_results.tex`,
  visualising the six-seed mechanism check and six-seed event-mixture check
  alongside the existing tables.
- Ran iterative visual checks on the generated figures and adjusted layout to
  avoid label collisions, legend overlap, and overly long taxonomy rows.
- Rebuilt `paper/main.pdf` successfully. The active PDF is now 40 pages, with no
  undefined citations or references. PDF text scan found no visible hits for
  `SCENEBAL`, `router`, `metpair`, `h075`, `seed45`, `CRST`,
  `strong-claim`, `subtype`, `raw unnormalised`, `weaker`, `actionaux`,
  `specialist-budget`, or `deployment threshold`. Remaining diagnostics are the
  existing four BibTeX empty-page warnings and one minor 1.8 pt overfull box.
- Rendered and inspected affected pages at 180 dpi:
  sensor table, generator validation table, generator statistics figure,
  24-seed evidence figure, behaviour audit figure, and mechanism/robustness
  tables. The final pass shows readable labels, no stale in-figure title, and no
  legend/data overlap.
- Added a final TikZ readability pass:
  `paper/figures/pdppo_framework_rewrite_tikz.tex` was simplified with shorter
  box text, clearer panel labels, and the shared soft color palette;
  `paper/figures/data_split_timeline_tikz.tex` was aligned to the same palette.
  Rendered checks confirmed no remaining title/text overlap.
- Final log check: no undefined references/citations; only the existing minor
  `1.79993pt` overfull box remains.

## 2026-06-23 PPO-GPT / PPO-LEMMA Application
- Created the required pre-edit paper backup:
  `paper_archives/paper_pre_ppo_gpt_lemma_20260623_014120.tar.gz`.
- Read and applied the paper-facing requirements from:
  - `docs/06-23-01-PPO-GPT.md`
  - `docs/06-23-01-PPO-LEMMA.md`
  - relevant recommendations from `docs/06-23-01.md`.
- Corrected the central PPO-method mismatch:
  - `paper/sections/03_problem_formulation.tex` now distinguishes candidate
    mask index `u_t` from executed mask `a_t=m^{u_t}`;
  - `paper/sections/04_framework_protocol.tex` now defines a masked categorical
    policy over currently feasible candidate masks and writes the PPO ratio on
    `u_t`, not on an untracked post-projection channel mask;
  - title, abstract, introduction, conclusion, framework figure, and highlights
    now use masked-candidate wording rather than projected channel-score wording.
- Added manuscript evidence objects:
  - `paper/tables/notation_summary.tex`
  - `paper/tables/action_space_instantiation.tex`
  - `paper/tables/online_observability_audit.tex`
  - `paper/tables/main_protocol_hyperparameters.tex`
  - `paper/tables/static_mask_selection_summary.tex`
  - `paper/tables/event_type_loss_decomposition.tex`
  - `paper/tables/static_mask_selection_ledger.tex`
  - `paper/figures/figure_event_type_diagnostics.{pdf,png}`
  - regenerated `figure_pdppo_framework_image2`, `figure_regime_balanced_24seed_evidence`,
    `figure_behavior_audit`, and `figure_mechanism_robustness`.
- Supplemented analysis from existing 24-seed held-out artifacts:
  - event-type loss decomposition shows macro average positive in 24/24 seeds;
  - specialist-use heatmap shows laser/FC4/surface-IR dominance in particle,
    flux, and thermal windows respectively;
  - fixed-mask selection ledger lists validation-selected constant masks and
    validation scores for seeds 117--140.
- Strengthened reproducibility reporting:
  - protocol table now includes chronological partitions, candidate masks,
    PPO settings, actor/critic dimensions, behaviour-cloning pretraining,
    fixed TCN forecaster architecture/fitting, normalisation, and checkpoint
    / reference selection.
- Tightened writing based on subagent round 1:
  - the six-seed mechanism check is now consistently labelled as a pilot;
  - sensor table no longer implies sensors directly observe event labels;
  - funding and data availability no longer read as unfinished draft notes;
  - event-mixture robustness table now includes the changed mixture, fresh seed
    IDs, and mean/median margins rather than only 6/6 counts.
- Verification after this pass:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeds;
  - `main.pdf` is 47 pages;
  - PDF text scan has no reader-visible hits for `SCENEBAL`, `metpair`,
    `router`, `subtype`, `remote-gpu`, `UniVPN`, `aTrust`, `h075`, `seed45`,
    `gate`, `favourable`, `projected`, `projection`, or `channel-score`.

## 2026-06-23 Final Subagent and Compile Closure
- Completed the required subagent review loop:
  - one completeness review over `docs/06-23-01-PPO-GPT.md` and
    `docs/06-23-01-PPO-LEMMA.md`;
  - three wording/terminology audit rounds;
  - a final read-only recheck after fixing round-three P2 items.
- Fixed the final review items:
  - removed duplicate rendered appendix prefixes by changing the visible
    references to bare `\ref{...}` outputs;
  - removed the unused `FW-MAE` abbreviation from the active manuscript;
  - expanded reader-facing `AW imitation` wording to
    `advantage-weighted imitation`.
- Final verification:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeds;
  - `main.pdf` is 47 pages;
  - PDF text scan has no reader-visible hits for the old internal/method terms
    and patterns: `gate`, `favourable`, `projected`, `projection`,
    `channel-score`, `SCENEBAL`, `metpair`, `router`, `subtype`, `remote-gpu`,
    `UniVPN`, `aTrust`, `oracle_loss`, `actionaux`, `raw unnormalised`,
    `deployment threshold`, `strong-claim`, `h075`, `seed45`,
    `event-type latent`, `shortcut`, `FW-MAE`, `Appendix Appendix`,
    `Supplementary Section Appendix`, `AWBC`, and `AW imitation`.

## 2026-06-23 Stricter 06-23-02 PPO-LEMMA Pass Started
- New goal created to implement `docs/06-23-02-PPO-LEMMA.md` more strictly than
  the prior pass.
- Required pre-edit backup completed before any article modification:
  `paper_archives/paper_pre_ppo_lemma_062302_20260623_025244.tar.gz`.
- This pass explicitly treats high-priority experiment and analysis supplements
  as blocking requirements unless the 06-23-02 checklist proves that an item is
  already satisfied by an existing completed artifact.
- Parsed the 06-23-02 review into
  `reports/aggregate/ppo_lemma_062302_requirement_matrix_20260623.md`.
  Initial status marks all three high-priority items as missing until verified
  by concrete artifacts.
- Local and remote artifact audit found only the previous six-seed mechanism
  ablation (`117--122`). No 24-seed mechanism ablation artifact exists locally
  or on `remote-gpu`.
- Started the high-priority R1.1 remote extension in tmux:
  `ppo_lemma_062302_ablation24`.
  Configuration: seeds `117--140`, variants `no_imitation_guide`,
  `no_regime_aux_path`, and `no_staticnorm_train`, router confidence `0.5`,
  `RUN_THRESHOLD_SENSITIVITY=0`, evaluation directory
  `eval_router_conf05_mechablation_20260623`, behaviour audit directory
  `behavior_audit_router_conf05_mechablation_20260623`.
- Early monitoring found that the existing pilot runner launches every seed in
  `SEEDS_TEXT` concurrently. For 24 seeds this would oversubscribe the six GPUs.
  The first tmux session was stopped before active GPU training. No active
  training processes remained. The corrected run will use six-seed batches and
  a new tag `20260623b` so partial directories from the stopped attempt are not
  treated as evidence.

## 2026-06-23 Stricter 06-23-02 PPO-LEMMA Pass Continued
- Completed the high-priority local manuscript/figure items that do not depend
  on the remote 24-seed mechanism ablation:
  explicit `L_event` formula, removal of unused `L_guide` from the main loss
  equation, appendix note that the guide prior is optional and unused in the
  reported configuration, three-stage framework figure, and distributional
  behaviour audit with event-type specialist heatmap.
- Generated the fixed TCN forecaster validation diagnostic against persistence:
  `reports/aggregate/tcn_forecaster_validation_20260623/`. The result is
  negative under ordinary validation nMAE, so it is recorded as a diagnostic and
  is not inserted as positive manuscript evidence.
- `paper/main.tex` currently compiles successfully. Source terminology scan over
  the active manuscript files finds no active-source hits for `SCENEBAL`,
  `regime-aware`, `event-context support`, or lowercase `scenebal`.
- The PDF still contains the six-seed pilot mechanism wording because Table 11
  has not yet been replaced by the required 24-seed mechanism ablation. This is
  an intentional temporary state and remains blocking for goal closure.
- The corrected remote tmux session `ppo_lemma_062302_ablation24` is active with
  date tag `20260623b`. It batches seeds `117--140` over variants
  `no_imitation_guide`, `no_regime_aux_path`, `no_staticnorm_train`, and
  `full_reference`.
- First batch (`117--122`) for `no_imitation_guide` completed and passed the
  six-seed stage checks. This is partial evidence only. The first batch for
  `no_regime_aux_path` is running on the six GPUs; Table 11 and the mechanism
  figure must wait for the full 24-seed collection.
- Added and smoke-tested the continuous-margin post-processing path for the
  eventual 24-seed mechanism result:
  `scripts/80_v31_summarize_mechanism_margins.py`,
  `paper/tables/gen_mechanism_ablation_table.py`, and
  `paper/figures/gen_fig_mechanism_continuous.py`. These scripts have been
  synced to `remote-gpu`; the smoke test used old six-seed data only and is not
  migrated into the manuscript.
- R1.3 is downgraded to a training-diagnostic item: available PPO histories do
  not contain validation macro or validation step-loss curves, so any curve from
  these files cannot be used as validation-convergence evidence.
- Completed two low-priority local cleanup items from the 06-23-02 matrix:
  Discussion now explains why domain-threshold and contextual-bandit baselines
  are future deployment comparisons rather than part of the current sequential
  protocol, and Figure 5 now shows seed-level step margins, macro margins, and
  the step threshold in the same panel.
- Rebuilt `paper/main.pdf` successfully after these edits. Current warnings are
  non-fatal table overfull/underfull warnings already tracked for dense tables;
  no compile failure or undefined-reference failure occurred.
- Remote status after the first `no_regime_aux_path` training wave: GPU training
  completed and the seed scripts are in replay-gate / evaluation collection.
  The tmux session is still active and should be monitored rather than killed.
- The first `no_regime_aux_path` batch (`117--122`, tag `20260623b`) has now
  completed collection and passes all six seeds under the router-conf0.5
  protocol. This differs from the older `20260621` six-seed pilot, where the
  same ablation had one strict-step failure. Mechanism wording must therefore
  wait for the full 24-seed rerun and should not preserve the old 5/6
  degradation claim by default.
- The next first-batch variant, `no_staticnorm_train`, started automatically at
  `2026-06-23T03:40:32+08:00` on the six GPUs.
- The first 6-seed batch (`117--122`, tag `20260623b`) has now completed all
  three ablation variants plus the full-reference collection:
  `no_imitation_guide`, `no_regime_aux_path`, `no_staticnorm_train`, and
  `full_reference` each produced macro, raw-macro, and old-claim aggregate
  directories, plus the batch-level
  `reports/aggregate/mechanism_ablation_pilot_117_118_119_120_121_122_20260623b/`.
  The wrapper advanced automatically to the second batch (`123--128`) at
  `2026-06-23T04:04:43+08:00`; `no_imitation_guide` training is active there.
- In the second batch (`123--128`), `no_imitation_guide` completed training,
  replay-gate checks, and collection at `2026-06-23T04:27:12+08:00`; the
  wrapper then started `no_regime_aux_path` for the same seeds.
- In the second batch (`123--128`), `no_regime_aux_path` completed training and
  collection at `2026-06-23T04:47:37+08:00`; `no_staticnorm_train` then started.
  Aggregate count for `20260623b` rose to 19, matching the expected additional
  macro/raw-macro/oldclaim outputs for the completed variant.
- The second 6-seed batch (`123--128`, tag `20260623b`) completed all three
  ablation variants plus full-reference collection at
  `2026-06-23T05:11:44+08:00`. The wrapper advanced to the third batch
  (`129--134`) and started `no_imitation_guide`.
- The third 6-seed batch (`129--134`, tag `20260623b`) completed all three
  ablation variants plus full-reference collection at
  `2026-06-23T06:19:10+08:00`. The wrapper advanced to the fourth and final
  batch (`135--140`) and started `no_imitation_guide`.
- The fourth 6-seed batch (`135--140`, tag `20260623b`) completed and the final
  24-seed collection finished at `2026-06-23T07:30:26+08:00`. Final aggregate:
  `reports/aggregate/mechanism_ablation_24seed_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133_134_135_136_137_138_139_140_20260623b/`.
  The table/figure post-processing watcher initially stopped after successful
  aggregate sync because zsh expanded the remote rsync log wildcard locally; the
  log sync and post-processing were rerun manually with a quoted remote path and
  explicit seed list.
- Generated the final 24-seed continuous-margin artifacts:
  `reports/aggregate/mechanism_ablation_continuous_margins_24seed_20260623b/`,
  regenerated `paper/tables/mechanism_ablation_summary.tex`, and regenerated
  `paper/figures/figure_mechanism_robustness.{pdf,png}`.
- Replaced the old six-seed mechanism narrative in Results, Discussion, and
  Conclusion. The new interpretation is bounded: Full PD-PPO passes the strict
  fixed-mask step check in 24/24 seeds; no-imitation and no-event-context
  variants pass 23/24 and 21/24; no-balanced-training-loss still passes 24/24.
  Macro-margin deltas against Full have bootstrap intervals overlapping zero,
  so the paper frames the ablation as reliability evidence rather than a large
  independent macro-margin effect.

## 2026-06-23 Strict 06-23-02 PPO-LEMMA Closure
- Subagent audit initially found one real P1: Table 11 did not include the
  required median macro margin and median bootstrap interval even though the
  aggregate artifact contained them.
- Fixed the P1 by updating `paper/tables/gen_mechanism_ablation_table.py` and
  regenerating `paper/tables/mechanism_ablation_summary.tex`. Table 11 now
  reports mean macro margin, median macro margin, and paired mean delta, with
  95% bootstrap intervals where applicable.
- Removed the stale six-seed mechanism plotting function from
  `paper/figures/gen_fig_framework_and_support.py`, so rerunning the framework
  figure script can no longer overwrite the current 24-seed mechanism figure.
- Regenerated `paper/figures/figure_pdppo_framework_image2.{pdf,png}` with
  more spacing between the stage headers and the internal flow boxes.
- Recompiled `paper/main.pdf` successfully with `latexmk`; the final PDF has 50
  pages. Remaining warnings are dense-table layout warnings in pre-existing
  active tables and BibTeX empty-page metadata warnings, not 06-23-02 blockers.
- Final source and PDF text scans found no reader-visible hits for the internal
  terminology set: `SCENEBAL`, `metpair`, `router`, `subtype`, `AWBC`,
  `FW-MAE`, `six-seed pilot`, `event-context support`, and related stale labels.
- The same subagent rechecked the fixes and reported no P0/P1 blockers. It
  judged all 06-23-02 high-priority requirements complete; remaining P2 notes
  are visual polish only.
- Updated
  `reports/aggregate/ppo_lemma_062302_requirement_matrix_20260623.md` so the
  matrix, artifacts, PDF checks, and subagent audit now agree.

## 2026-06-24 06-23-03 ESWA Review Pass
- Created a new active goal for `docs/06-23-03.md` and backed up the active
  manuscript to
  `paper_archives/paper_pre_062303_20260624_000251.tar.gz`.
- Parsed the review into
  `reports/aggregate/ppo_lemma_062303_requirement_matrix_20260624.md`.
  High-priority items are Table 11 layout and explicit expert-selection
  frequency heatmap wording/visualisation.
- Initial audit found that Figure 1 is already a three-stage protocol diagram
  in `paper/figures/gen_fig_framework_and_support.py`, and Figure 2 already
  includes partition step counts in `paper/figures/data_split_timeline_tikz.tex`.
- Initial audit also found that the existing fixed TCN validation diagnostic is
  not manuscript-ready positive evidence: persistence is vastly better under the
  recorded nMAE calculation, so it will remain a diagnostic gap rather than a
  supporting table unless the evaluation issue is separately resolved.

## 2026-06-24 Strict 06-23-03 Closure
- Closed the two high-priority review blockers. Table 11 now uses two physical
  rows per policy variant, aligning point estimates with bootstrap 95% CIs.
  Figure 7 Panel B now reads as a specialist-selection frequency heatmap, and
  the Results text explicitly connects that heatmap to the scalar behaviour
  diagnostics in Panel A.
- Added the bounded training-diagnostic appendix figure from saved PPO histories:
  `paper/figures/figure_training_diagnostics.{pdf,png}` and
  `reports/aggregate/training_diagnostics_20260624/`. The manuscript states
  that these are optimisation traces only, not validation macro-score or
  efficiency evidence.
- Corrected and reran the fixed TCN forecaster benchmark. The original
  diagnostic included padded warm-up windows that created near-zero-variance
  mask inputs and numerically extreme predictions. The corrected 24-seed audit
  excludes those warm-up windows but still shows TCN behind persistence:
  weighted nMAE `0.381874` vs `0.210347`, with 0/24 seeds and 0/9 targets
  beating persistence. The corrected artifact is
  `reports/aggregate/tcn_forecaster_validation_20260624_corrected/`; it is not
  included by `paper/main.tex`.
- Queued a stricter rare-flux event-mixture robustness run on the only valid
  remote server via tmux `eswa062303_flux10`. It has not started yet because
  all six GPUs are occupied by another user's jobs. Its log is
  `logs/eswa062303_flux10_queue_20260624.log`; no result from this run has been
  migrated into the manuscript.
- Updated Figure 1 labels away from explicit `Fixed TCN` wording to "fixed
  forecaster" / "forecaster loss" and changed Figure 2's timeline label to
  "forecaster + normalisation" to avoid confusing the reader with the failed
  TCN validation diagnostic.
- Rebuilt `paper/main.pdf` successfully with `latexmk`; the final PDF has 54
  pages and no undefined references. Remaining warnings are dense-table
  overfull/underfull warnings and BibTeX empty-page metadata warnings.
- Final PDF text scan found no visible hits for the stale/internal terminology
  set `SCENEBAL`, `metpair`, `router`, `subtype`, `Fixed TCN`, `TCN forecaster
  validation`, or `sample-efficiency`. The only flagged phrase is the factual
  "six-seed event-mixture check" in the conclusion.

## 2026-06-24 Remote Rare-Flux Queue Check
- Checked `remote-gpu` via the SSH alias only. Server is online as `LAB113`;
  project path `~/_code/microclimate_demo/rl_sensor_scheduling_framework` is
  accessible.
- tmux session `eswa062303_flux10` is still alive. It is waiting for the
  condition that all six GPUs have low utilisation and low memory use before
  launching `DATE_TAG=20260624flux10 SEEDS_TEXT="147 148 149 150 151 152"
  ROBUSTNESS_VARIANTS_TEXT=event_mix_flux10`.
- Queue log `logs/eswa062303_flux10_queue_20260624.log` shows repeated
  `gpu_free` checks. As of the latest check, GPUs 2--5 were free, while GPUs 0
  and 1 were occupied by another user's `UIBDiffusion.py` jobs. The experiment
  has therefore not started, and no `robustness_pilot_147_148_149_150_151_152_20260624flux10`
  aggregate directory exists yet.
- No remote processes were interrupted.

## 2026-06-25 Remote Rare-Flux Completion Check
- Rechecked `remote-gpu` via the SSH alias only. The queued tmux session
  `eswa062303_flux10` has finished; the queue log reports
  `robustness_pilot_done` at `2026-06-24T03:59:24+08:00`.
- Synced the completed rare-flux artifacts back to the local workspace:
  `reports/aggregate/robustness_pilot_147_148_149_150_151_152_20260624flux10`,
  the matching `event_mix_flux10` old-claim, macro, raw-macro, and decision
  audit outputs, plus `logs/eswa062303_flux10_queue_20260624.log`.
- The rare-flux six-seed robustness check is complete for seeds
  `147 148 149 150 151 152`: operational step `6/6`, true-static macro
  `6/6`, strict true-static step `6/6`, behaviour complexity `6/6`, mean step
  margin `0.092293`, and mean macro margin `0.087765`.
- The decision audit records `upgrade_allseed_strict` with no failed seeds.
  This is robustness support for the current bounded claim, not a replacement
  for the main 24-seed evidence.

## 2026-06-25 Rare-Flux Result Analysis
- Analysed the completed rare-flux event-mixture run against the main 24-seed
  benchmark, the 24-seed mechanism ablation, the earlier higher-flux
  event-mixture pilot, the weaker-latent boundary check, and the raw
  unnormalised macro diagnostic.
- Wrote the standalone analysis report:
  `reports/aggregate/rare_flux_event_mix_result_analysis_20260625.md`.
- Updated `findings.md` with the interpretation: rare-flux seeds `147--152`
  passed all gates, but the weakest seed has a tight strict true-static step
  margin (`0.002898`), so the correct reading is stronger changed-mixture
  sensitivity support rather than universal rare-event robustness.
- Updated `task_plan.md`: the remote rare-flux monitor item is complete, and
  manuscript migration is now tracked as a separate pending item so the paper
  is not treated as already updated.

## 2026-06-25 Rare-Flux Manuscript Migration
- Backed up the active manuscript inputs before editing:
  `paper_archives/paper_pre_rare_flux_migration_20260625_011923.tar.gz`.
- Updated `paper/tables/event_mix_robustness_summary.tex` to include both
  changed-mixture rows: higher-flux `0.35/0.30/0.35` and lower-flux
  `0.45/0.10/0.45`. The table now reports mean/median step and macro margins
  plus strict fixed-mask / behaviour gate counts.
- Revised `paper/sections/06_results.tex`,
  `paper/sections/07_discussion_future_work.tex`, and
  `paper/sections/08_conclusion.tex` with cautious wording: the result supports
  a benchmark-local changed-mixture sensitivity statement, not robustness to
  arbitrary event prevalence or event detectors.
- Rebuilt `paper/main.pdf` successfully with `latexmk`. The final PDF remains
  `54` pages. No undefined references remain; the only layout warnings are
  pre-existing overfull/underfull warnings in other tables and existing BibTeX
  empty-page metadata warnings.
- PDF text scan confirms the new table and boundary wording are visible:
  "Sensitivity checks under changed event mixtures", "Higher-flux mixture",
  "Lower-flux mixture", and "do not establish robustness to arbitrary event
  prevalence". The old "six-seed event-mixture check" wording no longer appears.

## 2026-06-25 Plan Completion Audit
- Reviewed the active plan
  `.planning/2026-06-10-eswa-terminology-rewrite/task_plan.md`, root
  `task_plan.md`, and inactive historical `.planning/*/task_plan.md` files.
- Fixed stale plan-state debt: the active plan no longer says rare-flux evidence
  is unmigrated, and the root historical plan no longer lists the superseded
  V3.1 static-comparator handoff items as active incomplete work.
- Added the remaining true active tasks explicitly to the active plan:
  pin the final code revision / versioned evidence archive, then run a final
  submission-package audit after the archive is pinned.
- Current active-plan unfinished items are therefore limited to those two
  submission-packaging tasks. Older unchecked items in
  `.planning/2026-06-07-pd-ppo-static-break-recalibration/` belong to an
  inactive historical plan and are superseded by the completed SCENEBAL-2 line.

## 2026-06-25 Post-Codex Manuscript Polish
- Confirmed no active Codex process was working inside the PD-PPO paper path
  before editing.
- Backed up the current compiled PDF before polishing:
  `paper/backups/main_before_polish_20260625_014131.pdf`; size
  `2370586` bytes; SHA256
  `a255527f5692ebccc043f85205db44729fc8917ba1f71c075ce05af68c91f083`.
- Applied a claim-preserving polish to `paper/main.tex` and the active included
  manuscript sources:
  `sections/01_introduction.tex`, `sections/02_related_work.tex`,
  `sections/03_problem_formulation.tex`, `sections/04_framework_protocol.tex`,
  `sections/05_simulation_setup.tex`, `sections/06_results.tex`,
  `sections/07_discussion_future_work.tex`, `sections/08_conclusion.tex`, and
  `sections/appendix_theory.tex`.
- The edits removed residual reader-steering / AI-style phrasing such as
  "This paper presents", "should be read", "This pattern", and "This design"
  without changing any reported numeric result, table entry, figure asset, or
  evidence boundary.
- Source scan over the 24 included TeX files found zero hits for internal codes
  or obsolete reader-facing terms (`SCENEBAL`, `V3.1`, `metpair`, `h075`,
  `frozen oracle`, `candidate prior`, `compact static`, `deployable static`,
  `FW-MAE`, `projected PPO`, and related terms), zero reader-steering hits, no
  missing BibTeX keys, and no long prose lines over 120 characters outside PDF
  metadata.
- Rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex`. Output: `53` pages, `2370299` bytes, SHA256
  `3df0f0ba55f0a8392b6577dfdb3cf4da5bb53a61fb717b77626c07fcd862dd17`.
- PDF text scan found no old/internal terminology or reader-steering phrases.
  Log scan found no undefined references/citations and no stale label warning.
  Remaining BibTeX warnings are the existing empty-page metadata warnings for
  `Liu2024`, `Murad2020`, `Pendyala2024`, and `Wei2020`; remaining layout
  warnings are table overfull/underfull boxes outside this prose-polish pass.
- `git diff --check` passes on the LaTeX files touched by this pass. A full
  paper-repo `git diff --check` is still blocked by trailing whitespace in
  generated SVG files modified by the broader Codex figure changes, not by the
  prose edits.

## 2026-06-25 06-25-01 PPD Deep Rewrite
- User requested a more thorough rewrite according to
  `docs/06-25-01-PPD.md`, because the current manuscript still carried too much
  historical residue.
- Confirmed no active Codex/Claude process was working inside the paper path
  before editing.
- Backed up the current compiled PDF and active source package before applying
  the rewrite:
  `paper/backups/main_before_062501_ppd_rewrite_20260625_015911.pdf`;
  size `2370299` bytes; SHA256
  `3df0f0ba55f0a8392b6577dfdb3cf4da5bb53a61fb717b77626c07fcd862dd17`.
  Source archive:
  `paper_archives/paper_pre_062501_ppd_rewrite_20260625_015911.tar.gz`;
  size `10608374` bytes; SHA256
  `c8035be5a407c8eaba121381a128b74b96790cd6e662f999798653aee11e54d2`.
- Applied a problem-first rewrite to the active manuscript: title, PDF metadata,
  abstract, keywords, Introduction, Related Work, Problem Formulation, Method,
  Simulation Setup, Results, Discussion, and Conclusion.
- Added `paper/tables/related_work_positioning.tex` to make the paper's
  relation to AoI/estimation scheduling, active perception, DRL adaptive
  sensing, and forecasting-model evaluation explicit.
- Structural changes: the main text now foregrounds forecasting-oriented
  specialist scheduling and the strict replay protocol; the Introduction states
  H1--H3; the formulation section is shorter and moves proof detail to the
  appendix; the training reward is located in the method section; result and
  discussion prose no longer repeatedly restates the same 24-seed conclusion.
- Rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex`. Output: `47` pages, `2357699` bytes, SHA256
  `ad5eb8b6678c4bae0a4258bebefe867261a27252270b83adcb97bd7f97aeb569`.
- Verification: source scan over the included TeX tree found zero hits for
  internal/stale terms and reader-steering residues (`SCENEBAL`, `V3.1`,
  `metpair`, `h075`, `frozen oracle`, `candidate prior`, `compact static`,
  `deployable static`, `FW-MAE`, `projected PPO`, `This paper presents`,
  `This pattern`, `This design`, `should be read`, and the old title phrase).
  PDF text scan found zero hits for the same set.
- Log scan found no undefined references/citations and no stale label warning.
  The remaining BibTeX warnings are the existing empty-page metadata warnings
  for `Liu2024`, `Murad2020`, `Pendyala2024`, and `Wei2020`; remaining layout
  warnings are existing table overfull/underfull boxes.
- Targeted `git diff --check` passes for the touched tracked/untracked LaTeX
  files in this pass.

## 2026-06-25 06-15 Comparison Style Repair
- Started after the user flagged that the active manuscript had again become too
  AI-like: too many colon-style hooks, too many hyphenated internal labels, and
  inconsistent coined terminology.
- Created working notes under
  `/home/horeb/agent/tmp/pdppo_0615_style_rewrite/` and backed up the current
  active paper sources to
  `paper_archives/paper_pre_0615_style_rewrite_20260625_030818.tar.gz` with
  SHA256 `a3ddcaf54f03e2cf9f04f4a3c1c7303f0ab9e217efb8a8b92e0d88358f78a258`.
  The then-current `main.pdf` was corrupt from an interrupted compile, so no PDF
  backup was copied at that point.
- Located the closest around-06-15 style reference as
  `paper/pdppo_crst_rewrite.pdf` dated 2026-06-12 and extracted it to
  `/home/horeb/agent/tmp/pdppo_0615_style_rewrite/old_pdppo_crst.txt` for
  comparison.
- Rewrote high-visibility and included manuscript material in a claim-preserving
  pass: `main.tex`, sections 01--08, appendix theory, notation/action/protocol
  tables, fixed mask selection tables, mechanism/robustness tables, generator and
  sensor tables, plus the framework, behaviour-audit, and mechanism figures.
- Removed the renewed style residues from active included source: first-person
  pronouns, visible prose colons, and stale style strings all scan as zero. The
  repaired set includes removal of labels such as `fixed-mask`, `final-test`,
  `event-context`, `static-selection`, `candidate-mask`, `fixed-forecaster`,
  `static-reference-normalised`, `one-specialist`, `regime-balanced`,
  `step-margin`, and `forecast-loss` from reader-visible active source.
- Removed the phone number from the active front matter and changed caption label
  separators to periods so table/figure captions no longer render as `Table 1:`
  and `Figure 1:` in the PDF.
- Regenerated affected figure PDFs/PNGs after fixing embedded text:
  `figure_pdppo_framework_image2`, `figure_behavior_audit`, and
  `figure_mechanism_robustness`.
- Rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex`. Output: 47 pages, 2,356,206 bytes, SHA256
  `39dea898850c66cfddc9005d2480713e0156954f35564e0d21ed7e708e2ea18c`.
  Final PDF backup:
  `paper/backups/main_after_0615_style_rewrite_final_20260625_035429.pdf` with
  the same SHA256.
- Final verification: included-source scan found zero hits for the stale style
  and term list, zero first-person hits, and zero visible source prose-colon
  hits. PDF text scan found zero hits for the same stale term list and zero
  first-person hits. The only remaining body colons are ESWA class front-matter
  labels `Email addresses:` and `Keywords:`.
- `main.log` has no undefined references, undefined citations, stale label
  warning, runaway argument, or fatal error. Targeted `git diff --check` passes
  in both the paper worktree and the framework root.

## 2026-06-29 Supervisor-Facing Experiment Update
- Created a concise Chinese supervisor handoff report focused on the current
  experiment changes, main figures, and supported results:
  `reports/supervisor_update_20260629/supervisor_report_20260629.md`.
- Copied four figure assets into the same handoff directory so the report is
  self-contained: experiment protocol, 24-seed main result, event-type behaviour
  audit, and mechanism ablation.
- Exported a PDF version with XeLaTeX and CJK fonts:
  `reports/supervisor_update_20260629/supervisor_report_20260629.pdf`.
- Verification: image links exist; `pdfinfo` reports 3 pages and 684,327 bytes;
  `pdftotext` confirms the Chinese text and key numerical results render; pages
  1--3 were rasterised and visually checked for layout/readability.

## 2026-07-01 LEMMA Fourth-Round Theory/Structure Revision
- Started from the user-provided LEMMA fourth-round review that had been
  organised as `docs/07-01-01-LEMMA.md`.
- Backed up the active paper before editing:
  `paper/backups/main_before_lemma0701_20260701_013700.pdf`, 2,356,206 bytes,
  SHA256 `39dea898850c66cfddc9005d2480713e0156954f35564e0d21ed7e708e2ea18c`.
- Archived the active recoverable manuscript bundle before editing:
  `paper_archives/paper_before_lemma0701_20260701_013700.tar.gz`, 12,990,580
  bytes, SHA256
  `5d52c91cb65e8a6307d34f3b78d21d0bdca9519eef6eeee60eacb2749048831d`.
- Applied the LEMMA high-priority theory fix: Proposition 1 now explicitly
  compares fixed specialist subsets against an oracle dynamic policy with access
  to event labels; the surrounding text connects this oracle to event label
  replay and treats the learned policy gap as empirical.
- Added benchmark support for the Proposition 1 incompatibility premise from the
  event-type loss table, with careful wording that avoids overclaiming
  per-event all-seed dominance.
- Expanded the Proposition 2 explanation and appendix proof preface as a
  sufficient-condition linear-Gaussian construction whose qualitative mechanism
  is supported by the event-type decomposition.
- Reorganised method/setup content: added fixed-forecaster architecture text and
  a new `tables/pdppo_training_hyperparameters.tex` in Section 4, shortened the
  Section 5.3 opening, and reduced `tables/main_protocol_hyperparameters.tex` to
  protocol-facing fields.
- Moved generator validation checks from Section 5.2 into the appendix under
  `Generator Validation Checks`, leaving the main text as a short pointer.
- Split Section 6.4 into `Mechanism ablation` and `Robustness checks`, clarified
  Figure 6/7 heatmap repetition, and expanded Table 12 and Table 10 notes.
- Added the Discussion boundary explaining why the current evaluation does not
  compare against other learned RL implementations such as DQN or SAC.
- Rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex`. Output: 51 pages, 2,368,168 bytes, SHA256
  `e6f880abd04863d5a047db7646e2b11a1eafe6c468629549dd751fb8409b420c`.
- Verification: included-source scan resolved 26 files and found zero duplicate
  labels among included files, zero missing references, and zero hits for the
  stale/AI-style watch list (`SCENEBAL`, `V3.1`, `frozen oracle`, `candidate
  prior`, `should be read`, `should not be interpreted`, `This pattern`, `This
  distinction`, `rather than`). PDF text scan found zero hits for the same list.
- Log verification found no undefined control sequences, references, citations,
  stale cross-reference warning, or rerun warning. Targeted `git diff --check`
  passed for all touched manuscript sources/tables.

## 2026-07-01 N-Series Follow-Up Fixes
- Started from the newly reported N-1--N-5 issues after the LEMMA fourth-round
  revision.
- Backed up the active post-LEMMA PDF before editing:
  `paper/backups/main_before_nfix_20260701_020321.pdf`, 2,368,168 bytes,
  SHA256 `e6f880abd04863d5a047db7646e2b11a1eafe6c468629549dd751fb8409b420c`.
- Archived the active recoverable manuscript bundle before editing:
  `paper_archives/paper_before_nfix_20260701_020321.tar.gz`, 12,955,749 bytes,
  SHA256
  `54f25b1b9e1a630c785eed6fb290a15030a511874275b2fd8aec404a6791fabf`.
- Fixed N-1 by adding explicit `\clearpage` barriers before Appendix B
  `Generator Validation Checks`, Appendix C `Training Diagnostics`, and Appendix
  D `Fixed mask reference selection`. PDF text verification now shows Table B.14
  and Figure B.8 under Appendix B, Figure C.9 under Appendix C, and Table D.15
  under Appendix D.
- Fixed N-2 by rendering Eq. (11) as
  `\mathcal{L}_{\mathrm{PD}\text{-}\mathrm{PPO}}`, so the extracted equation label
  is `LPD-PPO` rather than the misspelled `PD-PO` form.
- Fixed N-3 by using the section reference output directly for appendix section
  references that already expand to `Appendix A.1`, `Appendix A.2`, `Appendix
  A.5`, `Appendix B`, and `Appendix D`. PDF text scan found zero `Appendix
  Appendix` hits.
- Fixed N-4 by removing the paragraph-title period in `Benchmark verification of
  the incompatibility condition`, eliminating the rendered double period.
- Verified N-5: source `tables/baseline_reference_taxonomy.tex` is captioned
  `Comparison references used in the evaluation`, and PDF page 21 renders Table
  6 with the expected comparison reference taxonomy, including the event-aware
  diagnostic replay row.
- Rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex`. Output: 52 pages, 2,368,460 bytes, SHA256
  `a7a94d90d02f454a822a132606e4a26cb446273a10b7d0d0587e6b7764dac998`.
- Verification: included-source scan resolved 26 files and found zero duplicate
  labels, zero missing references, and zero hits for `Appendix Appendix`,
  `PD-PO`, `LPD-PPO`, doubled benchmark-heading periods, and duplicated
  `event event-sensitive`. PDF text scan found zero hits for the same error
  classes, and log verification found no undefined references/citations or rerun
  warnings. Targeted `git diff --check` passed.

## 2026-07-01 Supervisor Report Readability Revision
- Reworked the supervisor-facing experiment update after the user judged the
  2026-06-29 report too technical and difficult to read.
- Created a new easy-language HTML/PDF handoff:
  `reports/supervisor_update_20260701/supervisor_report_easy_20260701.html`
  and
  `reports/supervisor_update_20260701/supervisor_report_easy_20260701.pdf`.
- Removed the unfinished-looking framework figure from the handoff. Replaced it
  with a simple HTML/CSS flow diagram explaining the change from the older
  many-sensor competition line to the current background-channel plus
  event-specific expert-sensor scheduling task, with direct parenthetical
  comparisons.
- Kept only three result figures in the report: 24-repeat main result,
  event-type behaviour audit, and mechanism ablation.
- Verification: Chrome-generated PDF is 3 A4 pages and 876,987 bytes. Text scan
  found no reader-visible internal identifiers such as `SCENEBAL`, `metpair`,
  `router`, `V3.1`, `macro gate`, or `operational`; remaining terminology is
  explained in plain Chinese.

## 2026-07-02 Reference-Audit Corrections
- Started from the user-provided 23-reference audit summary. Before editing,
  backed up the active bibliography to
  `paper/backups/references_before_refaudit_20260701_235827.bib`, 13,711 bytes,
  SHA256 `0f7b04395efd33d9d3c3fb8f5385aac92cd46a359f1d571c7521f87011ea8358`.
- Patched `paper/references.bib`: changed Golovin--Krause from the erroneous
  2017 arXiv-style entry to the 2011 JAIR article; changed the AntAWS first
  author from Wille to Wang et al.; updated AlAhdab2025 from arXiv to the ICML
  2025/PMLR 267 proceedings entry; corrected Bajcsy to the 2018 journal volume
  year; protected `Ad\'{e}lie Land` capitalisation; added Qu page/article 6972;
  added Jonah volume/pages from DOI metadata; added Aloni volume/article fields;
  added Bai2018; and removed Liang2024.
- Patched active text: cited Schulman2017 at the masked PPO introduction, cited
  Bai2018 for the residual temporal convolutional forecaster, updated Golovin and
  Bajcsy citation keys, and removed Liang2024 from the Related Work DRL cluster.
- Metadata note: DOI/Crossref checks differed from two audit shorthand values.
  Aloni's DOI resolves to Environmental Modelling & Software volume 185, article
  106283, formal publication year 2025; Jonah's DOI resolves to IEEE Access 14,
  pages 40042--40059. The bibliography follows the DOI metadata.
- Rebuilt `paper/main.pdf` with `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex`. Output: 52 pages, 2,369,087 bytes, SHA256
  `53834192290628df81c70a79b0414b3fb8466000d8788818e0d6b607f7efd4b3`.
- Verification: active included-source citation scan resolved 26 files, found
  zero missing citation keys, zero duplicate BibTeX keys, and no active-source
  `Liang2024`, `golovin2017...`, or `Bajcsy_2017` residue. The `.bbl` and PDF
  text show Golovin and Krause (2011), Wang et al. (2023), Ahdab et al. (2025),
  Bajcsy et al. (2018), Qu 22:6972, Jonah 14:40042--40059, Aloni 185:106283,
  Bai et al. (2018), and Schulman et al. (2017); Liang has zero PDF hits.
  LaTeX reported no undefined citations/references and targeted `git diff
  --check` passed. Four remaining BibTeX warnings are pre-existing empty-page
  warnings for Liu2024, Murad2020, Pendyala2024, and Wei2020.

## 2026-07-01 Natural-Language Rewrite of 2026-06-29 Supervisor Report
- Reworked the original 2026-06-29 supervisor report in place, following the
  user's request to base the revision on
  `reports/supervisor_update_20260629/supervisor_report_20260629.pdf`.
- Backed up the previous Markdown and PDF before overwriting:
  `supervisor_report_20260629_before_natural_20260701_235226.md` and
  `supervisor_report_20260629_before_natural_20260701_235226.pdf`.
- Rewrote the report in natural Chinese with explicit old-vs-new experimental
  design explanation: older many-sensor competition and fixed-sensor shortcut
  risk versus the current background meteorological channel plus one expert
  sensing slot.
- Removed the earlier protocol/framework figure from the report body. Kept the
  main 24-repeat result, event-dependent behaviour result, and mechanism
  ablation figure, with plain-language captions.
- Rebuilt `supervisor_report_20260629.pdf` from Markdown via Pandoc/XeLaTeX.
  Output is 2 pages, 535,824 bytes. Text scan found no stale/internal terms from
  the checked list (`SCENEBAL`, `metpair`, `router`, `operational`,
  `macro gate`, `event macro`, `replay`, `seed`, `duty`, `V3.1`, etc.).
  Raster preview of pages 1--2 was checked for readability.

## 2026-07-02 Main-Manuscript Figure/Table Flow Repair
- Addressed the figure-placement audit for the canonical ESWA source
  `paper/main.tex` and active section files, without editing `raw.tex` or legacy
  rewrite sources.
- Backed up the pre-edit PDF to
  `paper/backups/main_before_figflow_20260702_010023.pdf`, 2,369,087 bytes,
  SHA256 `53834192290628df81c70a79b0414b3fb8466000d8788818e0d6b607f7efd4b3`,
  and archived the paper directory to
  `rl_sensor_scheduling_framework/paper_archives/paper_before_figflow_20260702_010023.tar.gz`,
  10,607,602 bytes, SHA256
  `8630a97e2706a4002ab1dbb42bb4e8f5d1eac9dd33a427c911e0d500f5f9fd16`.
- Moved the framework figure to the start of Section 4 and forced it to render
  before Section 4.1; added a Section 3 qualitative specialist/regime matrix as
  an unnumbered visual anchor so Figure 1 remains the framework figure.
- Confirmed Figure 3 and Table 7 already appear in Section 5.1 and strengthened
  their bridge text/caption so the AWS rendering is a physical anchor and the
  table is the schedulable abstraction.
- Reordered Section 6.1 to text -> Table 10 -> Figure 4, moved the mechanism
  ablation distribution figure to Appendix C, and added conclusion-oriented
  caption sentences plus table anchor sentences for the main action, observability,
  reference, event-decomposition, ablation, and robustness tables.
- Rebuilt `paper/main.pdf` with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Output: 54
  pages, 2,373,685 bytes, SHA256
  `2491b758a94c7435cbf6d74ff18967ebc1bb77a29e3c16e1033631873f63b179`.
- PDF text verification confirmed the intended order: Section 4 title -> Figure
  1 -> Section 4.1; Section 5.1 contains Figure 3 and Table 7; Section 6.1 text
  precedes Table 10 and Figure 4; mechanism ablation distribution appears in
  Appendix C. LaTeX log scan found no undefined citations/references or rerun
  warnings; remaining warnings are pre-existing table overfull/underfull messages
  and four existing empty-page BibTeX warnings.

## 2026-07-02 Figure 4 Step-Threshold Clarification
- Audited the apparent black zero-valued points in Figure 4 Panel A. The plotted
  black markers were not zero margins; they were the prespecified strict
  step-margin thresholds, computed as
  `max(0.001, 0.002 * replay_static_reference_loss)`.
- Verified the 24-seed threshold range from the aggregate CSVs:
  `0.002398--0.003564`, while true-static step margins range
  `0.028498--0.119145` and true-static macro margins range
  `0.023601--0.119734`. The black markers appeared visually near zero only
  because they were drawn on the same y-axis as much larger margins.
- Patched `paper/figures/gen_fig_scenebal_evidence.py` to replace the per-seed
  black threshold markers with a grey threshold band labelled
  `Step threshold 0.0024-0.0036`.
- Regenerated `paper/figures/figure_regime_balanced_24seed_evidence.pdf` and
  `.png`, and updated the Figure 4 caption in `paper/sections/06_results.tex`
  to state that all step margins exceed the shown threshold range.
- Rebuilt `paper/main.pdf` with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Output: 54
  pages, 2,373,132 bytes, SHA256
  `91beca21d333c0ee1c3a0ee59aa0ad24e095c4f176e29dc86a1c85b56401fe3f`.
  Log scan found no undefined citations/references or rerun warnings; PDF text
  contains the updated Figure 4 threshold label and caption.

## 2026-07-02 Table 10 Fixed-Mask Wording Correction
- Confirmed the user's audit was correct: Table 10 contained two reversed row
  labels, `Fixed mask replay wins, macro score` and `Fixed mask replay wins,
  step margin`, even though the counts and margin convention mean PD-PPO wins
  against the fixed-mask replay reference.
- Verified the source aggregate:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133_134_135_136_137_138_139_140_decision_audit_20260621.json`
  reports `learned_replay_static_gate_count=24`,
  `learned_replay_macro_static_gate_count=24`, and
  `true_static_step_strict_gate_count=24`; margins are defined as reference
  loss minus PD-PPO loss.
- Patched `paper/tables/regime_balanced_24seed_summary.tex` to read
  `PD-PPO step-loss wins vs. fixed mask replay`, `PD-PPO macro-score wins vs.
  fixed mask replay`, and `Prespecified fixed-mask step-margin criterion`.
  The same wording was corrected in the older generated
  `paper/tables/scenebal2_24seed_staticnorm_macro_summary.tex` and in
  `scripts/77_v31_write_scenebal_summary_table.py` to prevent regeneration from
  restoring the reversed labels.
- Rebuilt `paper/main.pdf` with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Output: 54
  pages, 2,373,145 bytes, SHA256
  `b93296ba8dc2cffd3a00c870d11b457a186edbe31e965579d9287eaa3a9e7143`.
- PDF text verification found Table 10 rows as
  `PD-PPO step-loss wins vs. fixed mask replay`, `PD-PPO macro-score wins vs.
  fixed mask replay`, and `Prespecified fixed-mask step-margin criterion`; the
  reversed `Fixed mask replay wins` wording no longer appears in the active
  paper PDF. LaTeX log scan found no undefined citations/references or rerun
  warnings.

## 2026-07-02 Deep-Research Report Route Correction
- Read `docs/07-01-01deep-research-report.md` and the active ESWA planning
  files after the user rejected the benchmark/protocol-only packaging route.
- Appended a route-correction section to
  `docs/07-01-01deep-research-report.md`. The added section states that the
  active paper should be framed as a prediction-driven constrained RL scheduling
  framework, not primarily as a benchmark paper and not as a novel PPO algorithm.
- Recorded the same decision in `findings.md`: the benchmark remains the
  evaluation environment, while the method contribution is the combination of
  forecast-loss reward, executable candidate masks, online feasibility masking,
  fixed forecast evaluator separation, and replay/behaviour diagnostics.
- Added the framework-oriented supplementary experiment priorities:
  forecast-greedy / one-step lookahead baseline, contextual-bandit baseline,
  PPO reward ablation against AoI and uncertainty/covariance rewards,
  lightweight forecaster-sensitivity check, event-proxy quality audit, and
  target-wise raw-metric audit. Masked DQN is useful but lower priority than
  these checks; SAC is low priority for the current small feasible-mask action
  geometry.
- No manuscript source, figure asset, table asset, or PDF was modified in this
  pass, following the user's instruction that the paper should not be edited
  yet.

## 2026-07-02 Framework Supplement GOAL Start
- Created an explicit Codex GOAL: supplement the PD-PPO framework evidence,
  monitor remote experiments to completion, and aggregate the results.
- Extended `task_plan.md` with a framework-first supplementary experiment
  phase. This is separate from submission packaging and is now live work.
- Added `scripts/81_v31_framework_baseline_supplements.py` as a replay-based
  supplement runner. It reuses completed final-benchmark run directories,
  frozen forecast evaluators, candidate masks, final-test starts, and existing
  static-candidate tables.
- Implemented two first-stage baselines:
  1. `context_alert_bandit_t0p5`, a station-context proxy bandit that maps
     particle/flux/thermal alert columns to validation-selected subtype masks;
  2. `forecast_greedy_one_step`, a privileged myopic diagnostic that saves and
     restores environment state while testing each feasible mask by one-step
     final forecast loss.
- Local smoke verification only:
  - `context_alert_bandit_t0p5` on seed `117` completed on the same 512 x 8
    final-test protocol. It was slightly worse than PD-PPO:
    `oracle_loss_mean=1.241395` vs PD-PPO `1.240080`; static-normalised macro
    margin vs PD-PPO `+0.000861`. Positive margin means PD-PPO is better.
  - `forecast_greedy_one_step` ran on a short `32`-step-per-start smoke test to
    verify environment state restoration. That short diagnostic is not
    comparable to the full result and is not a claim.
- Next action: sync the new script to `remote-gpu`, run the 24-seed
  context-bandit supplement first, then decide whether full forecast-greedy is
  computationally justified or should remain a bounded diagnostic.

## 2026-07-02 Framework Supplement Experiments Completed
- Completed the active GOAL's supplementary experiment package on `remote-gpu`
  and synced the finished artifacts back to the local repository.
- Completed 24-seed `context_alert_bandit_t0p5` supplement:
  `reports/aggregate/framework_baseline_supplements_context_20260702/`.
  The context-alert bandit is stronger than PD-PPO on the current 24-seed
  diagnostic: PD-PPO wins 7/24 by step loss and 6/24 by static-normalised macro
  score; mean step margin vs PD-PPO is `-0.008331` and mean macro margin is
  `-0.004533`.
- Completed 24-seed `forecast_greedy_one_step` supplement:
  `reports/aggregate/framework_baseline_supplements_forecast_greedy24_20260702/`.
  PD-PPO beats this privileged one-step final-future-loss greedy diagnostic in
  24/24 seeds, with mean step margin `0.264152` and mean static-normalised macro
  margin `0.155232`.
- Completed two-seed AoI reward-proxy pilot:
  `reports/aggregate/scenebal2_reward_aoi_117_118_macro_20260702aoi/`.
  It passes 2/2 step, macro, replay, and behaviour gates, but is worse than the
  original forecast-reward mainline on seeds 117--118 by mean step-loss delta
  `0.008698` and mean macro-loss delta `0.006855`.
- Completed two-seed coverage reward-proxy pilot:
  `reports/aggregate/scenebal2_confirm_conf05_117_118_macro_20260702coverage/`.
  It passes 2/2 step, macro, replay, and behaviour gates. Against the original
  forecast-reward mainline on seeds 117--118, its mean step-loss delta is
  `-0.002139` and mean macro-loss delta is `-0.011234`, so it is not a negative
  reward ablation.
- Generated the durable combined summary:
  `reports/aggregate/framework_supplement_summary_20260702.md`,
  `reports/aggregate/framework_supplement_summary_20260702.csv`, and
  `reports/aggregate/framework_supplement_summary_20260702.json`.
- Interpretation recorded in the summary: the experiments support that PD-PPO is
  not just a one-step myopic greedy controller, but the context-alert bandit
  challenges any claim that RL currently dominates all context-aware rule
  policies. Reward specificity is mixed: forecast reward beats AoI in the
  two-seed check, while coverage reward remains competitive.

## 2026-07-02 Supplement-Handling Report Review
- Read `docs/07-02-01.md`. The report's main diagnosis is reasonable: the new
  supplement does not invalidate the ESWA paper, but it requires a cleaner claim
  hierarchy that promotes the 24-seed one-step forecast-greedy win, treats the
  context-alert bandit as a serious challenge, and downgrades the two-seed
  reward ablations to exploratory evidence.
- The report should not be applied mechanically. Whether the context-alert
  bandit belongs as a main baseline or a diagnostic/privileged challenger depends
  on its exact information access and tuning path. That status must be audited
  before rewriting the main Results section.
- The report's strongest actionable point is a consolidated comparison table or
  paired-margin figure separating fixed/static references, simple dynamic rules,
  one-step forecast-greedy, context-alert bandit, and event-label replay.

## 2026-07-03 CA-PD-PPO Implementation and Dev Runner
- Continued the active GOAL to improve PD-PPO against
  `context_alert_bandit_t0p5` without turning the primary method into a
  bandit-dependent patchwork.
- Implemented feature-parity state support in `src/v2/env.py`: optional online
  alert scores, threshold flags, event-type argmax one-hot over calm/particle/
  flux/thermal, max alert confidence, time since alert onset, rolling alert
  trend, previous specialist one-hot, and remaining minimum-on time. The
  simulator event flag can now be removed from the policy state.
- Implemented CA-PD-PPO support in `src/v2/custom_ppo.py`: a simple
  context-feature encoder fused with the main observation encoder before the
  masked actor head. The forecast-loss reward and feasibility-masked action
  distribution are unchanged.
- Extended `scripts/25_v2_train_custom_ppo.py`,
  `scripts/58_v31_split_protocol_run.py`, and
  `scripts/64_v31_eval_saved_run_operational_baselines.py` so alert/context
  features and event-flag removal are saved and restored through metadata.
- Added skip controls to
  `scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh` so the new
  dev run can skip historical subtype-router/replay/behavior add-ons while
  leaving default historical behavior unchanged.
- Added `scripts/run_v31_feature_parity_ca_pdppo_dev_20260703.sh`. It runs
  three primary variants over fresh dev seeds 201--224:
  `original_clean`, `feature_parity`, and `ca_pdppo`. All disable final-state
  simulator event labels and subtype-router execution; only the latter two add
  online alert-derived features, and only `ca_pdppo` uses the context encoder.
- Added `scripts/82_v31_collect_contextaware_pdppo_dev.py` to aggregate
  step/macro margins against context-alert bandit, forecast-greedy, static, and
  original dynamic baseline families. Margins are `baseline - PD-PPO`, so
  positive means the PD-PPO variant is better.
- Local validation:
  - `python -m py_compile` passed for the modified Python modules and the new
    collector.
  - `bash -n` passed for the modified runner and the new dev runner.
  - Collector smoke on existing 24-seed context-bandit output reproduced the
    known result: 7/24 step wins and 6/24 macro wins for the old PD-PPO against
    `context_alert_bandit_t0p5`, confirming the margin direction and input
    parsing.
- Next action: sync the modified files to `remote-gpu`, run remote preflight,
  and launch `ca_pdppo_dev_20260703` in tmux for continuous monitoring.

## 2026-07-03 Remote CA-PD-PPO Dev Launch
- Synced the current GOAL files to `remote-gpu` with corrected directory
  placement after an initial rsync path mistake that briefly copied files into
  the remote framework root. The stray root-level copies were removed, and the
  corrected `rsync -R` deployment placed files under `src/v2/`, `scripts/`, and
  `tests/v2/`.
- Remote preflight passed in the `darts` environment:
  `python -m py_compile` for modified modules/scripts and `bash -n` for the
  modified/new shell runners.
- GPU check showed GPUs 0--4 idle and GPU 5 already using about 17GB. The dev
  tmux run was therefore launched with `GPU_IDS="0 1 2 3 4"` to avoid GPU 5.
- Launched remote tmux session `ca_pdppo_dev_20260703` from
  `~/_code/microclimate_demo/rl_sensor_scheduling_framework`:
  `DATE_TAG=20260703capdppo bash scripts/run_v31_feature_parity_ca_pdppo_dev_20260703.sh`.
- Main remote log:
  `logs/ca_pdppo_dev_20260703capdppo/main.log`.
- First monitoring pass confirmed the run is healthy. Five original-clean
  workers entered `25_v2_train_custom_ppo.py`; logs show PPO updates progressing
  and key clean-comparison flags active:
  `--no-subtype-router`, `--no-event-aware-critic`,
  `--no-event-gated-actor`, and `--no-include-event-flag-in-state`.
- The first five seeds had created `v2_tcn_oracle.pt`,
  `reward_staticnorm_candidates.csv`, and live PPO training histories by
  2026-07-03 00:54 CST. No seed-level `v2_custom_ppo_metrics.csv` was complete
  yet at that check.
- By 2026-07-03 01:09 CST, original-clean had completed 4/24 seed metrics
  (`seed202`--`seed205`) and the corresponding workers had advanced to their
  next assigned seeds. `seed201` was still training and was the slowest current
  worker, but it continued to emit PPO updates, so no intervention was made.
- By 2026-07-03 01:56 CST, original-clean had completed 13/24 metrics:
  seeds 201--205, 207--210, and 212--215. Seed 206 was in progress; seeds
  217--219 had started; seeds 211, 216, and 220--224 had not yet reached their
  worker turns. Feature-parity and CA-PD-PPO variants had not started yet.
- By 2026-07-03 02:15 CST, original-clean had completed 17/24 metrics:
  seeds 201--205, 207--210, 212--215, and 217--220. Seed 206 remained in
  progress; seeds 211, 216, and 221--224 were not yet complete. Because GPU4
  had become idle and worker0 would otherwise serialize 211/216/221, a
  supplemental tmux session `ca_pdppo_orig_catchup_20260703` was launched to
  run original-clean seeds 216 and 221 with the same method/settings. This is a
  scheduling acceleration only; the main runner will skip those seeds later if
  their `custom_ppo.pt` exists.
- By 2026-07-03 03:20 CST, original-clean had completed 23/24 metrics:
  seeds 201--210 and 212--224. The only missing seed was 211. Remote worker0
  had reached the end of the 200k PPO update loop for seed211 and was expected
  to enter evaluation/metric writing next. Feature-parity and CA-PD-PPO had not
  started yet.
- By 2026-07-03 03:24 CST, original-clean completed all 24/24 metrics and the
  main runner entered `81_v31_framework_baseline_supplements.py` for the
  original-clean comparison against `context_bandit` and `forecast_greedy`.
  The baseline output root is
  `reports/aggregate/contextaware_pdppo_original_clean_dev_20260703capdppo/`.
- By 2026-07-03 03:29 CST, the original-clean baseline replay was healthy and
  had completed seeds 201--206, with seed207 started. The aggregate
  `framework_baseline_seed_metrics.csv` is expected only after the full replay
  finishes, so the zero row-count at this point is not a failure.
- By 2026-07-03 03:40 CST, original-clean baseline replay had completed
  seeds 201--216 and started seed217. No feature-parity or CA-PD-PPO seed had
  started yet.
- By 2026-07-03 03:52 CST, original-clean baseline replay completed with 48
  policy rows plus header. Results: original-clean PD-PPO still beat
  `forecast_greedy_one_step` in 24/24 seeds with mean step margin `0.240783`
  and mean macro margin `0.159410`; it still lost to
  `context_alert_bandit_t0p5` on average, with 6/24 step wins, 6/24 macro wins,
  mean step margin `-0.026022`, and mean macro margin `-0.007329`.
  Feature-parity training then started on five GPUs with alert-context features
  enabled, simulator event flag disabled, and the CA context encoder disabled.
- By 2026-07-03 04:23 CST, feature-parity had completed the first five metrics
  (seeds 201--205) and all five workers had advanced into the next assigned
  seeds. CA-PD-PPO had not started yet.
- By 2026-07-03 04:53 CST, feature-parity had completed 11/24 metrics
  (seeds 201--211). No feature-parity baseline replay or CA-PD-PPO training had
  started yet.
- By 2026-07-03 05:24 CST, feature-parity had completed 19/24 metrics. Worker1
  failed on seed217 during oracle initialization with a CUDA OOM, leaving seeds
  217 and 222 unrun. The other active feature-parity workers continued on seeds
  221, 223, and 224. A catch-up tmux session
  `ca_pdppo_feature_catchup_20260703` was launched on idle GPU4 to run only
  feature-parity seeds 217 and 222 with the same method flags and output naming.
- By 2026-07-03 05:35 CST, the main feature-parity tmux session had exited
  after the worker1 failure, so seeds 223 and 224 also remained missing. The
  first catch-up session was healthy on seed217. A second catch-up tmux session
  `ca_pdppo_feature_catchup2_20260703` was launched on idle GPU0 to run only
  feature-parity seeds 223 and 224. GPU1 still showed very high memory use after
  the OOM, so it should be avoided for subsequent CA-PD-PPO launches unless it
  clearly recovers.
- By 2026-07-03 06:01 CST, feature-parity had 22/24 metrics. Seeds 217 and
  223 were recovered by the catch-up sessions; seeds 222 and 224 were actively
  running. The main dev runner had exited, so feature-parity baseline replay and
  CA-PD-PPO will need to be launched manually after all feature-parity seed
  metrics are present.
- By 2026-07-03 06:22 CST, feature-parity reached 24/24 metrics after the two
  catch-up sessions completed. A manual feature-parity baseline replay session
  `ca_pdppo_feature_baseline_20260703` was launched to run the same
  `context_bandit` and `forecast_greedy` comparisons that the original runner
  would have executed.
- By 2026-07-03 06:52 CST, feature-parity baseline replay completed. It still
  lost to `context_alert_bandit_t0p5` on average: 8/24 step wins, 8/24 macro
  wins, mean step margin `-0.014382`, and mean macro margin `-0.007684`. It
  still beat `forecast_greedy_one_step` strongly: 24/24 step wins, 23/24 macro
  wins, mean step margin `0.258144`, and mean macro margin `0.164435`.
- By 2026-07-03 06:53 CST, CA-PD-PPO training was launched manually as five
  independent tmux workers over seeds 201--224, using GPUs 0--4. The CA variant
  keeps the same forecast-loss reward and feasibility-masked PPO action
  distribution, enables alert-context features, disables the simulator event
  flag in the policy state, and enables the context encoder.
- By 2026-07-03 07:02 CST, all five CA-PD-PPO workers were healthy, each running
  PPO updates for the first assigned seed. No CUDA OOM was observed and GPU
  memory use was low across GPUs 0--4.
- By 2026-07-03 07:32 CST, CA-PD-PPO completed the first five metrics
  (seeds 201--205). The next five seeds 206--210 were active and all five tmux
  workers remained healthy.
- By 2026-07-03 08:02 CST, CA-PD-PPO completed 10/24 metrics
  (seeds 201--210). Seeds 211--215 were active and all five worker sessions
  remained present.
- By 2026-07-03 08:33 CST, CA-PD-PPO completed 20/24 metrics
  (seeds 201--220). Seeds 221 and 223 were visible as active training
  processes; worker logs had advanced to seeds 222 and 224 as well, but those
  were not yet visible as `25_v2_train_custom_ppo.py` processes at the check.
- By 2026-07-03 08:56 CST, CA-PD-PPO reached 24/24 metrics and all CA training
  worker sessions exited cleanly. A manual CA baseline replay session
  `ca_pdppo_ca_baseline_20260703` was launched for the same `context_bandit`
  and `forecast_greedy` comparisons.
- By 2026-07-03 09:27 CST, CA baseline replay completed with 48 policy rows
  plus header. Result: CA-PD-PPO had positive mean margins against
  `context_alert_bandit_t0p5` but did not meet the win-count gate:
  12/24 step wins, mean step margin `0.004125`; 13/24 macro wins, mean macro
  margin `0.008257`. Against `forecast_greedy_one_step`, CA-PD-PPO won 24/24
  seeds with mean macro margin `0.176646`.
- The collector `scripts/82_v31_collect_contextaware_pdppo_dev.py` was updated
  to distinguish this middle case as
  `competitive_positive_mean_final_gate_not_passed` instead of incorrectly
  falling through to `context_alert_bandit_remains_stronger`.
- Final aggregate artifacts were generated and synced locally:
  `reports/aggregate/contextaware_pdppo_dev_20260703capdppo/contextaware_pdppo_dev_summary.md`,
  `.csv`, `.json`, and `contextaware_pdppo_dev_seed_metrics.csv`. The three
  per-variant baseline directories were also synced under
  `reports/aggregate/contextaware_pdppo_{original_clean,feature_parity,ca_pdppo}_dev_20260703capdppo/`.
- Decision: do not continue to fresh final 24-seed evaluation from this run,
  because the user-defined gate requires at least 15/24 macro wins. Keep
  CA-PD-PPO as the clean primary improvement over original/feature-parity, and
  report it as competitive with a strong context-aware hand-coded baseline.

## 2026-07-03 CA-PD-PPO Report Deliverable
- Wrote a structured Chinese report for the completed CA-PD-PPO vs
  `context_alert_bandit_t0p5` development experiment:
  `reports/aggregate/contextaware_pdppo_dev_20260703capdppo/contextaware_pdppo_vs_context_bandit_report_20260703.md`.
- The report records the method boundary, feature-parity audit, CA-PD-PPO
  architecture, 24-seed development comparison, decision-rule outcome, artifact
  paths, and next diagnostic recommendations.
- The report preserves the main conclusion: CA-PD-PPO is a clean improvement
  and reaches positive mean macro margin against the context-alert bandit, but
  it does not pass the final-evaluation gate because it has 13/24 macro wins
  rather than the required 15/24.

## 2026-07-03 CA-PD-PPO Failure Analysis and Dev2 Launch
- Ran a diagnostic-only failure analysis for CA-PD-PPO versus
  `context_alert_bandit_t0p5` and synced the artifacts under
  `reports/analysis/ca_pdppo_failure_20260703/`.
- Losing macro seeds are concentrated in flux windows: 11/24 seeds lose macro
  score, with dominant losing event type `flux` for 9 seeds and `thermal` for
  2 seeds. The worst seed is 214 with macro margin `-0.009557`.
- Context-confidence bins show CA-PD-PPO is already positive in high-confidence
  alert regions (`[0.70,0.85)` and `[0.85,1.00]`) and loses mainly in lower
  confidence/no-alert regions. Alert-lag bins are positive in early/mid event
  periods and negative around late/post-offset/outside-alert periods.
- The diagnostic does not support bandit imitation as the fix: seed-level mask
  agreement with the bandit is negatively correlated with macro margin. The next
  improvement wave should therefore target clean context calibration/fusion and
  PPO stability rather than bandit-dependent labels, priors, residual actions,
  or margin rewards.
- Implemented clean bounded-dev2 hooks:
  `context_fusion_mode={concat,gated_add}`, optional context LayerNorm,
  learning-rate passthrough, and a collector
  `scripts/84_v31_collect_ca_pdppo_bounded_dev2.py`.
- Added and launched
  `scripts/run_v31_ca_pdppo_bounded_dev2_20260703.sh` on `remote-gpu` in tmux
  session `ca_pdppo_dev2_20260703`, using idle GPUs 2/3/4 and dev seeds
  201--224. It runs four method-consistent variants only: `ctx128`, `gated`,
  `gated_ctx128`, and `nsteps2048`.
- Local validation passed: Python `py_compile`, `bash -n`, and
  `conda run -n darts pytest -q rl_sensor_scheduling_framework/tests/v2/test_custom_ppo.py`.
  Remote preflight also passed the same Python/shell checks and the test file.
  Fresh final seeds are still locked out until a dev2 variant passes the
  predeclared gate.
- `ctx128` completed 24/24 training and replay. It did not pass the dev2 gate:
  against `context_alert_bandit_t0p5`, macro wins were 14/24 and mean macro
  margin was `0.004083`, below both the 15/24 win gate and the `>0.010` mean
  gate. It remained strong against forecast-greedy: 23/24 macro wins, mean
  macro margin `0.165670`.
- Because `ctx128` failed the final-launch gate, the main tmux proceeded to the
  next clean variant `gated`. A GPU1 catch-up session
  `ca_pdppo_dev2_gated_catchup_20260703` was launched only for the late gated
  seeds 222--224; this mirrors the safe ctx128 catch-up pattern and does not
  alter method parameters.
- `gated` completed 24/24 training and replay. It also did not pass the dev2
  gate: against `context_alert_bandit_t0p5`, macro wins were 13/24 and mean
  macro margin was `0.002763`, below both the 15/24 win gate and the `>0.010`
  mean gate. It remained strong against forecast-greedy: 24/24 macro wins,
  mean macro margin `0.171250`.
- Because `gated` failed the final-launch gate, the main tmux proceeded to
  `gated_ctx128` with `context_hidden_dim=128`, `gated_add` fusion, and context
  LayerNorm. No fresh final evaluation has been launched.
- `gated_ctx128` completed 24/24 training and replay. It did not pass the dev2
  gate: against `context_alert_bandit_t0p5`, macro wins were 13/24 and mean
  macro margin was `0.006706`; step wins were 11/24 with mean step margin
  `-0.000270`. It remained strong against forecast-greedy: 24/24 macro wins,
  mean macro margin `0.169535`.
- Because `gated_ctx128` failed the final-launch gate, the main tmux proceeded
  to the last bounded clean variant `nsteps2048`, which keeps the CA-PD-PPO
  architecture but uses longer PPO rollout steps. No fresh final evaluation has
  been launched.
- `nsteps2048` completed 24/24 training and replay, and the main tmux session
  `ca_pdppo_dev2_20260703` exited cleanly. It did not pass the dev2 gate:
  against `context_alert_bandit_t0p5`, macro wins were 10/24 and mean macro
  margin was `0.002962`; step wins were 10/24 with mean step margin
  `-0.000719`. It remained strong against forecast-greedy: 24/24 macro wins,
  mean macro margin `0.168785`.
- The bounded dev2 collector completed under
  `reports/aggregate/ca_pdppo_bounded_dev2_20260703capdppodev2/`. No dev2
  variant passed the predeclared fresh-final gate. The best candidate remains
  the original `ca_current` development result for context-bandit mean macro
  margin (`0.008257`, 13/24 macro wins), but it also fails the `>0.010` mean
  and 15/24 win gates. Fresh final seeds 301--324 remain locked and were not
  launched.
## 2026-07-10 Post-Hermes Experiment-Evidence Audit: Started
- Scope is a read-only audit of the active ESWA manuscript and the final
  SCENEBAL-2 24-seed evidence, requested after several Hermes manuscript edits.
- The canonical manuscript remains `paper/main.tex` with active
  `paper/sections/*.tex`; `raw.tex` and historical rewrite sources are out of
  scope.
- Remote check through `ssh remote-gpu` found no active tmux experiment session
  or scheduler/collector process. The audit therefore examines completed,
  durable artifacts rather than partial runs.
- Initial boundary to verify: the paper's 24/24 claim is tied to ordinary step
  loss and a static-normalised event-regime macro score. Historical raw
  unnormalised subtype-macro diagnostics are not equivalent and must not be
  silently substituted for the reported primary metric.

## 2026-07-10 Post-Hermes Experiment-Evidence Audit: Midpoint
- Verified local/remote SHA-256 equality for the final aggregate CSV and JSON.
  There is no evidence that Hermes' manuscript edits changed the underlying
  primary aggregate.
- Recomputed ordinary step statistics from the durable seed CSV: the reported
  best-operational margin `0.149379`, all 24 positive paired margins, and the
  reported sign-test direction are consistent with the source values.
- Diagnosed a macro normalizer mismatch rather than a PPO-performance failure.
  `70_v31_split_replay_gate.py` builds its static normalizers from final-test
  replay candidates, while the active paper states that normalizers are fixed
  from validation. A strict validation-frozen recomputation from existing NPZ
  rollouts retains 24/24 positive macro margins; no training rerun is required.
- Two read-only analysis attempts initially failed because the router metrics
  CSV intentionally stores only summary loss values, not subtype columns. The
  audit switched to the saved policy rollout NPZ files plus the truth labels,
  which is the correct source for subtype macro recomputation.
- Confirmed that seeds 117 and 122 were the initial SCENEBAL-2 pivot pilot.
  The 22 subsequent seeds retain the positive result. This calls for a modest
  confirmatory-scope correction, not for discarding the whole 24-seed result.
- Next: recompute the privileged event-label diagnostic under the same frozen
  normalizers on `remote-gpu`, inspect action-trace/baseline artifact completeness
  there, then classify the required manuscript and evidence-package corrections.

## 2026-07-10 Post-Hermes Experiment-Evidence Audit: Complete
- Wrote durable audit report:
  `reports/analysis/pdppo_posthermes_evidence_audit_20260710/audit_report.md`.
- The primary aggregate hashes match local and remote. Step-loss 24/24 and
  behavior evidence are confirmed. The post-pilot 22-seed subset also remains
  all-positive.
- The main correction requirement is deterministic macro reaggregation. Current
  replay collectors derive macro denominators from final-test static candidates,
  despite the active manuscript defining validation-frozen denominators. A
  validation-frozen recalculation preserves 24/24 macro signs, so no PPO retrain
  is required.
- Do not edit the manuscript before new frozen-normalizer assets exist. Future
  paper edits must separate the validation-selected fixed baseline from the
  stronger post-hoc operational/replay diagnostics, and must label the 117/122
  pivot pilot versus the 22 post-pilot scenario seeds.
- Remote has the complete per-seed replay metrics and NPZ artifacts; local does
  not. A versioned evidence sync/archive is now a submission blocker.

## 2026-07-10 Evidence Repair Goal: Started
- User authorized completion of the audit repairs. Scope is deterministic
  aggregation, evidence freezing, and manuscript correction; no PPO retraining
  is authorized or required at this stage.
- Binding principle: direct primary macro claims must use validation-frozen
  static normalizers. Final-test-selected fixed replay and event-label schedules
  may remain only as explicitly privileged diagnostics.

## 2026-07-10 Evidence Repair Goal: Near Completion
- Implemented the validation-frozen collector and regenerated the primary,
  mechanism-ablation, and changed-mixture evidence without retraining PPO.
- Updated the active manuscript tables, figures, Results, Discussion,
  Conclusion, and theory appendix so that direct validation-selected comparisons
  are distinct from fixed-replay and privileged event-label diagnostics.
- Generated and archived the post-pilot 22-seed replication: 22/22 positive
  primary macro margins, mean `0.0779690`, bootstrap interval
  `[0.0669584, 0.0895956]`.
- Added a versioned reproducibility package under
  `reproducibility/pdppo_eswa_evidence_20260710/` and an external immutable
  remote-artifact manifest under `/home/horeb/_Data/pdppo_eswa_evidence_20260710/`.
- This intermediate status is superseded by the completion record below.

## 2026-07-10 Evidence Repair Goal: Complete
- Final checks passed: `py_compile` for the new collectors and figure/table
  generators; `pytest -q tests/v2/test_custom_ppo.py` (10 passed); both
  framework and paper `git diff --check`; and `latexmk` for `paper/main.tex`.
- `paper/main.pdf` compiles to 57 pages with no undefined citation/reference or
  fatal LaTeX error. The remaining messages are existing table underfull boxes,
  a 1.8-pt page-width overfull box, and four bibliography entries lacking page
  fields.
- Final archive snapshot:
  `reproducibility/pdppo_eswa_evidence_20260710/` (11 MiB), with current
  source patches, a paper source tarball, deterministic aggregate inputs, and
  SHA-256 checksums. Its remote manifest covers 530 read-only artifacts
  totaling 1.13 GiB; the local archive deliberately pins rather than duplicates
  that raw source set.

## 2026-07-17 Manuscript Completeness and Structure Audit: Started
- User requested a fresh whole-paper review after later Hermes edits. Scope is
  read-only: canonical `paper/main.tex`, active `paper/sections/*.tex`, active
  tables/figures, and rendered `paper/main.pdf`; `raw.tex` remains out of scope.
- Initial state: the paper is 56 pages and the nested paper worktree contains a
  new untracked `tables/development_contextaware_summary.tex`. The audit will
  verify that development-only CA-PD-PPO evidence has not displaced or blurred
  the validation-frozen 24-seed main claim.

## 2026-07-17 Manuscript Completeness and Structure Audit: Complete
- Audited the complete active source, front matter, eight main sections, nine
  appendices, 38-entry bibliography, compile log, and representative rendered
  pages. No manuscript files were edited.
- Confirmed that the validation-frozen fixed schedule remains the primary
  comparison and that later context-aware experiments are confined to a
  development-only appendix with no confirmatory promotion.
- Identified four revision blockers: missing estimator/partial-observation
  specification, non-executable baseline definitions, overbroad rule-based and
  24-held-out-seed wording, and absence of matched learned-policy/reward
  baselines for an algorithm-framework submission.
- Identified secondary work: fuller simulator specification, a stronger
  Discussion and nearest-work comparison, one budget/capacity or forecaster
  sensitivity, justification of the superiority threshold, appendix pagination
  cleanup, and a concrete submission archive identifier.
- Verified the current 56-page PDF is source-synchronized and has no undefined
  citation/reference or fatal LaTeX error. Appendix H has an isolated heading on
  page 49 because of forced page breaks; sampled main-text pages have no visible
  clipping or overlap.

## 2026-07-18 Method and Experimental Closure Goal: Started
- Created the explicit Codex goal to complete experiments, aggregate evidence,
  revise the canonical English manuscript coherently, and finish with a compiled
  claim-to-evidence audit.
- Read `docs/07-18-01.md`, the active ESWA planning history, the completed 07-05
  narrative plan, framework supplement reports, and the relevant experiment and
  remote-server skills.
- Corrected `.planning/.active_plan` from the completed 07-05 narrative draft to
  the authoritative 06-10 ESWA plan so later planning hooks preserve this work.
- Confirmed that the local worktree is heavily dirty from prior user/Hermes work;
  no existing changes will be reverted. Formal experiments remain remote-only.
- Located the prior reward-control hook in `scripts/25_v2_train_custom_ppo.py`:
  `reward_proxy_mode` currently accepts forecast, AoI, coverage, and instant
  error. The formal uncertainty control and matched learned-policy baseline still
  require implementation/interface audit.
- Connected to `remote-gpu` through the SSH alias. No PD-PPO tmux or scheduler
  process is active; GPUs 0--4 are free, while GPU 5 has an unrelated/unresolved
  allocation and will be excluded from this experiment wave.
- Verified local/remote SHA-256 equality for `25_v2_train_custom_ppo.py`,
  `58_v31_split_protocol_run.py`, `src/v2/env.py`, and `src/v2/custom_ppo.py`.
- The first remote manifest read used `python`, which is unavailable in the
  non-interactive shell. Retried with system `python3`; no environment or file
  modification was needed.
- Compared seed-117 mainline, AoI, and coverage manifests/metadata. Truth and
  protocol controls match, but each old reward variant retrained the TCN
  evaluator and regenerated static normalizers. The formal experiment therefore
  needs a reuse/freeze path before seed expansion.
- Audited `src/v2/dqn.py`, `src/v2/custom_ppo.py`, and `src/v2/env.py`. The DQN
  action mask is conceptually compatible, but both learned trainers rebuild
  environment configs incompletely. This exposed a missing propagation path for
  subtype reward weights/static normalizers in the existing PPO run. The next
  implementation phase is therefore a shared complete-config clone, an explicit
  uncertainty proxy, and a frozen-source-run launcher, followed by a seed-117/118
  remote pilot before any full expansion.
- A local lookup for the raw seed-117 run failed because raw mainline artifacts
  are remote-only. The authoritative directory was then located directly on
  `remote-gpu`; it stores truth, oracle, normalizers, validation candidates,
  checkpoint, metrics, and rollouts at the run root rather than under `raw/`.
- The first local regression-test command used bare `pytest`, which is not on
  this shell's PATH. Tests are rerun through the existing `darts` conda
  environment; no package installation or environment mutation is required.
- Implemented uncertainty-reward state tracking and online alert routing in
  `src/v2/env.py`, complete environment copies in `src/v2/custom_ppo.py` and
  `src/v2/dqn.py`, strict source reuse/provenance in
  `scripts/25_v2_train_custom_ppo.py`, and forwarding support in
  `scripts/58_v31_split_protocol_run.py`.
- Added focused regression tests for uncertainty updates, online-alert priority,
  full PPO config propagation/snapshot restore, and DQN config propagation.
  Local validation passes: 25 tests, Python compilation, shell syntax, and
  `git diff --check`.
- Added `scripts/run_v31_matched_reward_controls_20260718.sh` and the five-GPU
  bounded launcher `scripts/run_v31_matched_reward_pilot_20260718.sh`. Remote
  preflight completed with exit 0 for seeds 117 and 118; both source truth,
  frozen evaluator, six masks, validation starts, and final starts passed the
  strict validator.
- Launched the six-job reward pilot in remote tmux session
  `pdppo_matched_reward_pilot_20260718` on GPUs 0--4. GPU 5 remains excluded.
  The sixth job is queued behind the first worker by the bounded launcher.
- Added `scripts/89_v31_train_matched_dqn.py` for the same-mask Double-DQN
  comparator and `scripts/90_v31_collect_matched_reward_controls.py` for paired
  reward-control aggregation. The DQN launcher is syntax/import validated but
  not yet formally run.
- A remote `conda run` test invocation became detached by the command transport
  after exceeding its wait window; the resulting owned test/dry-run processes
  were explicitly cleaned. Long remote work now runs only in tmux with durable
  logs. No training process was killed or modified.
- Completed the executable method and baseline audit at
  `reports/analysis/pdppo_method_reproducibility_audit_20260718.md`. It records
  the sample-and-hold observation update, the reward-control uncertainty
  recursion, all six candidate actions, hard feasibility/minimum-duration
  semantics, information privileges, and selection/tie-breaking rules for the
  fixed, conventional, context-aware, myopic, event-label, and full-observation
  references.
- The remote reward pilot remains active in tmux
  `pdppo_matched_reward_pilot_20260718`: five 200k-step jobs are running on GPUs
  0--4 and the sixth is queued. The formal matched-DQN queue is waiting in tmux
  `pdppo_matched_dqn_pilot_20260718` and will start automatically after the PPO
  pilot. GPU 5 remains excluded.
- Corrected forecast-reward PPO completed on pilot seeds 117 and 118. Its
  validation-selected-static macro margins are `+0.0313873` and `+0.0554237`;
  its ordinary step-loss margins are `+0.0459912` and `+0.0820836`. Both have
  zero warm-up aborts.
- The corrected pilot action traces pass the existing complexity gate in both
  seeds. Each uses four specialist masks, has mask entropy above 1.85 bits, is
  neither fixed-like nor cycle-like, and leaves only the low-value radiometer at
  zero duty. The mandatory weather backbone is the sole always-active channel.
- Launched the frozen 24-seed corrected forecast expansion in tmux
  `pdppo_corrected_forecast24_20260718` on GPUs 2--4, with seeds 117--140 and no
  further tuning. A second tmux session,
  `pdppo_corrected_controls24_20260718`, is queued to run the matched AoI and
  uncertainty controls on GPUs 1--4 after the forecast expansion.
- Added resumable artifact checks to the reward-control launcher and a bounded
  worker-pool launcher for the eventual 24-seed Double-DQN expansion. The
  formal DQN expansion remains gated by the two-seed pilot.
- Expanded Discussion with result-independent interpretation of strong fixed
  schedules, the supplied-warning context rule, sequential value under minimum
  activation time, deployment-time computation, and extension to several
  specialist slots. `paper/main.tex` recompiles successfully to 66 pages; no
  new fatal, undefined-reference, or clipping issue was introduced.
- Completed the two-seed matched reward-control aggregate. Forecast versus AoI
  is `1/2` wins with mean macro-loss improvement `+0.004340`; forecast versus
  uncertainty is `1/2` wins with mean improvement `+0.000139`. All three reward
  variants beat validation-selected static in `2/2` seeds with zero aborts.
  Result path:
  `reports/aggregate/pdppo_matched_reward_pilot_20260718/`.
- Removed the collector's optional `tabulate` dependency after the first remote
  Markdown write failed; the CSV computations were unaffected. The repaired
  collectors are self-contained and the aggregate now writes successfully.
- Found and fixed residual exact-event use in PPO bootstrap and BC warm-start
  actor/critic gates. Rollout, bootstrap, warm start, and final inference now
  pass the online warning proxy; exact subtype labels are confined to
  training-only guide and auxiliary targets. Focused validation is now 26 tests.
- Stopped the partially trained formal `corrected24`/`corrected24r1` batches to
  avoid mixing information contracts. Restarted the authoritative 24-seed
  forecast sweep as `pdppo_corrected_forecast24r2_20260718`; matched AoI and
  uncertainty controls are queued in
  `pdppo_corrected_controls24r2_20260718`.
- Corrected the manuscript's behavior-cloning description: the formal guide is
  subtype-to-mask supervision on the policy-training partition, while the
  frozen forecaster supplies the PPO reward. `paper/main.tex` recompiles to 66
  pages without fatal or undefined-reference errors; result pages remain
  provisional until the corrected aggregate replaces the historical figures.
- Audited deterministic inference and found that the optional auxiliary
  subtype router bypassed actor logits in the earlier pilot. Stopped all partial
  `corrected24`, `corrected24r1`, and `corrected24r2` expansions before treating
  them as evidence. The earlier two-seed reward-control aggregate is now marked
  exploratory rather than confirmatory.
- Launched two actor-only forecast pilots on `remote-gpu`: plain no-router
  PD-PPO (`20260718cleanpilot`, seeds 117/118 on GPUs 2/3) and no-router
  context-aware PD-PPO (`20260718capilot`, seeds 117/118 sequentially on GPU 4).
  Direct process inspection verifies `--no-subtype-router` for both and the
  intended context-encoder flags. The same-mask Double-DQN pilot remains active
  on GPUs 0/1.
- Added an automatic, lock-safe CA seed-118 launcher in tmux
  `pdppo_ca_norouter_seed118_20260718`. It waits for the plain pilot to release
  GPU 2, then runs the second CA seed concurrently with CA seed 117; the
  existing sequential launcher will wait for or skip the same complete
  artifact set rather than duplicate it.
- Added `forecast_loss_samples` and
  `scripts/93_v31_secondary_forecaster_rescore.py` for the compact forecaster
  sensitivity promised by the 07-18 plan. The tool fits a multi-output ridge
  forecaster only on the original forecaster-fitting partition, selects its
  static reference only on validation windows, and rescores frozen final
  trajectories without policy updates. Focused local validation now passes 28
  tests, Python compilation, and the new global-step/subtype alignment test.
- The clean architecture gate completed and selected plain actor-only PD-PPO.
  Both plain and CA variants passed the two-seed static, behavior, and zero-abort
  criteria. CA improved the macro score in both seeds, but its mean paired gain
  was `+0.002891`, below the frozen `+0.005` materiality threshold; the added
  encoder is therefore not promoted.
- Launched selected-architecture AoI and uncertainty controls in tmux
  `pdppo_clean_reward_controls_pilot_20260718`, a post-DQN aggregation watcher
  in `pdppo_dqn_clean_collect_20260718`, and a one-seed independent ridge
  forecaster smoke run in `pdppo_secondary_forecaster_smoke_20260718`.
- The matched Double-DQN pilot completed and passed protocol/behavior audits.
  Plain PD-PPO wins `2/2` on ordinary and macro loss, with mean DQN-minus-PPO
  margins `+0.083934` and `+0.049926`; both policies pass behavior in `2/2`
  and have zero aborts. Double-DQN beats static in only `1/2` macro comparisons,
  so it remains a valid but weaker learned-policy comparator.
- The independent ridge-forecaster smoke completed on seed 117 after fixing the
  resolver to follow the clean run's checksummed control-source truth rather
  than require a duplicate CSV. Frozen PD-PPO remains ahead of both the original
  validation static and the ridge-validation-selected static on ordinary and
  macro loss. This validates the rescore implementation only; the robustness
  claim remains gated on the full selected-policy seed set.
- The clean same-architecture reward-control pilot completed. Forecast reward
  is `1/2` against AoI (mean improvement `-0.004428`) and `0/2` against the
  diagonal-uncertainty reward (mean `-0.004510`). All variants still beat
  static in `2/2`, have zero aborts, and share the selected plain architecture.
  This is a genuine objective-alignment risk, not an implementation failure;
  the frozen 22-seed expansion will resolve it without reward-specific tuning.
- Launched the frozen post-pilot expansion in tmux
  `pdppo_clean_reward_formal22_20260718`: seeds 119--140 for forecast, AoI, and
  uncertainty reward, reusing the completed 117/118 artifacts under one tag.
  Ten bounded workers use GPUs 0--4; GPU 5 remains excluded. A 24-seed matched
  collector is queued in `pdppo_clean_reward_formal_collect_20260718`.
- Started Double-DQN seeds 119--122 and queued seeds 123--140 plus a 24-seed
  collector. The DQN configuration remains byte-for-byte the pilot config; no
  result-dependent hyperparameter correction was made.
- Queued read-only postprocessing after the PPO wave: validation-frozen main
  aggregation and behavior audit, 24-seed ridge-forecaster rescore, and clean
  context-alert/forecast-greedy/event-label references. These jobs consume only
  frozen final trajectories and cannot alter architecture or checkpoints.
- Completed all 24 selected actor-only forecast-reward policies. A repaired
  validation-frozen collector resolves the checksummed control-source truth
  when the clean run does not duplicate `truth_v31_split.csv`; local compile,
  remote seed-117 smoke, and the full aggregate pass.
- The clean main comparison is positive in `24/24` seeds against the
  validation-selected fixed schedule for both ordinary loss and macro score.
  Mean macro margin is `+0.080126` (95% CI
  `[+0.067398,+0.093035]`; minimum `+0.019128`), and mean ordinary-loss margin
  is `+0.157971` (`[+0.116100,+0.205185]`). Macro wins against AoI, round
  robin, random, and the post-hoc best conventional rule are also `24/24`.
- The predeclared post-pilot subset (seeds `119--140`) remains positive in
  `22/22` seeds, with mean macro margin `+0.083774`
  (`[+0.070685,+0.096705]`) and ordinary-loss margin `+0.167133`
  (`[+0.123090,+0.216425]`). This removes dependence of the primary direction
  on the two architecture-selection seeds.
- The complete action-trace audit passes in `24/24` seeds: the mandatory
  weather backbone is the only always-on channel, the radiometer is the only
  always-off channel, `3--4` specialists have intermediate duty, switching is
  `0.002849--0.004721` per step, and aborts are zero. These are clean actor
  traces rather than hard-router traces.
- Synced the main and behavior aggregates locally. Matched reward controls,
  remaining Double-DQN seeds, strong reference replays, and the second-
  forecaster rescore continue under their frozen protocols.
- Added and ran a clean-rollout mechanism collector. Offline subtype grouping
  shows mean specialist duties of `0.9928` laser in particle windows, `0.9783`
  FC4 in flux windows, and `0.9914` surface IR in thermal windows. Validation
  selects a single FC4/static specialist in 13 seeds, surface IR in 9, and
  laser in 2, so the clean learned policy implements the incompatible
  specialist allocation that no one fixed schedule can express.
- The new mechanism aggregate is
  `reports/aggregate/pdppo_clean_mechanism_24seed_20260718/`. It records that
  exact subtype labels are used only offline for grouping; the collector rejects
  hard-router and exact-event-online metadata.
- Completed and synced the clean 24-seed strong-reference replay aggregate.
  PD-PPO beats the privileged one-step forecast-greedy diagnostic in `24/24`
  seeds (mean ordinary/macro margins `+0.269041/+0.178989`). It is effectively
  tied with the context-alert and exact-label references: ordinary win counts
  are `10/24` for both, macro win counts are `11/24` and `12/24`, and mean
  macro margins are only `+0.001222` and `+0.001768`. These are explicit
  context-information boundaries, not primary deployable baselines.
- Added `scripts/95_v31_build_clean_paper_assets.py` as the single frozen-
  evidence source for the replacement result tables and figures. It requires
  all 24 seeds for the primary, mechanism, strong-reference, matched-reward,
  Double-DQN, and independent-forecaster inputs before writing any manuscript
  asset, records input checksums, and fails on incomplete seed sets. Python
  compilation passes; execution remains intentionally gated on the pending
  formal controls.
- Removed an unnecessary serial dependency from the remaining Double-DQN queue.
  GPUs 0--4 had low memory occupancy and ample utilization headroom, so seeds
  123--140 now train concurrently with the reward controls in tmux
  `pdppo_matched_dqn_fast_20260718`. The original frozen configuration, seed
  list, and output paths are unchanged. The existing `expand2` queue now waits
  for both active waves and reruns only missing artifacts before the unchanged
  24-seed collector starts.
- Reprofiled the matched Double-DQN execution path after the CPU-evaluator
  queue advanced too slowly for a bounded closure run. A 10,000-step CUDA
  evaluator timing run, including the full 4,096-step multi-policy evaluation,
  completed successfully in 2 minutes 22 seconds. This changes only the device
  used to compute the already frozen TCN reward during training.
- Split the DQN evaluator device contract: training now uses CUDA, while final
  rollout scoring reloads the same checksummed forecaster on CPU. The latter
  preserves the exact numerical comparison path used by PD-PPO and the fixed
  replay. Local DQN tests, Python compilation, shell syntax checks, and remote
  syntax checks pass.
- Stopped the incomplete CPU-DQN queue without deleting its artifacts and
  launched a uniform 24-seed replacement under tag `20260718cuda` in tmux
  `pdppo_matched_dqn_cuda24_20260718`; its frozen collector is
  `pdppo_matched_dqn_cuda_collect_20260718`. All 24 seeds retain 200,000
  timesteps and the original DQN hyperparameters.
- Strengthened the matched-reward collector with a normalized full-metadata
  fingerprint. A remote three-mode seed-117 audit passed; paths and reward mode
  are the only permitted differences. The running formal collector will use
  this stricter script.
- Quantified CPU/CUDA frozen-evaluator agreement over all 4,096 held-out
  seed-117 forecast windows. Mean absolute reward difference is `0.000165`
  (`0.0236%` of the CPU mean), while all reported metrics remain CPU-scored.
  Recorded the check in
  `reports/analysis/pdppo_dqn_oracle_backend_audit_20260718.md`.
- Completed, synced, and independently rechecked the 24-seed ridge-forecaster
  sensitivity aggregate. PD-PPO records `24/24` ordinary and `23/24` macro
  wins against ridge-validation-selected static schedules; mean macro margin
  is `+0.133435` with CI `[+0.111065,+0.154450]`. Only seed `129` is negative.
  Appended this result to the root `CHANGELOG.md` as a separate completed
  evidence block.
- Completed the controlled eval-only pass for all 24 frozen Double-DQN
  checkpoints after limiting evaluation to six workers and four CPU threads per
  worker. This changed resource scheduling only; every checkpoint retains the
  predeclared 200,000-step training history and all final metrics use the CPU
  forecaster.
- The strict DQN collector passed all source-truth, oracle, candidate-mask,
  observation, partition, final-window, and evaluator-device checks. PD-PPO
  wins `24/24` macro comparisons against Double-DQN (mean DQN-minus-PD-PPO
  margin `+0.069719`, 95% CI `[+0.053916,+0.085406]`) and `23/24` ordinary
  comparisons (mean `+0.140775`, CI `[+0.104129,+0.178748]`). Double-DQN
  beats static in `12/24` macro comparisons and passes behavior in `21/24`.
- Synced the lightweight aggregate to
  `reports/aggregate/pdppo_matched_dqn_clean_24seed_20260718/`, independently
  recomputed directions and counts, and appended the result to the root
  `CHANGELOG.md`.
- Completed and strictly aggregated all 72 matched reward runs. The normalized
  full-metadata fingerprint confirms that, within a seed, forecast, AoI, and
  uncertainty runs differ only in reward mode and output/model paths.
- Forecast reward is comparable to both proxy rewards rather than uniformly
  better. Against AoI it records `10/24` ordinary and `13/24` macro wins, with
  mean AoI-minus-forecast differences `-0.000874` and `+0.001005`; both 95%
  intervals cross zero. Against uncertainty it records `11/24` ordinary and
  `12/24` macro wins, with mean differences `+0.000105` and `+0.000807`; both
  intervals also cross zero.
- All three reward modes beat their own validation-selected fixed schedule in
  `24/24` macro comparisons and pass the basic execution checks in every seed.
  The final paper must retain theoretical objective non-equivalence but cannot
  claim empirical reward-proxy superiority under the shared labelled guide and
  auxiliary protocol.
- Synced and independently checked
  `reports/aggregate/pdppo_clean_matched_reward_24seed_20260718/`; appended the
  frozen result to the root `CHANGELOG.md`.

## 2026-07-18 Method and Experimental Closure Completed

- An independent scope audit established that the primary comparison uses eight
  prespecified 512-epoch subtype-balanced, transport-rich windows rather than
  every epoch in the final partition. The manuscript now states this selection
  rule, its truth-only inputs, and the resulting 4,096 evaluated epochs
  explicitly.
- Added a no-retraining coverage sensitivity in
  `scripts/96_v31_run_full_final_partition_replay.sh` and extended scripts 64,
  86, and 95 to record, validate, aggregate, and render its scope. The same
  frozen checkpoints, validation-selected masks, and validation-fitted
  normalizers were replayed over `[64750,69992)`, the 5,242 epochs with a
  complete eight-step target.
- All 24 formal replay artifacts use CPU evaluation and record one common
  `full_scoreable_final_partition` scope. PD-PPO wins `24/24` ordinary
  comparisons (mean margin `+0.124728`, minimum `+0.009150`) and `24/24` macro
  comparisons (mean `+0.079260`, minimum `+0.013825`). Both manuscript
  bootstrap intervals remain above zero.
- Appended the completed result to the root `CHANGELOG.md` and wrote the
  provenance audit to
  `reports/analysis/pdppo_full_final_partition_audit_20260718.md`.
- Revised the canonical manuscript globally. The estimator and online
  observation chain, executable baselines, simulator parameters, forecaster
  fitting mixture, reward controls, Double-DQN control, independent forecaster,
  event-information references, primary window scope, and continuous replay
  now form one consistent method/evidence hierarchy. No v1, no-warmup, or hard
  subtype-router result is imported.
- Regenerated all frozen-evidence tables and figures with script 95. The new
  coverage table is maintained by the generator rather than edited by hand.
  The abstract is 230 words.
- The final `paper/main.pdf` compiles to 66 pages with no undefined
  references, undefined citations, float-size errors, or content-table
  overflows. Appendix G and its fixed-schedule ledger begin on the same page.
  The only remaining layout message is a pre-existing 1.8-point frontmatter
  overfull box.
- Focused validation passes: 28 pytest tests, Python compilation for all
  closure scripts, shell syntax for all 2026-07-18 launchers, independent
  CSV/manifest/LaTeX numerical checks, and package checksum verification.
- Frozen the complete lightweight evidence package at
  `reproducibility/pdppo_eswa_evidence_20260718/`. It contains 24 per-seed
  continuous-replay metadata/metric pairs, all aggregate families, code and
  paper snapshots, tracked-worktree patches, the compiled PDF, and 107
  SHA-256 entries.
- Confirmed that `remote-gpu` has no remaining PD-PPO process or tmux session
  from this run, and stopped the local long-interval monitor.

## 2026-07-19 Submission consistency and anonymization

- Reconciled the pilot-seed history from frozen evidence rather than inherited
  prose. The active clean actor-only method uses seeds 117/118 for the bounded
  architecture choice and seeds 119--140 for the unchanged expansion. The
  older 117/122 boundary belongs to the superseded July 10 evidence contract
  and remains only in provenance archives.
- Verified the frozen selection artifact
  `reports/aggregate/pdppo_clean_method_gate_20260718/clean_candidate_decision.json`:
  both candidates pass both pilot seeds; `plain_pdppo` is selected because the
  context encoder's +0.002891 mean macro improvement is below the predeclared
  +0.005 materiality threshold.
- Added the concise estimator update order, exact conventional comparator names,
  the shared-guide reward-control boundary, and the aggregation-specific claim
  boundary to the canonical English manuscript.
- Added a one-source ESWA double-anonymized build and a separate title page. The
  anonymous PDF contains no author identity, affiliation, email, CRediT text,
  acknowledgements, or author metadata. Its `.fls` dependency record also
  confirms that no author-identity source file is loaded by the anonymous build.
- Compiled `main.pdf` (67 pages), `anonymous_manuscript.pdf` (66 pages), and
  `title_page.pdf` (2 pages). The abstract is 228 words; all builds have resolved
  references and citations. The only retained warnings are the known empty-page
  BibTeX fields and minor box warnings.
- Rewrote the five submission highlights so that each is result/method focused
  and no line exceeds ESWA's 85-character limit.
- Recorded the verification and bounded supplementary-material split plan in
  `reports/analysis/pdppo_eswa_submission_pass_20260719.md`. No invalid budget
  sweep, baseline expansion, or mechanical Discussion cut was introduced.

## 2026-07-19 Submission compression and package audit completed

- Created `paper/supplementary_material.tex` and moved implementation/audit
  material there while preserving the complete problem, executable policy,
  reward, PPO objective, training-information boundary, main evidence, and
  mechanism discussion in the manuscript.
- The named manuscript is now 50 pages, the anonymous manuscript 49 pages, and
  the main text reaches Conclusion on page 41. The independent supplement is
  9 pages and the title page is 2 pages.
- Recompiled all four LaTeX targets with `latexmk -halt-on-error`. There are no
  undefined references, undefined citations, LaTeX errors, or fatal errors.
  The abstract is 230 words and both maintained five-line highlight files stay
  within 85 characters per line.
- Visually audited the compressed method/results/appendix transition and all
  nine supplement pages. No clipping, overlap, misleading blank page, or float
  separation remains.
- Verified Figure 1 as a tracked Matplotlib-generated vector PDF with no
  embedded raster image object; no generative-image asset is used by the active
  manuscript.
- Built `submission/eswa_pdppo_20260719/` with anonymous and named PDFs,
  editable sources, vector figures, declarations, cover letter, three upload
  archives, and SHA-256 checksums. Anonymous text, metadata, source dependency,
  path, and archive scans found no author leakage.
- The anonymous evidence archive passes its aggregate verifier and the complete
  focused `tests/v2` suite in the `darts` environment, with one existing skip.
  Its README now states accurately that full truth reconstruction additionally
  requires the external meteorological anchor data.
- The only remaining external submission action is uploading
  `anonymous_repository_upload.zip` to an anonymous accessible host and then
  replacing the provisional Data Availability wording with the real URL.

## 2026-07-19 Journal-language consolidation completed

- Read `docs/07-19-01.md` against the active LaTeX source rather than applying
  its suggestions mechanically. The supervisor-approved title was retained;
  no evidence, metric, comparator, or claim boundary was changed.
- Manually standardized the abstract, Introduction, Methods, Results,
  Discussion, Conclusion, active tables, supplementary text, Highlights, and
  framework figure around one vocabulary: evaluation seeds, macro score,
  evaluable epochs, warning-score rule, exact-label reference, training-only
  guide signals, action-trace diagnostics, and secondary/ridge forecaster.
- Replaced audit-style transitions and internal experiment shorthand with
  direct journal prose. Clarified that the action surface contains five
  specialist choices, while four receive nonzero final-evaluation duty and the
  low-value basic radiometer receives none.
- Rebuilt `paper/main.pdf` (51 pages), `paper/anonymous_manuscript.pdf` (50
  pages), and `paper/supplementary_material.pdf` (9 pages). The abstract is 233
  words; all five Highlights remain below 85 characters. Final logs contain no
  undefined citations, undefined references, LaTeX errors, or fatal errors.
- Synchronized and independently rebuilt
  `submission/eswa_pdppo_20260719/anonymous_source`, regenerated Figure 1 and
  `anonymous_source_bundle.zip`, and refreshed the 12-entry SHA-256 manifest.
  All checksums pass; canonical and packaged anonymous PDFs have identical
  extracted-text hashes, empty author metadata, and no author-identity strings.
- Data Availability remains the sole external gate. The provisional statement
  was retained because no accessible anonymous repository URL has been
  provided.

## 2026-07-19 Figure-system and final editorial refinement completed

- Applied `docs/07-19-02.md` as an editorial and figure-system plan without
  changing the supervisor-approved title, experiment set, numerical evidence,
  or claim boundary.
- Rebuilt Figure 1 as a source-controlled protocol diagram separating the four
  chronological partitions, online feasibility-masked execution, frozen
  forecaster reward, validation-only choices, comparators, and training-only
  auxiliary supervision.
- Rebuilt Figure 2 with independent seed markers and three explicit evidence
  groups. Matched Double-DQN now appears with the strong comparators; ridge
  sensitivity remains in Figure 4, eliminating the previous duplication.
- Harmonized Figures 1--4 with embedded DejaVu Sans fonts, one comparator-minus-
  PD-PPO sign convention, consistent panel labels, line weights, and palette.
  Rendered manuscript pages 3, 30, 32, and 36 were inspected directly.
- Added an unnumbered terminology guide to the Supplement, fixed its initial
  width overflow, and cited it from the method section without shifting the
  existing supplementary table numbering.
- The final named, anonymous, and supplementary builds contain 51, 50, and 10
  pages. The abstract is 235 words; all references and citations resolve; all
  four main-figure fonts are embedded. Anonymous metadata and text contain no
  author identity.
- Independently rebuilt `submission/eswa_pdppo_20260719/anonymous_source`,
  refreshed the four separate figures, source bundle, PDFs, and 12-entry SHA-256
  manifest. Every checksum passes, and canonical and packaged anonymous PDFs
  have identical extracted text.
- Data Availability remains the only external submission gate because an
  accessible anonymous repository URL does not yet exist.

## 2026-07-19 Proposition matrix and Figure 1 redesign completed

- Restyled the Proposition 1 specialist--regime matrix in TikZ using the same
  sans-serif language and colorblind-safe teal, amber, and slate palette as the
  main figures. Explicit cell text and a compact legend preserve the semantic
  distinction in grayscale.
- Rebuilt Figure 1 from its tracked Matplotlib source as three coordinated
  layers: chronological development/evaluation, online feasibility-masked
  scheduling, and training-only policy updates. The frozen forecaster is now
  visibly outside the runtime path, and hard feasibility masking precedes
  action selection.
- Rejected four intermediate versions after actual-size review for crossed
  arrows, text overflow, redundant annotation, or insufficient manuscript-size
  typography. The fifth revision passes standalone, full-page, and grayscale
  inspection without overlaps or ambiguous routing.
- Rebuilt the named and anonymous manuscripts successfully; the anonymous
  manuscript remains 50 pages and its final log has no undefined citations or
  references. Figure 1 is pure vector artwork with all fonts embedded.
- Synchronized the independently compilable anonymous source, separate Figure
  1, and top-level anonymous PDF. Rebuilt the source archive and 12-entry
  checksum manifest; every checksum passes, and canonical/package anonymous
  PDFs have identical extracted-text hashes.

## 2026-07-19 Figure typography and objective-prose pass completed

- Audited the embedded fonts of all four active figures and found that the
  Matplotlib assets used DejaVu Sans while the TikZ Proposition 1 matrix used
  TeX Gyre Heros. The plotting base sizes also differed across generators.
- Standardized all active figure text on DejaVu Sans with an 8.5-point base
  hierarchy. The Proposition matrix now uses the same family at a fixed size
  and no longer relies on whole-object scaling. Two geometry revisions removed
  clipped headings and overlapping legend blocks.
- Reworked the abstract, Introduction, Problem Formulation, Methods, Setup,
  Results, Discussion, Conclusion, Supplement, and maintained table notes by
  hand. Active prose now contains no first-person authorial subject, no
  `rather than` or `instead of` contrast, and no colon-led prose list.
- Replaced compact colon-based table headers with complete two-line labels.
  Visual review led to a one-step size reduction for the five-column learner
  control table; all numeric columns are now separated and legible.
- Updated `scripts/95_v31_build_clean_paper_assets.py` so regenerated evidence
  tables preserve the revised wording and typography. A clean regeneration
  reproduces the maintained table files.
- Rebuilt the named manuscript (50 pages), anonymous manuscript (50 pages),
  and Supplement (10 pages). Final logs contain no undefined references,
  undefined citations, or material overfull boxes.
- Rebuilt the anonymous source archive and submission package. Canonical and
  packaged main/Supplement PDFs have identical extracted-text hashes, all four
  separate figures embed only DejaVu Sans variants, and all 12 checksums pass.

## 2026-07-19 Draw.io MCP installation and Figure 1 replacement completed

- Verified Codex CLI 0.144.1, draw.io Desktop 30.0.4, and the official
  `@drawio/mcp` package 1.4.1. Installed the MCP globally for the current user
  with `codex mcp add drawio -- npx -y @drawio/mcp`; `codex mcp list` reports
  the server enabled alongside the unchanged GitHub MCP.
- Exercised the MCP through a direct protocol client. `list_pages` returned the
  `PD-PPO framework` page, `get_page` returned its native `mxGraphModel`, and
  `open_drawio_xml` generated a valid diagrams.net editor URL with obstacle-
  avoiding routing enabled.
- Added `paper/figures/figure_pdppo_framework.drawio` and exported
  `figure_pdppo_framework_drawio.pdf` plus a PNG review copy. Four visual
  revisions corrected clipping, encoder layout, connector routing, text
  hierarchy, and grayscale separation.
- Updated `paper/sections/01_introduction.tex` to use the new vector asset. The
  existing caption remains accurate because the three-panel semantics are
  unchanged; no experiment, number, or claim was edited.
- Rebuilt `paper/main.pdf` and `paper/anonymous_manuscript.pdf` successfully;
  both remain 50 pages. Final logs contain no undefined references, undefined
  citations, or PDF-version inclusion warnings.
- Converted the exported figure to PDF 1.5 without rasterization. DejaVu Sans
  fonts remain embedded, and `pdfimages -list` reports no image objects.
- Synchronized the anonymous source, added the editable draw.io source, updated
  the separately uploaded `Figure_1.pdf`, rebuilt the source bundle, and
  regenerated the 12-entry checksum manifest. Every package checksum passes.

## 2026-07-19 Figure 1 iconography refinement in progress

- Audited the accepted draw.io source against the active Matplotlib style and
  selected only native vector symbols that remain editable inside diagrams.net.
- Rendered three review versions at 2400-pixel width. Revision 1 confirmed icon
  compatibility but exposed label overflow; revision 2 corrected body copy and
  title spacing; revision 3 left only the masked-action description to tighten.
- One combined XML patch failed because its context included a line already
  changed by an earlier hunk. The patch was atomic and changed no files. Smaller
  targeted patches were then applied successfully and validated with `xmllint`.

## 2026-07-19 Figure 1 iconography refinement completed

- Corrected the editing workflow after user feedback. The final native page was
  updated through Draw.io MCP, read back through MCP, and then exported with the
  local diagrams.net renderer.
- Added restrained native vector symbols to all four chronological stages, the
  sensing and observation path, policy header, feasibility and masked-action
  modules, executable subset, frozen forecaster, reward, PPO objective, and
  training supervision.
- Completed four full-size source reviews plus colour and grayscale manuscript-
  page reviews. No label clipping, connector collision, or icon/text overlap
  remains.
- Rebuilt `paper/main.pdf` and `paper/anonymous_manuscript.pdf`; both remain 50
  pages and contain no undefined references, citations, or PDF-version warnings.
- Verified that Figure 1 is a PDF 1.5 pure-vector asset with only embedded
  DejaVu Sans fonts. No raster objects are present.
- Synchronized the editable source and vector export into the anonymous source
  bundle, rebuilt the staged manuscript, and regenerated the submission package.
  All 12 checksums pass; canonical and packaged manuscript text hashes match,
  and the standalone Figure 1 hashes are identical.

## 2026-07-19 Figure 1 icon-to-text spacing correction completed

- Reviewed all four active main-figure pages at 180 dpi and isolated the spacing
  defect to five iconized modules inside Figure 1.
- Performed two Draw.io MCP revisions. The first was rejected because a larger
  inset caused excessive wrapping. The second regularized the cards at 34 units,
  centered their internal symbols, and retained a 6--7-unit icon-to-text gap.
- Exported and inspected the accepted standalone diagram and page 4 in colour
  and grayscale. Text remains legible, no icon overlaps a label, and the action
  and training paths remain distinguishable without colour.
- Rebuilt named and anonymous manuscripts successfully. Both remain 50 pages;
  final logs contain no undefined citations, references, or PDF-version warnings.
- The accepted Figure 1 remains pure vector and embeds only DejaVu Sans fonts.
  The submission package and anonymous source bundle were rebuilt, all 12
  checksums pass, and canonical/package text and Figure 1 hashes match.

## 2026-07-19 ESWA-reference Figure 1 hierarchy refinement in progress

- Audited three representative ESWA method figures and extracted their shared
  layout principles without copying article artwork.
- Preserved the previous accepted Draw.io source and vector export under
  `/home/horeb/agent/tmp/microclimate-codex-coordination/figure1_backups/2026-07-19-pre-eswa-reference/`.
- Completed three Draw.io MCP revisions. The current source uses a compact
  protocol strip, dominant online loop, and compact training-only strip; all
  obsolete waypoints, overflowing labels, and redundant flow annotations have
  been removed.
- Standalone colour and grayscale review has passed. Manuscript-scale review,
  final export, manuscript rebuild, and submission-package synchronization are
  pending.

## 2026-07-19 ESWA-reference Figure 1 hierarchy refinement completed

- Accepted the third Draw.io MCP revision after standalone, grayscale, named-
  manuscript, and anonymous-manuscript review. The final canvas is 1600 x 900
  and contains no overflowing labels, obsolete route points, redundant edge
  annotations, or connector-module intersections.
- Exported Figure 1 as PDF 1.5 pure vector with embedded DejaVu Sans fonts; the
  companion PNG is retained only as a preview.
- Added a local `-1.2em` pre-figure adjustment in `sections/01_introduction.tex`
  after manuscript-scale review exposed excess top whitespace. No global float
  spacing was changed.
- Rebuilt `main.pdf` and `anonymous_manuscript.pdf`; both are 50 pages and the
  logs contain no undefined citation, reference, or PDF-version warning.
- Rebuilt the staged anonymous source, synchronized standalone Figure 1, source
  archive, and manuscript PDF, regenerated `checksums.sha256`, and passed all
  12 package checks. Canonical/package text hashes and Figure 1 hashes match.

## 2026-07-19 Proposition 1 figure reconstruction completed

- Removed the old unnumbered qualitative compatibility matrix from Section 3.
- Added `figures/proposition_dynamic_value_tikz.tex` and a standalone build
  wrapper/PDF using the paper's DejaVu Sans and NewTX visual system.
- Added numbered Figure 2 immediately after Proposition 1, with a caption that
  separates the abstract sufficient condition from the Section 5 simulator
  mapping. Fixed placement prevents the figure from interrupting the theorem.
- Completed five standalone revisions plus colour, grayscale, and manuscript-
  page reviews. The accepted figure fits the 390-pt text width without scaling
  or overflow and contains no raster objects.
- Rebuilt named and anonymous manuscripts successfully. The submission package
  now contains Figures 1--5, its staged source includes the new TikZ source and
  standalone wrapper, and all 13 checksum entries pass.

## 2026-07-19 Figure 2 manuscript-spacing correction completed

- Corrected the embedded-layout collisions found in the manuscript-scale audit
  and increased the formula-to-panel clearance to approximately 0.36 cm.
- Rebuilt `main.pdf` (51 pages) and `anonymous_manuscript.pdf` (50 pages). The
  logs contain no undefined citation, reference, or targeted overflow warning.
- Synchronized Figure 2 and the anonymous source bundle into
  `submission/eswa_pdppo_20260719`; all 13 checksum entries pass, and the
  canonical and packaged Figure 2 PDFs have identical SHA-256 hashes.

## 2026-07-19 07-19-03 refinement in progress

- Updated the canonical title, PDF metadata, abstract, keywords, Introduction,
  method prose, results, discussion, conclusion, active supplement sources, and
  reader-facing baseline tables without changing numerical evidence.
- Added `paper/figures/gen_fig_pdppo_workflow.py` and generated the workflow PDF
  and PNG.
- Updated the authoritative clean-asset generator for frozen plot terminology
  and a non-color switching cue.
- Generation warning: the default Matplotlib cache directory is read-only.
  Resolution: use a task-specific `MPLCONFIGDIR` under `/tmp` for subsequent runs.

## 2026-07-19 07-19-03 refinement completed

- Generated Figure 1 with
  `env MPLCONFIGDIR=/tmp/pdppo_071903_mpl python3 paper/figures/gen_fig_pdppo_workflow.py`.
- Regenerated Figures 3--5 and their active tables with
  `env MPLCONFIGDIR=/tmp/pdppo_071903_mpl conda run -n darts python scripts/95_v31_build_clean_paper_assets.py --manifest-output /tmp/pdppo_071903_assets/paper_asset_manifest.json`.
  The manifest remained outside the frozen `reports/aggregate` tree.
- Ran all four required `latexmk -pdf -interaction=nonstopmode -halt-on-error`
  builds from `paper/`. The named manuscript is 51 pages, the anonymous
  manuscript is 50 pages, the supplement is 10 pages, and the title page is
  2 pages.
- Compared the active main and section sources with the pre-edit backup.
  Numeric tokens, citation keys, equation bodies, and proposition bodies match.
- Verified a 223-word abstract with an 18-word longest sentence, five highlight
  lines of at most 81 characters, nonempty PDF/PNG figure pairs, clean targeted
  `git diff --check` output, and no forbidden vocabulary or first-person prose.
- Inspected manuscript pages 1, 3, 31, 33, and 36 at 120 dpi. The title page and
  Figures 1 and 3--5 have no clipping, overlap, or connector collision.
- Confirmed that the anonymous PDF has empty author metadata and contains no
  author name or email marker in extracted text.

## 2026-07-19 Main-figure insertion repair completed

- Restored the complete Draw.io PD-PPO architecture as the active Figure 1 and
  added pre-figure anchors for all five main figures.
- Rebuilt `paper/main.pdf` and `paper/anonymous_manuscript.pdf`; both are 50
  pages and have no undefined citation or reference warning.
- Inspected pages 3--5, 11--13, and 29--36 at manuscript scale. Figures 1--5
  remain in the intended sections with no overlap, clipping, blank page, or
  post-reference ordering defect.
- Rebuilt `submission/eswa_pdppo_20260719/anonymous_source`, removed the
  unreferenced compact workflow from that bundle, synchronized standalone
  Figure 1, and passed all 13 package checks. Canonical and packaged anonymous
  PDF text hashes match; all three active Figure 1 copies have the same SHA-256
  hash.

## 2026-07-19 Figure 1 execution-boundary label corrected

- Updated the editable Figure 1 source through Draw.io MCP and re-exported the
  vector PDF/PNG with the online-action boundary stated explicitly.
- Revised the Figure 1 caption, rebuilt both 50-page manuscripts, and inspected
  the rendered page 4 at 180 dpi.
- Synchronized the anonymous source and standalone Figure 1; all 13 package
  checks pass and canonical/package PDF text and Figure 1 hashes match.
- Shortened the crowded Panel C badge to `OFFLINE ONLY`, reduced its width, and
  repeated the page-scale and submission-package checks successfully.

## 2026-07-22 strict theory revision completed

- Created pre-edit archives in
  `paper_archives/2026-07-22-theory-strict-revision/` for both `paper/` and the
  ESWA submission package and recorded SHA-256 checksums.
- Reconciled `docs/07-22-01.md` with the active manuscript and inspected the
  action-mask builder, projector, environment duration guard, reward metadata,
  and the six candidate masks before revising the mathematics.
- Revised the formulation, notation table, PPO protocol, reward equation,
  Proposition 1, Proposition 2, theoretical appendix, action table, online
  observability audit, and recurring constraint terminology. No experimental
  result or reported numerical comparison was changed.
- Updated Figure 1 through Draw.io MCP to show separate power/startup masking,
  action proposal, and duration-guard stages. Re-exported and converted the
  vector asset to PDF 1.5. Updated the Proposition 1 TikZ figure and rebuilt its
  standalone PDF.
- Built `main.pdf` (51 pages), `anonymous_manuscript.pdf` (50 pages),
  `supplementary_material.pdf` (10 pages), and `title_page.pdf` (2 pages) with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error`.
- Rendered manuscript pages 4, 9--16, and 42--44 and inspected the framework,
  action equations, notation, both propositions, and appendix proofs. No overlap,
  clipping, unreadable glyph, or broken cross-reference was found.
- Synchronized the anonymous source whitelist, Figures 1--2, anonymous PDF, and
  supplement into `submission/eswa_pdppo_20260719`. The source bundle compiles
  independently in `/tmp/eswa_pdppo_source_verify_20260722`.
- Added self-contained definitions of $\mathcal{C}_{\mathrm{mis}}(S_0)$ and
  $\Delta_c$ to the Proposition 1 caption, repeated the page-scale review, and
  regenerated the source archive. All 15 entries in `checksums.sha256` pass;
  the anonymous identity scan also passes.

## 2026-07-22 strict citation revision completed

- Archived the pre-edit paper and submission package under
  `paper_archives/2026-07-22-citation-strict-revision/` with SHA-256 checksums.
- Verified high-risk records against Crossref and publisher records, corrected
  the Fernández-Bes DOI, added the Tran arXiv DOI, recast the PANGAEA item as a
  dataset with complete authors, and completed the Wei, Pendyala, Murad, and
  Ogbodo records.
- Rewrote the active Related Work, forecaster description, and simulation-source
  paragraphs so citations support the nearby claim and manuscript-specific
  choices are not attributed to external papers.
- Built `main.pdf` and `anonymous_manuscript.pdf` successfully at 51 pages and
  confirmed the 10-page supplement remains current. The main bibliography has
  23 entries with no undefined citation or reference.
- Inspected rendered pages 5--7, 21, 27, and 48--51. Long author lists,
  proceedings titles, DOI links, the preprint record, and the dataset label are
  readable without overlap or clipping.
- Synchronized the anonymous PDF and edited source files into
  `submission/eswa_pdppo_20260719`, rebuilt `anonymous_source_bundle.zip`, and
  compiled that bundle independently. Canonical and packaged anonymous PDFs are
  byte-identical, extracted source-build text matches, all 15 checksums pass,
  and the anonymous identity scan passes.
- The first checksum-regeneration command failed because zsh did not split a
  newline-delimited scalar into paths. Re-running the operation through `xargs`
  completed the manifest without changing scientific content.

## 2026-07-27 Figure 1 text-overflow repair completed

- Archived the pre-edit Draw.io source, Figure 1 PDF, and named manuscript under
  `paper/paper_archives/2026-07-27-figure1-overflow-fix/` with SHA-256 checksums.
- Used the Draw.io MCP to revise 11 overflowing or crowded nodes. The validation
  stage, scheduler input, proposed subset, minimum-duration check, frozen
  forecaster, and forecast-loss reward now use shorter labels and consistent
  18--19 pt typography without changing the method or experimental protocol.
- Exported the revised diagram as a PDF 1.5 vector asset and inspected both the
  standalone rendering and manuscript page 4. No text crosses a node boundary,
  overlaps another element, or is clipped at manuscript scale.
- Built `main.pdf` (56 pages) and `anonymous_manuscript.pdf` (55 pages) with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error`.
- Synchronized the revised Draw.io source, Figure 1 PDF, anonymous manuscript,
  and curated anonymous source tree in `submission/eswa_pdppo_20260719`.
  The regenerated source bundle independently builds a 55-page anonymous PDF,
  and all 15 submission checksums pass.
