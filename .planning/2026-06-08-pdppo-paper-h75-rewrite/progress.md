# Progress: PD-PPO H75 Paper Rewrite

## 2026-06-08

- Created isolated paper rewrite plan.
- Confirmed scope:
  - active manuscript: `paper/pdppo_crst_rewrite.tex`;
  - active sections: `paper/rewrite_sections/*.tex`;
  - do not edit `paper/raw.tex`.
- Imported h75 5-seed evidence boundary from the completed algorithm plan.

### Manuscript rewrite completed

- Reframed the manuscript from Antarctic-first to algorithm-first:
  PD-PPO is now introduced as a prediction-driven RL scheduler for power-limited
  sensing; the Antarctic AWS blowing-snow simulator is introduced later as the
  controlled case study.
- Replaced the previous 3-seed env-dwell12 operational evidence with the locked
  h75 5-seed evidence:
  - dynamic heuristic wins: 5/5;
  - deployable selected static wins: 4/5;
  - duty-constrained non-PD-PPO wins: 4/5;
  - compact-static diagnostic wins: 3/5;
  - deployment behaviour valid: 5/5.
- Rewrote active English sections:
  - `paper/pdppo_crst_rewrite.tex`;
  - `paper/rewrite_sections/01_introduction.tex`;
  - `paper/rewrite_sections/02_related_work.tex`;
  - `paper/rewrite_sections/03_problem_formulation.tex`;
  - `paper/rewrite_sections/04_framework_protocol.tex`;
  - `paper/rewrite_sections/05_simulation_setup.tex`;
  - `paper/rewrite_sections/06_results.tex`;
  - `paper/rewrite_sections/07_discussion_future_work.tex`;
  - `paper/rewrite_sections/08_conclusion.tex`.
- Updated active highlight files:
  - `paper/highlights.txt`;
  - `paper/pdppo_crst_rewrite_highlights.txt`.
- Added h75 operational result assets:
  - `paper/tables/env_dwell12_operational_results.tex`;
  - `paper/figures/figure_h75_operational_summary.png`;
  - `paper/figures/figure_h75_operational_summary.svg`;
  - `scripts/66_v31_plot_h75_operational_summary.py`.
- Rendered and visually spot-checked key PDF pages:
  - Figure 1 framework page;
  - Figure 2 chronological split page;
  - Figure 3 AWS rendering page;
  - Table 3 / Figure 5 h75 operational result page.
- Compile command:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
- Compile result: success, output `paper/pdppo_crst_rewrite.pdf`, 36 pages.
- Remaining compile issue: one harmless 1.8 pt overfull hbox on page 1.

### Figure 4/5 style audit and redraw

- Audited final PDF pages for Figure 4 and Figure 5 after the h75 rewrite.
- Figure 4 source:
  - script: `scripts/53_redraw_generator_validation_figure.py`;
  - outputs: `paper/figures/figure3_synthetic_statistics.png` and `.svg`.
- Figure 5 source:
  - script: `scripts/66_v31_plot_h75_operational_summary.py`;
  - outputs: `paper/figures/figure_h75_operational_summary.png` and `.svg`.
- Updated both plotting scripts to use a palette consistent with the TikZ
  framework and split figures:
  - pale blue `#D9E7FC`;
  - pale green `#DEEFDE`;
  - pale yellow/orange `#FFF4CC` / `#FFECD8`;
  - pale red `#FFE2DE`;
  - slate/ink gray for outlines and text.
- Switched plots to a serif font family to better match the LaTeX/TikZ figures.
- Fixed Figure 4 label/title overlap by shortening panel titles and moving
  panel labels outside the axes.
- Fixed Figure 5 legend/title overlap by removing the in-image suptitle and
  placing the legend above the subplots.
- Recompiled:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
- Result: success, output remains `paper/pdppo_crst_rewrite.pdf`, 36 pages.
- Final rendered PDF pages 20 and 22 were visually checked; no obvious figure
  text obstruction remains.

### Rhetoric simplification and Figure 4(a) fix

- Removed visible internal experiment labels from the compiled PDF:
  - `V3.1`;
  - `h75`;
  - `no-warm-up`;
  - overly specific deployment wording such as `environment-level dwell`.
- Tightened main-text wording so abstract, introduction, framework, setup,
  results, discussion, and conclusion present the algorithm and evidence rather
  than implementation/engineering logs.
- Replaced implementation-heavy terms where possible:
  - `startup/warm-up` -> `activation`;
  - `guard drops` -> `energy clips` / `energy-constraint violations`;
  - detailed projector constraints -> deployment/resource constraints.
- Updated Figure 1 projector labels to `resource limits`, `activation state`,
  and `duty rules`.
- Fixed Figure 4(a) obstruction by removing the in-panel legend; Figure 4(c)
  retains the shared AntAWS/Synthetic legend.
- Recompiled `paper/pdppo_crst_rewrite.tex`; result: success, 35-page PDF.
- `pdftotext` check found no visible `V3.1`, `h75`, or `no-warm-up` strings in
  the compiled PDF.

### Figure-density check and supporting figures

- Compared current `paper/pdppo_crst_rewrite.pdf` with old `paper/main.pdf`:
  current rewrite had 5 figures, old main had 7.
- Decision: the rewrite was somewhat visually thin for Results, but the old
  state-machine/timeline figures should not be restored verbatim because they
  expose implementation details or fixed-budget behaviour that is not the lead
  operational claim.
- Added manuscript-facing supporting figures:
  - `paper/figures/figure_operational_behavior.png` and `.svg`, showing
    intermediate-duty channel use and switching rates across the five operational
    seeds;
  - `paper/figures/figure_fixed_budget_power_error.png` and `.svg`, redrawing the
    old fixed-budget power-error trade-off in the current paper style.
- Added generator script:
  `scripts/67_plot_paper_supporting_figures.py`.
- Updated `paper/rewrite_sections/06_results.tex` to include the two figures and
  explanatory text.
- Recompiled:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
- Result: success, `paper/pdppo_crst_rewrite.pdf`, 36 pages, 7 figures.
- `pdftotext` check found no visible `V3.1`, `h75`, `no-warm`, `corrected`,
  `boundary case`, `single failure`, `static shortcut`, `only a protocol`, or
  `negative result` strings.

### ESWA compression pass

- User requested compression and removal of earlier Cold Regions Science and
  Technology journal marking.
- Updated active manuscript target `paper/pdppo_crst_rewrite.tex`:
  - `\journal{Expert Systems with Applications}`;
  - title changed to `Prediction-driven reinforcement learning for power-limited
    sensor scheduling under deployment constraints`;
  - PDF metadata title updated accordingly;
  - abstract shortened and made more journal-neutral.
- Removed the appendix from the compiled main manuscript. The appendix source is
  preserved but no longer input by the active manuscript.
- Compressed:
  - `rewrite_sections/01_introduction.tex`;
  - `rewrite_sections/03_problem_formulation.tex`;
  - `rewrite_sections/05_simulation_setup.tex`;
  - `rewrite_sections/06_results.tex`;
  - `rewrite_sections/07_discussion_future_work.tex`;
  - `tables/main_results_v31.tex`;
  - `highlights.txt` and `pdppo_crst_rewrite_highlights.txt`.
- Main compression choices:
  - removed central-question quote block;
  - compressed simulator construction from detailed subsubsections to a compact
    benchmark-construction description;
  - removed redundant feasibility theorem and appendix reference;
  - shortened results text that duplicated tables/figures;
  - replaced five-part discussion with a shorter ESWA-style discussion.
- Recompiled:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
- Result: success, `paper/pdppo_crst_rewrite.pdf`, 30 pages.
- Quantitative reduction:
  - PDF total words: about 7,912 -> 5,896;
  - pre-statement manuscript words: about 6,981 -> 4,964;
  - pages: 36 -> 30.
- PDF text check: `Cold Regions Science and Technology` and `CRST` both absent;
  `Expert Systems with Applications` appears once as journal metadata/text.

### Reference correction: CRYOWRF first author

- User flagged that the seventh reference / Lenaerts-labeled citation had an
  incorrect first author.
- Verified the CRYOWRF v1.0 paper metadata: first author is Varun Sharma, with
  Franziska Gerber and Michael Lehning as coauthors.
- Updated `paper/references.bib`:
  - renamed `Lenaerts2023` to `Sharma2023`;
  - changed author field to `Sharma, Varun and Gerber, Franziska and Lehning,
    Michael`.
- Updated active rewrite citations in:
  - `paper/rewrite_sections/01_introduction.tex`;
  - `paper/rewrite_sections/02_related_work.tex`;
  - `paper/rewrite_sections/05_simulation_setup.tex`.
- Recompiled
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
- Result: success, `paper/pdppo_crst_rewrite.pdf`, 30 pages.
- Verification: `pdppo_crst_rewrite.bbl` and PDF text now cite/reference
  `Sharma et al., 2023`; no visible `Lenaerts` or `Lenaerts2023` remains in the
  active PDF/bibliography.

### Manuscript language pass: AI-style wording reduction

- User requested a careful detail pass for AI-style manuscript wording.
- Applied `academic-paper-composer` quality heuristics to the active English
  manuscript only; `raw.tex` and legacy `sections/*.tex` were not maintained.
- Edited:
  - `paper/pdppo_crst_rewrite.tex`;
  - `paper/rewrite_sections/01_introduction.tex`;
  - `paper/rewrite_sections/02_related_work.tex`;
  - `paper/rewrite_sections/03_problem_formulation.tex`;
  - `paper/rewrite_sections/04_framework_protocol.tex`;
  - `paper/rewrite_sections/05_simulation_setup.tex`;
  - `paper/rewrite_sections/06_results.tex`;
  - `paper/rewrite_sections/07_discussion_future_work.tex`;
  - `paper/rewrite_sections/08_conclusion.tex`.
- Main style changes:
  - replaced generic self-description (`we show`, `we present`, `framework`,
    `The results show`, `qualified but positive claim`) with direct technical
    prose;
  - rewrote the abstract into a cleaner problem--method--evidence structure;
  - converted the contribution list from repetitive first-person claims to
    concrete contribution items;
  - removed internal draft wording from the CRediT statement;
  - tightened related-work and results transitions without changing numerical
    claims.
- Validation:
  - Recompiled with
    `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
  - Result: success, output `paper/pdppo_crst_rewrite.pdf`, 29 pages.
  - Abstract word count: about 185 words.
  - PDF text scan found no visible `qualified but positive`, `negative result`,
    `only a protocol`, `This is the primary`, `The results show`, `we show`,
    `we present`, `we propose`, `V3.1`, `h75`, `no-warm`, `Lenaerts`,
    `Cold Regions Science and Technology`, or `CRST`.
  - Remaining nonblocking warnings: four bibliography entries still have empty
    `pages` fields; one small overfull hbox is reported by LaTeX.

### Public repository safety reset

- User clarified that the manuscript has not been published and the public
  repository should expose only a `coming soon` placeholder.
- Target public repository:
  `https://github.com/JekYUlll/forecast-driven-sensor-scheduling`.
- Local public-repo checkout:
  `/home/horeb/_code/microclimate_demo/forecast-driven-sensor-scheduling`.
- Actions:
  - saved the one local uncommitted public-repo paper edit in a local stash;
  - created an orphan placeholder commit containing only `README.md`;
  - force-updated remote `main` to the placeholder commit
    `088c22d Replace public repository with coming soon placeholder`;
  - deleted the old `v0.1.0` GitHub release and its tag;
  - aligned the local public-repo checkout back to `main`.
- Verification:
  - GitHub contents API for `main` returns only `README.md`;
  - `gh release list` returns no releases;
  - `git ls-remote --tags origin` returns no tags;
  - `git ls-remote --heads origin` returns only `refs/heads/main` at
    `088c22d49dc6f82d0a48159588e9765527165692`.
- Updated active manuscript Data Availability in
  `paper/pdppo_crst_rewrite.tex` to state that the code and reproduction package
  will be released after the manuscript is ready for public distribution, with the
  GitHub URL now described as a placeholder repository.
- Recompiled:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
- Result: success, `paper/pdppo_crst_rewrite.pdf`, 29 pages.

### Prediction-driven scenario clarification

- User noted that the manuscript under-explained the unusual conditions under
  which "prediction-driven" scheduling is the right framing.
- Updated:
  - `paper/pdppo_crst_rewrite.tex`;
  - `paper/rewrite_sections/01_introduction.tex`;
  - `paper/rewrite_sections/02_related_work.tex`.
- Added concise explanation that prediction-driven scheduling differs from
  ordinary observation scheduling because delayed or high-cost sensors may need to
  be activated before their immediate measurements are useful, especially when
  rare regimes dominate forecast error.
- Added introduction-level boundary: this is narrower than generic low-power data
  collection and applies when present measurements have asymmetric value for future
  targets.
- Added related-work boundary: the formulation is most relevant when forecast loss
  depends on rare regimes, delayed measurements, or variables whose value is not
  proportional to instantaneous freshness.
- Validation:
  - Abstract word count: about 215 words, still below the 250-word limit.
  - Recompiled with
    `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`.
  - Result: success, `paper/pdppo_crst_rewrite.pdf`, 29 pages.
  - PDF text scan confirmed the new scenario clarification appears in the abstract,
    introduction, and related work.

### Theory/formula supplement 2026-06-08T10:10+08:00

User asked to read `docs/06-08-01-fo.md` and first strengthen the theory/formula
parts of the active English manuscript.

Actions:

- Edited only the active English rewrite section files:
  - `paper/rewrite_sections/03_problem_formulation.tex`;
  - `paper/rewrite_sections/04_framework_protocol.tex`.
- Added or clarified the following mathematical definitions:
  - startup peak feasibility constraint, `eq:startup_peak`;
  - steady energy draw and simplified SOC update, `eq:energy_cost`;
  - scaled and target-weighted forecast-weighted MAE, `eq:fw_mae`;
  - training reward with switching, duty, violation, and optional event weight,
    `eq:training_reward`;
  - PPO clipped surrogate objective, `eq:ppo_clip`;
  - implemented PD-PPO loss with value, entropy, AWBC, and candidate-prior KL
    regularisation, `eq:pdppo_loss`;
  - AWBC and prior-KL regulariser definitions, `eq:awbc` and `eq:prior_kl`.
- Tightened the theoretical comparison with AoI/covariance:
  - unified sensor and target indices;
  - made AoI summation explicit over `t=0..T-1` and `j \in \mathcal{Y}`;
  - softened the over-strong "no monotone transformation" wording to an ordering
    non-equivalence proposition.
- Clarified that the action projector is an executable feasibility map, not an
  exact knapsack optimiser.
- Stated that the SOC auxiliary head exists but is disabled in the reported
  deployment-constrained PD-PPO runs.

Validation:

- Recompiled with:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
- Result: success, output `paper/pdppo_crst_rewrite.pdf`, 32 pages.
- Remaining nonblocking warnings are inherited/minor: four bibliography entries
  have empty `pages` fields and one small overfull hbox.

### AI-style and over-caveating audit 2026-06-08T10:23+08:00

User requested another AI-style audit focused on repeated/inconsistent wording,
`not ... but`-style constructions, and empty boilerplate. User also noted that
the final limitations were too self-limiting and the future-work section revealed
too much.

Actions:

- Edited the active English manuscript and active support files:
  - `paper/pdppo_crst_rewrite.tex`;
  - `paper/rewrite_sections/01_introduction.tex`;
  - `paper/rewrite_sections/02_related_work.tex`;
  - `paper/rewrite_sections/03_problem_formulation.tex`;
  - `paper/rewrite_sections/04_framework_protocol.tex`;
  - `paper/rewrite_sections/05_simulation_setup.tex`;
  - `paper/rewrite_sections/06_results.tex`;
  - `paper/rewrite_sections/07_discussion_future_work.tex`;
  - `paper/rewrite_sections/08_conclusion.tex`;
  - `paper/rewrite_sections/appendix_theory.tex`;
  - `paper/figures/pdppo_framework_rewrite_tikz.tex`;
  - `paper/tables/sensor_specs.tex`.
- Removed repeated caveat-first phrasing around prediction-driven scheduling,
  static baselines, simulation status, and data splits.
- Rewrote defensive `not/rather than/instead of` sentences into positive
  definitions where possible.
- Replaced `Limitations` with `Scope of evidence`, keeping external-validity
  boundaries without making the section self-defeating.
- Shortened `Future work` to broad external-validation steps and removed detailed
  algorithmic disclosure.
- Cleaned table/figure prose that used repeated negative boundary statements such
  as device-watt and polling-period disclaimers.

Validation:

- Recompiled with:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
- Result: success, output `paper/pdppo_crst_rewrite.pdf`, 30 pages.
- PDF text scan found no visible `not field validation`, `negative result`,
  `only a protocol`, `qualified but positive`, `V3.1`, `h75`, `no-warm`,
  `rather than ordinary reward`, `end-to-end co-trained`, `we show`,
  `we present`, `we propose`, `therefore produce`, `not a field`,
  `not a physical battery`, `future work should`, `dynamic Lagrangian`,
  `training-authorised data only`, `not device-watt`, `not native polling`, or
  `not eight separate`.
- Remaining `not/only/protocol` hits are technical or administrative:
  `not tractable`, `not equivalent to enforcing a deployment constraint`,
  math proof wording, table captions, and Data Availability.

### Data freshness / missing-experiment audit 2026-06-08T10:35+08:00

User asked whether the manuscript still uses old data and whether obvious
additional experiments are needed.

Findings:

- Active manuscript inputs only:
  - `rewrite_sections/01_introduction.tex` through `08_conclusion.tex`;
  - `tables/env_dwell12_operational_results.tex`;
  - `tables/main_results_v31.tex`;
  - `tables/energy_account_storm_oracle.tex`;
  - `tables/sensor_specs.tex`;
  - `tables/g1_generator_validation.tex`;
  - figures `aws_deployment.png`, `figure3_synthetic_statistics.png`,
    `figure_operational_summary.png`, `figure_operational_behavior.png`,
    `figure_fixed_budget_power_error.png`, and two TikZ figures.
- New main operational table is aligned with:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/env_dwell12_h75_operational_summary_5seed.csv`
  and its comparison CSV. The table values match the h75 5-seed source:
  PD-PPO beats best original dynamic 5/5, deployable selected static 4/5, and
  best duty-constrained non-PD-PPO 4/5.
- `figure_operational_summary.png` is generated by
  `scripts/66_v31_plot_h75_operational_summary.py`, whose defaults point to the
  h75 5-seed summary and comparison CSVs.
- `figure_operational_behavior.png` is generated by
  `scripts/67_plot_paper_supporting_figures.py`, whose operational raw input is
  the h75 5-seed raw directory.
- The fixed-budget table/figure remain from
  `reports/v31_split_protocol_main/v31_s2_main_stats.csv`, i.e. the older
  fixed-budget diagnostic run, not the h75 deployment-constrained run. Current
  text frames it as a fixed-budget reference/static-shortcut diagnostic, which is
  acceptable if not presented as the main operational evidence.
- The energy-account table is a separate reference-policy/storm-window analysis;
  its caption already states the separate three-hour logical-epoch configuration
  and that values are not pooled with the fixed-budget table.
- Main old-data risk: `tables/g1_generator_validation.tex` and
  `figure3_synthetic_statistics.png` come from the older
  `reports/v3_supplement_assets` G1 generator-validation assets. They validate a
  similar V3.1 generator family but are not regenerated from the current h75
  90,000-step truth sequences. This is the clearest remaining source-alignment
  issue.

Recommended additions before a stronger submission draft:

- Low-cost, no-retraining:
  - regenerate generator validation table/figure from the current h75 truth
    design, or relabel current Figure 4/Table 2 as generator-family validation;
  - build an operational event/non-event loss and sensor-duty table from saved
    h75 rollouts where rollout tensors are available;
  - add a concise source note in Results/Table captions distinguishing h75
    operational evidence from fixed-budget diagnostic evidence.
- Training/eval additions:
  - extend h75 deployment-constrained mainline from 5 seeds to 10 seeds;
  - add at least one operational budget sensitivity point, preferably B=1.65 and
    B=1.75 under the same dwell/duty constraints;
  - optional but useful: ablate prior/AWBC/event-context on the h75 operational
    setting if the paper wants a stronger algorithm-component claim.

Follow-up fix:

- Corrected `paper/figures/data_split_timeline_tikz.tex` note from
  `budget and warm-up constraints` to `budget and deployment constraints`, because
  the current main operational evidence is no-warmup with dwell/duty guards.
- Recompiled successfully with:
  `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
- PDF text scan no longer finds `warm-up constraints`, `warmup`, `no-warm`, `h75`,
  `V3.1`, or `v31`.

### Event-conditioned h75 diagnostic addition

- User requested continuing the PD-PPO paper plan after the data-freshness and
  `06-08-02.md` review.
- Audited existing h75 diagnostic assets:
  - `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/h75_pdppo_vs_deployable_static_loss_audit.csv`;
  - `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/h75_pdppo_vs_deployable_static_sensor_audit.csv`;
  - `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/h75_pdppo_vs_deployable_static_top_masks.csv`.
- Added:
  - `paper/tables/env_dwell12_event_diagnostics.tex`.
- Updated:
  - `paper/rewrite_sections/06_results.tex` to input the new event-conditioned
    table and add a concise interpretation of overall/event/non-event loss,
    mask diversity, top-mask concentration, switching, and laser duty;
  - `paper/rewrite_sections/05_simulation_setup.tex` to describe the generator
    table/figure as generator-family validation and not as final-test policy
    selection.
- Key values now in the manuscript:
  - PD-PPO vs deployable static mean oracle loss:
    `0.143213` vs `0.144220`;
  - event loss: `0.182876` vs `0.183668`;
  - non-event loss: `0.125584` vs `0.126672`;
  - PD-PPO win counts: `4/5` overall, `3/5` event, `4/5` non-event;
  - average unique masks: `10.8` vs `7.0`;
  - top mask share: `37.4%` vs `48.5%`;
  - switching rate: `3.10%` vs `3.02%`.
- Validation:
  - Command:
    `cd rl_sensor_scheduling_framework/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
  - Result: success.
  - Output: `paper/pdppo_crst_rewrite.pdf`, 31 pages.
  - Abstract word count: about 210 words.
  - PDF scan found no visible `V3.1`, `h75`, `no-warm`, `Cold Regions Science and
    Technology`, `CRST`, `negative result`, `only a protocol`, or `static
    shortcut` strings.
  - Remaining nonblocking warnings: four bibliography entries with empty pages
    and a small overfull hbox on page 1.

### Remote h75 10-seed extension launched

- Used `microclimate-experiment-server` skill and connected via the configured
  `ssh remote-gpu` alias.
- Remote project path:
  `/home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework`.
- Pre-launch verification:
  - h75 result directory existed;
  - done markers and metrics existed for seeds 41--45 only;
  - GPU 5 was idle.
- Launched tmux session:
  `pdppo_h75_extend_46_50_20260608`.
- Remote command uses:
  - output directory:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced`;
  - seeds `46 47 48 49 50`;
  - budget `1.70`;
  - GPU id `5`;
  - h75 deployment parameters: hard duty guard `[0.12, 0.75]`, min dwell `12`,
    total timesteps `40000`, AWBC `0.02`, prior KL `0.05`, candidate-prior
    scale `0.5`, entropy coefficient `0.003`, duty-constrained baselines enabled.
- Launch log:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/logs/extend_46_50_tmux.log`.
- First log lines confirmed:
  `[split-grid] tasks=5 skipped=0 pending=5 workers=1`;
  seed46 started.
- Next action after completion:
  collect seeds 41--50 with `scripts/65_v31_collect_operational_pdppo.py`, sync
  the 10-seed summary locally, and update paper tables/figures only if the
  10-seed gate remains coherent.

### H75 10-seed sync, manuscript update, and budget sensitivity launch

- Monitored remote tmux `pdppo_h75_extend_46_50_20260608` until completion.
- Completion detected at 2026-06-08 12:25+08:00:
  - remote done markers: `10`;
  - remote metrics CSV count: `10`.
- Synced remote directory to local:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced`.
- Aggregated 10 seeds:
  `python scripts/65_v31_collect_operational_pdppo.py --base-dir reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced --budget-label budget1p70 --seeds 41 42 43 44 45 46 47 48 49 50 --out-name env_dwell12_h75_operational_summary_10seed.csv`
- 10-seed result:
  - vs full-open reference: `0/10`, mean delta `-0.020634`;
  - vs compact/selected static: `5/10`, mean delta `+0.003394`;
  - vs deployable selected static: `4/10`, mean delta `-0.000320`;
  - vs best original dynamic heuristic: `10/10`, mean delta `+0.008493`;
  - vs best duty-constrained non-PD-PPO baseline: `9/10`, mean delta `+0.005477`;
  - schedule-validity gate: `10/10`.
- Added rollout audit script:
  `scripts/68_v31_operational_rollout_audit.py`.
- Generated 10-seed rollout audit CSVs:
  - `h75_pdppo_vs_deployable_static_10seed_loss_audit.csv`;
  - `h75_pdppo_vs_deployable_static_10seed_sensor_audit.csv`;
  - `h75_pdppo_vs_deployable_static_10seed_top_masks.csv`.
- Event-conditioned result vs deployable static:
  - overall loss: `0.140635` vs `0.140315`;
  - event loss: `0.184192` vs `0.184050`;
  - non-event loss: `0.124176` vs `0.123832`;
  - PD-PPO win counts: `4/10`, `3/10`, `5/10`;
  - average unique masks: `10.9` vs `7.0`;
  - top-mask share: `36.4%` vs `47.4%`;
  - laser duty: `26.8%` overall and `29.5%` during events.
- Updated active paper assets:
  - `paper/pdppo_crst_rewrite.tex`;
  - `paper/rewrite_sections/01_introduction.tex`;
  - `paper/rewrite_sections/06_results.tex`;
  - `paper/rewrite_sections/07_discussion_future_work.tex`;
  - `paper/rewrite_sections/08_conclusion.tex`;
  - `paper/tables/env_dwell12_operational_results.tex`;
  - `paper/tables/env_dwell12_event_diagnostics.tex`;
  - `paper/highlights.txt`;
  - `paper/pdppo_crst_rewrite_highlights.txt`;
  - `paper/figures/figure_operational_summary.png/.svg`;
  - `paper/figures/figure_operational_behavior.png/.svg`.
- Updated plotting scripts:
  - `scripts/66_v31_plot_h75_operational_summary.py` now defaults to 10-seed
    summary/comparison files and avoids win-count label overlap for near-zero
    bars;
  - `scripts/67_plot_paper_supporting_figures.py` now uses seeds 41--50 for the
    operational behaviour figure.
- Interpretation change:
  - the paper no longer claims stable superiority over deployable static;
  - the main positive operational claim is now PD-PPO's valid nondegenerate
    schedule plus wins over original dynamic heuristics `10/10` and
    duty-constrained non-PD-PPO baselines `9/10`;
  - deployable selected static is reported as a close fixed-design reference.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, 32 pages;
  - abstract word count: about 211 words;
  - PDF scan found no old 5-seed claims or internal labels;
  - visual check of Results pages 19--22 found no table overflow or figure
    overlap.
- Launched next supplemental experiment:
  - remote tmux: `pdppo_h75_budget_sensitivity_20260608`;
  - output directory:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_budget_sensitivity_10seed`;
  - budgets `1.65` and `1.75`;
  - seeds `41`--`50`;
  - workers `4`, GPU ids `1,2,3,5`;
  - same h75 deployment parameters as the main 10-seed run.

### Resume after compaction: budget sensitivity monitoring

- Timestamp: 2026-06-08T13:27+08:00.
- Restored the active h75 paper/rewrite plan and corrected the plan-level
  evidence lock from older 5-seed wording to the current 10-seed mainline.
- Remote tmux monitor remains active for
  `pdppo_h75_budget_sensitivity_20260608`.
- Latest monitor count:
  - done markers: `10/20`;
  - metrics CSVs: `10/20`;
  - B=1.65 appears complete;
  - B=1.75 seeds 41--44 were in progress in the latest tails.
- No error pattern was visible in the latest log tails.

### Budget sensitivity completed and written into draft

- Timestamp: 2026-06-08T14:25+08:00.
- Remote tmux `pdppo_h75_budget_sensitivity_20260608` completed with:
  - done markers: `20/20`;
  - metrics CSVs: `20/20`.
- Synced remote directory locally:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_budget_sensitivity_10seed`.
- Sync stats:
  - files transferred: `541`;
  - total file size: `1,220,938,296` bytes.
- Aggregated:
  - `env_dwell12_h75_budget1p65_operational_summary_10seed.csv`;
  - `env_dwell12_h75_budget1p65_operational_summary_10seed_comparisons.csv`;
  - `env_dwell12_h75_budget1p75_operational_summary_10seed.csv`;
  - `env_dwell12_h75_budget1p75_operational_summary_10seed_comparisons.csv`.
- Budget sensitivity summary:
  - B=1.65: PD-PPO loss `0.140670`; vs original dynamic `9/10`,
    delta `+0.004005`; vs duty non-PD-PPO `9/10`, delta `+0.003618`;
    vs deployable static `5/10`, delta `-0.000390`; valid `10/10`.
  - B=1.70: PD-PPO loss `0.140635`; vs original dynamic `10/10`,
    delta `+0.008493`; vs duty non-PD-PPO `9/10`, delta `+0.005477`;
    vs deployable static `4/10`, delta `-0.000320`; valid `10/10`.
  - B=1.75: PD-PPO loss `0.142710`; vs original dynamic `9/10`,
    delta `+0.004209`; vs duty non-PD-PPO `7/10`, delta `+0.002333`;
    vs deployable static `1/10`, delta `-0.004874`; valid `10/10`.
- Paper updates:
  - added `paper/tables/env_dwell12_budget_sensitivity.tex`;
  - updated `paper/rewrite_sections/06_results.tex`;
  - updated `paper/rewrite_sections/07_discussion_future_work.tex`;
  - fixed one abstract grammar issue in `paper/pdppo_crst_rewrite.tex`.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output PDF: `paper/pdppo_crst_rewrite.pdf`, 33 pages;
  - abstract word count: `209`;
  - PDF scan found no old 5-seed/internal labels or high-risk defensive phrases;
  - page 23 visual check confirmed the new budget table is readable;
  - remaining nonblocking warning: one 1.8pt overfull hbox on page 1.

### No-candidate-prior ablation launched

- Timestamp: 2026-06-08T14:26+08:00.
- Remote tmux: `pdppo_h75_no_prior_20260608`.
- Remote output directory:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_candidate_prior_10seed`.
- Budget/seeds:
  - budget `1.70`;
  - seeds `41`--`50`.
- Workers/GPU ids:
  - workers `4`;
  - GPU ids `1,2,3,5`.
- Same h75 deployment constraints as the main run:
  hard duty guard `[0.12,0.75]`, min dwell `12`, total timesteps `40000`,
  AWBC `0.02`, entropy coefficient `0.003`, duty-constrained baselines enabled.
- Ablation changes:
  - `--no-use-candidate-prior`;
  - `--prior-kl-coef 0.0`;
  - `--candidate-prior-scale 0.0`.
- Launch log confirmed:
  `[split-grid] tasks=10 skipped=0 pending=10 workers=4`, seeds 41--44 started.

### No-candidate-prior ablation completed and written

- Timestamp: 2026-06-08T15:32+08:00.
- Remote tmux `pdppo_h75_no_prior_20260608` completed with:
  - done markers: `10/10`;
  - metrics CSVs: `10/10`.
- Synced remote directory locally:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_candidate_prior_10seed`.
- Sync stats:
  - files: `285`;
  - total file size: `610,437,168` bytes.
- Aggregated:
  - `env_dwell12_h75_no_candidate_prior_operational_summary_10seed.csv`;
  - `env_dwell12_h75_no_candidate_prior_operational_summary_10seed_comparisons.csv`.
- No-prior summary:
  - PD-PPO loss `0.142476`;
  - vs original dynamic `10/10`, delta `+0.006816`;
  - vs duty non-PD-PPO `9/10`, delta `+0.003706`;
  - vs deployable static `2/10`, delta `-0.001873`;
  - valid behaviour `10/10`.
- Paired comparison with the main B=1.70 configuration:
  - main loss `0.140635`;
  - no-prior minus main mean `+0.001841`;
  - main is lower in `8/10` seeds.
- Paper updates:
  - added `paper/tables/env_dwell12_candidate_prior_ablation.tex`;
  - updated `paper/rewrite_sections/06_results.tex`;
  - updated `paper/rewrite_sections/07_discussion_future_work.tex`.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, 33 pages;
  - abstract word count `208`;
  - PDF text scan found no stale internal labels or high-risk AI-style phrases;
  - page 23 visual check confirmed the budget-sensitivity and candidate-prior
    ablation tables are readable;
  - remaining nonblocking issues: one 1.8pt page-1 overfull hbox and four
    BibTeX empty-page warnings for existing bibliography entries.

### Min-dwell sensitivity follow-up launched

- Timestamp: 2026-06-08T15:36+08:00.
- Rationale:
  test whether the h75 deployment result is sensitive to the minimum sustained
  activation length. This stays within the current PD-PPO paper scope and avoids
  code changes.
- Planned sequence:
  - first run dwell `6`;
  - after completion, run dwell `24`;
  - both use B=1.70, seeds `41`--`50`, and the h75 parameters otherwise.
- First launch attempt failed:
  - tmux session exited immediately;
  - log showed `scripts/59_v31_split_protocol_grid.py: Permission denied`;
  - root cause was premature `$PY` expansion in the remote tmux command, which
    made bash try to execute the Python script directly.
- Corrected launch:
  - tmux session: `pdppo_h75_dwell6_sensitivity_20260608`;
  - output directory:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell6_h75_dwell_sensitivity_10seed`;
  - workers `4`, GPU ids `1,2,3,5`;
  - launch verification:
    `[split-grid] tasks=10 skipped=0 pending=10 workers=4`;
  - first four seeds `41`--`44` started.

### Min-dwell sensitivity: dwell=6 completed, dwell=24 launched

- Timestamp: 2026-06-08T16:34+08:00.
- Dwell=6 remote run completed:
  - tmux session: `pdppo_h75_dwell6_sensitivity_20260608`;
  - output directory:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell6_h75_dwell_sensitivity_10seed`;
  - completion: `done=10`, `metrics=10`;
  - sync stats: `294` files, total size `610,466,025` bytes.
- Aggregated dwell=6:
  - `env_dwell12_h75_dwell6_operational_summary_10seed.csv`;
  - `env_dwell12_h75_dwell6_operational_summary_10seed_comparisons.csv`.
- Dwell=6 summary:
  - vs original dynamic `7/10`, delta `+0.000362`;
  - vs duty non-PD-PPO `9/10`, delta `+0.005748`;
  - vs deployable static `4/10`, delta `-0.001182`;
  - vs compact/static `6/10`, delta `+0.008064`.
- Interim interpretation:
  dwell=6 is weaker against unconstrained dynamic heuristics than the main
  dwell=12 setting, so it should not be written as a standalone positive result
  before dwell=24 is aggregated.
- Dwell=24 launched:
  - tmux session: `pdppo_h75_dwell24_sensitivity_20260608`;
  - output directory:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell24_h75_dwell_sensitivity_10seed`;
  - parameters: same h75 B=1.70 setting, but `--min-dwell-steps 24`;
  - launch verification:
    `[split-grid] tasks=10 skipped=0 pending=10 workers=4`.

### Min-dwell sensitivity completed and written

- Timestamp: 2026-06-08T17:34+08:00.
- Dwell=24 remote run completed:
  - tmux session: `pdppo_h75_dwell24_sensitivity_20260608`;
  - output directory:
    `reports/v31_split_protocol_no_warmup_hguard_envdwell24_h75_dwell_sensitivity_10seed`;
  - completion: `done=10`, `metrics=10`;
  - remote marker: `DWELL24_COMPLETE 2026-06-08T17:27:29+08:00`;
  - sync stats: `294` files, total size `610,466,523` bytes.
- Aggregated dwell=24:
  - `env_dwell12_h75_dwell24_operational_summary_10seed.csv`;
  - `env_dwell12_h75_dwell24_operational_summary_10seed_comparisons.csv`.
- Minimum-dwell sensitivity summary at $B=1.70$:
  - dwell 6: PD-PPO loss `0.136948`; vs original dynamic `7/10`,
    delta `+0.000362`; vs duty non-PD-PPO `9/10`, delta `+0.005748`;
    vs deployable static `4/10`, delta `-0.001182`; switch rate `0.059`;
    valid `10/10`.
  - dwell 12: PD-PPO loss `0.140635`; vs original dynamic `10/10`,
    delta `+0.008493`; vs duty non-PD-PPO `9/10`, delta `+0.005477`;
    vs deployable static `4/10`, delta `-0.000320`; switch rate `0.031`;
    valid `10/10`.
  - dwell 24: PD-PPO loss `0.142483`; vs original dynamic `10/10`,
    delta `+0.010106`; vs duty non-PD-PPO `10/10`, delta `+0.008249`;
    vs deployable static `6/10`, delta `-0.000129`; switch rate `0.016`;
    valid `10/10`.
- Interpretation:
  - dwell=6 shows that loose actuation constraints allow fast cycling heuristics
    to approach PD-PPO;
  - dwell=12/24 preserve PD-PPO's advantage over dynamic baseline families while
    reducing switching frequency;
  - this supports the paper's deployment-constrained comparison instead of
    over-weighting unconstrained round-robin/AoI cycling.
- Paper updates:
  - added `paper/tables/env_dwell12_dwell_sensitivity.tex`;
  - updated `paper/rewrite_sections/06_results.tex`;
  - updated `paper/rewrite_sections/07_discussion_future_work.tex`.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, 34 pages;
  - abstract word count `208`;
  - stale/internal-label and high-risk AI-style scan returned no hits;
  - visual checks of pages 23 and 24 confirmed the new dwell table and paragraph
    are readable with no overflow.

### Vanilla-PPO ablation launched

- Timestamp: 2026-06-08T17:37+08:00.
- Rationale:
  ESWA-oriented audit identified the need for a stronger algorithmic baseline.
  Existing code supports a PPO-only ablation by disabling the AWBC imitation
  term, candidate prior, and prior-KL regularizer while preserving the same
  forecast reward, projection layer, duty guard, dwell guard, and evaluation
  protocol.
- Remote tmux:
  `pdppo_h75_vanilla_ppo_20260608`.
- Remote output directory:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_vanilla_ppo_10seed`.
- Parameters:
  - budget `1.70`;
  - seeds `41`--`50`;
  - workers `3`, GPU ids `2,3,5`;
  - total timesteps `40000`;
  - hard duty guard `[0.12,0.75]`;
  - min dwell `12`;
  - `--awbc-coef 0.0`;
  - `--no-use-candidate-prior`;
  - `--prior-kl-coef 0.0`;
  - `--candidate-prior-scale 0.0`;
  - duty-constrained baselines enabled.
- Launch verification:
  `[split-grid] tasks=10 skipped=0 pending=10 workers=3`, seeds `41`--`43`
  started.

### Vanilla-PPO ablation completed and written

- Timestamp: 2026-06-08T18:14+08:00.
- Remote tmux `pdppo_h75_vanilla_ppo_20260608` completed:
  - metrics `10`;
  - done markers `10`;
  - remote marker: `VANILLA_PPO_COMPLETE 2026-06-08T18:04:28+0800`.
- Synced remote output directory locally:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_vanilla_ppo_10seed`.
- Aggregated with:
  `python scripts/65_v31_collect_operational_pdppo.py --base-dir reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_vanilla_ppo_10seed --budget-label budget1p70 --seeds 41 42 43 44 45 46 47 48 49 50 --out-name env_dwell12_h75_vanilla_ppo_operational_summary_10seed.csv`.
- Vanilla/PPO-only result:
  - mean PD-PPO loss `0.149170`;
  - valid deployment behaviour `10/10`;
  - beats best original dynamic heuristic `5/10`, mean delta `-0.001107`;
  - beats best duty-constrained non-PD-PPO baseline `3/10`, mean delta
    `-0.003615`;
  - beats deployable selected static `0/10`, mean delta `-0.009748`.
- Paired against the main PD-PPO configuration:
  - vanilla minus main mean `+0.008535`;
  - main is lower in `10/10` seeds.
- Paper updates:
  - extended `paper/tables/env_dwell12_candidate_prior_ablation.tex` to include
    the PPO-only row;
  - updated `paper/rewrite_sections/06_results.tex`;
  - updated `paper/rewrite_sections/07_discussion_future_work.tex`.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, `35` pages;
  - abstract word count `206`;
  - stale/internal-label and high-risk AI-style scan returned no hits;
  - visual check of page 24 confirmed the ablation table is readable.

### AWBC-off / prior-on ablation launched

- Timestamp: 2026-06-08T18:16+08:00.
- Rationale:
  The PPO-only ablation disables both AWBC and the candidate prior. A cleaner
  component split disables AWBC while retaining the candidate prior and prior-KL
  term under the same deployment contract.
- Remote tmux:
  `pdppo_h75_no_awbc_prior_on_20260608`.
- Remote output directory:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_awbc_prior_on_10seed`.
- Parameters:
  - budget `1.70`;
  - seeds `41`--`50`;
  - workers `3`, GPU ids `1,2,3`;
  - total timesteps `40000`;
  - hard duty guard `[0.12,0.75]`;
  - min dwell `12`;
  - `--awbc-coef 0.0`;
  - `--use-candidate-prior`;
  - `--prior-kl-coef 0.05`;
  - `--candidate-prior-scale 0.5`;
  - duty-constrained baselines enabled.
- Dry-run passed:
  `[split-grid] tasks=10 skipped=0 pending=10 workers=3`.
- Launch verification:
  tmux session exists and seeds `41`--`43` started writing logs/manifests.

### AWBC-off / prior-on ablation completed and written

- Timestamp: 2026-06-08T18:55+08:00.
- Remote tmux `pdppo_h75_no_awbc_prior_on_20260608` completed:
  - metrics `10`;
  - done markers `10`;
  - tmux session exited cleanly.
- Synced remote output directory locally:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_awbc_prior_on_10seed`.
- Sync stats:
  - files `294`;
  - total size `610,458,921` bytes;
  - regular files transferred `270`.
- Aggregated with:
  `python scripts/65_v31_collect_operational_pdppo.py --base-dir reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_no_awbc_prior_on_10seed --budget-label budget1p70 --seeds 41 42 43 44 45 46 47 48 49 50 --out-name env_dwell12_h75_no_awbc_prior_on_operational_summary_10seed.csv`.
- AWBC-off/prior-on result:
  - mean PD-PPO loss `0.145448`;
  - valid deployment behaviour `10/10`;
  - beats best original dynamic heuristic `3/10`, mean delta `-0.001588`;
  - beats best duty-constrained non-PD-PPO baseline `7/10`, mean delta
    `+0.003098`;
  - beats deployable selected static `0/10`, mean delta `-0.004175`.
- Paired against the main PD-PPO configuration:
  - no-AWBC/prior-on minus main mean `+0.004813`;
  - main is lower in `9/10` seeds.
- Comparative training-scaffold summary:
  - main: loss `0.140635`, dynamic `10/10`, duty `9/10`, deployable static
    `4/10`;
  - no prior: loss `0.142476`, dynamic `10/10`, duty `9/10`, deployable static
    `2/10`;
  - no AWBC: loss `0.145448`, dynamic `3/10`, duty `7/10`, deployable static
    `0/10`;
  - PPO only: loss `0.149170`, dynamic `5/10`, duty `3/10`, deployable static
    `0/10`.
- Paper updates:
  - extended `paper/tables/env_dwell12_candidate_prior_ablation.tex`;
  - updated `paper/rewrite_sections/06_results.tex`;
  - updated `paper/rewrite_sections/07_discussion_future_work.tex`.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, `35` pages;
  - abstract word count `206`;
  - stale/internal-label and high-risk AI-style scan returned no hits;
  - visual check of page 24 confirmed the expanded ablation table is readable;
  - remote check found no active PD-PPO tmux session after completion.

## 2026-06-08 - Manuscript stale-wording and consistency audit

- Scope:
  active English rewrite only, centred on
  `paper/pdppo_crst_rewrite.tex` and `paper/rewrite_sections/*.tex`.
  `raw.tex` was not maintained.
- Fixed stale or inconsistent wording:
  - final-test protocol wording now applies to scheduler comparisons generally,
    not only fixed-budget results;
  - simulation setup distinguishes fixed-budget sensitivity
    (`B in {1.65, 1.70, 1.75}`) from the main deployment-constrained
    `B=1.70` run;
  - "activation failures" was standardised to "warm-up aborts";
  - Data Availability now uses a release-after-ready statement and no public
    placeholder URL;
  - ablation table caption now describes row-policy win counts;
  - seed45 duty-constrained exception is explained as a close seed-level
    alignment case without changing the main dynamic-baseline claim.
- Validation:
  - stale/internal-term scan returned no main-source hits for V3.1/h75/h85,
    no-warmup labels, CRST/Code Region remnants, defensive protocol-paper
    language, placeholder URLs, or old seed-count phrasing;
  - only remaining scan hits are inside the archived, explicitly labelled
    non-independent energy-account curriculum table;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, `35` pages;
  - TeX abstract environment count `209`, under the `250`-word limit;
  - visual check of pages 24--25 passed.

## 2026-06-08 - Numeric audit

- Recomputed the main operational values from:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/env_dwell12_h75_operational_summary_10seed.csv`.
- Confirmed:
  - PD-PPO mean loss `0.140635`;
  - best original dynamic comparison `10/10`, `+0.008493`,
    one-sided Wilcoxon `p=0.00097656`;
  - best duty-constrained non-PD-PPO comparison `9/10`, `+0.005477`,
    one-sided Wilcoxon `p=0.00488281`;
  - deployable static comparison `4/10`, `-0.000320`;
  - compact static comparison `5/10`, `+0.003394`;
  - schedule validity `10/10`, with no always-on/off channels and no warm-up
    aborts.
- Confirmed oracle checkpoint history:
  - final train loss `0.07455±0.00253`;
  - validation loss `0.07315±0.00285`.
- Confirmed fixed-budget table against `v31_s2_main_stats.csv` and fixed-budget
  significance against `v31_s2_significance.csv`; `p <= 0.0235` is consistent
  with the rounded Bonferroni-adjusted value `0.023438`.
- Confirmed generator-validation values against
  `reports/v3_supplement_assets/exp_g1_generator_validation.csv`.
- Confirmed energy-account table values against
  `reports/physical_event_v4_energy_cal_h092_cap180_storm_tcn_b120_seed41/oracle_lift_candidate_table.csv`.
- Fixes:
  - renamed energy-account table columns from stale `Energy clips` /
    `Activation fails` to `Guard drops` / `Warm-up aborts`;
  - corrected uncompiled appendix table setting from `100,000` to `40,000`
    PPO timesteps.
- Validation:
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - compiled PDF text has no `Activation fails`, `Energy clips`, or
    `100,000 timesteps` remnants.

## 2026-06-08 - Style defensiveness audit

- Re-audited the active English rewrite for over-defensive, self-critical, or
  old failure-first wording.
- Edited:
  - `paper/pdppo_crst_rewrite.tex`;
  - `paper/rewrite_sections/01_introduction.tex`;
  - `paper/rewrite_sections/02_related_work.tex`;
  - `paper/rewrite_sections/06_results.tex`;
  - `paper/rewrite_sections/07_discussion_future_work.tex`;
  - `paper/rewrite_sections/08_conclusion.tex`.
- Removed or softened repeated caveat-first patterns:
  hidden-fixed-subset disclaimers, "easy to overstate" phrasing, "failure point"
  language, and repeated static-shortcut warnings in the lead narrative.
- Current framing:
  PD-PPO is presented as a deployable prediction-driven scheduler that beats
  dynamic and duty-constrained baselines under dwell/duty constraints; static
  allocation remains a fixed-design reference and limitation topic rather than
  the opening thesis.
- Validation:
  - high-risk defensive phrase scan on sources returned no hits;
  - the same scan on compiled PDF text returned no hits;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded before this log-only update;
  - abstract word count is `213`.

## 2026-06-08 - AI-writing trace audit

- Ran a second audit focused on AI-writing traces rather than only defensive
  wording.
- Edited:
  - `paper/rewrite_sections/06_results.tex`;
  - `paper/rewrite_sections/07_discussion_future_work.tex`.
- Fixed:
  - "The operational result is clear ..." -> direct dynamic-baseline result
    phrasing;
  - "genuine schedule variation" -> concrete mask-variation wording;
  - "The sweep supports ..." -> direct nearby-budget result wording;
  - "This supports reporting ..." -> deployment-constrained baseline-contract
    wording;
  - "The static results sharpen the engineering reading" -> direct fixed-design
    boundary wording.
- Validation:
  - source AI-phrase scans returned no hits for Tier-1 vocabulary, transition
    fillers, chatbot artefacts, hedge stacks, or defensive-paper phrases;
  - compiled-PDF scan found one hit only in a bibliography title, not body prose;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, 36 pages;
  - abstract word count remains `213`.

## 2026-06-08 - Whole-manuscript continuity/style review

- Re-read the active English rewrite in manuscript order after the AI-style and
  defensive-wording passes.
- Edited:
  - `paper/pdppo_crst_rewrite.tex`;
  - `paper/rewrite_sections/02_related_work.tex`;
  - `paper/rewrite_sections/04_framework_protocol.tex`;
  - `paper/rewrite_sections/05_simulation_setup.tex`;
  - `paper/rewrite_sections/06_results.tex`;
  - `paper/rewrite_sections/07_discussion_future_work.tex`;
  - `paper/rewrite_sections/08_conclusion.tex`.
- Fixed continuity issues:
  - abstract no longer repeats "fixed-design reference" in close succession;
  - Related Work now transitions from RL literature to evaluation protocol through
    concrete leakage/baseline risks;
  - method prose around projection and candidate-prior fitting is less redundant;
  - Results no longer reads like a patch after the seed45 duty-constrained
    exception;
  - Discussion and Conclusion now use smoother deployment-valid and design-rule
    phrasing.
- Validation:
  - source and PDF scans found no high-risk phrasing from the review list;
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error pdppo_crst_rewrite.tex`
    succeeded;
  - output `paper/pdppo_crst_rewrite.pdf`, 35 pages;
  - abstract word count is `205`.
