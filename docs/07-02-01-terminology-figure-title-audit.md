# 2026-07-02 Terminology, Figure, and Title Audit for the Active ESWA Manuscript

## Scope

This audit covers the active English manuscript only: `paper/main.tex`, the 26 TeX files included from it, the currently compiled `paper/main.pdf`, included table files, and the figure assets that render into the PDF. Historical drafts such as `raw.tex`, `rewrite_sections/`, and `_archive/` are excluded unless they are included by `main.tex`.

The audit follows `docs/07-01-02-deep-research-report.md`: source-native extraction first, rendered-PDF extraction second, and targeted OCR only for included raster figures. The goal is to find reader-visible internal experiment names, suspected AI-coined labels, proprietary/instrument names needing official spelling, abbreviations, and inconsistent variants in charts, titles, captions, headings, and tables.

## Extraction coverage

- Active included TeX files resolved from `main.tex`: 26.
- Source-native records: 237 headings, captions, and table/source rows.
- Rendered PDF records: 303 page headings, table titles, figure captions, and caption-like lines.
- Included figure asset text records: 360 lines. Vector/PDF figures were read with `pdftotext`; the included raster AWS illustration was passed through Tesseract PSM 11 as the targeted OCR fallback.
- First visible figure found in the rendered PDF: page 15, `Figure 1. Forecasting reward scheduling framework. Forecaster fitting, policy training, static-`.
- First table/title object found before that: page 6, `Table 1. Position of this work relative to nearby scheduling and sensing literatures.`.

Evidence files written under `.planning/2026-07-02-terminology-figure-audit/`: `source_records.json`, `pdf_records.json`, `figure_text_records.json`, `cluster_hits.json`, and `candidate_terms.json`.

## Executive findings

1. The first framework figure is too late. The first true figure appears on PDF page 15, after the Introduction, Related Work, a positioning table, much of the formal problem formulation, and the action-space table. This makes the opening feel concept-heavy before readers see the pipeline.
2. The most confusing terms are not isolated typos. They form clusters: fixed/static/mask schedule names, event-label/oracle diagnostic names, macro/static-normalised metric labels, backbone/specialist sensor names, and internal result labels such as 24-seed/evidence/audit/boundary.
3. Several chart titles and captions use internal experiment language or template-like phrasing: “seed evidence”, “behavioural diagnostics”, “reference taxonomy”, “static normalisers”, “candidate mask rollouts”, “event-mask mutual information”, “macro fixed”, and “step-margin”. These labels are meaningful to the experiment log but too opaque for first-time readers.
4. Instrument and dataset labels need official-first-use discipline. `AntAWS`, `AWS`, `FC4 flux`, `FC4 Blowing-Snow Flux Sensor`, `Thermo-hygro`, and `Surface IR` appear in shortened or mixed forms across tables, captions, and the raster platform figure.
5. Some terms are technically defensible but overexposed in high-visibility positions. `PD-PPO`, `masked PPO`, `static-normalised macro score`, and `event-label diagnostic schedule` can stay, but they should be defined once and kept out of chart titles where plain descriptions work better.

## Flagged term clusters and evidence

| ID | Cluster | Category | Count | Priority | Recommended canonical treatment |
|---|---|---|---:|---|---|
| C01_method_name_pdppo_prediction_driven | Method/title framing and forecast-objective terms | internal method name / abbreviation / possible coined title phrase | 18 | High | Keep PD-PPO only as the algorithm name after definition; replace broad “forecasting reward”/“prediction-driven” surfaces with plain “forecasting objective” or “forecast-based scheduling” where possible. |
| C02_macro_staticnorm_margin | Macro score, static normalisation, and margin terms | internal metric label / inconsistent variants | 27 | High | Use “static-normalised macro score” only next to the formal metric. Use “macro forecast score” or “macro loss difference” in captions/figures when readers do not need the formula name. |
| C03_fixed_static_reference | Fixed/static schedule references | comparator naming / inconsistent variants | 29 | High | Use “validation-selected fixed schedule” for the comparator; use “fixed-schedule replay” only for the diagnostic replay object. Avoid mixing fixed mask, static mask, fixed specialist, and selected-static labels. |
| C04_event_label_privileged_diagnostic | Event-label / privileged diagnostic terms | internal diagnostic comparator / inconsistent variants | 4 | High | Use one label: “event-label diagnostic schedule”, with one definition that it uses event labels and is not deployable. |
| C05_rule_dynamic_rotation | Rule-based, rotation, and cycle terms | comparator naming / opaque shorthand | 4 | Medium | Use “rule-based schedule” for the family; reserve “rotation schedule” for the cyclic baseline only. |
| C06_specialist_backbone_slot | Backbone and specialist sensor terms | concept naming / inconsistent variants | 24 | High | Use “weather backbone” and “selectable specialist sensor”. Avoid switching among mandatory backbone, meteorological backbone, background channel, specialist slot, specialist instrument, and expert sensor. |
| C07_operational_feasibility_rules | Operational constraints and feasibility terms | constraint wording / inconsistent variants | 9 | Medium | Use “operational constraints” for power/start-up/minimum-duration rules. Use “feasibility mask” only for the algorithmic masking mechanism. |
| C08_instrument_dataset_names | Dataset and instrument names | proper names / abbreviation / short forms | 25 | High | Use official names at first use: AntAWS dataset; FC4 blowing-snow flux sensor; laser disdrometer; surface infrared temperature sensor; thermo-hygrometer. Short forms only after first use. |
| C09_internal_experiment_versions | Internal experiment/version labels | internal version and seed-count labels | 14 | High | Remove internal version labels from reader-facing text. Use “24 independent seeds” only where seed count is necessary; avoid “24-seed” as a title adjective. |
| C10_ai_style_chart_titles | AI-style chart/caption wording | suspected AI-coined or template-like labels | 42 | Medium | Replace behaviour audit/evidence/boundary/reference taxonomy phrasing with concrete result wording: specialist use by event type, ablation study, sensitivity analysis, comparison schedules. |
| C11_training_internal_regularizers | Training internal regularizers | internal abbreviation / training detail | 1 | Medium | Expand in the method table and keep out of high-level captions and titles. |

### C01_method_name_pdppo_prediction_driven. Method/title framing and forecast-objective terms

- Category: internal method name / abbreviation / possible coined title phrase
- Priority: High
- Recommended treatment: Keep PD-PPO only as the algorithm name after definition; replace broad “forecasting reward”/“prediction-driven” surfaces with plain “forecasting objective” or “forecast-based scheduling” where possible.
- Representative evidence:
  - `sections/03_problem_formulation.tex:3` (heading:subsection): Forecast objective for sensor scheduling
  - `sections/04_framework_protocol.tex:1` (heading:section): PD-PPO Scheduler and Evaluation Protocol
  - `sections/04_framework_protocol.tex:7` (caption): Forecasting reward scheduling framework. Forecaster fitting, policy training, static-reference selection, and final evaluation are separated chronologically. The diagram shows where forecast loss enters trai...
  - `sections/04_framework_protocol.tex:39` (caption): Masked PD-PPO training
  - `tables/pdppo_training_hyperparameters.tex:3` (caption): PD-PPO training hyperparameters for the final benchmark.
  - `tables/baseline_reference_taxonomy.tex:28` (table-row/source): & Retrain PD-PPO variants with selected components removed.
  - `tables/regime_balanced_24seed_summary.tex:17` (table-row/source): PD-PPO step-loss wins vs. fixed mask replay & 24/24 \\
  - `tables/regime_balanced_24seed_summary.tex:18` (table-row/source): PD-PPO macro-score wins vs. fixed mask replay & 24/24 \\

### C02_macro_staticnorm_margin. Macro score, static normalisation, and margin terms

- Category: internal metric label / inconsistent variants
- Priority: High
- Recommended treatment: Use “static-normalised macro score” only next to the formal metric. Use “macro forecast score” or “macro loss difference” in captions/figures when readers do not need the formula name.
- Representative evidence:
  - `tables/notation_summary.tex:18` (table-row/source): MATH & Macro score normalised by static references across event regimes \\
  - `tables/main_protocol_hyperparameters.tex:25` (table-row/source): Checkpoint/reference selection & final policy checkpoint fixed after policy learning; fixed mask selected by lowest static selection macro score \\
  - `sections/06_results.tex:37` (caption): Seed evidence for the final 24-seed benchmark. Panel A sorts seeds by fixed mask step margin and overlays the corresponding macro margin and the range of prespecified step margin thresholds. Panel B shows se...
  - `sections/06_results.tex:58` (caption): Event type diagnostics for the final benchmark. Panel A is the primary specialist selection frequency heatmap: each cell reports the mean fraction of steps, averaged over 24 seeds, in which the learned polic...
  - `tables/regime_balanced_24seed_summary.tex:3` (caption): Final 24-seed evidence for the macro score across regimes.
  - `tables/regime_balanced_24seed_summary.tex:15` (table-row/source): Event label replay improves over fixed mask replay, macro score & 24/24 \\
  - `tables/regime_balanced_24seed_summary.tex:20` (table-row/source): Mean step margin vs. best selected baseline & 0.1494 \\
  - `tables/regime_balanced_24seed_summary.tex:21` (table-row/source): Median step margin vs. best selected baseline & 0.1060 \\

### C03_fixed_static_reference. Fixed/static schedule references

- Category: comparator naming / inconsistent variants
- Priority: High
- Recommended treatment: Use “validation-selected fixed schedule” for the comparator; use “fixed-schedule replay” only for the diagnostic replay object. Avoid mixing fixed mask, static mask, fixed specialist, and selected-static labels.
- Representative evidence:
  - `tables/online_observability_audit.tex:17` (table-row/source): Static normalisers and selected fixed mask & Validation/static selection partition & No & Reference selection and metric normalisation only \\
  - `tables/baseline_reference_taxonomy.tex:13` (table-row/source): & Select one feasible fixed mask before final testing.
  - `tables/baseline_reference_taxonomy.tex:19` (table-row/source): & Replay one constant mask without duty guards or fallback rotation.
  - `tables/main_protocol_hyperparameters.tex:25` (table-row/source): Checkpoint/reference selection & final policy checkpoint fixed after policy learning; fixed mask selected by lowest static selection macro score \\
  - `tables/static_mask_selection_summary.tex:3` (caption): Validation-selected fixed mask references over the final seeds.
  - `sections/06_results.tex:37` (caption): Seed evidence for the final 24-seed benchmark. Panel A sorts seeds by fixed mask step margin and overlays the corresponding macro margin and the range of prespecified step margin thresholds. Panel B shows se...
  - `sections/06_results.tex:58` (caption): Event type diagnostics for the final benchmark. Panel A is the primary specialist selection frequency heatmap: each cell reports the mean fraction of steps, averaged over 24 seeds, in which the learned polic...
  - `sections/06_results.tex:99` (caption): Behavioural diagnostics for the final 24-seed benchmark. Panel A reports seed distributions of three scalar diagnostics: mask entropy for action diversity, event-mask mutual information for state dependence ...

### C04_event_label_privileged_diagnostic. Event-label / privileged diagnostic terms

- Category: internal diagnostic comparator / inconsistent variants
- Priority: High
- Recommended treatment: Use one label: “event-label diagnostic schedule”, with one definition that it uses event labels and is not deployable.
- Representative evidence:
  - `tables/baseline_reference_taxonomy.tex:23` (table-row/source): & Privileged-information diagnostic, not the learned controller. \\
  - `tables/regime_balanced_24seed_summary.tex:14` (table-row/source): Event label replay improves over fixed mask replay, step loss & 24/24 \\
  - `tables/regime_balanced_24seed_summary.tex:15` (table-row/source): Event label replay improves over fixed mask replay, macro score & 24/24 \\
  - `PDF:12` (heading-or-numbered-line): 0.968, 0.747, and 0.797. The event label replay strictly improves over the fixed

### C05_rule_dynamic_rotation. Rule-based, rotation, and cycle terms

- Category: comparator naming / opaque shorthand
- Priority: Medium
- Recommended treatment: Use “rule-based schedule” for the family; reserve “rotation schedule” for the cyclic baseline only.
- Representative evidence:
  - `tables/baseline_reference_taxonomy.tex:19` (table-row/source): & Replay one constant mask without duty guards or fallback rotation.
  - `sections/06_results.tex:99` (caption): Behavioural diagnostics for the final 24-seed benchmark. Panel A reports seed distributions of three scalar diagnostics: mask entropy for action diversity, event-mask mutual information for state dependence ...
  - `sections/appendix_theory.tex:80` (heading:subsection): Scheduling from state evidence and simple cycles
  - `PDF:43` (heading-or-numbered-line): Appendix A.3. Scheduling from state evidence and simple cycles

### C06_specialist_backbone_slot. Backbone and specialist sensor terms

- Category: concept naming / inconsistent variants
- Priority: High
- Recommended treatment: Use “weather backbone” and “selectable specialist sensor”. Avoid switching among mandatory backbone, meteorological backbone, background channel, specialist slot, specialist instrument, and expert sensor.
- Representative evidence:
  - `tables/action_space_instantiation.tex:11` (table-row/source): 0 & Weather backbone & 0.25 & Backbone only & 0/24 \\
  - `tables/action_space_instantiation.tex:12` (table-row/source): 1 & Weather backbone + Radiometer & 0.33 & Radiation context & 0/24 \\
  - `tables/action_space_instantiation.tex:13` (table-row/source): 2 & Weather backbone + Thermo-hygro & 0.43 & Non-event thermo-hygro context & 0/24 \\
  - `tables/action_space_instantiation.tex:14` (table-row/source): 3 & Weather backbone + Surface IR & 0.74 & Thermal-event specialist & 9/24 \\
  - `tables/action_space_instantiation.tex:15` (table-row/source): 4 & Weather backbone + Laser disdrometer & 0.74 & Particle-event specialist & 2/24 \\
  - `tables/action_space_instantiation.tex:16` (table-row/source): 5 & Weather backbone + FC4 flux & 0.74 & Flux-event specialist & 13/24 \\
  - `sections/05_simulation_setup.tex:24` (caption): Author-rendered AWS platform used to motivate the benchmark sensing system. The simulator represents the platform as a required meteorological backbone plus one specialist sensing slot, so the image is a phy...
  - `tables/sensor_specs.tex:12` (table-row/source): 1 & Meteorological backbone

### C07_operational_feasibility_rules. Operational constraints and feasibility terms

- Category: constraint wording / inconsistent variants
- Priority: Medium
- Recommended treatment: Use “operational constraints” for power/start-up/minimum-duration rules. Use “feasibility mask” only for the algorithmic masking mechanism.
- Representative evidence:
  - `tables/notation_summary.tex:14` (table-row/source): MATH & Candidate-index set that remains feasible at epoch MATH after power, startup, and minimum-duration checks \\
  - `tables/notation_summary.tex:19` (table-row/source): MATH & Switching or minimum-duration penalty term used during policy learning \\
  - `sections/04_framework_protocol.tex:14` (heading:subsection): State, action, and feasibility masking
  - `tables/action_space_instantiation.tex:3` (caption): Feasible action masks in the final single specialist benchmark.
  - `tables/online_observability_audit.tex:13` (table-row/source): Previous mask and warm-up state & Scheduler runtime & Yes & Feasibility mask and switching/minimum-duration accounting \\
  - `tables/pdppo_training_hyperparameters.tex:18` (table-row/source): Minimum on-time / switch cost & 6 epochs / 0.001 \\
  - `tables/main_protocol_hyperparameters.tex:20` (table-row/source): Minimum on-time & 6 epochs \\
  - `PDF:15` (heading-or-numbered-line): 4.1. State, action, and feasibility masking

### C08_instrument_dataset_names. Dataset and instrument names

- Category: proper names / abbreviation / short forms
- Priority: High
- Recommended treatment: Use official names at first use: AntAWS dataset; FC4 blowing-snow flux sensor; laser disdrometer; surface infrared temperature sensor; thermo-hygrometer. Short forms only after first use.
- Representative evidence:
  - `tables/action_space_instantiation.tex:13` (table-row/source): 2 & Weather backbone + Thermo-hygro & 0.43 & Non-event thermo-hygro context & 0/24 \\
  - `tables/action_space_instantiation.tex:15` (table-row/source): 4 & Weather backbone + Laser disdrometer & 0.74 & Particle-event specialist & 2/24 \\
  - `tables/action_space_instantiation.tex:16` (table-row/source): 5 & Weather backbone + FC4 flux & 0.74 & Flux-event specialist & 13/24 \\
  - `sections/05_simulation_setup.tex:24` (caption): Author-rendered AWS platform used to motivate the benchmark sensing system. The simulator represents the platform as a required meteorological backbone plus one specialist sensing slot, so the image is a phy...
  - `tables/sensor_specs.tex:18` (table-row/source): 3 & Shielded thermo-hygro
  - `tables/sensor_specs.tex:24` (table-row/source): 5 & Laser disdrometer
  - `tables/sensor_specs.tex:27` (table-row/source): 6 & FC4 flux
  - `tables/static_mask_selection_summary.tex:10` (table-row/source): 5 & Weather backbone + FC4 flux & 13/24 & 0.899 & 0.74 \\

### C09_internal_experiment_versions. Internal experiment/version labels

- Category: internal version and seed-count labels
- Priority: High
- Recommended treatment: Remove internal version labels from reader-facing text. Use “24 independent seeds” only where seed count is necessary; avoid “24-seed” as a title adjective.
- Representative evidence:
  - `sections/06_results.tex:37` (caption): Seed evidence for the final 24-seed benchmark. Panel A sorts seeds by fixed mask step margin and overlays the corresponding macro margin and the range of prespecified step margin thresholds. Panel B shows se...
  - `sections/06_results.tex:58` (caption): Event type diagnostics for the final benchmark. Panel A is the primary specialist selection frequency heatmap: each cell reports the mean fraction of steps, averaged over 24 seeds, in which the learned polic...
  - `sections/06_results.tex:99` (caption): Behavioural diagnostics for the final 24-seed benchmark. Panel A reports seed distributions of three scalar diagnostics: mask entropy for action diversity, event-mask mutual information for state dependence ...
  - `tables/regime_balanced_24seed_summary.tex:3` (caption): Final 24-seed evidence for the macro score across regimes.
  - `tables/mechanism_ablation_summary.tex:3` (caption): Mechanism ablation over 24 seeds with continuous macro margins.
  - `tables/event_mix_robustness_summary.tex:13` (table-row/source): Lower-flux mixture & 0.45/0.10/0.45 & 6 & 0.092 / 0.102 & 0.088 / 0.098 & 6/6 / 6/6 \\
  - `sections/appendix_theory.tex:170` (caption): Mechanism ablation over 24 seeds. Panel A shows seed macro margin distributions for the full policy and three training variants. Panel B reports paired bootstrap confidence intervals for the change in mean m...
  - `PDF:30` (caption-or-table-title): Table 10. Final 24-seed evidence for the macro score across regimes.

### C10_ai_style_chart_titles. AI-style chart/caption wording

- Category: suspected AI-coined or template-like labels
- Priority: Medium
- Recommended treatment: Replace behaviour audit/evidence/boundary/reference taxonomy phrasing with concrete result wording: specialist use by event type, ablation study, sensitivity analysis, comparison schedules.
- Representative evidence:
  - `tables/online_observability_audit.tex:3` (caption): Online observability audit for scheduler inputs and diagnostics.
  - `tables/online_observability_audit.tex:15` (table-row/source): Event type labels & Simulator truth sequence & No & Training auxiliary labels and diagnostic grouping only \\
  - `tables/baseline_reference_taxonomy.tex:23` (table-row/source): & Privileged-information diagnostic, not the learned controller. \\
  - `tables/sensor_specs.tex:22` (table-row/source): & snow-surface temperature, thermal-regime evidence
  - `tables/sensor_specs.tex:25` (table-row/source): & particle diameter, particle velocity, particle microstructure evidence
  - `tables/sensor_specs.tex:28` (table-row/source): & snow mass flux, snow-flux evidence
  - `sections/06_results.tex:46` (heading:subsection): Event type decomposition and diagnostic replay
  - `sections/06_results.tex:93` (heading:subsection): Behavioural diagnostics

### C11_training_internal_regularizers. Training internal regularizers

- Category: internal abbreviation / training detail
- Priority: Medium
- Recommended treatment: Expand in the method table and keep out of high-level captions and titles.
- Representative evidence:
  - `tables/mechanism_ablation_summary.tex:18` (table-row/source): No event context auxiliary signal & 21/24 & 24/24 & 0.0665 & 0.0713 & -0.0045 \\

## High-visibility figure and table wording problems

| Surface | Current wording problem | Why it is difficult | Suggested direction |
|---|---|---|---|
| Figure 1 caption and figure text | `Forecasting reward scheduling framework`, `Static/reference selection partition`, `static normalisers`, `candidate mask rollouts`, `Behaviour checks` | The figure is the first pipeline overview but uses internal labels and appears only on page 15. | Move earlier; retitle as an overview of the evaluation protocol; replace internal labels with plain stage names. |
| Figure 4 | `Seed evidence`, `Step and macro margins`, `Step threshold`, `Macro refs`, `Macro fixed`, `positive favours PD-PPO` | The result is important, but chart labels assume the reader already knows step/macro/static-normalised notation. | Use “Forecast loss difference” and “Macro score difference”; spell out reference schedule in the caption. |
| Figure 5 | `Event type diagnostics`, `Specialist-selection frequency`, `Margin vs. fixed mask` | Diagnostic and fixed-mask language blurs method result, comparator, and analysis category. | Use “Specialist use by event type” and “Loss difference relative to the fixed schedule”. |
| Figure 6 | `Behavioural diagnostics`, `event-mask mutual information`, `no-imitation`, `co-locates` | Reads like an internal audit notebook; not natural for a paper result title. | Use “Action diversity and event dependence” or “Checks on learned specialist choices”. |
| Table 6 | `Comparison references used in the evaluation`, `Event-aware diagnostic replay`, `Fixed-mask replay` | This is conceptually useful but labels are inconsistent with surrounding `event label replay` / `fixed mask` prose. | Use one comparator vocabulary: “fixed-schedule replay” and “event-label diagnostic schedule”. |
| Table 10 | `Final 24-seed evidence`, `macro score across regimes`, `fixed mask replay` | The title foregrounds seed count and evidence rather than the result being measured. | Use “Final comparison with the validation-selected fixed schedule”. |
| Appendix headings | `Generator Validation Checks`, `Training Diagnostics`, `Fixed mask reference selection` | Mostly acceptable, but capitalization and comparator naming should match the body. | Use sentence case and the same “fixed schedule” vocabulary. |

## Proposed normalization glossary

| Concept | Use this canonical form | Avoid / normalize away | First-use rule |
|---|---|---|---|
| Method | PD-PPO | Prediction-driven RL as a repeated high-visibility phrase; PD PPO; PD-PPO scheduler in every caption | Define once as the masked PPO policy used for this forecasting objective. |
| Problem objective | forecasting objective / forecast-based scheduling | forecasting reward scheduling, prediction-driven scheduling as a general field label | Introduce in Introduction; use plain wording afterward. |
| Main comparator | validation-selected fixed schedule | fixed mask, static mask, fixed specialist, selected-static reference | Define in Methods/Protocol; captions can shorten to fixed schedule. |
| Fixed replay diagnostic | fixed-schedule replay | fixed mask replay / fixed-mask replay | Use only when replaying a constant schedule as a diagnostic. |
| Privileged diagnostic | event-label diagnostic schedule | event label replay, event-aware diagnostic replay, oracle dynamic policy in reader-facing captions | Define once as using event labels and not deployable. |
| Main metric | static-normalised macro score | staticnorm, Mstaticnorm, static normalisers, macro margin without context | Use formal term near equations; captions can use macro forecast score. |
| Step metric | forecast loss difference | step margin, ordinary step, step threshold unless formula-specific | Use formula language only near metric definition. |
| Sensor structure | weather backbone and selectable specialist sensor | mandatory backbone, meteorological backbone, background channel, specialist slot, specialist instrument, expert sensor | Define at the problem setup; repeat consistently. |
| Constraints | operational constraints | hard operating rules, operating rules, schedule-validity check | List power, start-up, and minimum-duration constraints at first use. |
| Dataset | AntAWS dataset | AntAWS anchor, AntAWS scalar variables without explanation | Cite/expand at first occurrence in application description. |
| Instrument | FC4 blowing-snow flux sensor | FC4 flux as first mention; FC4 Blowing-Snow mixed title case | Use official/full form at first mention; later short form “FC4 flux sensor”. |

## Editing plan derived from this audit

1. Move the framework overview figure from Section 4 to the Introduction, immediately before the contribution list or just after the problem setup. This gives readers a visual map by page 3–4 instead of page 15.
2. Rewrite the Introduction slightly to remove repeated protocol explanations after the figure moves. Keep the central setting, result preview, and contributions, but reduce repeated check lists.
3. Standardize comparator naming across Table 6, Table 10, Results captions, and discussion: fixed schedule / fixed-schedule replay / event-label diagnostic schedule.
4. Standardize backbone/specialist wording across action-space tables, sensor tables, figure captions, and setup prose.
5. Replace chart and table titles that sound like internal notebooks: “seed evidence”, “behavioural diagnostics”, “reference taxonomy”, “mechanism robustness”, “static normalisers”, “step-margin”.
6. Regenerate or patch figure assets only when reader-visible labels inside the figure contain deprecated terms. Source scripts should be patched when available so future regeneration does not reintroduce old labels.
7. Compile and run residual scans for deprecated variants over active included sources and rendered PDF text. Report any intentional remaining terms, especially formal metric names or official dataset/instrument names.

## Terms intentionally allowed after normalization

- `PD-PPO`: allowed as the method name after definition.
- `PPO`: allowed in method and reference contexts.
- `AntAWS`: allowed as the official dataset name after citation/first use.
- `FC4`: allowed only after `FC4 blowing-snow flux sensor` is introduced.
- `static-normalised macro score`: allowed in formula/metric contexts because it is the formal metric, but avoid using it as a chart-title slogan.
- `event-label diagnostic schedule`: allowed as a carefully defined non-deployable diagnostic comparator.

## Verification commands for the follow-up edit

```bash
cd /home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftotext main.pdf - | grep -E "SCENEBAL|V3\.1|metpair|static normalisers|seed evidence|Behavioural diagnostics|fixed mask replay|event-aware diagnostic replay|event label replay"
git diff --check -- <touched-files>
```
