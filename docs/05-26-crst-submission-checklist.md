# CRST Pre-Submission Checklist - Active Rewrite

## Current Status

The current active manuscript source is `paper/main.tex`. It compiles to
`paper/main.pdf`, but the submission package is not yet final because the
protocol-controlled energy-account gate produced only a weak single-seed result
and has been expanded to an `n=5` check, two declarations require author
confirmation, CRediT roles require author confirmation, and the corrected evidence
package has not yet been deposited as a versioned public artifact.

## Required Before Submission

| Item | State | Required action |
| --- | --- | --- |
| Corrected fixed-budget evidence | Ready | Retain the split-protocol tables and conservative static-comparator interpretation. |
| Energy-account learned-policy evidence | n=5 extension running | Do not add comparative learned-policy claims from the weak-positive seed-41 gate; aggregate seeds 41--45 before deciding whether the claim remains in scope. |
| Code/data availability | Provisional wording in manuscript | Before submission, deposit the corrected scripts, configs and supporting result artifacts as a versioned release or archival record, then replace the provisional sentence with the final identifier/link. |
| CRediT contribution statement | Missing | Authors provide approved CRediT roles for Yongzhe Li and Zhuyu Zhang. |
| Funding statement | Missing | Authors provide funder/grant details or approve a no-specific-funding statement. |
| Competing-interests declaration | Missing | Authors approve the applicable declaration text. |
| Independent reviews | In progress | Close all material findings or record the justification for retaining text. |
| Figure 1 provenance/permission | Awaiting author confirmation | Confirm the 3D rendering source, model/texture rights and absence of generative-AI image creation or alteration; replace the artwork if that cannot be confirmed. |

## Active Manuscript Inputs

The submission source package must be assembled from the dependency chain of
`paper/main.tex`, including:

- `paper/main.tex`, `paper/references.bib`, and `paper/highlights.txt`
- `paper/sections/01_introduction.tex` through `paper/sections/08_conclusion.tex`
- `paper/algorithms/pd_ppo.tex`
- `paper/tables/sensor_specs.tex`
- `paper/tables/g1_generator_validation.tex`
- `paper/tables/main_results_v31.tex`
- `paper/tables/condition_results_v31.tex`
- `paper/tables/physical_unit_mae.tex`
- `paper/tables/energy_account_storm_oracle.tex`
- `paper/tables/ablation_full.tex`
- `paper/figures/aws_deployment.png`
- `paper/figures/sensor_state_machine_tikz.tex`
- `paper/figures/data_split_timeline_tikz.tex`
- `paper/figures/pdppo_architecture_tikz.tex`
- `paper/figures/figure3_synthetic_statistics.png`
- `paper/figures/figure5_sensor_timeline.png`
- `paper/figures/figure6_power_error_tradeoff_v31.png`

## Exclude From Submission Package

These files are present for historical/reproducibility context but are not active
submission inputs and must not be bundled as manuscript evidence:

- `paper/raw.tex` and `paper/raw.pdf`: untranslated/legacy companion state with
  superseded fixed-budget narrative.
- `paper/tables/energy_account_curriculum_results.tex`: archived
  non-independent learned-policy diagnostic excluded from the main text.
- `paper/tables/main_results.tex` and `paper/tables/condition_results.tex`:
  superseded pre-repair table variants.
- Legacy PDFs, LaTeX auxiliary files and unused historical figure/table assets
  not referenced by `paper/main.tex`.

## Verification To Repeat At Packaging

- Build `paper/main.tex` from a clean staging directory containing only active
  dependencies.
- Scan the staged source for old fixed-budget values, archived curriculum claims,
  placeholder declarations and undefined references/citations.
- Verify highlights count/length and abstract length.
- Render pages containing all tables, figures and declarations.
- Verify the versioned code/data deposit identifier in the final availability
  statement.
- Verify Figure 1 provenance/permission and submission-artwork compliance; the
  current image is rendered at reduced display width to avoid upscaling its
  1369-pixel source beyond approximately 300 dpi effective resolution.
