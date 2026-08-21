# CRST Draft Verification Memo - 2026-05-26

## Scope

This memo audits the current active rewrite source, `paper/main.tex` with
`paper/sections/*.tex`, after replacement of the fixed-budget results with the
chronological split-protocol evidence. It is an interim verification: any
promotion of pending energy-account learned-policy results requires this audit to
be rerun.

## Passed Checks

| Check | Evidence | Status |
| --- | --- | --- |
| Active fixed-budget table source | `paper/sections/06_experiments.tex` includes `tables/main_results_v31`; table cites `reports/v31_split_protocol_main/v31_s2_main_stats.csv` | Pass |
| Active conditioned table source | Section 6 includes `tables/condition_results_v31`; table cites the corrected split-protocol condition CSV | Pass |
| Excluded non-independent comparison | `energy_account_curriculum_results.tex` is not included in the active manuscript; Section 6 explicitly excludes archived curriculum outputs | Pass |
| Stale fixed-budget headline value | Active source contains no old `0.1620` headline result | Pass |
| Statistical wording | `v31_s2_significance.csv` records significant adjusted comparisons against round-robin, AoI and random at all three budgets; all comparisons against validation-selected static have adjusted `p = 1.0` | Pass |
| Abstract length | Mechanical count after the page-1 prose repair: 238 words, within the recorded CRST 250-word limit | Pass |
| Highlights | Five bullets; maximum bullet length is 79 characters | Pass |
| Source build | `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` completed successfully after the abstract repair; output is 47 pages | Pass |
| Visual check after prose repair | Rendered PDF page 34 contains the repaired hyperparameter paragraph without overlap, clipping or duplicated text | Pass |
| Declarations already present | Code and Data Availability corrected to distinguish the public development snapshot from the not-yet-deposited manuscript evidence package; Declaration of Generative AI and AI-Assisted Technologies | Pass |

## Repair Performed

The Hyperparameter Sensitivity paragraph in
`paper/sections/06_experiments.tex` contained a duplicated incomplete clause.
It has been reduced to one complete, evidence-preserving sentence. No numerical
result, interpretation boundary or figure/table source changed.

An independent evidence review identified an unsupported statement that the
reported PD-PPO hyperparameters had been chosen by held-out validation grid
search. `paper/sections/05_methodology.tex` now describes the corrected
fixed-budget configuration as pre-specified and limits H1 to a post hoc local
sensitivity diagnostic that does not select from final-test outcomes.

The same review identified ambiguity in the event-stratified table caption.
`scripts/54_rebuild_table3_main_results.py` and the regenerated
`paper/tables/condition_results_v31.tex` now state that the event labels define
the strata and are available as simulated context to PD-PPO; operational
deployment would require inference of that context from available measurements.
The evidence reviewer performed a focused re-check after these edits and
confirmed that both findings are resolved.

## Open Submission Items

| Item | Reason it remains open |
| --- | --- |
| CRediT author-contribution statement | Requires author-approved role allocation; no formal statement is present in the active source. |
| Funding declaration | Requires author-confirmed funding information or explicit no-funding wording. |
| Declaration of competing interests | Requires author confirmation; it cannot be inferred from experiment artifacts. |
| Pending energy-account learned-policy gate | The corrected `semi_markov` split-protocol gate is still running; a single seed is only a scale/no-scale gate, not manuscript comparative evidence. |
| Independent review closure | Two read-only reviewers have been launched for claim-boundary and CRST-compliance checks; findings must be addressed or documented. |
| Versioned code/data deposit | The public repository remote HEAD predates the corrected split-protocol scripts and result package; these artifacts must be deposited and cited before final submission. |
| Figure 1 provenance and permission | The displayed asset is an English-labelled conceptual 3D rendering, not a field photograph; authors must confirm rendering provenance, rights and no prohibited generative-image use. It is now included at reduced width to avoid upscaling the 1369-pixel source beyond approximately 300 dpi effective resolution. |

## Non-Blocking Build Notes

The latest XeLaTeX log contains 15 overfull-box and 2 underfull-box warnings.
No undefined-reference or undefined-citation message was found. The warnings are
layout polish items unless later visual review exposes unreadable content.
After correction of the availability statement, `paper/main.pdf` was rebuilt and
page 38 was visually inspected; the revised statement is legible and does not
introduce a layout defect.
After the evidence-review repairs, pages 24 and 28 were rendered and inspected;
the pre-specified-configuration wording and event-context caption render cleanly.
After a later page-1 visual audit found a broken abstract sentence, `paper/main.tex`
was repaired and rebuilt; the rendered first page now reads cleanly.
