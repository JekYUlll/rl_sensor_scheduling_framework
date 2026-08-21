# Task Plan: Narrative and Claim Draft

## Goal
Produce a compileable English manuscript draft for
`rl_sensor_scheduling_framework/paper/main.tex` that converges the paper around
PD-PPO as a prediction-driven constrained RL scheduling framework. The draft
must use only existing evidence and completed CA-PD-PPO/dev2 conclusions, keep
CA-PD-PPO as an exploratory appendix/supplementary extension, and preserve the
context-alert bandit as a serious diagnostic challenge baseline.

## Scope
- Active paper source: `paper/main.tex`, `paper/sections/*.tex`, and any
  included `paper/tables/*.tex` or appendix material needed for this draft.
- Required backup to verify before edits:
  `paper/backups/pre_narrative_claim_draft_20260705_025040/`.
- No new large-scale remote experiments.
- Do not promote CA-PD-PPO to the main method.
- Do not claim PD-PPO or CA-PD-PPO dominates all context-aware handcrafted
  heuristics.

## Phases
- [x] Evidence audit: verify the backup, inspect required evidence/context
  files, and record current claim boundaries.
- [x] Narrative design: define the new main storyline and placement of
  PD-PPO, forecast-greedy, context-alert bandit, and CA-PD-PPO/dev2.
- [x] Source editing: revise the high-visibility manuscript sections and any
  necessary appendix/table inputs using existing evidence only.
- [x] Figure/table/appendix integration: add or adjust compact evidence
  summaries only when they are traceable to existing Markdown/CSV/JSON
  artifacts.
- [x] Compile validation: rebuild `paper/main.pdf` using the project LaTeX
  command and record fatal errors, page count, PDF size, and undefined
  references/citations.
- [x] Residual wording scan: search for high-risk claim phrases, internal
  experiment codes, and CA-PD-PPO/main-method confusion; clean targeted issues.
- [x] Delivery summary: record changed files, compile/scan results, remaining
  non-final-submission work, and hand off the draft status.

## Status Notes
- Created 2026-07-05 for the user-requested narrative/claim draft pass.
- This plan supersedes no historical plan; it is a focused manuscript drafting
  layer over the completed ESWA rewrite and CA-PD-PPO/dev2 evidence.
- Completed 2026-07-05 with a compileable narrative/claim draft.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| Initial risky-phrase scan still matched negated boundary phrases such as `all context-aware` and a `fresh-final` gate label. | 1 | Rephrased to avoid the exact high-risk wording while preserving the claim boundary. |
| New diagnostic table produced minor overfull boxes from long `texttt` baseline names. | 1 | Replaced long code-style names with short reader-facing labels. |
