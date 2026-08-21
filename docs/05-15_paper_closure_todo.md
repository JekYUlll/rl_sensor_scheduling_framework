# Paper Closure TODO

Last updated: 2026-05-15

Goal: close the manuscript using the completed V3.1 S2 results, while keeping every
claim traceable to existing experiments. The interrupted V3.1-aligned A1/A2/H1 rerun
is treated as non-blocking unless it is later resumed and completed.

## A. Blocking Consistency Fixes

- [x] Create this checklist and use it as the closure tracker.
- [ ] Remove or replace all submission placeholders in `paper/paper.tex`.
  Remaining: author names, affiliations, corresponding-author contact, and PDF
  metadata require user-provided submission information.
- [x] Replace the incorrect "Lagrangian dual convergence" claim with the actual
  action-masking feasibility proposition.
- [x] Remove any statement implying DQN is part of the V3.1 main-result table.
- [x] Make the scheduling time-step language consistent with the V3.1 simulator:
  use "logical scheduling epoch/step", not "one-second decision epoch".
- [x] Align the PD-PPO method description with the implementation in
  `src/v2/custom_ppo.py` and `scripts/25_v2_train_custom_ppo.py`.
- [x] Remove empty appendix placeholders or convert them into real appendix material.

## B. Experimental Narrative Fixes

- [x] Keep V3.1 S2 as the only main performance result.
- [x] Keep V3.1 event-fraction strata as the condition-stratified result.
- [x] Keep V3.1 generator validation as the simulation credibility result.
- [x] Explicitly label V2 A1/A2/H1 and DQN experiments as development diagnostics.
- [ ] Do not claim V3.1-aligned A1/A2/H1 completion unless the interrupted rerun is
  resumed and collected.
- [ ] Do not claim significant superiority over round-robin; write mean improvement
  only.
- [ ] Preserve full observation as an unconstrained upper bound, not a feasible policy.

## C. Language and Structure

- [ ] Tighten the abstract so it states only completed evidence.
- [ ] Shorten the introduction contribution list and remove redundant architecture
  details already covered in Methodology.
- [ ] Improve transitions between V3.1 main results and V2 diagnostics.
- [ ] Make limitations explicit but not self-undermining.
- [ ] Ensure "PD-PPO" is used consistently in all formal text.

## D. Figure, Table, and Reference Checks

- [x] Recompile the paper after edits.
- [ ] Inspect Figures 2, 3, 5, and the data-split timeline in the generated PDF.
- [ ] Check that all table labels are unique and referenced correctly.
- [ ] Search for stale V2 numbers outside diagnostic sections.
- [x] Search for `FILL`, square-bracket placeholders, `Custom PPO`, and stale
  "future S2" wording.
  Remaining bracket placeholders are limited to author/affiliation metadata.
- [ ] Check references for unused or missing bibliography entries.

## E. Final Snapshot

- [ ] Commit the paper subrepo after the closure pass.
- [ ] Commit/update the main repo pointer and closure docs.
- [ ] Save a final local PDF snapshot for review.
