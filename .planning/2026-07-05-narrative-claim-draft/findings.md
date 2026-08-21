# Findings: Narrative and Claim Draft

## Initial Claim Boundaries
- Main method: PD-PPO is the prediction-driven constrained RL scheduler.
- Main benchmark story: fixed weather backbone, one specialist slot,
  operating-rule-constrained scheduling, fixed evaluation forecaster,
  feasible-mask PPO, and replay/behaviour diagnostics.
- Main positive claim: within this operating-rule benchmark, PD-PPO learns
  useful context-dependent specialist schedules and passes replay/behaviour
  diagnostics under the completed evidence package.
- Forecast-greedy placement: supporting diagnostic baseline. PD-PPO's stable
  advantage over one-step forecast greedy supports the claim that the learned
  policy is not merely a myopic selector.
- Context-alert bandit placement: strong hand-coded context-aware challenge
  baseline. The manuscript must not say PD-PPO or CA-PD-PPO dominates all
  context-aware handcrafted heuristics.
- CA-PD-PPO/dev2 placement: appendix or supplementary-style exploratory
  ablation. It is method-consistent and improves competitiveness, but no
  variant passed the predeclared fresh-final gate, so it is not the main method
  and not confirmatory final evidence.

## Evidence Files To Audit
- `CHANGELOG.md`
- `reports/aggregate/ca_pdppo_bounded_dev2_20260703capdppodev2/variant_summary.md`
- `reports/aggregate/framework_supplement_summary_20260702.md`
- `reports/aggregate/contextaware_pdppo_dev_20260703capdppo/contextaware_pdppo_vs_context_bandit_report_20260703.md`
- `.planning/2026-06-10-eswa-terminology-rewrite/{task_plan.md,findings.md,progress.md}`
- `paper/main.tex`
- `paper/sections/*.tex`
- `paper/tables/*.tex`
- Existing `paper/main.pdf` and LaTeX logs if present.

## Evidence Audit Results
- `CHANGELOG.md` confirms the 2026-07-03 bounded CA-PD-PPO dev2 wave did not
  pass the predeclared fresh-final gate. Fresh final seeds `301--324` were not
  launched. The interpretation remains: CA-PD-PPO is a clean,
  method-consistent improvement that is competitive with the context-alert
  bandit and robustly beats forecast-greedy, but does not support stable
  superiority over the bandit.
- `framework_supplement_summary_20260702.md` reports:
  `forecast_greedy_one_step` over 24 seeds supports sequential non-myopic
  policy value (`24/24` macro wins, mean macro margin `0.155232`);
  `context_alert_bandit_t0p5` challenges broad context-rule dominance
  (`6/24` macro wins, mean macro margin `-0.004533`).
- The same supplement reports two reward-proxy pilots: AoI proxy supports
  forecast reward over AoI in a two-seed pilot, while coverage proxy remains
  competitive in that tiny slice. The paper should not say forecast reward
  alone explains the full gain.
- `contextaware_pdppo_vs_context_bandit_report_20260703.md` reports that
  CA-PD-PPO on development seeds `201--224` improves over original clean
  PD-PPO against the context-alert bandit: macro wins move from `6/24` to
  `13/24`, and mean macro margin moves from `-0.007329` to `+0.008257`.
  It still fails the `15/24` macro-win fresh-final gate.
- `ca_pdppo_bounded_dev2_20260703capdppodev2/variant_summary.md` reports that
  all bounded dev2 variants fail the dev2 gate. Context macro wins range
  `10/24`--`14/24`; context macro means range `0.002763`--`0.006706` for the
  dev2 variants, while `ca_current` remains `13/24`, `0.008257`.
- Existing `paper/main.log` shows the pre-edit `main.pdf` built as `53` pages
  and `2369716` bytes, with no `Undefined`, `Citation`, `Reference`, `Fatal`,
  or `Emergency stop` hits in the searched warning patterns.

## Narrative Design Applied
- Title and abstract now foreground contextual specialist scheduling under
  operating rules rather than a generic microclimate scheduling claim.
- Introduction states the main claim as benchmark-local: fixed weather
  backbone, one specialist slot, forecast-loss-trained feasible-mask PPO,
  replay checks, and behaviour diagnostics.
- Results add a supplementary framework-diagnostics subsection. It separates
  forecast-greedy support from context-alert bandit challenge evidence.
- Appendix now contains a traceable diagnostic summary table linking the
  forecast-greedy/context-alert/AoI/coverage/CA-PD-PPO/dev2 evidence to the
  existing aggregate reports.
- Discussion and conclusion explicitly keep CA-PD-PPO as an exploratory
  extension and keep the context-alert bandit as a serious handcrafted
  challenge baseline.

## Validation Results
- Build command: `latexmk -pdf -interaction=nonstopmode main.tex` from
  `rl_sensor_scheduling_framework/paper`.
- Build result: success, exit code `0`.
- Generated PDF: `paper/main.pdf`, `57` pages, `2377563` bytes.
- Fatal/undefined scan of `main.log`: no hits for fatal errors, emergency stops,
  undefined references, undefined citations, changed-label rerun requests, or
  rerun warnings after the final build.
- BibTeX log retains four existing empty-page warnings:
  `Liu2024`, `Murad2020`, `Pendyala2024`, and `Wei2020`.
- Typesetting notes remain non-fatal: one small global overfull box
  (`1.79993pt`) and several underfull boxes in narrow table columns.
- Source scan and PDF-text scan found no hits for the requested high-risk
  phrases/internal codes:
  `dominates all`, `all context-aware`, `all rule-based`, `fresh final`,
  `fresh-final`, `confirmatory`, `SCENEBAL`, `pdppo_crst`, `CRST`, `metpair`,
  `seed45`, `h075`, or CA-PD-PPO/main-method confusion patterns.

## Backup Status
- Verified present:
  `paper/backups/pre_narrative_claim_draft_20260705_025040/`.
- The manifest records archive creation at `2026-07-05T02:50:40+08:00`,
  archive SHA256
  `8ee1571b87e6f02a7ebd91beee710dcd265134b72b07543edd22ef37ec78b8f0`,
  size `13057770` bytes, and contents covering `main.tex`, `main.pdf`,
  `sections`, `tables`, `figures`, `references.bib`, and `highlights.txt`.
- No additional backup is currently required before scoped manuscript edits,
  because the pre-edit paper source/PDF bundle has already been captured.

## Repository State
- `git status --short` in `rl_sensor_scheduling_framework` shows extensive
  pre-existing dirty/untracked state. This draft pass will avoid reverting or
  cleaning unrelated files and will report only files touched in this session.
- Existing `paper/main.pdf` is present at `2369716` bytes; `main.log`,
  `main.aux`, and `main.blg` also exist.
