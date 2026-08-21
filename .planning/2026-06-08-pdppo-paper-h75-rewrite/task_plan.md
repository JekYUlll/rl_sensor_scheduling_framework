# Task Plan: PD-PPO H75 Paper Rewrite

## Goal

Rewrite the English PD-PPO first-paper manuscript around the new h75 operational
evidence and a more general algorithm-first framing: PD-PPO is a
prediction-driven reinforcement-learning scheduler for power-limited sensing in
extreme environments; the Antarctic AWS case is the controlled scenario used to
instantiate and calibrate the problem.

## Scope

- Active manuscript: `paper/pdppo_crst_rewrite.tex`.
- Active sections: `paper/rewrite_sections/*.tex`.
- Do not edit `paper/raw.tex`.
- Do not edit legacy `paper/sections/*.tex` unless only used as archival
  reference.
- Use h75 10-seed results as the main operational evidence.
- Keep original compact static as a diagnostic shortcut, not as the fair
  deployment comparator.

## Current Phase

Phase 9 is complete; manuscript is compiled after the 10-seed operational
rewrite, budget-sensitivity supplement, candidate-prior ablation, minimum-dwell
sensitivity supplement, vanilla-PPO/PPO-only ablation, and language pass. Phase
10 is complete after the AWBC-off/prior-on ablation isolated training-scaffold
components.

## Phases

### Phase 1: Orient And Lock Sources
- [x] Confirm active English manuscript target.
- [x] Confirm h75 5-seed result source.
- [x] Inspect current section structure, figures, and tables.
- [x] Decide exact manuscript files to edit.
- **Status:** completed

### Phase 2: Evidence Integration
- [x] Replace old env-dwell12 3-seed table with h75 5-seed operational table.
- [x] Update Results lead subsection and comparison claims.
- [x] Preserve h85/historical results only if needed as diagnostic context.
- **Status:** completed

### Phase 3: Algorithm-First Reframing
- [x] Rewrite title/abstract/contributions to introduce PD-PPO generally.
- [x] Rewrite Introduction so the general power-limited predictive scheduling
      problem comes first.
- [x] Move Antarctic AWS/scenario calibration after the general method
      motivation.
- **Status:** completed

### Phase 4: Figure Refresh
- [x] Audit current included figures and figure order.
- [x] Redraw at least the PD-PPO framework/result figures needed for style
      consistency.
- [x] Keep user-rendered AWS image if still useful, but position it as the case
      study setup rather than the paper's conceptual starting point.
- **Status:** completed

### Phase 5: Style And Compile
- [x] Remove AI-like overqualified phrasing and repeated caveats.
- [x] Ensure limitations are localized to Discussion.
- [x] Compile `pdppo_crst_rewrite.tex`.
- [x] Record compile result and remaining issues.
- **Status:** completed

### Phase 6: Evidence Freshness And Operational Diagnostics
- [x] Audit h75 rollout-derived event/non-event diagnostic assets.
- [x] Add a compact event-conditioned deployment diagnostic table.
- [x] Clarify that generator validation is a generator-family boundary check,
      not a final-test policy-selection criterion.
- [x] Compile `pdppo_crst_rewrite.tex` and scan for stale labels.
- **Status:** completed

### Phase 7: Budget Sensitivity Supplement
- [x] Launch same-parameter h75 budget-sensitivity runs at B=1.65 and B=1.75
      for seeds 41--50.
- [x] Monitor remote tmux until all 20 runs finish.
- [x] Sync remote sensitivity artifacts locally.
- [x] Aggregate B=1.65 and B=1.75 separately.
- [x] Decide whether the sensitivity evidence belongs in the main results,
      appendix, or discussion text only.
- [x] Recompile manuscript after adding sensitivity result.
- **Status:** completed

### Phase 8: Candidate-Prior Ablation
- [x] Launch same-parameter B=1.70 h75 deployment runs with
      `--no-use-candidate-prior`, `--prior-kl-coef 0.0`, and
      `--candidate-prior-scale 0.0`.
- [x] Monitor remote tmux until all ten runs finish.
- [x] Sync remote ablation artifacts locally.
- [x] Aggregate the ten-seed ablation and compare it with the main B=1.70
      result.
- [x] Decide whether the ablation belongs in Results, Discussion, or only the
      coordination log.
- **Status:** completed

### Phase 9: Next Supplemental Experiment Selection
- [x] Identify a high-value supplemental experiment that can run within the
      current PD-PPO paper scope.
- [x] Launch remotely if it does not require risky code changes.
- [x] Monitor, sync, aggregate, and decide whether to write it into the paper.
- **Status:** completed

### Phase 10: Continuing Supplement / Audit Loop
- [x] Check remote server state and confirm no current PD-PPO run is unfinished.
- [x] Select the next highest-ROI supplement or manuscript audit.
- [x] Run, sync, aggregate, write, compile, and review if the supplement is
      launched.
- **Status:** completed

## Evidence Lock

- H75 10-seed table:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/env_dwell12_h75_operational_summary_10seed.csv`.
- H75 10-seed comparison table:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/env_dwell12_h75_operational_summary_10seed_comparisons.csv`.
- H75 10-seed rollout audit:
  `reports/v31_split_protocol_no_warmup_hguard_envdwell12_h75_reduced/h75_pdppo_vs_deployable_static_10seed_loss_audit.csv`.
- Claim summary:
  `.planning/2026-06-07-pd-ppo-static-break-recalibration/h75_operational_claim_summary.md`.

## Claim Boundary

Supported:
- PD-PPO learns nondegenerate schedules under symmetric duty/dwell deployment
  constraints.
- PD-PPO beats original dynamic heuristics in all ten h75 seeds.
- PD-PPO beats duty-constrained non-PD-PPO baselines in 9/10 h75 seeds.
- PD-PPO remains close to deployable selected static, but does not stably
  dominate it across ten seeds.

Not supported:
- Full-open superiority.
- Universal dominance over original compact static.
- Every duty-constrained baseline beaten in every seed.

## Decisions

| Decision | Rationale |
|---|---|
| Use `pdppo_crst_rewrite.tex` as the active manuscript | It is the maintained English rewrite and already compiled in the previous paper session. |
| Keep h75 as the main operational evidence | It is locked-parameter 10-seed evidence and better matches the deployment-constraint claim. |
| Do not center Antarctic AWS in the opening | User requested algorithm-first framing; AWS should become the case-study instantiation. |

## Progress Log

- 2026-06-08: Plan created after h75 5-seed evidence lock.
- 2026-06-08: Completed algorithm-first manuscript rewrite, h75 evidence
  integration, h75 operational summary figure generation, highlight sync, PDF
  visual spot-check, and successful LaTeX compile.
- 2026-06-08: Added event-conditioned h75 deployment diagnostics, clarified
  generator-family validation wording, recompiled successfully, and confirmed
  stale internal labels remain absent from the PDF.
- 2026-06-08: Promoted h75 evidence from 5 to 10 seeds, rewrote claims around
  the 10-seed gate, recompiled successfully, and launched B=1.65/B=1.75
  budget-sensitivity supplement on the remote GPU server.
- 2026-06-08: Completed B=1.65/B=1.75 budget sensitivity, added a compact
  sensitivity table to Results, recompiled successfully, and launched the
  no-candidate-prior ablation for B=1.70 seeds 41--50.
- 2026-06-08: Completed no-candidate-prior ablation, added a compact ablation
  table to Results, recompiled successfully, and verified the new table page.
- 2026-06-08: Launched min-dwell sensitivity follow-up at dwell=6; dwell=24 will
  be launched after dwell=6 completes to avoid overloading the four free GPUs.
- 2026-06-08: Completed min-dwell sensitivity at dwell=6/12/24, added the
  sensitivity table and deployment-constraint interpretation to Results and
  Discussion, recompiled successfully, and visually checked the affected pages.
- 2026-06-08: Completed vanilla-PPO/PPO-only ablation, integrated it into the
  training-scaffold ablation table, recompiled successfully, and launched the
  AWBC-off/prior-on ablation.
- 2026-06-08: Completed AWBC-off/prior-on ablation, integrated it into the
  training-scaffold ablation table, recompiled successfully, checked page 24,
  and confirmed no active PD-PPO tmux session remains.
