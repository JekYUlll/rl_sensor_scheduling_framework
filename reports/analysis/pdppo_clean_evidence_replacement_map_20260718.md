# PD-PPO clean-evidence replacement map

This ledger freezes the manuscript update boundary before the actor-only runs
finish. It prevents historical classifier-router results from being mixed with
the clean masked-PPO evidence. No item in the **replace** list may remain in the
submission build unless it is regenerated from the selected no-router policy.

## Evidence contract

- Primary policy: feasibility-masked PPO actor; hard subtype routing disabled.
- Online context: sample-and-hold history, masks, AoI/runtime state, previous
  action, time/budget state, and supplied noisy warning scores.
- Privileged subtype labels: policy-learning guide/auxiliary targets, reward
  stratification, and offline diagnostic grouping only.
- Development seeds: 117 and 118 select between plain and context-encoder
  actor-only candidates. Seeds 119--140 form the unchanged post-pilot
  replication after the architecture is frozen.
- Final score: the same seed-specific frozen forecaster, final starts, candidate
  masks, validation-selected static schedule, and calibration normalizers for
  every matched comparator.

## Retain

- `paper/sections/02_related_work.tex`: literature positioning.
- Result-independent definitions and propositions in
  `paper/sections/03_problem_formulation.tex`.
- The sample-and-hold observation chain, feasibility mask, reward definition,
  training-only supervision boundary, and chronological protocol in
  `paper/sections/04_framework_protocol.tex`.
- `paper/tables/action_space_instantiation.tex`,
  `paper/tables/online_observability_audit.tex`,
  `paper/tables/main_protocol_hyperparameters.tex`, simulator parameter tables,
  observation-model table, and executable baseline table, subject to a final
  configuration cross-check.
- Generator validation assets whose data-generation settings are unchanged.

## Replace from clean 24-seed evidence

1. Main performance evidence
   - `paper/tables/regime_balanced_24seed_summary.tex`
   - `paper/figure_regime_balanced_24seed_evidence.pdf` and its source data
   - all main-result numbers in `paper/sections/06_results.tex`
   - summary claims in `paper/main.tex`, `paper/sections/01_introduction.tex`,
     and `paper/sections/08_conclusion.tex`

2. Policy behavior and event interpretation
   - `paper/tables/event_type_loss_decomposition.tex`
   - `paper/figure_event_type_diagnostics.pdf`
   - `paper/figure_behavior_audit.pdf`
   - every statement that all seeds use a particular specialist by subtype or
     pass a behavior gate

3. Comparator and objective isolation
   - replace the historical component-ablation narrative and
     `paper/tables/mechanism_ablation_summary.tex` with the matched
     forecast/AoI/uncertainty PPO comparison;
   - add the same-mask Double-DQN result as the second learned-policy baseline;
   - regenerate one-step forecast-greedy and context-alert comparisons on the
     selected clean rollouts before retaining their numerical claims.

4. Privileged diagnostic
   - rerun event-label replay under the clean protocol or remove its numerical
     comparison from the main text;
   - remove old values `0.0824`, `0.0770`, `19/24`, and `0.0054` from
     `paper/sections/03_problem_formulation.tex`,
     `paper/sections/06_results.tex`, and
     `paper/sections/appendix_theory.tex` unless regenerated.

5. Robustness and optimization diagnostics
   - remove the old changed-mixture table and prose unless an actor-only run is
     completed under the frozen architecture;
   - regenerate training diagnostics from the selected clean histories;
   - add one compact actor-only robustness check only after reward controls and
     Double-DQN are complete.

## Remove if not regenerated

- Historical component-ablation figure
  `paper/figure_mechanism_robustness.pdf` and Appendix mechanism section.
- Historical changed-mixture appendix and its limitations text.
- Development-only context-extension appendix if the context encoder becomes
  the primary method; otherwise replace it with a brief, non-numerical method
  note or remove it.
- Any table or figure generated from `corrected24`, `corrected24r1`, or
  `corrected24r2` partial directories.

## Final manuscript pass order

1. Freeze and checksum seed-level clean evidence.
2. Generate performance, behavior, objective-control, learned-baseline, and
   robustness tables/figures from that evidence.
3. Rewrite Results as one continuous evidence hierarchy.
4. Update Discussion to interpret the new hierarchy rather than restating old
   component ablations.
5. Update abstract, introduction contributions, theory cross-references, and
   conclusion last.
6. Compile, scan source and PDF for superseded numbers, and verify every claim
   against the versioned aggregate.

## Source-level stale-evidence inventory

The 2026-07-18 source scan found historical numerical dependencies outside the
obvious Results table. These files must be regenerated, rewritten, or removed
as one dependency set after the clean 24-seed collectors finish:

- summary layer: `paper/main.tex`, `paper/sections/01_introduction.tex`, and
  `paper/sections/08_conclusion.tex`;
- theoretical empirical instances: `paper/sections/03_problem_formulation.tex`
  and `paper/sections/appendix_theory.tex` (old event-label values `0.0824`,
  `0.0770`, `19/24`, and `0.0054`);
- main evidence narrative: every numerical paragraph in
  `paper/sections/06_results.tex`;
- interpretation: historical component-ablation and changed-mixture statements
  in `paper/sections/07_discussion_future_work.tex`;
- primary tables: `paper/tables/regime_balanced_24seed_summary.tex`,
  `paper/tables/event_type_loss_decomposition.tex`, and
  `paper/tables/action_space_instantiation.tex`;
- superseded diagnostic tables: `paper/tables/mechanism_ablation_summary.tex`,
  `paper/tables/event_mix_robustness_summary.tex`,
  `paper/tables/framework_diagnostics_summary.tex`, and the older
  `metpair_*`/`scenebal2_*` summary tables if still included;
- configuration-derived checks: `paper/tables/static_mask_selection_summary.tex`
  and `paper/tables/tcn_forecaster_validation.tex` must be retained only after
  confirming that their source assets are unchanged by the clean-policy restart;
- figures: the old main performance, event-type, behavior, mechanism-ablation,
  and changed-mixture assets must not survive merely because the files still
  compile.

The replacement hierarchy is fixed as: clean learned-policy performance,
matched reward controls and Double-DQN, behavior/event interpretation,
handcrafted/myopic/privileged references, and independent-forecaster rescore.
The final source scan must reject every historical number above even when it
appears only in an appendix caption or theory cross-reference.
