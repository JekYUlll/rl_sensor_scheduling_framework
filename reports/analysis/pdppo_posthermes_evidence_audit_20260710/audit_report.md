# PD-PPO Post-Hermes Experiment-Evidence Audit

Date: 2026-07-10

## Scope and Result Binding

This is a read-only audit of the active ESWA manuscript (`paper/main.tex` and
`paper/sections/*.tex`) and the final SCENEBAL-2 aggregate over seeds 117--140.
No manuscript source, model checkpoint, or completed experimental result was
changed during the audit.

The local and remote copies of the binding aggregate agree exactly:

- `metpair_seed_summary.csv`: SHA-256
  `807bc416d46780859c296fb1153bafc9d72d6c4bd492dc36f396aedfb8d5872a`
- `metpair_claim_summary.json`: SHA-256
  `29ef6ccc0cca9738bf92b87ec43ab6d59b4110d737ad16e6de5159a0ea9311be`

All 24 seed-specific truth CSVs have different SHA-256 hashes. The seed axis is
therefore a set of distinct generated scenarios, not duplicate replays.

## Confirmed Results

### Ordinary forecast loss

The ordinary step-loss result is internally consistent with the durable seed
table and old-claim collector:

| Comparison | Wins | Mean paired margin | 95% seed bootstrap CI | Minimum margin |
|---|---:|---:|---:|---:|
| PD-PPO vs strongest evaluated operational reference | 24/24 | 0.149379 | [0.112430, 0.189458] | 0.031445 |
| PD-PPO vs post-hoc fixed replay reference (step loss) | 24/24 | 0.076901 | [0.068156, 0.085562] | 0.028498 |

Margins are reference loss minus PD-PPO loss, so positive values favour
PD-PPO. The exact one-sided all-positive sign-test value is
`2^-24 = 5.960464477539063e-08`.

### Behaviour diagnostics

The binding seed table records behaviour-gate pass for all 24 runs. Across
seeds, the minimum number of unique masks is 4, the largest top-mask fraction
is 0.439697, the minimum mask entropy is 1.622161 bits, and the minimum
event--mask mutual information is 0.082645 bits. The evidence supports a
non-constant, non-simple-cycle specialist policy under the reported action-trace
definitions.

### Post-pilot replication check

Seeds 117 and 122 formed the earlier SCENEBAL-2 pivot pilot. Excluding those
two seeds leaves 22 later scenario seeds with all-positive margins:

| Comparison | Wins | Mean paired margin | 95% seed bootstrap CI | Minimum margin |
|---|---:|---:|---:|---:|
| PD-PPO vs strongest evaluated operational reference, step loss | 22/22 | 0.149654 | [0.111340, 0.191626] | 0.031445 |
| PD-PPO vs post-hoc fixed replay reference, step loss | 22/22 | 0.079079 | [0.070702, 0.087617] | 0.047888 |

The positive result therefore is not driven by the two pivot seeds. However,
the manuscript must distinguish 24 independent generated scenarios from a
strictly post-pilot confirmatory set.

## Correction-Required Findings

### 1. Macro normalizer mismatch

The manuscript defines the macro score with per-event-type denominators fitted
from validation static candidates. Training and standard policy evaluation save
those validation normalizers. In contrast:

- `scripts/70_v31_split_replay_gate.py` computes a new normalizer from
  final-test static replay candidates;
- `scripts/72_v31_collect_metpair_strongclaim.py` recalculates router macro
  scores with those final-test denominators;
- `scripts/73_v31_collect_oldclaim_gate.py` backfills macro scores through the
  same replay-scale helper.

This conflicts with the formula and prose in the active paper, which state that
final test windows are not used for normalizer fitting.

A read-only recomputation from saved final PD-PPO rollout NPZs, truth labels,
and validation candidate tables preserves the substantive result without any
training rerun:

| Validation-frozen macro comparison | Wins | Mean margin | 95% seed bootstrap CI | Minimum margin |
|---|---:|---:|---:|---:|
| PD-PPO vs validation-selected fixed schedule | 24/24 | 0.077820 | [0.066505, 0.089620] | 0.031532 |
| PD-PPO vs best available rule-based operational reference | 24/24 | 0.077106 | [0.066132, 0.088641] | 0.031532 |
| PD-PPO vs strongest fixed replay candidate under the frozen metric | 24/24 | 0.070473 | [0.060206, 0.080714] | 0.021986 |

The current `0.0710` replay-static value is a final-test-normalized diagnostic,
not the validation-frozen macro metric described by the manuscript.

### 2. Comparator scope is mixed in the main table

`paper/tables/regime_balanced_24seed_summary.tex` is captioned as a comparison
with the validation-selected fixed schedule, but its `0.1494` and `0.0841`
entries are margins against the strongest evaluated operational reference for
each seed. This selection is conservative for PD-PPO, but it is a different
comparator from the validation-selected fixed schedule named in the caption and
the first two rows.

The repair is presentation-level: report direct validation-selected fixed
schedule margins in their own rows, and label the stronger per-seed reference as
the strongest evaluated operational reference. Do not imply that all table rows
share one comparator.

### 3. Fixed replay and event-label results are privileged diagnostics

The true fixed replay table is generated on final test windows. Its static
reference is selected from final-test fixed candidates, and the event-label
schedule uses simulator event labels. Both are useful stress tests because they
make the static comparator strong and quantify adaptive opportunity, but neither
is the same thing as the validation-selected deployable baseline.

The current `0.0710` PD-PPO and `0.0773` event-label replay margins, and the
derived claim that PD-PPO captures about 92% of the label-informed advantage,
are internally comparable only as final-test-normalized diagnostic quantities.
They should move to a clearly marked diagnostic/appendix role or be recomputed
under the validation-frozen metric before remaining in the main text.

For reference, preserving the existing final-test-selected fixed action while
rescoring it with frozen validation denominators gives PD-PPO `24/24` wins and
mean margin `0.093472`; the privileged event-label policy gives `24/24` and
mean `0.098882`. These are not recommended as the primary metric because the
fixed action itself was selected using final-test data.

### 4. The raw unnormalised macro 0/24 result is not a valid sensitivity result

The historical `raw_macro` collector path compares values expressed with
different normalizers in the learned/replay fields. Its reported 0/24 gate is
therefore a scale-mismatch artifact, not evidence that PD-PPO fails a valid raw
macro metric. It is correctly absent from the active manuscript and should not
be reintroduced as a limitation without a repaired raw-metric collector.

### 5. Evidence archive is incomplete locally

Remote result directories contain all 24 replay metric tables and replay NPZ
files. The local workspace contains only a subset of these files, even though
the binding aggregate CSV and JSON are synchronized. This blocks a self-contained
reproduction package. The submission archive must include all per-seed raw
rollouts, replay metric tables, truth/metadata files, aggregate CSV/JSON, table
generation scripts, and a commit hash or git bundle of the code used.

## Supplementary Tables and Figures Affected

The same replay-normalizer code is used by the mechanism-ablation and
event-mixture result collectors. Their ordinary step-loss statements remain
separate, but their macro margins, macro confidence intervals, and figures must
be regenerated from validation-frozen normalizers before the paper treats those
values as direct counterparts to the main macro score.

## Required Repair Sequence

No PPO training needs to be repeated. The required work is deterministic
reaggregation and manuscript repair:

1. Add a validation-frozen macro collector that loads the normalizers already
   stored in each run's validation/reward candidate artifacts and never replaces
   them with final-test static candidates.
2. Regenerate the primary 24-seed table, event decomposition, mechanism
   ablation, robustness summaries, and any figure that plots replay macro
   margins. Preserve the current final-test-selected static/event-label numbers
   only as labelled privileged diagnostics.
3. Rewrite the main table and Results text to separate: validation-selected
   fixed baseline, strongest evaluated operational reference, post-hoc fixed
   replay diagnostic, and privileged event-label diagnostic.
4. Add a short statement that 117 and 122 were the configuration-pivot pilot;
   report the 22 later seeds as a post-pilot replication check or obtain a new
   fully locked final seed set for a stricter confirmatory claim.
5. Freeze a versioned evidence archive before further manuscript polishing.

## Overall Judgment

The primary step-loss result and the state-dependent behaviour evidence are
sound in the completed 24-seed experiment. The macro result remains positive
under the intended validation-frozen protocol. The current paper is not ready
to submit because it mixes validation-frozen and final-test-normalized macro
quantities, conflates fixed comparator scopes in one table, and lacks a complete
versioned evidence archive. These are correctable evidence-accounting issues;
they do not require retraining PD-PPO.

## Evidence-Repair Completion Addendum (2026-07-10)

All five correction items above have been addressed without retraining.

1. `scripts/86_v31_collect_validation_frozen_macro.py` now uses only the
   validation candidate normalizers for primary macro aggregation. The repaired
   24-seed result is 24/24 positive margins versus the validation-selected
   static schedule, mean `0.0778198`, with a 95% bootstrap interval
   `[0.0664682, 0.0896082]`.
2. The direct validation-selected comparator is now the only comparator in the
   main table. Final-test fixed replay and simulator event-label schedules are
   retained only as explicitly labelled diagnostics.
3. The manuscript states that seeds 117 and 122 formed the configuration pivot
   and reports the frozen post-pilot replication: 22/22 positive primary macro
   margins, mean `0.0779690`, interval `[0.0669584, 0.0895956]`.
4. Ablation and higher-flux sensitivity tables and figures were rebuilt under
   the same validation-frozen metric. The lower-flux setting is omitted from
   that comparison because seed 147 has no validation flux windows.
5. `reproducibility/pdppo_eswa_evidence_20260710/` contains the collector,
   extracted seed rows, aggregate copies, source snapshots, checksums, and a
   530-file remote path/SHA-256 manifest. It is a deterministic provenance and
   rebuild package, not a second 1.13-GiB raw-artifact mirror.

The canonical paper compiled successfully after the repair. The remaining raw
artifacts stay read-only on `remote-gpu`; their manifest hashes are frozen in
the reproducibility package.
