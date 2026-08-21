# PD-PPO ESWA Evidence Package: 2026-07-10

This package repairs the macro-score provenance for the final PD-PPO paper
evidence without retraining. Historical collectors normalized event-type losses
with static candidates replayed on final-test windows. The manuscript protocol
instead fixes each subtype denominator as the median loss across feasible static
candidates on the validation partition.

## Frozen result

The primary comparison is PD-PPO versus the validation-selected fixed schedule
over seeds 117--140. The corrected macro result is 24/24 wins, mean margin
0.0778198, 95% percentile bootstrap interval [0.0664682, 0.0896082], and a
one-sided sign-test probability of 5.960464477539063e-08. The corresponding
ordinary step-loss comparison is also 24/24 wins, with mean margin 0.1530815.

Margins are reference loss minus PD-PPO loss; positive values favor PD-PPO.

## Code and manuscript bases

- Framework Git base: `300e7321b47b2708bd131f4c8fa30161b06cd7b7`.
- Paper Git base: `e3af6ec828c08a5d385fbc80aca1d6140ec84ff1`.
- Both worktrees contain the manuscript and evidence-repair changes that must
  be preserved as a patch or committed together before submission.

## Authoritative artifacts

The canonical local summaries are:

- `reports/aggregate/scenebal2_validation_frozen_24seed_20260710_remote_verified/`
- `reports/aggregate/mechanism_ablation_validation_frozen_24seed_20260710/`
- `reports/aggregate/robustness_flux30_validation_frozen_6seed_20260710/`

The primary raw artifact manifest and remote SHA-256 list are stored outside the
repository at:

`/home/horeb/_Data/pdppo_eswa_evidence_20260710/manifests/`

They enumerate 530 files (truth sequences, validation candidates, final
rollouts, replay artifacts, and metadata). The remote source is read-only under
the `remote-gpu` alias because the remote project filesystem is currently above
the user's write quota.

The package intentionally stores the immutable remote path/size/SHA-256 manifest
and the extracted per-seed rows needed to regenerate every reported aggregate,
rather than a second physical copy of all 530 raw artifacts. This keeps the
archive locally rebuildable while preserving a byte-level provenance record for
the remote source files.

## Rebuild commands

The main result can be regenerated from the archived remote-extracted seed rows:

```bash
cd rl_sensor_scheduling_framework
python scripts/86_v31_collect_validation_frozen_macro.py \
  --seed-metrics-csv reproducibility/pdppo_eswa_evidence_20260710/remote_extracted_seed_rows/main_seed_metrics.csv \
  --diagnostic-csv reproducibility/pdppo_eswa_evidence_20260710/remote_extracted_seed_rows/main_diagnostics.csv \
  --out-dir reports/aggregate/scenebal2_validation_frozen_24seed_20260710_remote_verified \
  --bootstrap-samples 100000
```

The 24-seed ablation summary is built with:

```bash
python scripts/87_v31_summarize_validation_frozen_variants.py \
  --entries \
    full_reference=reports/aggregate/scenebal2_validation_frozen_24seed_20260710_remote_verified/validation_frozen_seed_metrics.csv \
    no_imitation=reports/aggregate/mechanism_ablation_no_imitation_validation_frozen_24seed_20260710/validation_frozen_seed_metrics.csv \
    no_regime_aux=reports/aggregate/mechanism_ablation_no_regime_aux_validation_frozen_24seed_20260710/validation_frozen_seed_metrics.csv \
    no_staticnorm=reports/aggregate/mechanism_ablation_no_staticnorm_validation_frozen_24seed_20260710/validation_frozen_seed_metrics.csv \
  --out-dir reports/aggregate/mechanism_ablation_validation_frozen_24seed_20260710 \
  --bootstrap-draws 100000
```

The post-pilot replication check excludes configuration-pivot seeds 117 and 122:

```bash
python scripts/86_v31_collect_validation_frozen_macro.py \
  --seed-metrics-csv reproducibility/pdppo_eswa_evidence_20260710/remote_extracted_seed_rows/main_seed_metrics.csv \
  --diagnostic-csv reproducibility/pdppo_eswa_evidence_20260710/remote_extracted_seed_rows/main_diagnostics.csv \
  --seeds 118 119 120 121 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 \
  --out-dir reports/aggregate/scenebal2_validation_frozen_postpilot22_20260710 \
  --bootstrap-samples 100000
```

Regenerate the manuscript assets and PDF with:

```bash
python paper/tables/gen_mechanism_ablation_table.py \
  --summary-csv reports/aggregate/mechanism_ablation_validation_frozen_24seed_20260710/validation_frozen_variant_summary.csv \
  --out-tex paper/tables/mechanism_ablation_summary.tex
python paper/figures/gen_fig_scenebal_evidence.py \
  --seed-csv reports/aggregate/scenebal2_validation_frozen_24seed_20260710_remote_verified/validation_frozen_seed_metrics.csv \
  --out-prefix paper/figures/figure_regime_balanced_24seed_evidence
python paper/figures/gen_fig_mechanism_continuous.py \
  --summary-csv reports/aggregate/mechanism_ablation_validation_frozen_24seed_20260710/validation_frozen_variant_summary.csv \
  --seed-csv reports/aggregate/mechanism_ablation_validation_frozen_24seed_20260710/validation_frozen_variant_seed_metrics.csv \
  --out paper/figures/figure_mechanism_robustness
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Boundary record

The lower-flux stress setting (seeds 147--152) is retained as an archive-only
stress experiment. Seed 147 contains no validation flux windows, so it cannot
define the pre-specified three-regime validation-frozen macro score. Its record
is `reports/aggregate/robustness_flux10_validation_frozen_6seed_20260710/metric_unavailable.md`.
