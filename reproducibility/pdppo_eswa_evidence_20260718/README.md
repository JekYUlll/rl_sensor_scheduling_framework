# PD-PPO ESWA Evidence Package: 2026-07-18

This package freezes the method-closure evidence used by the canonical English
manuscript in paper/main.tex. It contains the clean actor-only PD-PPO result,
matched reward and learner controls, strong diagnostic references, an
independent forecaster rescore, and continuous full-final-partition replay.

No result in this package uses the historical hard subtype router, the separate
no-warmup paper line, or v1. Exact subtype labels are confined to training-only
auxiliary signals or common offline grouping; the final PD-PPO actor executes
from online observations and warning scores.

## Main result

The primary evaluation uses eight prespecified subtype-balanced,
transport-rich windows of 512 epochs in each final partition. The frozen
actor-only policy is compared with the fixed mask selected on validation data.

- Seeds: 117--140 (24 evaluation seeds).
- Mean forecast loss: PD-PPO 1.401227 versus fixed 1.559199.
- Mean-loss margin: +0.157971; 24/24 wins.
- Validation-normalized subtype macro: PD-PPO 0.940433 versus fixed 1.020559.
- Macro margin: +0.080126; 24/24 wins.
- Unchanged post-pilot replication: seeds 119--140, 22/22 macro wins.
- Behavior: 24/24 traces pass the non-fixed/non-cyclic gate; zero warm-up
  aborts.

Margins are reference loss minus PD-PPO loss, so positive values favor PD-PPO.

## Coverage sensitivity

The same checkpoints and validation-selected fixed masks are replayed over the
complete scoreable interval [64750,69992), giving 5,242 epochs per seed.
Evaluation uses the frozen CPU forecaster in all 24 source runs.

- Mean-loss margin: +0.124728; 24/24 wins; 95% bootstrap CI
  [+0.090058,+0.164236].
- Macro margin: +0.079260; 24/24 wins; CI
  [+0.064229,+0.095031].
- Minimum seed-level margins: +0.009150 ordinary and +0.013825 macro.

The final eight epochs are excluded because they lack a complete eight-step
future target. No policy, reference, or normalizer is selected from this replay.

## Experimental controls

- Conventional references: PD-PPO beats validation-selected fixed, AoI,
  round-robin, random, and one-step forecast-greedy schedules in 24/24 macro
  comparisons.
- Context references: warning-score and exact-label policies are competitive;
  PD-PPO macro wins are 11/24 and 12/24, and their paired confidence intervals
  include zero.
- Matched Double-DQN: PD-PPO wins 24/24 macro and 23/24 ordinary comparisons;
  mean differences are +0.069719 and +0.140775.
- Same-architecture rewards: forecast, AoI, and uncertainty reward variants
  show no detected mean difference. Every variant beats its own selected fixed
  schedule in 24/24 macro comparisons.
- Independent ridge forecaster: PD-PPO beats the ridge-validation-selected
  fixed schedule in 23/24 macro comparisons, with mean margin +0.133435.

## Package layout

- aggregates/: lightweight top-level CSV, JSON, and Markdown summaries for all
  eight evidence families.
- full_partition_source_rows/: per-seed CPU evaluation metadata and metric CSVs
  for the continuous replay.
- code_snapshot/: exact collectors, evaluators, launchers, and paper-asset
  builder used for closure.
- analysis/: the complete final-partition audit.
- paper/main.pdf: compiled canonical manuscript.
- paper_source_snapshot.tar.gz: canonical LaTeX source, active sections,
  tables, figures, and bibliography.
- framework_method_source_snapshot.tar.gz: implementation, tests, configs, and
  closure scripts.
- framework_tracked_worktree.patch and paper_tracked_worktree.patch: tracked
  changes relative to the Git bases below.

Framework Git base:
300e7321b47b2708bd131f4c8fa30161b06cd7b7

Paper Git base:
e3af6ec828c08a5d385fbc80aca1d6140ec84ff1

## Rebuild and verification

From rl_sensor_scheduling_framework:

~~~bash
conda run -n darts python scripts/95_v31_build_clean_paper_assets.py

conda run -n darts python -m py_compile \
  scripts/64_v31_eval_saved_run_operational_baselines.py \
  scripts/86_v31_collect_validation_frozen_macro.py \
  scripts/89_v31_train_matched_dqn.py \
  scripts/90_v31_collect_matched_reward_controls.py \
  scripts/91_v31_collect_matched_dqn.py \
  scripts/93_v31_secondary_forecaster_rescore.py \
  scripts/94_v31_collect_clean_policy_mechanism.py \
  scripts/95_v31_build_clean_paper_assets.py

conda run -n darts pytest -q \
  tests/v2/test_custom_ppo.py \
  tests/v2/test_dqn.py \
  tests/v2/test_forecast_eval.py \
  tests/v2/test_warmup_env.py

cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

The paper asset builder verifies the complete 24-seed sets, full-partition CPU
metadata, scope fields, and all aggregate inputs before writing manuscript
tables and figures. checksums.txt provides byte-level hashes for this package.
