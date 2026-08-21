# Complete Final-Partition Replay Audit

## Purpose

The primary paper metric uses eight prespecified 512-epoch,
subtype-balanced, transport-rich windows per seed. This audit tests whether the
reported PD-PPO advantage is confined to that sampling rule. It replays the
already frozen actor-only policies and validation-selected fixed schedules over
every final-partition epoch for which the complete eight-step forecast target is
available.

## Frozen scope

- Seeds: 117--140 (24 total).
- Final partition: [64750,70000).
- Scoreable interval: [64750,69992), or 5,242 epochs per seed.
- Excluded tail: 8 epochs without a complete future target.
- Policies: saved custom_ppo checkpoint and saved
  validation_selected_static action for each seed.
- Normalization: subtype scales fitted from validation static candidates.
- Evaluator: common frozen TCN on CPU in every source run.
- Selection: no policy, fixed mask, checkpoint, normalizer, or parameter is
  selected from this replay.

## Provenance checks

Every source metadata file under
reports/v31_scenebal2_matched_reward_forecast_noexactevent_seed*_h075forecastctrl_20260718cleanpilot/full_final_partition_replay/
records:

- oracle_inference_device = cpu;
- evaluation_scope.mode = full_scoreable_final_partition;
- eval_steps = 5242;
- eval_start_indices = [64750];
- evaluation_scope.final_partition = [64750,70000];
- evaluation_scope.scoreable_interval = [64750,69992];
- excluded_tail_steps_without_complete_future = 8.

The synchronized aggregate contains exactly one row for every seed from 117
through 140. Independent recomputation from the seed CSV confirms all win
counts, means, minima, behavior fields, and source-scope fields.

## Result

Margins are fixed-schedule loss minus PD-PPO loss, so positive values favor
PD-PPO.

| Metric | Wins | Mean margin | 95% bootstrap CI | Minimum margin |
|---|---:|---:|---:|---:|
| Mean forecast loss | 24/24 | +0.124728 | [+0.090058, +0.164236] | +0.009150 |
| Validation-normalized subtype macro | 24/24 | +0.079260 | [+0.064229, +0.095031] | +0.013825 |

Mean ordinary loss is 0.663241 for PD-PPO and 0.787969 for the fixed schedule.
Mean macro score is 0.430069 for PD-PPO and 0.509329 for the fixed schedule.

All 24 runs have zero warm-up aborts. The required weather backbone is always
active and the basic radiometer remains inactive. All four useful specialists
have intermediate duty in 23/24 seeds; three do so in the remaining seed.
Switching ranges from 0.002099 to 0.004070 per step.

## Interpretation

The primary subtype-balanced windows remain the confirmatory evaluation because
they enforce comparable representation of the three incompatible specialist
requirements. The continuous replay is a coverage sensitivity. Its 24/24
paired direction shows that the fixed-schedule advantage is not created by the
primary window-selection rule.

## Authoritative files

- reports/aggregate/pdppo_full_final_partition_24seed_20260718/validation_frozen_seed_metrics.csv
- reports/aggregate/pdppo_full_final_partition_24seed_20260718/validation_frozen_claim_summary.json
- reports/aggregate/pdppo_clean_paper_assets_20260718/paper_asset_manifest.json
- paper/tables/clean_full_partition_sensitivity.tex

## Commands

~~~bash
ssh remote-gpu 'source /opt/miniconda3/etc/profile.d/conda.sh && \
  conda activate darts && \
  cd ~/_code/microclimate_demo/rl_sensor_scheduling_framework && \
  python scripts/86_v31_collect_validation_frozen_macro.py \
    --run-glob "reports/v31_scenebal2_matched_reward_forecast_noexactevent_seed*_h075forecastctrl_20260718cleanpilot" \
    --seeds $(seq 117 140) \
    --router-eval-dir full_final_partition_replay \
    --out-dir reports/aggregate/pdppo_full_final_partition_24seed_20260718 \
    --bootstrap-samples 100000'

conda run -n darts python scripts/95_v31_build_clean_paper_assets.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~
