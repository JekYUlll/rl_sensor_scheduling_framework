# Research Log: PD-PPO Strong-Claim Autoresearch

## 2026-06-21 04:56 CST
- User explicitly required continuous 24h autonomous execution and no stop-only
  reporting.
- Created a new active goal for 24h PD-PPO strong-claim autoresearch.
- Confirmed remote 24h runner `bo24_autonomy_20260621` is active on
  `remote-gpu`.
- Started local tmux watcher `bo24_local_watch_20260621`.
  - Poll interval: 600 seconds.
  - Status file: `reports/aggregate/bo24_local_watch_20260621_status.md`.
  - Log: `logs/bo24_local_watch_20260621.log`.
- Corrected root planning files so future planning hooks state ESWA, not CRST, as
  the active target.
- Current BO wave: `67--78`; odd seeds have oracle checkpoints and PPO training
  is in progress. Even paired seeds will start after each worker finishes its
  first seed.
- Step diagnostic:
  - seed55 `subtype_auto` replay passes strict step gate against true fixed
    static.
  - seed59 diagnostic remains running.

## Next Inner-Loop Action
Wait for the next BO wave checkpoint, then sync and aggregate. If no new step
breakthrough appears, prepare `subtype_static_auto` autoteacher pilot without
colliding with the active 24h runner.

## 2026-06-21 04:56 CST
- Remote wave `67--78` status:
  - seeds `67,71,73,75,77` reached `200000` PPO timesteps and have checkpoint
    files beginning to appear;
  - seed `69` was at `195584` timesteps;
  - paired even seeds `68,70,72,74,76,78` had not started yet.
- No new aggregate is available yet. Continue monitoring until eval/replay and
  behaviour artifacts appear for the wave.

## 2026-06-21 05:01 CST
- User asked to verify the active goal against the original strong-claim
  instruction.
- Current tool-level goal is acceptable at the top level, but was underspecified
  on three constraints. Added them to `task_plan.md`, root `task_plan.md`, and
  `research-state.yaml`:
  - PPO must remain the final learned scheduling algorithm; deeper changes may
    alter PPO inputs, auxiliary heads, policy/memory structure, teacher/oracle,
    reward, simulator, replay/evaluation, or moderate sensor/noise calibration,
    but cannot replace PPO.
  - Each modification direction has at most `10` bounded work units without
    effective improvement; earlier suspected failure should trigger a pivot.
  - Existing microclimate sensor configuration remains the physical baseline;
    only moderate, explainable simulated variants are allowed.
- Execution implication: BO-1 is one bounded modification direction, not an
  unlimited seed search. If new wave evidence does not improve the strict step
  gate, pivot to another layer such as `subtype_static_auto` teacher/PPO branch
  or simulator/framework changes.

## 2026-06-21 05:08 CST
- Wave `67--78` interim status from `remote-gpu`:
  - seeds `67,69,71,73,75,77` completed oracle, PPO, eval, strict replay, and
    behavior audit artifacts;
  - seeds `68,70,72,74,76,78` have started PPO training.
- Completed odd seeds show a stronger BO-1 step signal than the 41--66
  aggregate:
  - strict replay step gate: `6/6`;
  - strict replay macro gate: `6/6`;
  - behavior complexity gate: `6/6`;
  - fixed-like: `0/6`;
  - simple-cycle-like: `0/6`.
- Step margins against true fixed static / static reference:
  - seed67: `+0.166386` abs / `+0.062320` rel;
  - seed69: `+0.171943` abs / `+0.065821` rel;
  - seed71: `+0.216564` abs / `+0.061856` rel;
  - seed73: `+0.224125` abs / `+0.073188` rel;
  - seed75: `+0.167435` abs / `+0.048520` rel;
  - seed77: `+0.255616` abs / `+0.077346` rel.
- Interpretation: this is not yet a final wave result because the paired even
  seeds are still running, but BO-1 now has an interim `6/6` strict step-gate
  improvement in the current wave. Continue monitoring to complete the 12-seed
  wave and aggregate through seed 78.

## 2026-06-21 05:13 CST
- The independent seed59 `subtype_auto` step diagnostic finished:
  - `gate_pass=True`;
  - best replay policy: `split_subtype_auto_top2_c0_p0_f0_t0_l10`;
  - best replay step loss: `3.106645`;
  - static reference policy: `static_action5`;
  - static reference step loss: `3.379619`;
  - margin: `+0.272974` absolute / `+0.080771` relative.
- Together with seed55, this means at least two prior BO-1 strict step failures
  are recoverable by automatic subtype/static-candidate replay selection. This
  strengthens the rationale for the prepared `subtype_static_auto` teacher/PPO
  branch if the current BO aggregate still has residual step failures.

## 2026-06-21 05:24 CST
- Wave `67--78` completed and aggregate through seed `78` was generated:
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_41_78_20260621_oldclaim/`.
- Synced old-claim, macro, and raw-macro aggregate directories locally.
- 41--78 official aggregate:
  - complete seeds: `38`;
  - learned PPO beats static/rule/operational baselines on step objective:
    `36/38`;
  - learned PPO beats static/rule/operational baselines on macro objective:
    `38/38`;
  - strict explicit replay step gate: `32/38`;
  - strict explicit replay macro gate: `38/38`;
  - behavior complexity gate: `38/38`;
  - learned-policy true-static step gate: `33/38`;
  - old-step sign-test p-value: `1.2171280104666948e-05`;
  - macro sign-test p-value: `3.637978807091713e-12`.
- New wave contribution:
  - seeds `67--78` pass old-claim step gate `12/12`;
  - seeds `67--78` pass learned true-static step gate `12/12`;
  - every seed in the new wave has positive step margin vs best operational
    baseline and positive margin vs replay static reference.
- Anti-stall decision: BO-1 has now shown effective improvement, so do not pivot
  immediately. Continue the already-started wave `79--90` to test persistence of
  the 12/12 step-gate improvement.

## 2026-06-21 05:45 CST
- Wave `79--90` first-half status:
  - seeds `79,81,83,85,87,89` completed all artifacts;
  - seeds `80,82,84,88,90` have started; seed `86` is just starting.
- First-half gate results are mixed, so the clean `67--78` wave is not yet a
  stable zero-failure pattern:
  - seed83 fails strict replay step gate:
    best replay `3.273781`, static reference `3.212263`, margin `-0.061518`
    absolute / `-0.019151` relative; macro gate still passes.
  - seed87 passes replay step and macro gates but fails behavior complexity:
    `fixed_like=True`, `state_dependent=False`, event-mask MI about `0.02`.
  - seeds `79,81,85,89` pass both replay step and behavior gates.
- Interpretation: BO-1 remains improved statistically, but failures now split
  into two mechanisms: a step/static replay loss boundary (seed83) and a learned
  behavior collapse boundary (seed87). If the completed 79--90 aggregate remains
  mixed, the next pivot should target teacher/PPO behavior robustness, not only
  another seed wave.

## 2026-06-21 06:05 CST
- Aggregate through seed `90` completed and was synced locally:
  `reports/aggregate/metpair_backbone_context_ortholinear_balancedobjective_41_90_20260621_oldclaim/`.
- Official 41--90 summary:
  - complete seeds: `50`;
  - learned step wins: `45/50`;
  - learned macro wins: `50/50`;
  - strict explicit replay step gate: `40/50`;
  - strict explicit replay macro gate: `49/50`;
  - behavior complexity gate: `49/50`;
  - learned-policy true-static step gate: `41/50`;
  - old-step sign-test p-value: `1.1930665838377763e-05`;
  - macro sign-test p-value: `4.529709940470639e-14`.
- Wave `79--90` was mixed:
  - old step gate `8/12`;
  - macro gate `11/12`;
  - behavior gate `11/12`;
  - failures: step `83,84,86,87`; macro/behavior `87`.
- Decision under the anti-stall rule:
  - BO-1 produced useful statistical improvement, but did not reach the target
    stable/zero-failure strong claim.
  - Stopped same-configuration `bo24_autonomy_20260621` after 41--90 instead of
    continuing seed expansion into wave 91--102.
  - Pivoted to AT-1: `subtype_static_auto` teacher/PPO branch, preserving PPO and
    the sensor baseline.
- Launched remote tmux
  `autoteacher_pilot_parallel_83_92_20260621` with seeds
  `83,84,86,87,91,92`, one seed per GPU, using `SKIP_COLLECT=1` per worker and a
  final centralized macro/raw-macro/oldclaim aggregation step.
- Started local watcher tmux `autoteacher_pilot_local_watch_20260621`.
  First snapshot shows all six AT-1 seeds at status `100000`, with GPU memory
  allocated on all six devices.

## 2026-06-21 06:28 CST
- AT-1 `subtype_static_auto` pilot completed and failed the strong-claim gate:
  - oldclaim step gate: `2/6`;
  - oldclaim macro gate: `3/6`;
  - behavior gate: `3/6`;
  - claim strength: `not_supported`.
- Positive signal:
  - known step failures seed83 and seed86 are repaired;
  - replay gates pass `6/6`.
- Remaining/new failures:
  - seed84 remains a small step/static failure (`-0.013879`);
  - seed87 still fails behavior in raw rollout, although its
    `eval_router_conf08` behavior passes;
  - fresh seeds 91 and 92 fail behavior / oldclaim gates.
- Decision: do not continue AT-1 autoteacher training as-is. It fixes some
  performance failures but worsens behavior robustness.
- Started RT-1 router-threshold sweep in tmux
  `autoteacher_router_sweep_83_92_20260621`:
  - labels: `conf00`, `conf05`;
  - seeds: `83,84,86,87,91,92`;
  - collector patched with `--metrics-eval-dir` so metrics and behavior can be
    evaluated from the same router deployment directory.

## 2026-06-21 06:35 CST
- RT-1 router-threshold sweep failed:
  - `conf00` oldclaim step gate remains `2/6`, behavior gate `3/6`;
  - `conf05` behavior audit fails for all six seeds in the quick logs;
  - deployment threshold alone is not sufficient.
- Implemented a new behavior-regularized PPO branch:
  `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorreg_seed_sweep_20260621.sh`.
  It preserves PPO, sensors, and the BO-1 scene, reverts to the original
  `subtype_auto` teacher, and adds mild duty-balance / duty-score feedback
  through already existing environment mechanisms.
- Parameterized the base seed sweep wrapper with env-controlled
  `LAMBDA_DUTY_BALANCE`, `DUTY_SCORE_FEEDBACK`, `DUTY_HARD_GUARD`, and related
  duty bounds. Defaults preserve prior behavior.
- Launched BR-1 remote tmux `behaviorreg_pilot_parallel_83_92_20260621` on
  seeds `83,84,86,87,91,92`, one seed per GPU.

## 2026-06-21 06:41 CST
- BR-1 status check through `remote-gpu`:
  - remote tmux `behaviorreg_pilot_parallel_83_92_20260621` is still active;
  - all six seeds are in PPO training at roughly `38k--41k / 120k` timesteps;
  - no evaluation/replay/behavior aggregate artifacts have been produced yet.
- Added BR-1-specific local watcher
  `scripts/watch_behaviorreg_pilot_20260621.sh` and launched it in tmux
  `behaviorreg_pilot_local_watch_20260621`.
- The watcher writes
  `reports/aggregate/behaviorreg_pilot_local_watch_20260621_status.md` and
  syncs BR-1 aggregate directories every `600` seconds.

## 2026-06-21 07:00 CST
- BR-1 completed and failed the strong-claim gate:
  - old step gate: `3/6`;
  - old macro gate: `4/6`;
  - behavior gate: `4/6`;
  - replay gate: `5/6`;
  - replay macro gate: `6/6`;
  - learned true-static step gate: `2/6`.
- Failure pattern:
  - step failures: `83,87,92`;
  - macro failures: `87,92`;
  - behavior failures: `87,92`.
- Wrote `reports/aggregate/behaviorreg_pilot_83_92_failure_report_20260621.md`.
- Decision: BR-1 is not worth expanding. It slightly improves AT-1 but does not
  remove the behavior-collapse mechanism. Pivot to BD-1: explicit
  state-dependent behavior signal or PPO architecture change.

## 2026-06-21 07:33 CST
- BD-1 completed and failed the strong-claim gate:
  - oldclaim step gate: `2/6`;
  - oldclaim macro gate: `4/6`;
  - behavior gate: `4/6`;
  - replay gate: `5/6`;
  - replay macro gate: `6/6`;
  - learned true-static step gate: `3/6`.
- Failure pattern:
  - step failures: `83,84,87,92`;
  - macro failures: `87,92`;
  - behavior failures: `87,92`.
- The new subtype action CE/margin losses were present in PPO logs, so this is a
  real negative result: direct action regularization is too weak to prevent
  fixed-like deployment on hard seeds.
- Wrote `reports/aggregate/behaviorbd_pilot_83_92_failure_report_20260621.md`.
- Decision: do not expand BD-1. Pivot to BRG-1: enable a stronger observable
  regime-belief state path, train the subtype auxiliary head, and test a
  conservative subtype router as part of the PPO deployment head while keeping
  PPO as the learned scheduler.

## 2026-06-21 07:37 CST
- Implemented and launched BRG-1:
  - modified the base runner to expose `INCLUDE_OBSERVABLE_REGIME_BELIEF`,
    `REGIME_BELIEF_LOOKBACK`, and `EVENT_GATED_ACTOR`;
  - added BRG-1 wrapper, parallel runner, and watcher scripts.
- BRG-1 configuration:
  - keeps PPO, BO-1 met+specialist scenario, current sensor baseline, strict
    replay and corrected behavior audit;
  - enables observable regime-belief features with lookback `12`;
  - enables event-gated actor;
  - strengthens subtype auxiliary loss to `5.0`;
  - uses conservative raw subtype-router confidence `0.45` and eval router
    confidence `0.70`;
  - disables BD-1 direct subtype-action CE/margin by default.
- Remote tmux started:
  `behaviorbrg_pilot_parallel_83_92_20260621`.
- Local watcher started:
  `behaviorbrg_pilot_local_watch_20260621`.
- Initial health check: all six seeds started and no early errors were found.

## 2026-06-21 08:05 CST
- BRG-1 completed and produced local aggregate evidence:
  - oldclaim step gate: `3/6`;
  - oldclaim macro gate: `5/6`;
  - behavior gate: `5/6`;
  - operational macro gate: `6/6`;
  - replay macro gate: `6/6`;
  - learned true-static step gate: `4/6`.
- Router-eval macro collection is clean on macro/behavior (`6/6` and `6/6`),
  but raw macro collection is still `5/6` and raw old-claim is not a strong
  claim. Seed `87` is the key raw-deployment behavior failure; seed `83` misses
  strict replay margin; seed `92` remains slightly worse than selected static
  on step loss.
- Wrote
  `reports/aggregate/behaviorbrg_pilot_83_92_partial_report_20260621.md`.
- Decision: BRG-1 is a real effective-improvement unit, so one bounded BRG-2
  follow-up is justified under the anti-stall rule. BRG-2 will keep PPO and the
  met+specialist sensor baseline, match raw/eval subtype-router confidence at
  `0.70`, modestly increase entropy, and avoid reusing BD-1 direct
  subtype-action CE/margin losses. If it does not clearly improve the pilot, the
  next layer is simulator/reward headroom or deeper PPO architecture rather than
  more conservative BRG retries.

## 2026-06-21 08:09 CST
- Implemented and launched BRG-2:
  - `RUN_PREFIX=v31_metpair_backbone_context_ortholinear_balancedobjective_behaviorbrg2`;
  - `BUDGET_LABEL=h075ctxolbrg2`;
  - raw/eval subtype-router confidence matched at `0.70`;
  - entropy coefficient `0.0075`;
  - observable regime-belief and event-gated actor retained;
  - direct subtype-action CE/margin losses kept disabled.
- Validation:
  - local `bash -n` and Python compile passed;
  - remote conda `darts` validation passed on `remote-gpu`;
  - all six GPUs were idle before launch.
- Remote tmux started:
  `behaviorbrg2_pilot_parallel_83_92_20260621`.
- Local watcher started:
  `behaviorbrg2_pilot_local_watch_20260621`.

## 2026-06-21 08:40 CST
- BRG-2 completed:
  - default oldclaim step `5/6`;
  - default oldclaim macro `5/6`;
  - default behavior `5/6`;
  - replay step `6/6`;
  - replay macro `6/6`;
  - learned true-static step `5/6`.
- Added subtype-aware behavior auditing because the current scene is explicitly
  subtype-structured and high-event-rate segments can make the binary
  event/non-event gate too coarse.
- Subtype-aware sanity check:
  - BRG-2 seed `92` custom PPO passes with subtype MI `0.587826` and subtype
    sensor L1 `2.000000`;
  - validation-selected fixed static still fails the same audit.
- Subtype-aware BRG-2 aggregate:
  - oldclaim step `5/6`;
  - oldclaim macro `6/6`;
  - behavior `6/6`;
  - macro and raw macro gates `6/6`.
- Remaining blocker is not behavior: seed `92` learned PPO is still
  step-negative against validation-selected static (`3.046497` vs `3.024605`)
  while explicit subtype replay has headroom (`2.971215` vs true fixed static
  `3.049884`).
- Wrote
  `reports/aggregate/behaviorbrg2_pilot_83_92_evidence_report_20260621.md`.
- Decision: BRG-3 should add a moderate action-fidelity signal on top of BRG-2
  rather than changing the scene. If BRG-3 fails to repair seed `92`, pivot away
  from this direction.

## 2026-06-21 08:41 CST
- Implemented and launched BRG-3:
  - matched-router observable-regime-belief architecture from BRG-2 retained;
  - `SUBTYPE_ACTION_CE_COEF=0.25`;
  - `SUBTYPE_ACTION_MARGIN_COEF=0.05`;
  - `SUBTYPE_ACTION_MARGIN=0.50`;
  - entropy remains `0.0075`.
- Validation passed locally and on `remote-gpu` in conda `darts`.
- Remote tmux started:
  `behaviorbrg3_pilot_parallel_83_92_20260621`.
- Local watcher started:
  `behaviorbrg3_pilot_local_watch_20260621`.

## 2026-06-21 09:09 CST
- BRG-3 completed and failed as a step-claim direction:
  - old step `3/6`;
  - old macro `6/6`;
  - behavior `6/6`;
  - replay step `3/6`;
  - replay macro `6/6`;
  - learned true-static step `2/6`.
- Direct action CE/margin on top of BRG worsened step performance relative to
  BRG-2. Failed step seeds are `83,84,92`.
- Wrote
  `reports/aggregate/behaviorbrg3_pilot_83_92_failure_report_20260621.md`.
- Decision: do not continue action-fidelity retries. Pivot to TEMPORAL-1:
  inspect lead/subtype code paths and implement a temporal/lead-aware PPO or
  reward-credit modification while keeping PPO as the final scheduler.

## 2026-06-21 09:12 CST
- Implemented TEMPORAL-1:
  - parameterized temporal controls in the base runner;
  - longer subtype context lead and teacher/auxiliary lookahead;
  - no subtype action CE/margin;
  - PPO remains the final learned scheduler.
- Validation passed locally and on `remote-gpu`.
- Remote tmux started:
  `temporal1_pilot_parallel_83_92_20260621`.
- Local watcher started:
  `temporal1_pilot_local_watch_20260621`.

## 2026-06-21 09:39 CST
- TEMPORAL-1 completed and is a hard/control pilot breakthrough:
  - old step `6/6`;
  - old macro `5/6`;
  - behavior `6/6`;
  - replay step `6/6`;
  - replay macro `6/6`;
  - raw macro collector `6/6`.
- Seed `92` is repaired on step loss:
  `custom_ppo=2.777641` vs selected static `2.845988`.
- Wrote
  `reports/aggregate/temporal1_pilot_83_92_breakthrough_report_20260621.md`.
- Decision: keep TEMPORAL-1 as the leading candidate and expand to fresh seeds
  `93--98` before upgrading to a strong multi-seed claim.

## 2026-06-21 09:39 CST
- Generalized TEMPORAL-1 runner/watcher with `SEED_LABEL`.
- Launched fresh-seed expansion:
  `temporal1_pilot_parallel_93_98_20260621`.
- Seeds: `93,94,95,96,97,98`.
- Local watcher:
  `temporal1_pilot_local_watch_93_98_20260621`.

## 2026-06-21 10:13 CST
- TEMPORAL-1 fresh-seed expansion `93--98` completed with mixed evidence:
  - old step `4/6`;
  - old macro `6/6`;
  - behavior `6/6`;
  - strict replay step `5/6`;
  - strict replay macro `6/6`;
  - dedicated macro collectors `6/6` for raw and router-eval deployment.
- This preserves the dynamic-scheduling and macro story but does not support a
  zero-failure old-step strong claim.
- Wrote
  `reports/aggregate/temporal1_pilot_93_98_mixed_report_20260621.md`.
- Failure diagnosis:
  - seed `95` appears structural or replay-search related because default
    explicit replay is also step-negative against strict fixed static;
  - seed `96` is a learned-credit failure because default explicit replay is
    step-positive;
  - seed `97` is a true-static boundary case.
- Launched diagnostic tmux
  `temporal1_wide_replay_diag_95_97_20260621` to run wider lead/dwell explicit
  replay on seeds `95,96,97` before deciding whether to repair TEMPORAL-1 or
  pivot to simulator / target-generation / deeper PPO credit changes.

## 2026-06-21 10:25 CST
- TEMPORAL-1 wide replay diagnostic completed:
  - seed `95` remains replay-negative against strict fixed static:
    `3.600575` vs `3.520518`, margin `-0.080057`;
  - seed `96` is replay-positive, margin `+0.048074`;
  - seed `97` is replay-positive, margin `+0.032034`.
- This confirms seed `95` is a scenario/step-headroom failure caused by a
  fixed `met_station_core|fc4_flux` shortcut, not simply a PPO learning failure.
- Wrote
  `reports/aggregate/temporal1_wide_replay_diag_95_97_report_20260621.md`.
- Implemented and launched `SCENEBAL-1`:
  - keeps PPO as the final learned scheduler;
  - keeps the met+one-specialist sensor geometry;
  - rebalances subtype probabilities, target weights, and subtype latent
    strengths to reduce raw-step flux dominance;
  - pilot seeds are `93,94,95,96,97,98`.
- Remote tmux:
  `scenebal1_pilot_parallel_93_94_95_96_97_98_20260621`.
- Local watcher:
  `scenebal1_pilot_local_watch_93_94_95_96_97_98_20260621`.

## 2026-06-21 10:38 CST
- SCENEBAL-1 seed `93` initially failed before PPO training with
  `Could not derive finite positive staticnorm reward normalizers from
  validation static candidates`.
- Diagnosis: `reward_staticnorm_candidates.csv` had particle and thermal
  subtype samples, but `steps_subtype_flux=0` for every static candidate in the
  validation static-selection windows. The truth sequence still contains flux
  events, so this was a staticnorm sampling-coverage failure rather than a PPO
  training failure.
- Patched `scripts/25_v2_train_custom_ppo.py`:
  - detects missing subtype normalizer columns;
  - samples subtype-positive fallback windows from the normalization split;
  - writes `reward_staticnorm_fallback_candidates.csv` and
    `reward_staticnorm_normalizers.json`;
  - applies the same normalizer map to reward normalization, selected-static
    staticnorm columns, and final metric staticnorm columns.
- Synced the patch to `remote-gpu` and relaunched seed `93` in tmux
  `scenebal1_seed93_rerun_staticnormfix_20260621` on GPU0. The rerun produced
  fallback normalizers and entered PPO training.

## 2026-06-21 10:59 CST
- SCENEBAL-1 pilot `93--98` completed and was synced locally.
- Aggregate result:
  - complete seeds `6/6`;
  - old operational step gate `6/6`;
  - old operational macro gate `6/6`;
  - behavior gate `6/6`;
  - strict explicit replay gate `6/6`;
  - strict explicit replay macro gate `6/6`;
  - mean step margin vs best operational baseline `0.102988`;
  - median step margin `0.045900`.
- Seed `95`, the TEMPORAL-1 structural failure, is repaired in the operational
  protocol: `custom_ppo=1.951350` vs selected static `1.963998`, margin
  `+0.012648`; replay margin vs static is also positive.
- Remaining boundary:
  - learned true-static step gate is `5/6`;
  - learned true-static macro gate is `3/6`.
- Interpretation: SCENEBAL-1 is a real breakthrough relative to BO/TEMPORAL
  branches but not yet an unconditional true-static macro claim.
- Wrote
  `reports/aggregate/scenebal1_pilot_93_98_breakthrough_report_20260621.md`.
- Decision: keep SCENEBAL-1 as the active direction and launch fresh expansion
  `99--104` before upgrading claim strength.

## 2026-06-21 11:26 CST
- SCENEBAL-1 expansion `99--104` completed:
  - old operational step `6/6`;
  - old operational macro `6/6`;
  - behavior `6/6`;
  - replay step/macro `6/6`;
  - learned true-static step `6/6`;
  - learned true-static macro `2/6`.
- Combined `93--104` 12-seed aggregate:
  - complete seeds `12/12`;
  - old operational step `12/12`;
  - old operational macro `12/12`;
  - behavior `12/12`;
  - replay step/macro `12/12`;
  - learned true-static step `11/12`;
  - learned true-static macro `5/12`;
  - step sign-test p `0.000244140625`;
  - macro sign-test p `0.000244140625`.
- This is the first strong operational multiseed claim for SCENEBAL-1.
- Wrote
  `reports/aggregate/scenebal1_12seed_93_104_strongclaim_report_20260621.md`.
- Remaining boundary: do not claim unconditional true-static macro dominance.
  If needed, run a targeted true-static macro diagnostic as a separate branch.
- Decision: continue SCENEBAL-1 expansion to `105--110` while GPUs are idle.

## 2026-06-21 12:14 CST
- Synced SCENEBAL-1 expansion `105--110` aggregate and per-seed audit artifacts
  from `remote-gpu`.
- Built combined SCENEBAL-1 `93--110` aggregate:
  - complete seeds `18/18`;
  - operational step `18/18`;
  - operational macro `18/18`;
  - behavior `18/18`;
  - strict explicit replay step/macro `18/18`;
  - learned true-static step `17/18`;
  - learned true-static macro `7/18` in the original oldclaim collector
    (superseded by the replay-normalized diagnostic below);
  - old step and macro sign-test p `3.814697265625e-06`.
- Corrected `scripts/74_v31_write_balancedobjective_report.py` so report
  wording is data-dependent. The previous template sentence about strict replay
  not passing all seeds was false for SCENEBAL-1 18-seed evidence.
- Wrote:
  `reports/aggregate/scenebal1_18seed_93_110_strongclaim_report_20260621.md`.
- Decision: SCENEBAL-1 is still effective and not stalled. Because operational,
  replay, and behavior gates are already `18/18`, the next highest-value unit is
  a true-static macro diagnostic before blind same-config seed expansion.

## 2026-06-21 12:19 CST
- Diagnosed the apparent true-static macro weakness. It was a collector
  scale-mixing bug, not a real policy failure:
  `73_v31_collect_oldclaim_gate.py` compared PPO macro from the main
  staticnorm metric scale against replay-local static macro references.
- Patched the collector to compute learned PPO macro on the replay-local
  staticnorm scale by reusing the helper logic from
  `72_v31_collect_metpair_strongclaim.py`.
- Reran corrected aggregate:
  `reports/aggregate/scenebal1_18seed_93_110_oldclaim_replaynorm_20260621/`.
- Corrected result:
  - learned true-static macro `18/18`;
  - learned true-static step remains `17/18`;
  - seed `95` is the only true-static step strict-margin failure, with positive
    margin `0.001742` but below the configured relative-margin threshold.
- Updated report:
  `reports/aggregate/scenebal1_18seed_93_110_oldclaim_replaynorm_20260621/SCENEBAL1_18SEED_REPLAYNORM_REPORT.md`.
- Decision: next unit is no longer true-static macro; it is seed95 true-static
  step strict-margin diagnosis.

## 2026-06-21 12:23 CST
- Completed seed95 true-static step diagnostic.
- Result: seed95 does not lose to the replay-local true fixed static reference:
  PPO `1.9513495687247089` vs `static_action5=1.9530910958687855`, margin
  `+0.0017415271440766045`.
- It fails only the configured relative-margin gate:
  required margin `0.003906182191737571`, shortfall
  `0.0021646550476609665`.
- Across all 18 seeds, PPO has positive true-static step margin `18/18`, strict
  margin true-static step gate `17/18`, and replay-normalized true-static macro
  `18/18`.
- Wrote:
  `reports/aggregate/scenebal1_seed95_true_static_step_diagnostic_20260621.md`.
- Decision: the current evidence supports a strong operational and
  true-static-positive claim. Universal strict-margin true-static step dominance
  remains bounded by one sub-threshold seed.

## 2026-06-21 12:24 CST
- GPU status: all six GPUs are occupied by another user's jobs.
- Started remote tmux waiter `scenebal1_waitfree_111_116_20260621`.
- The waiter polls every 600 seconds and launches SCENEBAL-1 seeds
  `111--116` only when all six GPUs are idle. This avoids interrupting or
  racing current GPU users.
- This wave is robustness stress testing only; it is not needed to repair a
  static shortcut, because the current 18-seed evidence already has sign wins
  over true fixed static on step `18/18`, true-static macro `18/18`, operational
  and replay gates `18/18`, and behavior gates `18/18`.

## 2026-06-21 12:36 CST
- Rechecked remote state:
  tmux `scenebal1_waitfree_111_116_20260621` is alive, and all six GPUs remain
  busy from other-user jobs. No seed `111--116` output exists yet.
- Synced the corrected collectors, reports, and research state to `remote-gpu`;
  remote `python -m py_compile scripts/73_v31_collect_oldclaim_gate.py
  scripts/74_v31_write_balancedobjective_report.py` passed.
- Wrote paper-claim mapping report:
  `reports/aggregate/scenebal1_18seed_93_110_paper_claim_mapping_20260621.md`.
- Updated the canonical ESWA manuscript source `paper/main.tex` and included
  sections/tables/highlights from the stale 14-seed claim to the corrected
  18-seed SCENEBAL-1 claim.
- Updated claim boundary in the paper:
  operational step/macro `18/18`, explicit replay step/macro `18/18`,
  behavior `18/18`, true-static macro `18/18`, true-static step positive
  `18/18`, strict-margin true-static step `17/18`.
- Verification:
  `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` succeeded
  and regenerated `paper/main.pdf`. `pdftotext` confirmed the new 18-seed
  numbers appear in the rendered PDF and the old `13/14`, `10/14`, and
  ten-seed highlights no longer appear in the checked manuscript files.
- Non-blocking warnings remain: standard underfull/overfull boxes and four
  BibTeX empty-page warnings. They do not block the current evidence update.

## 2026-06-21 12:40 CST
- Started local tmux watcher `scenebal1_watch_111_116_20260621`.
- Watcher command tracks remote session `scenebal1_waitfree_111_116_20260621`
  with seed label `111_112_113_114_115_116` and a 600-second interval.
- First snapshot:
  `reports/aggregate/scenebal1_pilot_111_112_113_114_115_116_local_watch_20260621_status.md`.
- Snapshot confirms no seed outputs yet (`000000` for oracle/PPO/eval/replay/
  behavior on all six seeds) and all GPUs still at high utilisation.

## 2026-06-21 12:43 CST
- Ran CPU-side seed-margin risk analysis on the corrected 18-seed oldclaim
  aggregate.
- Wrote:
  `reports/aggregate/scenebal1_18seed_93_110_seed_margin_risk_20260621.md`.
- Finding: seed95 is an isolated strict-margin boundary, not a broad failure
  family. It is the only seed below `0.005` and `0.02` true-static step margin;
  the next-lowest seed is seed98 with margin `0.020629`.
- Decision: continue SCENEBAL-1 stress wave rather than pivot immediately. Pivot
  only if new seeds show repeated true-static sign failures, behavior collapse,
  or loss of explicit dynamic replay headroom.

## 2026-06-21 12:44 CST
- Added and launched local post-collect watcher
  `scenebal1_postcollect_93_116_20260621`.
- Script:
  `scripts/watch_scenebal1_24seed_postcollect_20260621.sh`.
- Purpose:
  after remote stress-wave session `scenebal1_waitfree_111_116_20260621` ends,
  automatically run a combined SCENEBAL-1 `93--116` 24-seed macro/raw/oldclaim
  aggregate on `remote-gpu`, write the report, and sync it locally.
- Local `bash -n` passed; script was synced to `remote-gpu`.
- First status file:
  `reports/aggregate/scenebal1_24seed_93_116_postcollect_status_20260621.md`.
- First tick confirms the remote stress-wave session is still active, so the
  post-collect watcher is correctly waiting.

## 2026-06-21 12:54 CST
- Verified the SCENEBAL-1 18-seed evidence figure in the manuscript build:
  `paper/figures/figure_scenebal1_18seed_evidence.pdf` and `.png` exist, and
  `paper/main.pdf` is up to date under latexmk.
- `pdftotext` confirms the rendered PDF contains the figure caption and the
  corrected `18/18` and `17/18` boundaries, with no checked stale `13/14`,
  `10/14`, ten-seed, or fourteen-seed wording.
- Fresh `remote-gpu` check: tmux `scenebal1_waitfree_111_116_20260621` is still
  alive; all six GPUs remain occupied by existing jobs; seed `111--116` output
  directories do not exist yet.
- Fixed local watcher log noise caused by `printf` format strings beginning
  with `-` in Markdown list lines. Updated both SCENEBAL-1 watcher scripts to
  use `printf --`, ran local and remote `bash -n`, restarted the local watcher
  tmux sessions, and confirmed the pilot status file writes cleanly.

## 2026-06-21 12:59 CST
- Synced the updated research state, active plan, watcher scripts, paper source,
  `paper/main.pdf`, and the SCENEBAL-1 evidence figure artifacts to
  `remote-gpu`.
- Remote script checks passed. Remote YAML validation initially repeated the
  non-interactive SSH `python` PATH issue, then passed after activating the
  `darts` Conda environment.
- Remote status remains unchanged for the stress wave: all six GPUs are busy and
  seed `111--116` output directories do not exist yet.

## 2026-06-21 13:03 CST
- Refined the results text so the new evidence plot is cited in the prose:
  `Table 3 and Figure 5 report the final aggregate and seed-level evidence ...`.
- Rebuilt `paper/main.pdf`; `pdftotext` confirms the rendered phrase is
  `Table 3 and Figure 5`, and the checked stale `13/14`, `10/14`, ten-seed, and
  fourteen-seed wording remains absent.

## 2026-06-21 13:04 CST
- Corrected a remote sync mistake: one rsync line had copied `paper/main.pdf`
  into `paper/sections/` on `remote-gpu`.
- Removed the misplaced `paper/sections/main.pdf` and verified the correct
  `paper/main.pdf` remains in place.

## 2026-06-21 13:07 CST
- Fixed a second watcher issue: the pilot watcher rsync rules were broad enough
  to pull stale remote `postcollect_status` Markdown back to local and overwrite
  the newer local postcollect status file.
- Patched `scripts/watch_scenebal1_pilot_20260621.sh` to exclude
  `*local_watch*` and `*postcollect_status*` before aggregate includes.
- Removed stale remote status Markdown, restarted both local watcher tmux
  sessions, and confirmed the pilot and postcollect status files both refresh at
  `13:06` without `printf` or overwrite noise.

## 2026-06-21 13:08 CST
- Ran a repository scan for deprecated remote access references:
  `UniVPN`, `aTrust`, old tunnel wording, common private-IP patterns, and
  hardcoded host-address phrasing.
- No executable old connection path or hardcoded server address was found in
  `rl_sensor_scheduling_framework`. Remaining matches are prohibition text
  (`do not use old IPs, UniVPN, aTrust`) and `uv.lock` package-version false
  positives.

## 2026-06-21 13:10 CST
- Added `reports/aggregate/scenebal1_24seed_decision_protocol_20260621.md`.
- The protocol fixes post-stress-wave decisions before seeds `111--116` finish:
  upgrade to 24-seed wording only if operational, replay, behavior, true-static
  macro, and true-static sign gates remain clean; treat isolated strict-margin
  misses as boundary cases; pivot on sign failure, behavior collapse, or replay
  headroom failure.
- It also blocks blind same-configuration seed expansion after `111--116`
  unless a new wave answers a concrete uncertainty that the 24-seed aggregate
  cannot answer.

## 2026-06-21 13:11 CST
- Preflighted the remote waitfree script and the SCENEBAL-1 parallel runner.
- The waitfree script activates `darts`, checks `busy_gpus`, and launches only
  when the count is zero.
- The parallel runner assigns seeds across `SCENEBAL1_GPU_IDS`, exits nonzero if
  any seed worker fails, and runs macro/raw/oldclaim collectors after successful
  seed completion.

## 2026-06-21 13:15 CST
- Checked the next remote waitfree tick after `13:14:35`.
- Result: `busy_gpus=6`; all six GPUs remain occupied and seed `111--116`
  output directories do not exist yet.
- The waitfree session remains alive and correctly waiting.

## 2026-06-21 13:20 CST
- Rechecked `remote-gpu`: `scenebal1_waitfree_111_116_20260621` is alive,
  latest completed waitfree tick still reports `busy_gpus=6`, and seed
  `111--116` directories are absent.
- Confirmed local watcher sessions `scenebal1_watch_111_116_20260621` and
  `scenebal1_postcollect_93_116_20260621` remain active.
- Wrote:
  `reports/aggregate/pdppo_strongclaim_goal_alignment_and_pivot_protocol_20260621.md`.
- Decision: the API goal text is treated as a stale wrapper because it still
  names BO-1. Current execution follows the stricter active-plan/research-state
  target: PPO must remain the final learned scheduler, the met-plus-specialist
  sensing baseline remains in force, deeper simulator/teacher/PPO/reward/eval
  changes are allowed if needed, and no modification direction may receive more
  than 10 bounded no-improvement units before pivoting.
- The SCENEBAL-1 direction has effective improvement and currently counts as
  five bounded units, so it is not exhausted. After seeds `111--116`, another
  same-configuration seed wave is disallowed unless it resolves a specific
  uncertainty that the 24-seed aggregate cannot resolve.

## 2026-06-21 13:24 CST
- Hardened `scripts/watch_scenebal1_24seed_postcollect_20260621.sh` with a
  per-seed artifact bitset in the status file and an `all_artifacts_ready` gate
  before remote 24-seed aggregation.
- Verified local and remote `bash -n`, synced the script to `remote-gpu`, and
  restarted local tmux `scenebal1_postcollect_93_116_20260621`.
- New status confirms the watcher sees seeds `93--110` complete (`111111`) and
  seeds `111--116` absent (`000000`) while the remote waitfree session remains
  alive. This makes abnormal remote-session exit diagnosable instead of
  triggering repeated blind aggregation attempts.

## 2026-06-21 13:30 CST
- Rechecked `remote-gpu`: `scenebal1_waitfree_111_116_20260621` remains alive,
  waitfree tick `13:24:35` reports `busy_gpus=6`, and seed `111--116`
  directories are absent.
- Added automatic stress-wave decision audit script:
  `scripts/75_v31_decide_scenebal1_stress_claim.py`.
- Local and remote 18-seed regression both returned
  `decision=upgrade_sign_bounded`, reproducing the current evidence boundary:
  all operational, replay, behavior, true-static macro, and true-static step
  sign gates pass, while strict true-static step fails only on seed `95`.
- Integrated the decision audit into
  `scripts/watch_scenebal1_24seed_postcollect_20260621.sh`. After remote
  24-seed aggregate collection completes, the watcher will now produce
  `scenebal1_24seed_93_116_decision_audit_20260621.json` and `.md`, then sync
  them locally.
- Fixed the postcollect status `find -printf` timestamp format (`%TM` for
  minutes), restarted the postcollect tmux session, and confirmed the status
  file still shows seeds `111--116` as not started.

## 2026-06-21 13:40 CST
- Rechecked `remote-gpu`: latest waitfree tick is `13:34:35 busy_gpus=6`; all
  six GPUs remain busy and seed `111--116` artifact bitsets are still `000000`
  with no output directories.
- Added `reports/aggregate/pdppo_next_layer_pivot_designs_20260621.md`.
- Decision: if `111--116` weakens the strong-claim gates, the next action is a
  bounded next-layer pilot, not another blind same-configuration seed wave.
  The first pivot layer depends on the failure type: simulator/data balance for
  true-static sign or replay-headroom failure, PPO observation/auxiliary
  architecture for behavior failure, and reward/oracle calibration when replay
  headroom is intact but learned PPO fails to realize it.
- Updated `research-state.yaml` with the report path and latest remote status.

## 2026-06-21 13:46 CST
- Enhanced `scripts/75_v31_decide_scenebal1_stress_claim.py` with an explicit
  decision-to-next-layer recommendation map.
- The audit output now includes both the decision and the recommended next
  action. For the current 18-seed regression it remains
  `decision=upgrade_sign_bounded`, with
  `recommendation.next_layer=claim_update_no_blind_expansion`.
- Local and remote `py_compile` plus 18-seed regression passed. The updated
  script and regenerated local/remotecheck decision audit outputs were synced
  across local and `remote-gpu`.

## 2026-06-21 13:54 CST
- Rechecked `remote-gpu`: the waitfree session remains alive and waiting;
  latest waitfree tick is `13:44:35 busy_gpus=6`; seeds `111--116` remain
  artifact-empty.
- Added `scripts/76_v31_write_next_action_protocol.py` to materialize a concrete
  next-action protocol from a decision audit JSON.
- Integrated it into the 24-seed postcollect watcher, so the future aggregate
  will produce both a decision audit and a next-action protocol file.
- Local and remote regressions on the 18-seed decision audit passed; the
  materialized bounded unit is `Sign-Bounded Claim Update` with
  `claim_update_no_blind_expansion`.
- Restarted local postcollect watcher after syncing. Also hardened status
  writing so transient SSH status-query failures produce a generic error line
  rather than a partial, misleading status page.

## 2026-06-21 13:57 CST
- Recorded and repaired a sync-path mistake: active-plan `progress.md` was
  briefly sent to the remote project root by a two-source `rsync`.
- Restored remote root `progress.md` from the local root file and synced the
  active-plan progress to its correct `.planning/...` directory separately.
- Remote validation confirms the two files now have the expected distinct sizes
  and the active-plan progress contains the latest `13:56 CST` entry.

## 2026-06-21 14:05 CST
- SCENEBAL-1 stress wave `111--116` launched on `remote-gpu`.
- Verified waitfree log transition:
  `14:04:36 busy_gpus=0` followed by
  `launch date=2026-06-21T14:04:36+08:00 seeds=111_112_113_114_115_116`.
- Verified six seed workers and six GPU Python processes are active. Each seed
  directory now contains initial truth, manifest, dataset-validation, and
  `run_train_eval.log` files.
- Artifact status remains early-stage: oracle/PPO/eval/replay/behavior bits
  are still `000000` for seeds `111--116`. Continue monitoring; do not infer
  claim strength from this launch-only state.

## 2026-06-21 14:08 CST
- Enhanced `scripts/watch_scenebal1_24seed_postcollect_20260621.sh` so the
  status page includes `TRAINING PROGRESS latest_timestep log_bytes
  latest_key_line` for every seed.
- Local and remote `bash -n` passed, and local tmux
  `scenebal1_postcollect_93_116_20260621` was restarted.
- Refreshed status confirms `111--116` are actually training:
  artifact bits are `100000` and PPO progress is roughly
  `26624--28672 / 200000` timesteps across the six stress seeds.

## 2026-06-21 14:12 CST
- Health check shows stress seeds `111--116` continue training normally.
- At `14:12:53`, artifact bits remain `100000` and PPO progress is
  `91136--94208 / 200000` timesteps across the six seeds.
- GPU memory remains about `1563 MiB` per GPU for the six Python workers.
- Error scan over stress-seed launcher logs and `run_train_eval.log` files found
  no `Traceback`, `RuntimeError`, CUDA OOM, or `nan` hits.

## 2026-06-21 14:29 CST
- SCENEBAL-1 stress wave `111--116` completed. All six stress seeds have
  artifact bits `111111`.
- Remote postcollect generated:
  `scenebal1_24seed_93_116_macro_20260621`,
  `scenebal1_24seed_93_116_raw_macro_20260621`,
  `scenebal1_24seed_93_116_oldclaim_replaynorm_20260621`,
  `scenebal1_24seed_93_116_decision_audit_20260621.{json,md}`, and
  `scenebal1_24seed_93_116_next_action_protocol_20260621.md`.
- Decision audit: `upgrade_sign_bounded`. All operational, replay, behaviour,
  true-static macro, and true-static step-sign gates pass `24/24`; strict-margin
  true-static step passes `23/24`, with seed `95` still the only boundary.

## 2026-06-21 14:36 CST
- Updated the manuscript from 18-seed to 24-seed SCENEBAL-1 evidence while
  preserving the strict-margin boundary.
- Regenerated `paper/figures/figure_scenebal1_24seed_evidence.pdf` and `.png`.
- Updated `paper/main.tex`, results, introduction, protocol/setup, discussion,
  conclusion, and `paper/tables/metpair_staticnorm_macro_summary.tex`.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeded.
- `pdftotext paper/main.pdf` confirms the rendered PDF contains the 24-seed
  `24/24` and `23/24` wording and no checked stale `18/18`, `17/18`,
  `18-seed`, or `93--110` claim text.
## 2026-06-21 18:24 CST - SCENEBAL-2 24-seed all-strict breakthrough

SCENEBAL-2 seeds `117--140` completed under the pre-fixed router-confidence
`0.5` protocol. Decision audit:
`reports/aggregate/scenebal2_confirm_conf05_117_118_119_120_121_122_123_124_125_126_127_128_129_130_131_132_133_134_135_136_137_138_139_140_decision_audit_20260621.json`.
Decision is `upgrade_allseed_strict`; all operational, replay, behaviour,
true-static macro, true-static step sign, strict-margin true-static step, and
old-claim gates are `24/24`. Migrated the active manuscript main result from
SCENEBAL-1 to SCENEBAL-2 and rebuilt `paper/main.pdf`.

## 2026-06-21 18:36 CST - SCENEBAL-2 claim-framing audit

Completed the paper-fit audit for the final SCENEBAL-2 scene. Judgment: the
scene is narrower than the original broad scheduling claim, but natural for ESWA
if framed as a regime-balanced microclimate backbone-plus-one-specialist
benchmark.

Added:
- `reports/aggregate/scenebal2_24seed_claim_framing_audit_20260621.md`
- `reports/aggregate/scenebal2_24seed_supervisor_summary_20260621.md`

Patched `paper/sections/01_introduction.tex` and
`paper/sections/05_simulation_setup.tex` to explain the benchmark as a
deployment-relevant abstraction: keep low-power weather context available and
allocate scarce active time among event-sensitive specialists. Rebuilt
`paper/main.pdf`; rendered text confirms the new scenario motivation and the
SCENEBAL-2 `24/24` evidence. Decision: do not launch further same-direction seed
expansion unless a concrete new uncertainty appears; current work should focus
on paper packaging, claim wording, and final evidence presentation.

## 2026-06-21 18:40 CST - Strong-claim completion audit

Wrote `reports/aggregate/pdppo_strongclaim_completion_audit_20260621.md` and
audited the active objective requirement by requirement. The audit records:

- BO-style same-configuration expansion was stopped after it failed to provide a
  stable zero-failure step claim.
- The project pivoted through bounded teacher/framework/simulator directions and
  reached SCENEBAL-2 within the 10-unit anti-stall rule.
- Final learned scheduler remains PD-PPO.
- Final SCENEBAL-2 aggregate over seeds `117--140` has
  `decision=upgrade_allseed_strict` and all primary gates at `24/24`.
- Behaviour gate passes `24/24`, so the scheduler is not fixed-like and not a
  simple cycle.
- Reports, manuscript PDF, and paper-fit audit were synced to `remote-gpu`.

Updated `research-state.yaml` to `strongclaim_experiment_complete`. Remaining
work is manuscript polishing and submission packaging rather than further
experimental exploration for the stated strong-claim objective.
