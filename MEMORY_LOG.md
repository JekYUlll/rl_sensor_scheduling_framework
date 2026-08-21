# Memory Log

## Key Results
- BO-1 14-seed checkpoint: macro old-claim gate 14/14, behaviour gate 14/14,
  learned PPO beats static/rule/operational baselines 14/14 on step and macro
  objectives; strict explicit replay step gate 11/14.
- BO-1 26-seed checkpoint: macro gate 26/26, behaviour gate 26/26, learned macro
  wins 26/26, strict explicit replay step gate 20/26. This supports a strong
  regime-balanced ESWA claim but not an unqualified step-weighted strict replay
  claim.
- Seed55 step diagnostic: `subtype_auto` replay with static duty guard disabled
  passes strict step gate against true fixed static, suggesting teacher/replay
  construction is a fixable source of step failures.
- Seed59 step diagnostic also passes: best replay step loss `3.106645` versus
  static reference `3.379619`, margin `+0.272974` absolute / `+0.080771`
  relative. Seed55+seed59 make the subtype/static-candidate replay fix more than
  a one-off result.
- BO-1 wave 67--78 interim odd-seed result: seeds 67,69,71,73,75,77 are 6/6 on
  strict replay step gate, 6/6 on macro gate, and 6/6 on behavior gate, with no
  fixed-like or simple-cycle-like policies. Even seeds are still running, so this
  is an interim signal rather than final aggregate evidence.
- BO-1 38-seed checkpoint through wave 67--78: complete seeds 38, learned step
  wins 36/38, learned macro wins 38/38, strict replay step gate 32/38, strict
  replay macro gate 38/38, behavior gate 38/38. The latest wave 67--78 is 12/12
  on old step gate and 12/12 on learned true-static step gate. This is effective
  improvement, so BO-1 continues into wave 79--90 rather than pivoting yet.
- BO-1 wave 79--90 first half is mixed: seed83 fails strict replay step gate
  despite macro/behavior pass, and seed87 passes replay gates but fails behavior
  as fixed-like / not state-dependent. This suggests the next pivot, if needed,
  should target teacher/PPO behavior robustness rather than more blind seed
  expansion.
- BO-1 stopped after 41--90: 50 seeds, learned step wins 45/50, learned macro
  wins 50/50, strict replay step 40/50, strict replay macro 49/50, behavior
  49/50. This is strong statistical evidence but not stable zero-failure. Same
  config expansion was stopped and AT-1 `subtype_static_auto` pilot launched on
  seeds 83,84,86,87,91,92 in parallel across GPUs.
- AT-1 `subtype_static_auto` failed: oldclaim step 2/6, macro 3/6, behavior
  3/6. It repaired seed83/86 but seed84 stayed slightly step-negative and
  seed87/91/92 failed behavior/oldclaim gates. RT-1 router-threshold sweep
  started with aligned metrics/behavior collection via new `--metrics-eval-dir`.
- RT-1 router-threshold sweep failed; threshold changes do not solve behavior.
  BR-1 behavior-regularized PPO branch launched with mild duty-balance and
  duty-score feedback, preserving PPO and the sensor baseline. The inherited
  ortholinear budget was `200k` PPO timesteps, and local watcher
  `behaviorreg_pilot_local_watch_20260621` was started.
- BR-1 completed and failed: old step `3/6`, old macro `4/6`, behavior `4/6`,
  replay `5/6`, replay macro `6/6`, learned true-static step `2/6`. Failure
  seeds are step `83,87,92`, macro/behavior `87,92`. Do not expand BR-1; pivot
  to BD-1, an explicit state-dependent behavior signal or PPO architecture
  change.
- BD-1 subtype-action CE/margin pilot completed and failed: old step `2/6`,
  old macro `4/6`, behavior `4/6`, replay `5/6`, replay macro `6/6`,
  learned true-static step `3/6`. Failure seeds are step `83,84,87,92`,
  macro/behavior `87,92`. The loss was active in logs, so this is a real
  negative result; do not expand BD-1.
- BRG-1 launched after BD-1 failed. It preserves PPO and the BO-1 met+specialist
  sensor baseline, but enables observable regime-belief features, event-gated
  actor, stronger subtype auxiliary learning, and a conservative subtype-router
  deployment head. Remote tmux: `behaviorbrg_pilot_parallel_83_92_20260621`.
- BRG-1 completed as an effective improvement but not a breakthrough: old step
  `3/6`, old macro `5/6`, behavior `5/6`, operational macro `6/6`, replay macro
  `6/6`, learned true-static step `4/6`. Router-eval macro/behavior are `6/6`,
  but raw seed `87` remains fixed-like and seed `92` remains step-negative
  against validation-selected static. Run exactly one bounded BRG-2 follow-up
  before pivoting if it does not improve.

## Recent Decisions
- Current active direction is BRG-2: matched-confidence observable-regime-belief
  PPO follow-up after BRG-1 improved macro/behavior but failed the full strong
  claim.
- BO-1 same-config expansion, AT-1 autoteacher, and RT-1 router-threshold sweep
  are closed directions unless later evidence explicitly reopens them. BR-1 is
  also closed and should not be expanded. BD-1 is closed and should not be
  expanded.
- BRG-2 should be tested as a bounded pilot on hard seeds `83,87,92` plus
  repaired/control seeds `84,86,91`, with strict replay and behavior audit.
  Match raw/eval subtype-router confidence at `0.70`, use modestly higher
  entropy, and keep BD-1 subtype-action CE/margin losses disabled.
- BRG-2 launched in remote tmux
  `behaviorbrg2_pilot_parallel_83_92_20260621` with local watcher
  `behaviorbrg2_pilot_local_watch_20260621`. Treat it as unit 2 in the current
  BRG direction; pivot if it does not clearly improve the pilot.
- BRG-2 completed as the best hard/control pilot so far: old step `5/6`,
  default macro/behavior `5/6`, explicit replay `6/6`, subtype-aware
  macro/behavior `6/6`. Added subtype-aware behavior audit; seed92 custom PPO
  is subtype-dependent (`MI=0.587826`, `L1=2.0`) while fixed static still fails.
  The remaining blocker is seed92 step performance (`3.046497` vs selected
  static `3.024605`). Proceed to BRG-3 with moderate action-fidelity on top of
  BRG-2; pivot if it fails.
- BRG-3 launched in remote tmux
  `behaviorbrg3_pilot_parallel_83_92_20260621` with local watcher
  `behaviorbrg3_pilot_local_watch_20260621`. It adds moderate subtype-action
  CE/margin to BRG-2 rather than replacing PPO.
- BRG-3 failed: old step `3/6`, macro/behavior `6/6`, replay step `3/6`,
  learned true-static step `2/6`. It worsened BRG-2 step performance, so
  action-fidelity should not be retried. Pivot to TEMPORAL-1: temporal/lead-aware
  PPO or reward-credit modification, because seed92 explicit replay headroom is
  lead-based.
- TEMPORAL-1 launched in remote tmux
  `temporal1_pilot_parallel_83_92_20260621` with local watcher
  `temporal1_pilot_local_watch_20260621`. It extends subtype context/teacher
  lead to `16`, uses replay leads `0,4,8,12,16`, and keeps action CE/margin off.
- TEMPORAL-1 completed as a hard/control breakthrough: old step `6/6`, behavior
  `6/6`, replay `6/6`, raw macro `6/6`. Seed92 is repaired (`custom_ppo
  2.777641` vs selected static `2.845988`). This is the leading candidate but
  still needs fresh-seed expansion; next seeds are `93--98`.
- TEMPORAL-1 fresh-seed expansion `93--98` launched in remote tmux
  `temporal1_pilot_parallel_93_98_20260621` with local watcher
  `temporal1_pilot_local_watch_93_98_20260621`.
- TEMPORAL-1 fresh-seed expansion `93--98` completed mixed: old step `4/6`,
  old macro `6/6`, behavior `6/6`, strict replay step `5/6`, strict replay
  macro `6/6`, and dedicated macro collectors `6/6`. It supports macro +
  non-fixed dynamic behavior but not a zero-failure step claim. Seed95 is the
  structural/replay-headroom warning; seed96 is learned-credit; seed97 is a
  true-static boundary case. Wide lead/dwell replay diagnostic
  `temporal1_wide_replay_diag_95_97_20260621` is running.
- TEMPORAL-1 wide replay diagnostic completed: seed95 remains replay-negative
  (`3.600575` vs strict fixed static `3.520518`), while seed96 and seed97 are
  replay-positive. Same TEMPORAL-1 expansion is closed. SCENEBAL-1 is now
  running on seeds 93--98 to rebalance simulator/target generation against the
  fixed `met_station_core|fc4_flux` shortcut while preserving PPO and the
  met+specialist sensor geometry.
- Current target journal is ESWA. CRST references are historical only.
- Hard exploration constraints: PPO must remain the final learned scheduler;
  each modification direction has at most 10 bounded work units without
  effective improvement; the current microclimate sensor setup remains the
  physical baseline, with only moderate explainable simulated variants allowed.
- SCENEBAL-1 seed93 surfaced a staticnorm sampling-coverage issue: validation
  static-selection windows can miss a subtype even when the full truth sequence
  contains it. `scripts/25_v2_train_custom_ppo.py` now fills missing subtype
  normalizers from subtype-positive normalization-split windows and records
  `reward_staticnorm_normalizers.json` plus
  `reward_staticnorm_fallback_candidates.csv`.
- SCENEBAL-1 pilot 93--98 completed as a breakthrough: old operational step
  `6/6`, old macro `6/6`, behavior `6/6`, replay `6/6`. Seed95 is repaired
  operationally (`custom_ppo=1.951350` vs selected static `1.963998`). Remaining
  boundary: learned true-static step `5/6`, true-static macro `3/6`; expand
  SCENEBAL-1 before upgrading the final claim.
- SCENEBAL-1 expansion 99--104 reproduced the effect. Combined 93--104:
  operational step `12/12`, operational macro `12/12`, behavior `12/12`,
  replay step/macro `12/12`, sign-test p `0.000244140625`. This supports a
  strong operational claim. Do not overstate true-static macro dominance:
  true-static step `11/12`, true-static macro `5/12`.
- SCENEBAL-1 expansion 105--110 completed and the combined 93--110 aggregate is
  now the strongest current evidence: operational step `18/18`, operational
  macro `18/18`, behavior `18/18`, strict explicit replay step/macro `18/18`,
  learned true-static step `17/18`, step/macro sign-test p
  `3.814697265625e-06`. A corrected replay-normalized collector shows learned
  true-static macro is actually `18/18`; the earlier `7/18` value mixed metric
  scales. The remaining boundary is seed95 true-static step strict-margin, not
  macro.
- Seed95 true-static step is a positive-margin but sub-threshold case, not a
  loss: PPO `1.9513495687247089` vs true fixed static
  `1.9530910958687855`, margin `+0.0017415271440766045`, required strict margin
  `0.003906182191737571`. Thus PPO beats true fixed static in sign `18/18`, but
  strict-margin true-static step is `17/18`.
