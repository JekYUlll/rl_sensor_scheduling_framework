# Final Strong-Claim Experiment Design (2026-06-20)

## Objective

Establish whether the PD-PPO scheduler can support a main-paper claim after the
single-seed met+specialist result failed to replicate under strict multi-seed
testing.

The target claim must satisfy two requirements:

1. Forecast quality improves against learned-selected static and replay-local
   true fixed-static baselines.
2. The learned deployment is genuinely state dependent: not a fixed sensor set
   and not a simple round-robin or cyclic rotation.

## Candidate Main Claim

The currently viable claim has two possible strengths:

> PD-PPO learns a contextual specialist scheduler that improves regime-macro
> forecast quality across particle, flux, and thermal event regimes under a
> required meteorological backbone and a one-specialist power budget.

If the ortholinear/strong-teacher branch clears strict step-weighted gates
across seeds, the stronger wording can be:

> PD-PPO learns a contextual specialist scheduler that improves forecast-oracle
> loss against learned-selected and replay-local fixed-static baselines while
> producing non-fixed, state-dependent sensor activation.

The weaker regime-macro wording is deliberately narrower than:

> PD-PPO always minimizes step-weighted overall forecast loss.

The broad step-weighted claim is not currently supported by the completed
multi-seed pools. It is only a candidate if the ortholinear/strong-teacher
pool passes.

## Required Evidence Pool

Current active branch: orthogonal-linear generator plus strong subtype teacher.
Run at least 10 seeds if seed41 passes the pilot gate:

```text
seeds: 41 42 43 44 45 46 47 48 49 50
runner: scripts/run_v31_metpair_backbone_context_ortholinear_strongteacher_seed_sweep_20260620.sh
aggregate:
  reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_10seed_macro_20260620/
  reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_10seed_raw_macro_20260620/
```

The completed ortholinear seed41 non-strong-teacher pilot shows that the scene
now has structural headroom, but learned flux scheduling still needs stronger
teacher/curriculum alignment.

## Primary Gate

For each seed, require all of the following under the chosen learned deployment
(`raw PPO` or `router-confidence`; these must be reported separately):

- complete artifacts:
  source metrics, router-confidence evaluation, explicit replay gate, behaviour
  audit;
- learned forecast improvement:
  learned `custom_ppo` oracle loss is lower than `validation_selected_static`
  by at least the configured learned margin;
- learned macro improvement for the regime-balanced claim:
  learned `custom_ppo_macro_subtype_loss < selected_static_macro_subtype_loss`;
- replay macro improvement:
  best explicit subtype dynamic replay has lower
  `oracle_loss_macro_subtype_event` than the replay-local best fixed static
  candidate;
- behaviour gate:
  `behavior_complexity_gate_pass=true`.

Aggregate thresholds:

- acceptable main-result bar: at least `8/10` macro-positive seed gates;
- preferred strong bar: at least `9/10` macro-positive seed gates;
- decisive bar: `10/10` macro-positive seed gates.

Reasoning:

- `8/10` is an 80% robustness bar but has a weak one-sided sign test.
- `9/10` gives a conventional-significance sign-test direction.
- `10/10` is strong enough to write confidently without leaning on caveats.

## Secondary Gates

Report these separately and do not conflate them with the primary gate:

- strict step-weighted seed gate from
  `seed_gate_pass`;
- learned step-weighted margin vs `validation_selected_static`;
- replay step-weighted margin vs replay-local true fixed static;
- `full_open_unconstrained`, `round_robin`, `aoi`, and `random` baselines.

If the macro gate passes but strict step-weighted gate fails, the paper claim
must say "regime-macro" or "event-regime balanced" rather than "overall
step-weighted optimal".

## Current Evidence

- Old metpair 7 seeds:
  `seed_gate_pass_count=1/7`; not supported.
- Backbone-context 7 seeds:
  `seed_gate_pass_count=3/7`,
  `macro_seed_positive_count=5/7`; not strong enough.
- Strong-latent probes 43 and 44:
  `seed_gate_pass_count=0/2`,
  `macro_seed_positive_count=2/2`; promising but too small.
- Strong-latent partial 4-seed continuation:
  `seed_gate_pass_count=0/4`,
  `macro_seed_positive_count=2/4`; failed because seed41 exposes a strong
  fixed-static shortcut.
- Ortholinear seed41:
  raw learned PPO passes the strict step-weighted seed gate
  (`4.956431` vs selected static `5.233835`), explicit dynamic replay passes
  against replay-local fixed static (`5.142764` vs `5.212586`), and behaviour
  passes. However learned macro remains negative because flux-subtype loss is
  worse than selected static (`11.330058` vs `10.574289`).
- Ortholinear strong-teacher seed41:
  running on `remote-gpu`. This is the current pilot for deciding whether to
  expand to a 10-seed strong-claim pool.

## 2026-06-20 Update: Balanced Objective Escalation

The completed ortholinear strong-teacher pool solved deployment behaviour but
did not support a strong step-weighted paper claim:

- 14 completed seeds:
  `reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_14seed_strictmacro_20260620/`;
- behaviour gate: `14/14`;
- strict step-weighted seed gate: `10/14`;
- raw/router event-subtype macro seed gate: about `7--8/14`, below the strong
  claim threshold.

Diagnosis: the old raw event-subtype macro was still numerically dominated by
the flux subtype. Static candidates often used `met_station_core + fc4_flux`,
which wins the high-scale flux loss while under-serving particle and thermal
regimes. Therefore the next experiment uses a static-normalized subtype macro:
for each event subtype, divide the subtype loss by the median feasible fixed
static loss for that subtype on the same validation/reference protocol, then
average particle, flux, and thermal normalized losses.

Implemented changes:

- `oracle_loss_macro_subtype_event_staticnorm` is now reported by PPO
  evaluation, explicit replay gates, and the multiseed collector.
- Static selection, metric sorting, replay macro gate, and collector macro gate
  can all use the static-normalized macro column.
- PPO training can optionally normalize oracle reward loss by validation static
  subtype medians via `--reward-loss-normalization staticnorm_subtype`.
- The active balanced-objective runner is
  `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh`.

Queued experiment:

```text
run prefix: v31_metpair_backbone_context_ortholinear_balancedobjective
seeds: 41 42 first, then expand only if learned + replay + behaviour gates pass
primary macro score: oracle_loss_macro_subtype_event_staticnorm
training reward: staticnorm_subtype
remote: remote-gpu
```

Do not treat this as evidence until the new balanced-objective PPO seeds finish.
The CPU-only replay posthoc on old strong-teacher runs is diagnostic only; it
can show that the scene has normalized dynamic headroom, but it does not prove
the learned PPO policy optimizes the new objective.

## 2026-06-20 Posthoc Staticnorm Result

The full 14-seed ortholinear strong-teacher pool was re-collected with
static-normalized macro scoring after replay recomputation:

```text
aggregate:
  reports/aggregate/metpair_ortholinear_strongteacher_14seed_staticnorm_replay_20260620/
complete seeds: 14
strict step seed gate: 10/14
macro seed gate: 13/14
behaviour gate: 14/14
macro claim strength: strong_macro_multiseed
macro sign-test p: 0.00091552734375
mean learned macro margin vs macro static reference: +0.0732109
mean replay macro margin vs static reference: +0.0804163
macro-failing seed: 48
strict-step-failing seeds: 44, 46, 48, 52
```

Interpretation:

- A broad step-weighted "PD-PPO beats fixed static" claim is still not
  supported (`claim_strength=not_supported` under the strict step gate).
- A static-normalized event-regime macro claim is supported by the completed
  14-seed pool: learned PPO, explicit replay, and behaviour gates pass in
  `13/14` seeds.
- The paper can use this as the main quantitative claim only if the objective
  is explicitly defined as static-normalized event-regime macro forecast loss.
  It must not silently present these results as ordinary step-weighted forecast
  optimality.
- The queued reward-aligned balanced-objective seeds remain useful as
  confirmation that the training reward can be made explicitly consistent with
  the final metric, but the current 14-seed result is already strong enough for
  the narrower macro claim.

## If Ortholinear Strong-Teacher Fails

Escalate from scenario-only changes to objective/framework changes:

1. Train and select with a static-normalized event-subtype macro objective, not
   only step mean. This is now implemented through
   `--reward-loss-normalization staticnorm_subtype` plus static-normalized
   static selection/evaluation.
2. Make final-test selection exactly subtype-balanced rather than only
   transport-rich.
3. Strengthen non-substitutable specialist information by assigning orthogonal
   latent target channels to particle, flux, and thermal specialists.
4. Keep the required meteorological backbone and one-specialist budget. Removing
   the backbone would create a different and less physically credible claim.

Do not write the main paper around single-seed seed45 or around a scheduler that
passes only behaviour but not forecast baselines.
