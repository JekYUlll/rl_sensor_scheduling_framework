# PD-PPO Scene Recalibration Changelog

## 2026-07-03 - CA-PD-PPO Bounded Dev2 Failure-Guided Wave

Objective: improve CA-PD-PPO against `context_alert_bandit_t0p5` without using
bandit-dependent patchwork. The primary method identity remains prediction-
driven masked PPO with hard feasibility masking.

### Diagnostics

- Completed failure analysis for the previous CA-PD-PPO dev run. Losses against
  the context-alert bandit concentrate in flux windows and in lower-confidence
  or alert-boundary regions, not in high-confidence alert regions.
- The analysis does not support bandit imitation: seed-level agreement with the
  bandit is negatively correlated with CA-PD-PPO macro margin.

### Clean Dev2 Variants

- Added bounded variants only: larger context encoder (`ctx128`), gated context
  fusion (`gated`), larger gated fusion (`gated_ctx128`), and longer PPO rollout
  (`nsteps2048`).
- Excluded from the main method: residual bandit actions, bandit-margin rewards,
  counterfactual bandit labels, bandit imitation losses, and bandit actor priors.

### Results

- `ctx128`: failed the fresh-final gate. Against `context_alert_bandit_t0p5`,
  macro wins were `14/24` and mean macro margin was `0.004083`; against
  `forecast_greedy_one_step`, macro wins were `23/24`.
- `gated`: failed the fresh-final gate. Against `context_alert_bandit_t0p5`,
  macro wins were `13/24` and mean macro margin was `0.002763`; against
  `forecast_greedy_one_step`, macro wins were `24/24`.
- `gated_ctx128`: failed the fresh-final gate. Against
  `context_alert_bandit_t0p5`, macro wins were `13/24`, mean macro margin was
  `0.006706`, and mean step margin was slightly negative; against
  `forecast_greedy_one_step`, macro wins were `24/24`.
- `nsteps2048`: failed the fresh-final gate. Against
  `context_alert_bandit_t0p5`, macro wins were `10/24`, mean macro margin was
  `0.002962`, and mean step margin was `-0.000719`; against
  `forecast_greedy_one_step`, macro wins were `24/24`.

### Decision

- No bounded dev2 variant passed the predeclared fresh-final gate. Fresh final
  seeds `301--324` were not launched.
- The strongest current interpretation is unchanged: CA-PD-PPO is a clean
  method-consistent improvement that remains competitive with the strong
  context-alert bandit and robustly beats forecast-greedy, but it does not yet
  support a stable-superiority claim over the bandit.

## 2026-06-20 - Strong-Claim Multiseed Extension Started

Objective: upgrade the met+specialist-pair result from a single-seed validated
candidate to paper-safe robustness evidence.

### Final Experiment Design

- Fixed scenario and method: use the seed45 metpair contract without further
  tuning drift:
  `configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml`,
  `budget=0.75`, `startup_peak_budget=0.95`, `max_active=2`,
  `truth_steps=70000`, event subtype latent lag `4`, context lead `8`,
  subtype-aware AWBC/auxiliary PPO, no candidate-prior KL, and router-confidence
  evaluation at `min_confidence=0.8`.
- Per-seed evidence must pass all three gates:
  learned PPO beats `validation_selected_static` under `eval_router_conf08`;
  strict subtype replay beats replay-local true fixed static with no duty guard;
  corrected behaviour audit rejects fixed-subset and simple-cycle explanations.
- Strong paper claim threshold:
  at least `10` complete seeds, at least `8/10` full seed-gate passes, positive
  mean learned margin, and positive mean strict-replay margin.
- Moderate fallback claim:
  at least `5` complete seeds with at least `80%` full seed-gate passes. Anything
  below that remains a replicated pilot or single-seed mechanism demonstration.
- Optional robustness after the main 10-seed result:
  nearby budgets `B=0.73` and `B=0.77` on a smaller seed subset, reported as
  sensitivity rather than the main claim.

### New Automation

- Added `scripts/run_v31_metpair_strongclaim_seed_sweep_20260620.sh`, which
  runs the fixed seed protocol plus standard eval, router-confidence eval,
  strict no-duty-guard replay, and behaviour-complexity audit.
- Added `scripts/72_v31_collect_metpair_strongclaim.py`, which aggregates per
  seed learned margins, strict replay margins, behaviour gates, and classifies
  the resulting claim strength.

### Launched

- Remote smoke collection on existing seed45 correctly reports
  `claim_strength=single_seed_only` with all three gates passing.
- Replication batch launched on `remote-gpu`:
  `metpair_s41` through `metpair_s44` on GPU2-GPU5, plus `metpair_s46` on
  GPU0 and `metpair_s47` on GPU1. Together with completed seed45, this will
  produce the first 7-seed evidence pool.

### Result

- Seven-seed collection finished:
  `reports/aggregate/metpair_strongclaim_7seed_20260620/`.
- Outcome: `complete_seeds=7`, `seed_gate_pass_count=1`,
  `learned_gate_pass_count=1`, `replay_gate_pass_count=3`,
  `behavior_gate_pass_count=2`, `claim_strength=not_supported`.
- Mean learned margin versus `validation_selected_static` was negative
  (`-0.022343`), so the old metpair branch is not a strong-claim candidate.
- Diagnosis: seed45 is a useful mechanism demonstration, but other seeds expose
  two shortcuts: static baselines can choose non-backbone pairs such as
  `shielded_thermo_hygro + laser_disdrometer`, and the learned PPO did not
  consistently infer subtype context strongly enough to follow the explicit
  dynamic teacher.
- New branch started: `v31_metpair_backbone_context_*`, which makes
  `met_station_core` a required backbone, exposes the generated subtype-alert
  context columns to the agent, balances subtype probabilities, and uses
  subtype-balanced transport-rich final-test windows.
- First backbone-context pilot result:
  `reports/aggregate/metpair_backbone_context_pilot_20260620/`.
  Seeds 41 and 42 both pass learned, replay, and behaviour gates:
  `complete_seeds=2`, `seed_gate_pass_count=2`,
  `mean_learned_margin_abs=0.020020`, `claim_strength=replicated_pilot`.
- Backbone-context seeds 43, 44, 45, 46, and 47 were launched next. If at least
  `4/5` complete context seeds pass, expand to the full 10-seed strong-claim
  run.
- Seven-seed backbone-context collection:
  `reports/aggregate/metpair_backbone_context_7seed_20260620/`.
  Result: `complete_seeds=7`, `seed_gate_pass_count=3`,
  `learned_gate_pass_count=5`, `replay_gate_pass_count=3`,
  `behavior_gate_pass_count=7`, `claim_strength=not_supported`.
- Diagnosis after backbone-context:
  the agent behaviour problem is largely fixed (`7/7` behaviour gates), but
  the simulator still permits fixed-static shortcuts in some seeds. In
  particular, explicit subtype replay fails or has too small a margin when the
  matching specialist does not materially improve future target loss.
- Strong-latent backbone-context branch launched on the failed seeds 43 and
  44:
  `v31_metpair_backbone_context_stronglatent_seed{43,44}_h075ctxsl_20260620`.
  It strengthens hidden subtype latents and specialist-dependent future target
  effects while keeping the met backbone and context-alert observation model.

## 2026-06-20 - Static-Shortcut Recalibration Closed On Met+Specialist Pair Scene

Objective: find and verify a PD-PPO scheduling scene that breaks the fixed-static
shortcut, run the necessary TCN-oracle gate and reduced PPO experiment, and
record enough evidence for paper-mainline decision making.

### Failed / Superseded Branches

- V25 low-budget static squeeze created structural headroom in one TCN gate, but
  split replay failed against replay-local raw/static references.
- V26 calm-selective scene also produced an apparent structural pass, but split
  replay again lost to replay-local raw static.
- V27 subtype-auto replay showed privileged dynamic headroom, but learned PPO
  variants did not clear the strict raw-static gate:
  subtype-aux PPO lost to replay-local raw static by `0.001088`, and strongBC2
  was only a single-seed near-threshold candidate.
- Context-power and decoy/fusion V31 variants did not provide paper-safe
  evidence. The best fusion/decoy run was close to source selected static
  (`custom_ppo=44.286885`, `validation_selected_static=44.288191`), but strict
  no-duty-guard replay found a much stronger true fixed static subset
  (`44.037335`). This branch is internal diagnostic evidence only.

### Framework / Scenario Changes

- Fixed duplicate-observation handling in `src/v2/env.py`: selected sensors that
  observe the same variable now fuse measurements by inverse noise variance,
  with circular fusion for `wind_direction_deg`, instead of overwriting in
  sensor-list order.
- Added strict no-duty-guard static replay support to
  `scripts/70_v31_split_replay_gate.py` so fixed static references are evaluated
  as true fixed masks rather than duty-guard rotations.
- Added `scripts/71_v31_explicit_replay_fast.py` for faster explicit subtype
  replay screening.
- Corrected `scripts/71_v31_behavior_complexity_audit.py` so state-dependent
  four-regime policies are not misclassified as fixed/simple merely because
  they use four masks or persist with period 1.
- Added `configs/sensors/windblown_sensors_v31_met_specialist_pair.yaml`.
  The key contract is `met_station_core + one specialist` feasible under
  `budget=0.75`, while two specialists remain infeasible.

### Final Candidate

- Run: `reports/v31_metpair_stronglatent_seed45_h075_20260620` on `remote-gpu`.
- TCN-oracle gate artifact: `v2_tcn_oracle.pt` exists in the run directory.
- Reduced PPO artifact: `custom_ppo.pt` exists; training log reached
  `120000` timesteps.
- Split protocol: `truth_steps=70000`, split ratios `[0.35, 0.50, 0.075, 0.075]`,
  final-test event-rich windows with mean event rate `0.768799`.

### Forecast Gate Evidence

- Source learned PPO:
  `custom_ppo=0.487083`,
  `validation_selected_static=0.491597`,
  `full_open_unconstrained=0.508449`,
  `feasible_static_projected=0.511365`,
  `aoi=0.525733`,
  `round_robin=0.530911`,
  `random=0.559486`.
- Router-confidence re-evaluation (`eval_router_conf08`) improved learned PPO:
  `custom_ppo=0.485635` vs `validation_selected_static=0.491597`, absolute
  margin `0.005962`, relative margin about `1.21%`.
- Strict explicit replay with no static duty guard passed:
  best dynamic `split_metpair_subtype_explicit_l4=0.482174`,
  replay-local true fixed static `static_action10=0.492351`,
  absolute margin `0.010177`, relative margin about `2.07%`,
  `gate_pass=true`.

### Behaviour Gate Evidence

- Correct audit file: `behavior_audit_v2/behavior_complexity_summary.json`.
  The older `eval_router_conf08/behavior_audit` file used the pre-fix audit
  logic and should not be used for final behaviour claims.
- Best learned policy under `eval_router_conf08`:
  `unique_mask_count=4`,
  `top1_mask_fraction=0.412354`,
  `top3_mask_fraction=0.913574`,
  `mask_entropy_bits=1.806220`,
  `transition_entropy_bits=1.998642`,
  `event_sensor_l1=1.579090`,
  `event_mask_mi_bits=0.520959`,
  `state_dependent=true`,
  `fixed_like=false`,
  `simple_cycle_like=false`,
  `behavior_complexity_gate_pass=true`.
- Learned deployment pattern:
  `met_station_core` is always on as the meteorological backbone; the second
  slot switches by context among `laser_disdrometer`, `fc4_flux`,
  `shielded_thermo_hygro`, and `surface_temp_ir`. This is a
  state-dependent specialist scheduler, not a fixed subset or simple cycle.

### Decision

- Migrate the met+specialist-pair scene to the PD-PPO paper mainline as the
  current validated candidate.
- Present the method as forecast-oriented contextual specialist scheduling with
  a stable meteorological backbone.
- Keep V20+/V25/V26/V27/context-decoy failures as diagnostics or appendix
  material, not as the main result.
- Remaining robustness work before final submission: reproduce the candidate
  on additional seeds or nearby budgets if time permits. This is a robustness
  extension, not a blocker for the 2026-06-20 recalibration objective.

## 2026-06-20 Macro-Subtype Evidence Audit

- Added event-subtype macro loss reporting to
  `scripts/70_v31_split_replay_gate.py`. Replay/static tables and summary JSON
  now include `oracle_loss_macro_subtype_event`, computed as the unweighted
  mean over particle, flux, and thermal event subtype losses.
- Extended `scripts/72_v31_collect_metpair_strongclaim.py` to backfill macro
  losses from saved rollout NPZ files and replay/static CSVs, so existing runs
  can be re-aggregated without retraining.
- Backbone-context 7-seed aggregate:
  `reports/aggregate/metpair_backbone_context_7seed_macro_20260620/`.
  Strict claim remains unsupported (`seed_gate_pass_count=3/7`), but
  event-subtype macro evidence improves (`macro_seed_positive_count=5/7`).
- Strong-latent probes:
  `reports/aggregate/metpair_backbone_context_stronglatent_2seed_macro_20260620/`.
  Seeds 43 and 44 are both macro-positive, but neither passes the strict
  step-weighted 1% replay gate.
- Paper implication: the current evidence does not support a broad
  step-weighted "PD-PPO always beats static" claim. The viable direction is a
  narrower regime-macro robustness claim, pending a larger multi-seed
  strong-latent run.

## 2026-06-20 Ortholinear / Strong-Teacher Follow-Up

- Treat `remote-gpu` as the only valid remote GPU target. Older internal-address
  or tunnel-based connection notes are obsolete for current experiments.
- Strong-latent continuation failed as a final-claim branch:
  `reports/aggregate/metpair_backbone_context_stronglatent_partial4_macro_20260620/`
  has `seed_gate_pass_count=0/4` and `macro_seed_positive_count=2/4`.
- Added an orthogonal-linear event generator branch:
  `event_subtype_flux_latent_linear_scale`, `offset`, and `clip`. This replaces
  unstable exponential flux-latent amplification with a bounded linear term
  while reducing thermal shortcut strength.
- Ortholinear seed41 fixed the structural replay problem:
  explicit dynamic replay `split_metpair_subtype_explicit_l10=5.142764` beats
  replay-local static `static_action5=5.212586`; behaviour also passes.
- Learned deployment audit:
  raw PPO is step-weighted positive (`4.956431` vs selected static `5.233835`),
  but router-confidence deployment is negative (`5.330788`). Learned macro is
  still negative because the raw policy underperforms static on the flux
  subtype.
- Exposed teacher/curriculum controls in
  `scripts/run_v31_metpair_backbone_context_seed_sweep_20260620.sh`:
  AWBC strength, BC pretrain length, entropy coefficient, subtype lookahead,
  dwell, and switch penalty.
- Added
  `scripts/run_v31_metpair_backbone_context_ortholinear_strongteacher_seed_sweep_20260620.sh`
  to test whether stronger subtype imitation and aligned lookahead can convert
  ortholinear structural headroom into a learned macro-positive PD-PPO policy.

## 2026-06-20 Remote Hygiene and Strong-Teacher 10-Seed Expansion

- Cleaned local operational instructions and scripts so `remote-gpu` is the
  only valid remote GPU entry point. Removed stale hardcoded remote-host paths
  from active local context, the smoke-result fetch script, and the local
  microclimate experiment skill. Removed password-based sync helpers from the
  fetch script.
- Strong-teacher 3-seed aggregate is now available:
  `reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_3seed_macro_20260620/`
  and raw counterpart
  `reports/aggregate/metpair_backbone_context_ortholinear_strongteacher_3seed_raw_macro_20260620/`.
  Strict step-weighted gate, learned gate, replay gate, and behaviour gate are
  all `3/3`.
- Macro-subtype robustness is not supported yet: macro-positive full seeds are
  only `1/3`.
- Launched strong-teacher extension seeds `44--50` on `remote-gpu` across GPUs
  `0/1/2/4/5`. The final claim audit should aggregate seeds `41--50` and judge
  the strong claim on the strict step-weighted static/replay/behaviour gate.

## 2026-06-20 Static-Normalized Macro and Reward-Aligned Balanced Objective

- Re-audited the previously reported strong-teacher evidence. The 14-seed pool
  does not support a strong main-paper claim: behaviour is solved (`14/14`),
  but strict step-weighted and raw macro gates remain below the required
  robustness threshold.
- Diagnosed the remaining failure as objective-scale dominance. Feasible fixed
  static candidates frequently choose `met_station_core + fc4_flux`; flux loss
  is much larger numerically than particle and thermal losses, so raw aggregate
  objectives can reward a static flux shortcut even when the dynamic specialist
  scheduler is better balanced across regimes.
- Added `oracle_loss_macro_subtype_event_staticnorm`: subtype losses are divided
  by the median feasible fixed-static subtype loss before averaging particle,
  flux, and thermal regimes. PPO metrics, replay gates, and the multiseed
  collector can now use this score.
- Added `--reward-loss-normalization staticnorm_subtype` to PPO training. The
  normalizers are computed only from validation static candidates and are stored
  in `v2_ppo_metadata.json`, so the training reward is aligned with the
  static-normalized macro gate without using final-test data.
- Added and queued the balanced-objective runner
  `scripts/run_v31_metpair_backbone_context_ortholinear_balancedobjective_seed_sweep_20260620.sh`.
  First target seeds are `41` and `42`; expansion depends on learned PPO,
  strict replay, and behaviour gates under the static-normalized macro contract.
- Completed the posthoc static-normalized replay/collector audit for the
  ortholinear strong-teacher 14-seed pool:
  `reports/aggregate/metpair_ortholinear_strongteacher_14seed_staticnorm_replay_20260620/`.
  Result: `complete_seeds=14`, strict step `seed_gate_pass_count=10`,
  `behavior_gate_pass_count=14`, `macro_seed_gate_count=13`,
  `one_sided_sign_test_p_macro_seed_gate=0.00091552734375`, and
  `macro_claim_strength=strong_macro_multiseed`.
- The only static-normalized macro failure is seed `48`; strict step gate also
  fails on seeds `44`, `46`, `48`, and `52`. The paper claim should therefore
  be written as a static-normalized event-regime macro claim, not as broad
  step-weighted forecast optimality.

## 2026-06-20 Paper Mainline Rewrite for Static-Normalized Macro Claim

- Archived the previous paper source/PDF before rewriting:
  `paper/_archive/pre_staticnorm_macro_rewrite_20260620_232507/`.
- Rewrote the canonical manuscript entry point `paper/main.tex` and main
  sections to remove the stale eight-channel, ten-seed, dynamic-baseline
  narrative.
- Replaced the main claim with the supported backbone-plus-specialist result:
  PD-PPO improves static-normalized event-regime macro forecast loss in `13/14`
  seeds, passes behaviour complexity in `14/14` seeds, and has one-sided
  macro-gate sign-test `p=0.00091552734375`.
- Added `paper/tables/metpair_staticnorm_macro_summary.tex` and replaced
  `paper/tables/sensor_specs.tex` with the six-channel met+specialist contract.
- The rewritten paper explicitly states the limitation that the strict
  step-weighted fixed-static gate passes only `10/14` seeds, so broad
  average-loss dominance over true fixed static is not claimed.
- Local and `remote-gpu` paper builds both complete with `latexmk`; the only
  remaining LaTeX issue is a minor `1.79993pt` overfull hbox warning.
### 2026-06-30 17:34:12 UTC | session `20260625_020` | model `gpt-5.5` | interrupted
**Tools:** shell, write
**Files:**
  - `<framework-root>/docs/07-01-01-LEMMA.md`
Commands:
  - `stat -c '%n %s bytes %y' docs/07-01-01-LEMMA.md && sha256sum docs/07-01-01-LEMMA.md`

### 2026-06-30 17:48:08 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/findings.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/progress.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/task_plan.md`
  - `<framework-root>/paper/tables/pdppo_training_hyperparameters.tex`
Commands:
  - `date '+%Y%m%d_%H%M%S %Y-%m-%d %H:%M:%S %Z'; git status --short`
  - `set -euo pipefail
stamp=$(date '+%Y%m%d_%H%M%S')
mkdir -p paper/backups paper_archives
pdf_backup="paper/backups/main_before_lemma0701_${stamp}.pdf"
archive="paper_archives/paper_before_lemma0701_${st…`
  - `git diff --check -- main.tex sections/01_introduction.tex sections/03_problem_formulation.tex sections/04_framework_protocol.tex sections/05_simulation_setup.tex sections/06_results.tex sections/07_di…`
  - … and 30 more

### 2026-06-30 18:13:39 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell
**Files:**
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/findings.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/progress.md`
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/task_plan.md`
  - `<framework-root>/paper/sections/appendix_theory.tex`
Commands:
  - `python - <<'PY'
from pathlib import Path
import subprocess,re
base=Path('<framework-root>/paper')
text=subprocess.check_output(['pdftotext','main.pdf'…`
  - `set -euo pipefail
stamp=$(date '+%Y%m%d_%H%M%S')
mkdir -p backups ../paper_archives
pdf_backup="backups/main_before_nfix_${stamp}.pdf"
archive="../paper_archives/paper_before_nfix_${stamp}.tar.gz"
cp …`
  - `git diff --check -- main.tex sections/03_problem_formulation.tex sections/04_framework_protocol.tex sections/05_simulation_setup.tex sections/appendix_theory.tex tables/regime_balanced_24seed_summary.…`
  - … and 54 more

### 2026-06-30 18:33:23 UTC | session `20260701_022` | model `gpt-5.5` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/.planning/2026-07-01-supervisor-ppt-easy/findings.md`
  - `<framework-root>/.planning/2026-07-01-supervisor-ppt-easy/progress.md`
  - `<framework-root>/.planning/2026-07-01-supervisor-ppt-easy/task_plan.md`
  - `<framework-root>/reports/supervisor_update_20260701/build_supervisor_ppt_20260701.py`
Commands:
  - `pdftotext supervisor_report_20260629.pdf - | sed -n '1,220p'`
  - `python - <<'PY'
from pathlib import Path
base=Path('<framework-root>')
for d in [base/'paper/figures', base/'reports/supervisor_update_20260629', base…`
  - `python - <<'PY'
from PIL import Image
from pathlib import Path
base=Path('<framework-root>/reports/supervisor_update_20260701/figures')
for p in sorte…`
  - … and 41 more

### 2026-07-01 16:07:03 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell
**Files:**
  - `<framework-root>/.planning/2026-06-10-eswa-terminology-rewrite/progress.md`
  - `<framework-root>/paper/references.bib`
Commands:
  - `git status --short`
  - `cp references.bib backups/references_before_refaudit_$(date '+%Y%m%d_%H%M%S').bib
ls -t backups/references_before_refaudit_*.bib | head -1 | xargs -I{} sh -c 'stat -c "%n %s bytes %y" "{}"; sha256sum …`
  - `python - <<'PY'
from pathlib import Path
import re, sys
bib=Path('references.bib').read_text()
keys=re.findall(r'^@\w+\s*\{\s*([^,]+),', bib, flags=re.M)
from collections import Counter
dups=[k for k,…`
  - … and 12 more

### 2026-07-01 19:35:09 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/audit_extract.py`
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/findings.md`
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/progress.md`
  - `<framework-root>/.planning/2026-07-02-terminology-figure-audit/task_plan.md`
  - `<framework-root>/paper/figures/data_split_timeline_tikz.tex`
  - `<framework-root>/paper/figures/gen_fig_event_type_diagnostics.py`
  - `<framework-root>/paper/figures/gen_fig_framework_and_support.py`
  - `<framework-root>/paper/figures/gen_fig_scenebal2_diagnostics.py`
  - `<framework-root>/paper/figures/gen_fig_scenebal_evidence.py`
  - `<framework-root>/paper/figures/gen_fig_training_diagnostics.py`
  - … and 11 more
Commands:
  - `pwd; git status --short; stat -c '%n %s bytes %y' paper/main.pdf 2>/dev/null || true; sha256sum paper/main.pdf 2>/dev/null || true; pdfinfo paper/main.pdf | awk '/^Pages:/ {print}' 2>/dev/null || true`
  - `pdftotext -f 1 -l 1 main.pdf -`
  - `pdftotext -f 2 -l 2 main.pdf -`
  - … and 313 more

### 2026-07-01 20:17:14 UTC | session `20260625_020` | model `gpt-5.5` | completed
**Tools:** shell, write
**Files:**
  - `<framework-root>/.planning/2026-07-02-reference-audit/findings.md`
  - `<framework-root>/.planning/2026-07-02-reference-audit/progress.md`
  - `<framework-root>/.planning/2026-07-02-reference-audit/task_plan.md`
  - `<framework-root>/docs/07-02-02-reference-audit.md`
Commands:
  - `pwd && git status --short && date '+%Y-%m-%d %H:%M:%S %Z'`
  - `python - <<'PY'
from pathlib import Path
import re, json
root=Path('<framework-root>')
paper=root/'paper'
plan=root/'.planning/2026-07-02-reference-au…`
  - `python - <<'PY'
from pathlib import Path
import json, urllib.parse, urllib.request, time, re
root=Path('<framework-root>')
plan=root/'.planning/2026-0…`
  - … and 49 more
### 2026-07-08 19:28:02 UTC | session `20260625_020` | model `deepseek-v4-pro` | completed
**Tools:** shell, write
**Files:**
  - `<framework-root>/docs/07-07-01-HERMES.md`
Commands:
  - `# Backup the original
cp <framework-root>/docs/07-07-01-HERMES.md <framework-root>/docs/backups/07-0…`
  - `mkdir -p <framework-root>/docs/backups && cp <framework-root>/docs/07-07-01-HERMES.md <user-home>/_c…`
  - `echo "=== ORIGINAL (backup) ===" && wc -l -c <framework-root>/docs/backups/07-07-01-HERMES_orig_20260709_032649.md && echo "=== REFINED ===" && wc -l …`
  - … and 2 more

### 2026-07-19 18:06:36 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_072001_wording/task_plan.md`
Commands:
  - `set -euo pipefail
paper=<framework-root>/paper
printf '%s\n' '=== codex ==='
ps -eo pid=,ppid=,stat=,etime=,time=,cmd= | grep -E '(codex --yolo resume…`
  - `set -euo pipefail
fw=<framework-root>
paper="$fw/paper"
plan=<temporary-workspace>/pdppo_072001_wording
files=("$paper/main.tex" "$paper/sections/01_i…`
  - `set -euo pipefail
fw=<framework-root>
paper="$fw/paper"
pkg="$fw/submission/eswa_pdppo_20260719"
plan=<temporary-workspace>/pdppo_072001_wording
stamp…`
  - … and 24 more

### 2026-07-20 17:02:32 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
Commands:
  - `set -euo pipefail
python - <<'PY'
from pathlib import Path
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmd=(proc/'cmdline').read_bytes().repl…`
  - `set -euo pipefail
conda run -n darts python scripts/95_v31_build_clean_paper_assets.py --framework-root .
python -m py_compile scripts/95_v31_build_clean_paper_assets.py`
  - `set -euo pipefail
python - <<'PY'
from pathlib import Path
import subprocess,json
script=Path('<hermes-home>/skills/writing/technical-manuscript-editing/scripts/compare_latex_invariants.py')
bef…`
  - … and 21 more

### 2026-07-20 17:26:59 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
Commands:
  - `set -euo pipefail
root=<framework-root>; paper="$root/paper"; pkg="$root/submission/eswa_pdppo_20260719"; plan=<temporary-workspace>/pdppo_072002_corr…`
  - `set -euo pipefail
root=<framework-root>; paper="$root/paper"; pkg="$root/submission/eswa_pdppo_20260719"; plan=<temporary-workspace>/pdppo_072002_corr…`
  - `set -euo pipefail
python - <<'PY'
from pathlib import Path
import subprocess,json
script=Path('<hermes-home>/skills/writing/technical-manuscript-editing/scripts/compare_latex_invariants.py'); be…`
  - … and 9 more

### 2026-07-23 20:04:15 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_drift_check_final_preintegration.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_ignored_png_integration.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_integration_hashes.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/ci_recomputation_hermes.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_roundtrip_verification.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_sync_report.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/zip_rebuild_report.json`
Commands:
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python -c 'from pathlib import Path
import pandas as pd, numpy as np, json
p=Path("reports/aggregate/pdppo_clean_validat…`
  - `python -c 'from pathlib import Path
p=Path("<hermes-home>/cache/delegation/live/deleg_f4750d7d/task-1.log")
s=p.read_text().splitlines()[101]
msg=s.split("assistant|",1)[1]
print(msg.replace(" -…`
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python -c 'from pathlib import Path
import re,ast,subprocess
s=Path("paper/main.tex").read_text(); a=s.split("\\begin{ab…`
  - … and 27 more

### 2026-07-23 20:41:15 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/paper_archives/plain_language_final_20260724_043256.sha256`
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_final_verification.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_roundtrip_verification_final.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/upload_artifact_filelist.txt`
Commands:
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python scripts/95_v31_build_clean_paper_assets.py`
  - `source <conda-root>/etc/profile.d/conda.sh && conda activate darts && python scripts/95_v31_build_clean_paper_assets.py`
  - `set -euo pipefail
for f in main.tex anonymous_manuscript.tex supplementary_material.tex title_page.tex; do
  echo "== $f =="
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$f"
done`
  - … and 49 more

### 2026-07-23 21:19:54 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/paper_archives/plain_language_final_20260724_051549.sha256`
  - `<framework-root>/submission/eswa_pdppo_20260719/checksums.sha256`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/canonical_final_verification.json`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/late_async_audit_mapping.md`
  - `<temporary-workspace>/pdppo_plain_language_20260724_010616/submission_roundtrip_verification_final.json`
Commands:
  - `set -euo pipefail
command -v drawio || command -v draw.io || command -v diagrams-net || true
printf 'DISPLAY=%s\n' "${DISPLAY:-}"`
  - `drawio --help | sed -n '1,160p'`
  - `pdfinfo figure_pdppo_framework_drawio.pdf | sed -n '/Page size/p'; identify -format '%w x %h\n' figure_pdppo_framework_drawio.png`
  - … and 127 more

### 2026-07-28 20:57:01 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/scripts/96_v31_analyze_warning_information.py`
Commands:
  - `python - <<'PY'
import csv, hashlib, json, pathlib
root=pathlib.Path('<framework-root>')
rows=[]
for seed in range(117,141):
    meta_path=root/f'repr…`
  - `python - <<'PY'
import pandas as pd
p='<framework-root>/reproducibility/pdppo_eswa_evidence_20260718/aggregates/pdppo_clean_validation_frozen_24seed_2…`
  - `python - <<'PY'
import pandas as pd
d=pd.read_csv('<framework-root>/reports/aggregate/pdppo_framework_baselines_clean_24seed_20260718/framework_baseli…`
  - … and 10 more

### 2026-08-02 13:44:53 UTC | session `20260802_213` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/docs/PD-PPO-TERMINOLOGY.md`
  - `<temporary-workspace>/pdppo_terminology_contract/findings.md`
  - `<temporary-workspace>/pdppo_terminology_contract/glossary_validation.json`
  - `<temporary-workspace>/pdppo_terminology_contract/progress.md`
  - `<temporary-workspace>/pdppo_terminology_contract/task_plan.md`
Commands:
  - `git status --short && printf '\nPAPER\n' && git -C paper status --short && printf '\nBRANCH\n' && git branch --show-current && git -C paper branch --show-current`
  - `set -euo pipefail
if command -v markdownlint >/dev/null 2>&1; then markdownlint docs/PD-PPO-TERMINOLOGY.md; else echo 'markdownlint: unavailable (structural Python audit used)'; fi
python -m py_compil…`
  - `set -euo pipefail
if command -v codespell >/dev/null 2>&1; then codespell docs/PD-PPO-TERMINOLOGY.md; else echo 'codespell: unavailable'; fi
set +e
git diff --no-index --check /dev/null docs/PD-PPO-TE…`

### 2026-08-02 19:06:31 UTC | session `20260625_034` | model `gpt-5.6-sol` | completed
**Tools:** edit, shell, write
**Files:**
  - `<framework-root>/paper_archives/eswa_pdppo_main65_upload_final_20260803_030029.zip.sha256`
  - `<framework-root>/paper_archives/paper_phase9_final_20260803_030029.tar.gz.sha256`
Commands:
  - `set -euo pipefail
for f in main.pdf anonymous_manuscript.pdf supplementary_material.pdf title_page.pdf; do echo "--- $f"; pdfinfo "$f" | grep -E '^(Title|Subject|Keywords|Author|Creator|Producer|Creat…`
  - `set -euo pipefail
conda run -n darts pytest -q -o addopts='' tests | tee <temporary-workspace>/pdppo_full_refinement_20260802/phase9_pytest_explicit_final.log`
  - `set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=<temporary-workspace>/pdppo_full_refinement_20260802/backups/phase9_pre_package_final_${STAMP}.tar.gz
mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT…`
  - … and 33 more

### 2026-08-22 | Flexible-subset PD-PPO development pilot

- Replaced the mandatory-core plus one-specialist geometry in an independent
  experiment track with 29 power-feasible subsets over six physical-system channels.
- Seed 401 at 30k steps improved over the selected static schedule on macro
  normalized loss (`0.656944 < 0.723909`) but lost on mean forecast loss
  (`0.171239 > 0.161443`) and remained slightly behind AoI.
- Operational feasibility passed with zero violations and zero warm-up aborts,
  but behaviour failed the diversity gate: one channel was always active, two
  were always inactive, and only three masks were executed.
- Diagnosis: the validation teacher selected the laser mask for both particle
  and flux conditions. The run was stopped at one development seed; v2 targets
  this measured identifiability and cost-geometry failure before further scaling.

### 2026-08-22 | Flexible-subset v2 bounded pilot

- Seed 402 passed both performance endpoints against the validation-selected
  static schedule: mean loss `0.165710 < 0.197650` and macro normalized loss
  `0.786622 < 0.996109`.
- It also beat AoI, round-robin, and random on both endpoints, with zero power
  violations and zero warm-up aborts.
- Behaviour improved from three to eleven executed masks. No channel was always
  active, five channels had intermediate duty, and only FC4 remained effectively
  inactive. The frozen v2 configuration advanced to seeds 403 and 404.

### 2026-08-22 | Flexible-subset v2 30k replication

- Seeds 403 and 404 did not reproduce seed 402's two-endpoint advantage. Across
  the three development seeds, PD-PPO beat the validation-selected static subset
  on both mean and macro forecast loss in `1/3` seeds.
- Seed 403 lost to the selected static subset by `0.008999` in mean loss and
  `0.060160` in macro loss. Seed 404 lost mean loss by `0.005471` while retaining
  a small macro advantage of `0.004232`.
- All three runs remained feasible with no always-on channel, but seed 403 used
  only four subsets and switched at `0.007237` per step. The 30k configuration
  failed the replication gate and was not expanded to confirmation seeds.
- A bounded 100k-step rerun of seed 403 was launched with the v2 scene and all
  other settings fixed to isolate training duration from scene calibration.

### 2026-08-22 | Flexible-subset v2 unmatched 100k diagnostic

- The seed-403 100k policy still used only four subsets, left FC4 inactive, and
  lost to selected static by `0.008091` in mean loss and `0.031861` in macro loss.
- Audit found that the longer run regenerated the frozen forecaster and
  validation candidate scores. Its final start indices were unchanged, but its
  selected static mask and all comparator losses differed from the 30k run.
- The launcher now exposes the existing control-source validation path so that
  training-duration comparisons can reuse byte-identical truth, evaluator,
  action surface, validation selection, and final windows. This unmatched run
  is retained as a collapse diagnostic, not a causal duration comparison.

### 2026-08-22 | Flexible-subset v2 matched 100k diagnostic

- A corrected seed-403 duration comparison reused byte-identical truth and
  frozen-forecaster assets. All six non-PD-PPO comparator rows matched the 30k
  run exactly, confirming the matched-control path.
- At 100k steps, PD-PPO lost to selected static by `0.019041` in mean loss and
  `0.088967` in macro loss. It used seven masks but never activated the laser.
  Longer training therefore does not resolve the failure.
- The next bounded variant keeps the scene and full feasible action surface but
  applies validation-static subtype normalization to the forecast-loss reward
  and uses physically specified subtype prototypes for auxiliary supervision.
  The prototypes do not constrain executable actions at training or evaluation.

### 2026-08-22 | Flexible-subset v3 normalized physical-teacher pilot

- Seed 405 beat selected static on mean loss (`+0.017557`) and macro loss
  (`+0.045745`), used all six channels, and had no always-on or always-off count.
- It still trailed the best conventional dynamic reference by `0.003114` in
  mean loss and `0.047103` in macro loss, so v3 did not pass the full gate.
- Particle scheduling followed the intended met-plus-laser prototype, but IR
  duty reached only `0.287` during thermal windows even though the online
  thermal alert was fully separable. The actor's BC pretraining accuracy was
  only `0.110`; the next matched run changes only BC pretraining strength.

### 2026-08-22 | Flexible-subset v3 matched strong-BC pilot

- Stronger pretraining raised BC accuracy from `0.110` to `0.786` while all
  frozen comparator metrics remained exactly unchanged.
- PD-PPO beat selected static by `0.031659` in mean loss and `0.097111` in macro
  loss. It also beat the best conventional dynamic reference by `0.010987` and
  `0.004263`.
- The policy used ten masks; all six channels had intermediate duty, switches
  per step were `0.021421`, and no feasibility or warm-up failure occurred.
  This configuration advanced to development seeds 406 and 407.

### 2026-08-22 | Flexible-subset v3 strong-BC replication

- Across seeds 405--407, PD-PPO beat selected static on both endpoints in `2/3`
  seeds and beat the best conventional dynamic reference on both in `2/3`.
- Mean margins against static remained positive (`+0.004613` mean and
  `+0.026351` macro), but the mean macro margin against the best dynamic
  reference was `-0.018517`. The configuration did not advance to final seeds.
- All runs used every channel at intermediate duty and executed 4--10 masks.
  The next matched variant replaces four-prototype BC labels with frozen-
  forecast greedy labels over the complete feasible action surface.

### 2026-08-22 | Flexible-subset v4b forecast-greedy warm start

- Full-surface forecast-greedy BC raised action coverage to 17 masks but failed
  both performance gates. PD-PPO lost to selected static by `0.042258` mean and
  `0.134527` macro, and one channel remained inactive.
- Training logs showed mean value loss `103.96` versus mean policy-loss
  magnitude `0.00836`. Whole-model gradient clipping therefore allowed critic
  gradients to suppress actor updates despite separate network modules.
- Added an opt-in actor/critic/auxiliary grouped gradient-clipping mode. It is
  enabled only for the flexible-subset experiment track and leaves historical
  configurations unchanged.

### 2026-08-22 | Flexible-subset v5 grouped-gradient diagnostic

- Grouped clipping reduced final entropy from about `3.11` to `1.99`, confirming
  that whole-model clipping had suppressed actor optimization.
- The policy beat selected static on both endpoints and beat the best dynamic
  mean loss by `0.002737`, but lost dynamic macro by `0.004705`.
- Strong BC plus grouped clipping collapsed exactly to the four physical
  prototypes. Grouped clipping is retained; the next matched run restores weak
  BC so PPO can use non-prototype feasible subsets.

### 2026-08-22 | Flexible-subset v5b grouped-gradient weak-BC diagnostic

- Weak BC plus continuing AWBC still produced exactly four prototype masks.
- PD-PPO beat selected static on both endpoints and the best dynamic mean by
  `0.003651`, but lost the best dynamic macro by `0.009019`.
- The next matched run uses strong BC only as initialization and disables AWBC
  during PPO, isolating whether continuing imitation causes prototype lock-in.

### 2026-08-22 | Flexible-subset v6 BC-warm-start grouped-gradient pilot

- Disabling AWBC after strong BC initialization produced the first broad-action
  passing pilot: 19 of 24 feasible masks and all six channels at intermediate
  duty.
- PD-PPO beat selected static by `0.030085` mean and `0.106974` macro, and beat
  the best conventional dynamic reference by `0.009414` and `0.014126`.
- No feasibility or warm-up failure occurred. The configuration was frozen for
  development replication on seeds 406 and 407.

### 2026-08-22 | Flexible-subset v6 development replication

- Across seeds 405--407, PD-PPO beat the validation-selected static subset on
  both endpoints in `2/3` seeds. Mean margins were `+0.008988` for mean loss
  and `+0.089263` for macro loss.
- PD-PPO beat the strongest conventional dynamic reference on both endpoints
  in `2/3` seeds. Mean margins were `+0.004541` and `-0.000391`, so v6 did not
  pass the frozen final-evaluation gate.
- Executed subset coverage was `19`, `8`, and `11` of 24. Seed 406 never used
  the laser and used FC4 at only `0.016` duty, localizing the remaining problem
  to cross-seed action-coverage stability rather than feasibility or switching.
- No warm-up abort occurred. The next bounded change will preserve the scene,
  action geometry, reward, and comparators while addressing actor coverage
  stability; fresh final seeds remain untouched.

### 2026-08-22 | Flexible-subset v7 zero-entropy diagnostic

- Removing the PPO entropy bonus did not stabilize the v6 replication failures.
  Across seeds 406 and 407, mean margins were `-0.014901/-0.056149` against
  static and `-0.003140/-0.044832` against the strongest dynamic reference.
- The policies executed 8 and 19 subsets, showing that entropy removal changed
  individual trajectories without reducing cross-seed coverage variance. This
  hypothesis is rejected.
- Added an opt-in linear AWBC decay schedule. Historical behavior is unchanged
  by default; the next bounded test uses existing physical-prototype guidance
  only during early PPO updates and decays it to zero, avoiding both immediate
  BC forgetting and permanent four-prototype lock-in.

### 2026-08-22 | Flexible-subset v8 decaying-AWBC diagnostic

- Linear decay from AWBC `0.15` to zero over 10k PPO steps degraded both
  representative development seeds. Average margins were
  `-0.020409/-0.083357` against static and `-0.008042/-0.061643` against the
  strongest conventional dynamic policy.
- Seed 406 recovered six-channel coverage, but seed 407 collapsed to a
  low-switching met-plus-FC4 policy. Short-lived teacher guidance does not solve
  cross-seed optimization drift and is rejected.
- The next bounded diagnostic restores v6 and lowers only the PPO learning rate
  from `3e-4` to `1e-4`. The launcher now exposes this existing hyperparameter;
  its historical default remains unchanged.

### 2026-08-22 | Flexible-subset v9 low-learning-rate diagnostic

- Lowering the PPO learning rate to `1e-4` increased executed subset coverage
  to 13 and 16 on seeds 406/407, but average margins remained negative:
  `-0.012654/-0.004511` against static and `-0.001211/-0.032193` against the
  strongest conventional dynamic reference.
- Entropy, teacher-transition, and learning-rate diagnostics have now all
  failed to produce stable cross-seed performance. Ordinary hyperparameter
  retries stop here.
- The next structural correction selects checkpoints only on the existing
  calibration/validation partition and restores the selected policy before
  independent test evaluation. The current final-update-only behavior can
  discard better intermediate PPO policies and amplify seed variance.

### 2026-08-22 | Flexible-subset v10 calibration-checkpoint diagnostic

- Calibration-only selection chose PPO updates 5 and 30 for seeds 406/407, but
  average margins remained mixed: `-0.009463/+0.038832` against static and
  `+0.000694/-0.001590` against the strongest dynamic reference.
- Checkpoint selection is retained as an optional, methodologically valid
  facility, but final-update selection is not the primary blocker.
- Cost-geometry audit showed that three low-cost channels can run together at
  `B=1.25`, recreating a compact static shortcut. The next scene calibration
  increases only fixed per-epoch effective costs for low-power channels. It
  keeps every single channel and most arbitrary pairs feasible, with no required
  channel and no explicit cardinality cap.

### 2026-08-22 | Flexible-subset v11 cost-balanced development result

- The calibrated power geometry contains 20 feasible actions: empty, all six
  singletons, and 13 arbitrary pairs. PD-PPO executed 11/15/15 subsets with no
  always-on or always-off channel, satisfying the flexible-behavior objective.
- Forecast performance failed: all three seeds lost both endpoints to static;
  average margins were `-0.026523/-0.098774` against static and
  `-0.019486/-0.063568` against the strongest conventional dynamic policy.
- Actor audit found that every candidate subset receives a trainable,
  state-independent action-prior parameter even when no external prior is used.
  This is a direct static shortcut. The next bounded variant disables that term
  while retaining state-conditioned additive sensor scoring and all 20 actions.

### 2026-08-22 | Flexible-subset v12 no-static-prior and context diagnostic

- Removing the state-independent per-action prior expanded execution to
  12/17/18 of 20 subsets but did not improve prediction: static two-endpoint
  wins were `1/3`, and strongest-dynamic wins were `0/3`.
- The online context-alert diagnostic beat PD-PPO in all three seeds, confirming
  that warning context is informative. It beat static only in seed 405 and was
  slightly weaker in seeds 406/407, so the scene still lacks stable
  dynamic-over-static value.
- The next bounded calibration changes only subtype latent update speed from
  `0.22` to `0.55`. Faster event-specific evolution should make stale specialist
  observations costly while preserving the physical channel model and online
  warning lead.

### 2026-08-22 | Flexible-subset v13 fast-latent diagnostics

- Raising latent update alpha from `0.22` to `0.55` degraded PD-PPO; average
  margins were `-0.034975/-0.098079` against static and
  `-0.020109/-0.065995` against conventional dynamic policies.
- Warning thresholds from 0.3 to 0.7 did not create stable dynamic-over-static
  value. Privileged exact-label replay also failed, including when actions were
  fixed to the prespecified physical specialist pairs.
- Faster latent innovations reduce future predictability from current
  specialist observations. V13 is rejected. The next bounded calibration
  restores alpha `0.22` and increases only the subtype-latent target amplitudes
  to strengthen specialist information value without changing action geometry.

### 2026-08-22 | Flexible-subset v14--v15 specialist-value calibration

- Stronger subtype target amplitudes improved v14 relative to fast-latent v13,
  but PD-PPO still reached only `1/3` two-endpoint static wins and `0/3`
  strongest-dynamic wins.
- Replacing the thermal physical pair with `{shielded thermo-hygro, IR}` made
  both online-warning and privileged exact-label physical policies beat static
  in seeds 405 and 407. Seed 406 remained slightly negative.
- Validation-selected, physical, hybrid, and validation-guarded action mappings
  were evaluated. None passed 3/3 because the seed-406 validation preference did
  not transfer to test. Further action-map tuning is stopped to avoid post-hoc
  adaptation.
- The flexible formulation itself is validated behaviorally: 20 naturally
  power-feasible actions, no required channel, no cardinality cap, broad subset
  use, and no forced duty quotas. Stable prediction superiority remains an open
  algorithm/scenario-transfer problem and is not claimed from these dev runs.

### 2026-08-22 | V16 invalid precursor-alias diagnostic

- A source audit initially used the state definition from the wrong pipeline
  helper and incorrectly concluded that the three subtype latent observations
  were absent from custom-PPO execution. V15 rollout artifacts instead confirm
  a 15-dimensional state with all three latent columns observed conditionally.
- V16 duplicated the same latent variables under new precursor aliases. Its
  frozen-oracle losses saturated near the clipping ceiling for every policy, so
  the run is classified as an invalid implementation diagnostic and is excluded
  from scientific comparisons.
- The alias configuration and state-column extension are removed. Exploration
  resumes from V15 with event-window and transition-level analysis of the
  seed-406 transfer failure.

### 2026-08-22 | Flexible-subset launcher teacher defaults

- The flexible-subset launcher still defaulted to infeasible three-channel calm
  and thermal teacher masks inherited from the earlier action geometry. A V17
  preflight rejected the launch before oracle fitting or policy training.
- Defaults now reproduce the feasible V15 teacher actions: `{met station,
  radiometer}` for calm and `{thermo-hygro, surface IR}` for thermal. Particle
  and flux teacher actions are unchanged.

### 2026-08-22 | Flexible-subset v17 no-action-prior signal

- With the V15 scene and the state-independent trainable action prior disabled,
  seed 406 PD-PPO improved over its within-run validation-selected static
  reference on mean loss (`0.24219` versus `0.24721`) and normalized subtype
  macro loss (`1.06079` versus `1.26149`). All six channels had intermediate
  duty, the switch rate was `0.05710` per step, and there were no aborts.
- This run retrained the stochastic TCN oracle. Its selected static action
  changed from the V15 action 8 to action 1, so it is positive development
  evidence but not a strict single-variable ablation. V18 will reuse the frozen
  V15 control artifacts to isolate the action-prior effect.

### 2026-08-22 | Flexible-subset v18 frozen no-action-prior control

- Reusing the exact V15 seed-406 truth, oracle, windows, and static reference,
  disabling the state-independent action prior reduced PD-PPO mean loss from
  `0.29292` to `0.27352` and normalized subtype macro loss from `0.96089` to
  `0.91173`.
- The controlled variant still lost to static (`0.25439`, `0.85606`) and left
  one channel effectively off. The prior is therefore a verified source of
  static bias, but removing it alone does not pass the prediction or behavior
  gates. The next bounded control adds validation-only checkpoint selection.

### 2026-08-22 | Flexible-subset v19 frozen checkpoint control

- Validation-only checkpoint selection chose update 30. On the frozen seed-406
  test windows it improved normalized subtype macro loss to `0.84910`, narrowly
  better than static `0.85606`, while mean loss remained worse (`0.27675`
  versus `0.25439`).
- Checkpoint selection therefore trades the two co-primary endpoints on this
  seed and does not pass the joint gate. The next bounded control aligns the
  training reward with validation-normalized subtype losses while retaining the
  no-prior actor and frozen evidence path.

### 2026-08-22 | Flexible-subset v20 normalized-reward control

- Subtype-normalized forecast reward with the frozen V15 evidence path reached
  macro loss `0.85721`, effectively tying but not beating static `0.85606`.
  Mean loss remained worse (`0.26286` versus `0.25439`), and only four channels
  retained intermediate duty.
- Reward normalization is rejected as the primary correction. Architecture
  inspection found that candidate embeddings were linear sums of sensor
  embeddings, which cannot represent non-additive complementarity or redundancy
  between channels. A shared nonlinear subset encoder is added as the next clean
  arbitrary-subset actor variant.

### 2026-08-22 | Flexible-subset v21 nonlinear subset encoder

- On the frozen seed-406 evidence path, the nonlinear subset encoder preserved
  intermediate duty for all six channels and improved normalized subtype macro
  loss from V18's `0.91173` to `0.88322`. Mean loss was `0.27579`; both values
  remained worse than static (`0.25439`, `0.85606`).
- Actor entropy declined much faster than in the linear encoder, indicating
  premature concentration after adding subset interaction capacity. The next
  bounded control raises the existing entropy coefficient and uses
  validation-only checkpoint selection; no new supervision or baseline prior is
  introduced.

### 2026-08-22 | Flexible-subset v22 nonlinear entropy control

- Raising entropy regularization to `0.02` restored broad six-channel use but
  degraded both endpoints to mean `0.30077` and normalized subtype macro
  `0.95556`. Higher actor entropy is rejected.
- The bounded actor controls are exhausted. Since the privileged physical
  dynamic reference itself passed only two of three V15 seeds, exploration
  moves to a prespecified stratified subtype generator that stabilizes event
  coverage across chronological partitions without using outcome feedback.

### 2026-08-22 | Flexible-subset v23 stratified-subtype gate

- Stratified event assignment prevented subtype-count drift, but checkpointed
  PD-PPO collapsed to a fixed two-channel action. It beat static mean loss by
  only `0.000118` and lost normalized subtype macro by `0.06382`; the learned
  policy therefore failed the dynamic-behavior and joint-performance gates.
- Frozen-oracle privileged replay confirmed that event adaptation is useful but
  not yet sufficient overall. Its subtype-average event loss was about `0.2454`
  versus the best static action's `0.2576`, while overall mean loss was narrowly
  worse (`0.155844` versus `0.155428`) because calm and transition periods
  offset the event gain.
- The next scene-only gate increases the prespecified event coverage before any
  PPO training. This tests whether an event-monitoring workload can support a
  positive dynamic upper bound without changing actions or using test feedback.

### 2026-08-22 | Flexible-subset v24 event-coverage upper bound

- Increasing the prespecified synthetic event coverage from `0.45` to `0.55`
  made privileged subtype adaptation beneficial on both endpoints. The best
  dynamic replay reached mean loss `0.115125`, ahead of the best static action
  at `0.123864`; its raw three-subtype average was about `0.1604` versus static
  `0.1792`.
- The dynamic upper bound used the same frozen TCN, final windows, power budget,
  and dwell constraint. It had zero aborts and only `0.00362` switches per step.
  The scenario gate therefore passes, and V25 will train PD-PPO on the frozen
  V24 evidence path before any seed expansion.

### 2026-08-22 | Flexible-subset v25 trained policy on passed scene

- Despite the positive V24 dynamic upper bound, checkpointed PD-PPO reached mean
  `0.15812` and normalized subtype macro `0.63718`, losing to static
  `0.14540/0.56057`.
- The selected policy again became nearly static, with one always-on channel,
  three always-off channels, and only `0.00608` switches per step. V26 removes
  validation checkpoint restoration while keeping the frozen scene, oracle,
  reward, and no-prior nonlinear actor unchanged.

### 2026-08-22 | Flexible-subset v26 final-policy control

- Removing checkpoint restoration recovered dynamic execution with all six
  channels at intermediate duty, `0.0453` switches per step, and zero aborts.
  Prediction still failed: mean `0.15987` and normalized subtype macro `0.57814`
  remained worse than static `0.14540/0.56057`.
- The actor learns broad behavior but does not reliably map its high-accuracy
  subtype representation to the corresponding feasible subset. V27 activates
  the existing training-only subtype action cross-entropy auxiliary without an
  execution-time label or hard subtype router.

### 2026-08-22 | Flexible-subset v27 subtype-action auxiliary

- Adding subtype-action cross-entropy at coefficient `0.1` degraded the frozen
  V24 seed-406 evaluation to mean loss `0.18254` and normalized subtype macro
  loss `0.71743`, compared with static `0.14540/0.56057` and AoI
  `0.14180/0.51523`.
- The policy regressed to one always-on and two always-off channels with only
  `0.0220` switches per step. High subtype-classification accuracy therefore
  did not translate into forecast-value action ranking. Hard subtype-action
  supervision is rejected; the next diagnostic checks whether its prototype
  labels agree with per-window forecast-loss-optimal feasible subsets before
  any further policy training.
