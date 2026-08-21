# Claim Audit - 2026-05-24

## Superseding Protocol Note - 2026-05-26

The learned energy-account PD-PPO claims below are retained as historical
same-protocol observations only. A subsequent audit
(`reports/energy_account_protocol_audit_20260526/energy_account_protocol_audit_summary.json`)
found that all five storm curriculum runs evaluate on the same recorded starts used
for training; all five full-distribution replays overlap training/storm windows;
reconstructed oracle windows overlap both evaluations in all five seeds; and no run
declares training-only normalisation. Do not use the learned-policy comparison as
held-out manuscript evidence without corrected split-protocol retraining.

This memo separates what the current experiments support from what should remain
qualified or unsupported in the paper. It is intended as a writing-control asset, not
as a new experiment.

## Fixed-Budget V3.1 S2

Supported:

- PD-PPO has lower mean FW-MAE than round-robin, AoI, and random scheduling across
  the fixed-budget V3.1 S2 sweep.
- At the primary fixed budget `B=1.70`, PD-PPO reports `0.1620 +/- 0.0142`, about
  `3.2%` lower than round-robin and `12.1%` lower than AoI.
- PD-PPO remains close to feasible static projection: the primary-budget gap is
  about `1.5%`.
- The feasible static projection remains a strong non-adaptive reference and is not
  statistically separated from PD-PPO in the fixed-budget sweep.

Required wording:

- Treat V3.1 S2 as evidence that the PD-PPO architecture learns a stable feasible
  policy that outperforms dynamic heuristic baselines under a fixed instantaneous
  budget.
- Do not use V3.1 S2 alone to claim that dynamic switching is the main source of
  the gain. The results are compatible with a strong static-allocation explanation.
- Describe `feasible_static_projected` as a fixed-priority policy passed through the
  common feasibility projector, not as an exhaustive best-static-subset solver.

## Energy-Account Oracle Diagnostic

Supported:

- The calibrated account uses `h=0.92`, `capacity=180`, and `reserve=20`, based on
  formal event-cluster statistics and normalized sensor costs.
- In the six highest-event storm windows, the hand-coded dynamic
  `snow_core -> event_laser_fc4` reference has lower oracle loss than the static
  snow-core subset: `0.4169` vs. `0.4248`.
- The advantage is event-period concentrated: event loss drops from `0.3517` to
  `0.3190`.
- The dynamic reference has no guard drops or warm-up aborts in this setting, while
  static laser diagnostics are clipped by the energy guard.

Required wording:

- Treat this as a mechanism diagnostic showing that long-term sustainability can
  create a real dynamic opportunity in event-rich storm windows.
- Do not present it as full-distribution superiority. The same calibrated setting
  does not make the dynamic reference best on full-distribution windows.
- Make clear that event labels are used for oracle diagnostics and storm-window
  selection; operational scheduling would require an event predictor from always-on
  channels.

## Energy-Account Learned PD-PPO

Supported:

- The locked learned-policy result is the 100k storm-window curriculum run over
  seeds `41--45`.
- Storm-window evaluation:
  - PD-PPO: `0.4153 +/- 0.0051`.
  - AoI: `0.4176 +/- 0.0105`.
  - Feasible static projection: `0.4742 +/- 0.0236`.
  - Round-robin: `0.4451 +/- 0.0167`.
  - Random: `0.4565 +/- 0.0140`.
- Storm-window win counts:
  - vs feasible static projection: `5/5`;
  - vs round-robin: `5/5`;
  - vs random: `5/5`;
  - vs AoI: `3/5`.
- Full-distribution evaluation:
  - PD-PPO: `0.3155 +/- 0.0133`;
  - AoI: `0.3168 +/- 0.0135`;
  - feasible static projection: `0.3318 +/- 0.0062`.
- Full-distribution win counts:
  - vs AoI: `4/5`;
  - vs feasible static projection: `4/5`;
  - vs round-robin/random: `5/5`.

Required wording:

- Claim consistent storm-window superiority over feasible static projection,
  round-robin, and random scheduling.
- Claim competitiveness with AoI and a small mean advantage, but not robust AoI
  dominance.
- Claim a small average full-distribution generalization gain, with narrow margins
  and non-uniform per-seed comparisons.

## Unsupported or Over-Strong Claims

Do not claim:

- PD-PPO robustly dominates AoI in the energy-account setting.
- PD-PPO reliably learns clean event-triggered laser gating.
- Fixed-budget V3.1 alone proves dynamic scheduling value.
- The energy-account model is a complete physical watt/battery/heater model.
- Simulator event labels are available to an operational controller.

## Recommended Paper Positioning

The defensible final narrative is:

1. Fixed-budget V3.1 shows that prediction-driven PPO learns a stable feasible
   scheduler that beats dynamic heuristic baselines and approaches a strong static
   reference.
2. The energy-account diagnostic explains when dynamic scheduling becomes valuable:
   momentary feasibility and long-term sustainability diverge in event-rich storm
   windows.
3. The locked curriculum PD-PPO result demonstrates that this opportunity is
   learnable enough to beat static projection, round-robin, and random scheduling
   consistently in storm windows, while AoI remains a strong and partially competitive
   freshness baseline.
4. The remaining limitation is mechanism-level: current PPO does not provide robust
   AoI dominance or clean event-triggered laser-gating behaviour, motivating future
   work on event prediction, long-horizon SOC credit assignment, and hardware power
   measurement.
