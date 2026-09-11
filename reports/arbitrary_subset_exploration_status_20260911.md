# Arbitrary-subset exploration status (2026-09-11)

## Decision

The bounded audit of the strongest remaining physical route, V642, is now
complete. The arbitrary-subset branch should not continue with the current
physical assets. Do not continue V695--V698 by adding further coupled
mechanisms, and do not start online transfer or PPO on any of these routes.

## Evidence

The current entity-compatible physical routes all fail the downstream
operating-value gate on their first completed seed:

| Route | Operating gap | Near-optimal static intersection | Decision |
|---|---:|---:|---|
| V695 physical entity correction | 0.00605018 | non-empty at 5% | closed |
| V696 particle-volume to future-flux | 0.000267 | 4 candidates at 1% | closed |
| V697 integrated heater-demand quality | 0.00013169 | 13 candidates at 1% | closed |
| V698 entity remap of V376 scene | 0 | non-empty at 1% | closed |

V642 remains the strongest candidate: it used the physical resource frontier
and causal sensor-quality relations, and passed the operating gap threshold in
three of four seeds (`0.046997`, `0.119788`, `0.017873`, `0`). The failing
seed 7180 retained a static radiometer optimum. This is evidence of a real
resource frontier, but not yet of stable forecast-value crossover.

## Audit result

The V642 seed7180 audit found full support for the fixed
`radiometer_basic;cr1000xe_backbone` candidate and all three deployable
operating groups. It was the lowest-loss candidate in every group, with
operating gap `0` and a singleton 1% near-optimal intersection. The failure is
therefore not a missing-window or candidate-support artifact. V642 is closed
before online transfer and PPO.

The only justified continuation is a new pre-specified physical measurement
or resource model. It must be frozen before truth generation and must first
pass the same policy-free geometry gates. Without that new input, stop the
arbitrary-subset branch and retain its outputs as diagnostic evidence.

## Rechecked historical frequency-cost route

The older fixed-frequency-cost route was also reviewed because it preserves
the minimum scheduling epoch and does not make sampling frequency an action.
It is not a viable rescue route: V566 produced operating gaps
`0.005465`, `0.000106`, `0.003823`, and `0`, while the later V610 route added
heater coupling and still produced approximately `0.0000007`, `0`, `0`, and
`0.0000198` at the deployable operating level. V612 subsequently confirmed
the same near-universal static subset. These historical results are not
reused as entity-system evidence or PPO evidence.

## Prohibited next steps

- No PPO training on V695--V698 or V642 before the geometry gate.
- No more quality multipliers, budget changes, or hidden-state target terms.
- No reuse of the old V376 geometry pass as entity-system evidence.

## Continuation decision (2026-09-11)

The exploration should continue only through a newly justified cumulative-energy
route. Local design documentation provides an independent system basis: a 24 V,
8640 Wh low-temperature battery, a 600 W photovoltaic module, and a 24 V, 400 W
wind generator. This is materially different from the historical arbitrary
`harvest-per-step` model and can support a predeclared SOC/resource screen.

The continuation is conditional and policy-free. First compute the documented
energy envelope and enumerate all 32 optional subsets using the existing hourly
truth. Require resource-state support, state-varying feasible frontiers, and a
non-empty set of deployable operating groups before fitting a forecaster. No
PPO, teacher, reward tuning, or scene retuning is authorized at this stage.

The existing `scripts/127_audit_entity_energy_trajectory.py` is only a starting
audit: its fixed 600 W hysteretic heater load is a stress diagnostic, not a
complete harvest model. The new screen must explicitly separate documented
installed loads, design-envelope assumptions, and any conservative PV/wind
conversion assumptions. If the SOC route fails the same geometry gates, the
arbitrary-subset branch should be frozen as negative evidence.

## V699/V700 closeout (2026-09-11)

The corrected cumulative-energy route passed the resource-feasibility screen
but not the all-seed forecast-value gate. V699 used trace-backed PV and gave
opportunity gaps `0.007659`, `0.004258`, `0.007456`, and `0.023342`. V700 then
included the documented GMX500/Parsivel2 heater loads in both feasibility and
SOC consumption; its gaps were `0.002591`, `0.002090`, `0.006981`, and
`0.023518`. V700 changed candidate support and covered seven resource
conditions, but only one seed exceeded the prespecified `0.01` gap threshold,
and 1% static intersections remained for two seeds.

Both routes are closed before online transfer or PPO. The corrected artifacts
remain under `reports/v699_energy_forecast_geometry_20260911_corrected/` and
`reports/v700_entity_soc_heater_geometry_20260911/`. The current evidence
supports state-dependent resource feasibility, not a robust PD-PPO value claim.

## Stage decision (2026-09-11)

The exploration is worth continuing only as a new, independently justified
physical-data branch. It is not worth continuing by retraining PPO, changing
heater thresholds, increasing quality multipliers, or retuning the budget in
V699/V700. Those routes have established the resource-side effect but failed
the predeclared all-seed forecast-value gate.

The next branch must freeze its hardware/resource assumptions and truth
generation before any forecast loss or policy result is inspected. It must
first report, for at least four seeds, resource-state occupancy, feasible-mask
support, operating opportunity gap, and the intersection of near-optimal
static subsets. Online transfer and PPO are admissible only if the complete
screen passes the existing geometry gate. If no independent physical input is
available, preserve V699/V700 as diagnostic evidence and return to paper or
implementation cleanup rather than adding another synthetic coupling.

Remote execution is currently quiescent. The experiment workspace occupies
about 1.8 GB under `/data/zhangzhuyu/pdppo_soc_screen_v699`; the account uses
about 27.9 GB of a 100 GB quota, so no urgent migration or deletion is needed.
