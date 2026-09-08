# V600-V602 Physics Screen (2026-09-09)

This is a resource-geometry diagnostic on existing V527 entity-effective
truth for seeds 7181--7184. It does not fit a forecaster, train a policy, or
change truth targets, event labels, or policy observations.

## Frozen mapping and rules

The selectable channels map to GMX500, Parsivel2, LPS10, SI-111-SS, and FC4;
CR1000Xe is the mandatory logger backbone. GMX500 and Parsivel2 receive
state-dependent heater proxy rules frozen before inspecting outputs. LPS10,
SI-111-SS, and FC4 remain fixed because no device-specific dynamic load
evidence was accepted in this screen. CNF4 is not used as an LPS10 proxy.

The absolute reference is the documented 1200 W controller ceiling. The
2.15 effective-resource budget is a development screen obtained by applying
the documented physical heater-to-base ratios to the existing channel costs;
it is not an installed-current calibration and cannot support a hardware
claim by itself.

## Results

| Seed | GMX heater fraction | Parsivel heater fraction | Effective feasible masks | Effective frontier forms | Absolute feasible masks |
|---:|---:|---:|---:|---:|---:|
| 7181 | 0.6915 | 0.8495 | 8--28 | 3 | 32/32 |
| 7182 | 0.6928 | 0.8498 | 8--28 | 3 | 32/32 |
| 7183 | 0.6933 | 0.8499 | 8--28 | 3 | 32/32 |
| 7184 | 0.6919 | 0.8497 | 8--28 | 3 | 32/32 |

Heater state transitions occur at approximately 0.0090--0.0103 per hourly
step. Median heater-state run lengths are 15 hours for GMX500 and 13 hours
for Parsivel2. The effective frontier changes in all four seeds, but eight
effective masks remain feasible throughout each seed. This is sufficient to
admit a resource-screen follow-up, not sufficient to claim dynamic scheduling
value.

## Gate decision

- Absolute physical envelope: **fails to bind**; all 32 subsets remain
  feasible at every row.
- Development effective envelope: **frontier variation passes the screen**;
  the number of feasible masks changes and the frontier has three forms in
  each seed.
- Complete-subset forecast geometry: **not run**.
- Chronological online transfer: **not run**.
- PD-PPO training: **not authorized**.

The next admissible step is to validate the effective scaling against installed
acquisition duty/current information or to define a deployment-derived finite
resource envelope. Only then may the frozen forecaster be reintroduced.
