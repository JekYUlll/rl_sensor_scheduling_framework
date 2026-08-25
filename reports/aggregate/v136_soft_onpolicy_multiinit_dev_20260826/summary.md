# V136 frozen multi-initialization development control

## Protocol

- Frozen V120 scenes and the complete V132 policy configuration on development
  seeds 1301--1305.
- Three policy seeds per scene: the existing V132 offset 12000 and new offsets
  13000 and 14000.
- One policy per scene was selected by the calibration/validation
  `max_static_ratio`; test losses were not used for selection.

## Selected-policy result

- Ordinary static wins: `4/5`.
- Normalized event-macro static wins: `3/5`.
- Joint static wins: `3/5`.
- Best-conventional-dynamic wins: `5/5`.
- Behavior-gate wins: `4/5`.
- Mean ordinary/macro margins versus static: `+0.024099/+0.048376`.
- Mean ordinary margin versus best conventional dynamic: `+0.044077`.

## Selection diagnosis

Across 15 candidates, the Spearman correlation between validation score and
held-out maximum static-loss ratio was `0.418` (`p=0.156`). Within-scene rank
correlation was `-1.0` for seeds 1301 and 1304, where validation selected the
worst held-out candidate. An oracle test-based choice could find a joint-positive
candidate in all five scenes, but that choice is unavailable under the protocol.

## Decision

V136 fails the prespecified `5/5` joint-static, dynamic, and behavior gate.
Additional policy restarts expose useful candidates but do not solve
validation-to-test selection transfer. Multi-initialization selection is closed;
no fresh confirmation is authorized from this result.
