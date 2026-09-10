# V673 shared heater-quality geometry

V673 kept the V672 resource controller and applied the declared
`heater_quality_relation_v1` semantics to the same truth-only operating risk.
Unheated channels receive the predeclared lower quality score under risk, and
the two heater-equipped channels recover to quality 1 when their heater is on.
The target generator, budget, candidate family, starts, and geometry audit
were otherwise unchanged. No policy was trained.

| seed | condition gap | operating gap | 1% operating near-optimal intersection | decision |
|---:|---:|---:|---|---|
| 7177 | 0.00329518 | 0.03421241 | empty | pass |
| 7178 | 0.00035174 | 0.00000000 | 7 candidates | fail |
| 7179 | 0.00013945 | 0.00009367 | 7 candidates | fail |
| 7180 | 0.00420043 | 0.00010058 | 2 candidates | fail |

The relation creates genuine discrete quality changes, but they do not produce
stable downstream subset-value separation across seeds. V673 is closed before
online transfer and PPO. The per-seed JSON files and condition-loss CSVs are
retained as a negative scene-screen result.
