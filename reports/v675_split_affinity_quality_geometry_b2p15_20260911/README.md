# V675 split specialist-affinity geometry

V675 reused the V672 physical resource trace and separated the four optional
specialist qualities into transport, particle, surface-thermal, and lagged
solar-radiation affinities. The truth relation uses only decision-time
nowcasts with a six-step lag; targets and event labels are not used by the
quality transform. Frozen TCN assets were refit before this audit. No policy
was trained.

| seed | condition gap | operating gap | 1% operating near-optimal intersection | decision |
|---:|---:|---:|---|---|
| 7177 | 0.00332100 | 0.03412441 | empty | pass |
| 7178 | 0.00034138 | 0.00000727 | 7 candidates | fail |
| 7179 | 0.00010846 | 0.00010035 | 7 candidates | fail |
| 7180 | 0.00149489 | 0.00019756 | 2 candidates | fail |

Separating the specialist affinities did not produce stable downstream subset
value at the 2.15 normalized budget. V675 is closed before online transfer and
PPO. A separate binding-budget probe is evaluated using the same frozen assets;
it is not part of the V675 result.
