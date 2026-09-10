# V672 shared-driver forecast geometry

V672 reused the frozen V669 truth and replaced only the resource trace with
the shared-driver hysteresis controller from `183_build_shared_driver_resource_trace.py`.
The controller creates four supported heater states in each final window and
changes the feasible 32-subset frontier. This report is the downstream,
forecast-loss screen; it is not policy evidence.

| seed | condition gap | operating gap | 1% operating near-optimal intersection | decision |
|---:|---:|---:|---|---|
| 7177 | 0.00332243 | 0.03418140 | empty | pass |
| 7178 | 0.00033988 | 0.00000691 | 7 candidates | fail |
| 7179 | 0.00005207 | 0.00010739 | 7 candidates | fail |
| 7180 | 0.00178753 | 0.00234366 | 2 candidates | fail |

The resource-only occupancy screen therefore does not translate into a stable
deployable forecast opportunity. Only one of four seeds exceeds the material
operating-gap threshold (0.01) and has no 1% near-optimal static intersection.
V672 is closed before online transfer and PPO. The individual seed JSON files
and raw condition-loss CSVs are retained for audit; no V672 policy result is
promoted.
