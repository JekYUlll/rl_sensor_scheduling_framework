# V646 explicit physical-watt unit reconciliation

Dynamic budget: 55 W; normalized interface budget retained: 2.15; materiality passes: 1/4.

| seed | operating gap | best static | condition winners | 1% intersection |
|---:|---:|---|---|---|
| 7401 | 0.000000 | candidate_003 | candidate_003 | candidate_003 |
| 7402 | 0.017026 | candidate_005 | candidate_003, candidate_005, candidate_017 | empty |
| 7403 | 0.002626 | candidate_005 | candidate_003, candidate_005 | empty |
| 7404 | 0.005402 | candidate_005 | candidate_001, candidate_005 | empty |

The explicit physical-watt mapping changes the evaluated resource feasibility path but does not pass the all-seed downstream opportunity gate. No PPO was launched.
