# V680 observable-target resource geometry audit

## Decision

Closed before online transfer and PPO. The V527-r2 observable target truth
did not produce stable subset geometry when paired with the independently
generated V672 resource trace.

## Results

| Seed | condition gap | operating gap | 1% operating intersection |
|---:|---:|---:|---|
| 7177 | 0.0028911606 | 0.0491808106 | empty |
| 7178 | 0.0003107718 | 0 | 11 candidates |
| 7179 | 0.0001092284 | 0.0009802611 | 11 candidates |
| 7180 | 0.0007326626 | 0 | 3 candidates |

Only seed7177 passes the material operating-gap and static-intersection gate.
The route is not promoted because the resource trace was generated from the
separate V672 scene rather than from the V527-r2 target/quality process.

## Interpretation

This is a valid negative compatibility result, not evidence against the V527
truth-only relation. Future resource traces must be generated from the same
observable target/quality drivers as the target truth, with a declared
controller and no post-hoc policy feedback.
