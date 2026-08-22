# V100--V101 execution diagnostics

Both pilots reuse the frozen V96 seed-1101 scene, evaluator, feasible action
geometry, budget, and comparator trajectories. They are diagnostics and do not
replace the V97 development configuration.

## V100: soft forecast-value pretraining

Soft targets derived from the frozen forecaster broadened deterministic action
use to 28 feasible subsets and gave every channel intermediate duty. Prediction
performance failed decisively. PD-PPO ordinary/macro losses were
`0.529617/1.819422`, corresponding to margins of `-0.094912/-0.498743` against
the strongest static family and `-0.081831/-0.526026` against the best
conventional dynamic family. The soft-label variant is rejected.

## V101: validation-selected execution temperature

Validation-only selection chose temperature `0.1` from
`0, 0.005, 0.01, 0.02, 0.05, 0.1`. Final PD-PPO ordinary/macro losses were
`0.437768/1.333261`. Margins were `-0.003062/-0.012583` against the strongest
static family and `+0.010018/-0.039865` against the best conventional dynamic
family. All six channels had nonzero literal duty, but the endpoint gate failed.
Low-temperature sampling is therefore rejected as the primary execution rule.

These results show that broad channel use can be induced, but not without a
forecast-quality cost. They support retaining deterministic V97 execution and
the original behavior requirement of no *multiple* constant channels, which
V97 satisfies in all five development seeds.
