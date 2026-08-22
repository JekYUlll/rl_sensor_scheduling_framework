# Flexible-subset v15 thermal-pair development result

V15 changed only the thermal prototype from `{radiometer, IR}` to
`{shielded thermo-hygro, IR}`. The physical warning and exact-label policies
then beat static on both endpoints in seeds 405 and 407, but remained slightly
behind in seed 406. PD-PPO itself did not beat static on both endpoints in any
seed; average margins were -0.016470/-0.023422 against static and
-0.003692/-0.037712 against the strongest conventional dynamic policy.

Validation-selected, physical, hybrid, and validation-guarded regime mappings
were all audited. None produced 3/3 dynamic-over-static wins because seed 406
showed a calibration-to-test transfer failure. Further action-map tuning is
stopped to avoid post-hoc test adaptation.
