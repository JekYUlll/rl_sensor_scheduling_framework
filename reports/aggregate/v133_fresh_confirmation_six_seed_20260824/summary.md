# V133 fresh six-seed confirmation checkpoint

|   seed |   ordinary_margin_vs_static |   macro_margin_vs_static |   ordinary_margin_vs_best_dynamic | best_dynamic   |   always_on |   always_off |   mid_duty |   switches_per_step |   warmup_aborts |   selected_update |   validation_ratio |   particle_margin |   flux_margin |   thermal_margin |
|-------:|----------------------------:|-------------------------:|----------------------------------:|:---------------|------------:|-------------:|-----------:|--------------------:|----------------:|------------------:|-------------------:|------------------:|--------------:|-----------------:|
|   1401 |                   -0.029174 |                -0.202507 |                         -0.004238 | aoi            |           0 |            0 |          5 |            0.023737 |               0 |                15 |           1.073226 |         -0.394370 |      0.023351 |        -0.236501 |
|   1402 |                    0.062077 |                 0.132225 |                          0.062907 | aoi            |           0 |            0 |          6 |            0.014040 |               0 |                20 |           1.015393 |          0.033543 |      0.301807 |         0.061325 |
|   1403 |                    0.021927 |                 0.002812 |                          0.034722 | aoi            |           0 |            0 |          6 |            0.040527 |               0 |                20 |           0.958842 |         -0.097470 |      0.035963 |         0.069943 |
|   1404 |                   -0.012240 |                -0.014891 |                          0.051996 | round_robin    |           0 |            0 |          5 |            0.024027 |               0 |                10 |           0.966727 |          0.053978 |      0.029440 |        -0.128091 |
|   1405 |                    0.050311 |                 0.132636 |                          0.046903 | aoi            |           0 |            0 |          6 |            0.035750 |               0 |                20 |           0.943902 |          0.021176 |      0.024425 |         0.352307 |
|   1406 |                    0.047226 |                 0.251853 |                          0.041151 | round_robin    |           0 |            0 |          4 |            0.021711 |               0 |                15 |           0.996354 |          0.412109 |      0.094024 |         0.249427 |

## Aggregate

- ordinary static wins: `4/6`, mean `+0.023355`, 95% bootstrap CI `[-0.004587, +0.048694]`
- macro static wins: `4/6`, mean `+0.050355`, 95% bootstrap CI `[-0.064241, +0.156257]`
- joint static wins: `4/6`
- best conventional dynamic wins: `5/6`, mean `+0.038907`, CI `[+0.020151, +0.053087]`
- behavior passes: `6/6`

The frozen direction transfers, but static confidence intervals remain inconclusive at six seeds. Extend the same locked protocol to seeds1407--1424 without tuning.
