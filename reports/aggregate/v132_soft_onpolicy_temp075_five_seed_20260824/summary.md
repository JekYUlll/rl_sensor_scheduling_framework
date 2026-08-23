# V132 frozen temperature-0.75 development summary

|   seed |   ordinary_margin_vs_static |   macro_margin_vs_static |   ordinary_margin_vs_best_dynamic | best_dynamic   |   always_on |   always_off |   mid_duty |   switches_per_step |   warmup_aborts |   selected_update |   validation_ratio |   particle_margin |   flux_margin |   thermal_margin |
|-------:|----------------------------:|-------------------------:|----------------------------------:|:---------------|------------:|-------------:|-----------:|--------------------:|----------------:|------------------:|-------------------:|------------------:|--------------:|-----------------:|
|   1301 |                    0.004485 |                 0.077972 |                          0.081660 | random         |           0 |            0 |          6 |            0.038211 |               0 |                20 |           1.031196 |          0.164728 |      0.076061 |        -0.006874 |
|   1302 |                    0.049074 |                 0.105595 |                          0.052049 | aoi            |           0 |            0 |          6 |            0.041829 |               0 |                15 |           1.012912 |         -0.020722 |      0.100456 |         0.237053 |
|   1303 |                    0.042544 |                -0.007524 |                          0.072391 | random         |           0 |            0 |          5 |            0.031119 |               0 |                20 |           1.114059 |         -0.295170 |     -0.013712 |         0.286310 |
|   1304 |                    0.041823 |                 0.170972 |                          0.039304 | aoi            |           0 |            0 |          5 |            0.023448 |               0 |                15 |           1.002650 |          0.049355 |      0.056701 |         0.406859 |
|   1305 |                    0.026305 |                 0.110441 |                          0.018719 | aoi            |           0 |            0 |          5 |            0.021856 |               0 |                 5 |           0.999337 |         -0.053185 |     -0.013539 |         0.398046 |

## Aggregate

- ordinary static wins: `5/5`, mean `+0.032846`
- macro static wins: `4/5`, mean `+0.091491`
- joint static wins: `4/5`
- conventional dynamic wins: `5/5`, mean `+0.052825`
- behavior passes: `5/5`

The frozen development gate passes. Scene seeds 1401--1406 are locked for fresh confirmation before generation or training.
