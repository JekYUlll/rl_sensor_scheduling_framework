# V128 masked soft forecast-value three-seed checkpoint

|   seed |   ordinary_margin_vs_static |   macro_margin_vs_static |   ordinary_margin_vs_best_dynamic | best_dynamic   |   always_on |   always_off |   mid_duty |   switches_per_step |   warmup_aborts |   selected_update |   validation_ratio |
|-------:|----------------------------:|-------------------------:|----------------------------------:|:---------------|------------:|-------------:|-----------:|--------------------:|----------------:|------------------:|-------------------:|
|   1301 |                   -0.009274 |                 0.021800 |                          0.067901 | random         |           0 |            0 |          6 |            0.030829 |               0 |                20 |           1.023709 |
|   1302 |                    0.024866 |                 0.072983 |                          0.027841 | aoi            |           0 |            0 |          6 |            0.042553 |               0 |                20 |           0.983999 |
|   1303 |                    0.035874 |                 0.014179 |                          0.065721 | random         |           0 |            1 |          5 |            0.039369 |               0 |                20 |           1.107072 |

## Aggregate

- ordinary static wins: `2/3`, mean `+0.017156`
- macro static wins: `3/3`, mean `+0.036321`
- joint static wins: `2/3`
- best conventional dynamic wins: `3/3`, mean `+0.053821`
- behavior passes: `3/3`

The frozen three-scene result authorizes completion on development seeds 1304 and 1305.
