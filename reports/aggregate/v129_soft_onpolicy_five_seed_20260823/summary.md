# V129 masked soft forecast-value five-seed summary

|   seed |   ordinary_margin_vs_static |   macro_margin_vs_static |   ordinary_margin_vs_best_dynamic | best_dynamic   |   always_on |   always_off |   mid_duty |   switches_per_step |   warmup_aborts |   selected_update |   validation_ratio |   particle_margin |   flux_margin |   thermal_margin |
|-------:|----------------------------:|-------------------------:|----------------------------------:|:---------------|------------:|-------------:|-----------:|--------------------:|----------------:|------------------:|-------------------:|------------------:|--------------:|-----------------:|
|   1301 |                   -0.009274 |                 0.021800 |                          0.067901 | random         |           0 |            0 |          6 |            0.030829 |               0 |                20 |           1.023709 |          0.122478 |     -0.020715 |        -0.036365 |
|   1302 |                    0.024866 |                 0.072983 |                          0.027841 | aoi            |           0 |            0 |          6 |            0.042553 |               0 |                20 |           0.983999 |          0.026277 |      0.039478 |         0.153195 |
|   1303 |                    0.035874 |                 0.014179 |                          0.065721 | random         |           0 |            1 |          5 |            0.039369 |               0 |                20 |           1.107072 |         -0.198067 |     -0.068578 |         0.309181 |
|   1304 |                   -0.014059 |                -0.035968 |                         -0.016578 | aoi            |           0 |            1 |          5 |            0.011579 |               0 |                 0 |           1.058676 |          0.082448 |      0.004437 |        -0.194789 |
|   1305 |                    0.027281 |                -0.006804 |                          0.019696 | aoi            |           0 |            1 |          3 |            0.013316 |               0 |                20 |           0.956742 |         -0.435337 |      0.021240 |         0.393686 |

## Aggregate

- ordinary static wins: `3/5`, mean `+0.012938`
- macro static wins: `3/5`, mean `+0.013238`
- joint static wins: `2/5`
- best conventional dynamic wins: `4/5`, mean `+0.032916`
- behavior passes: `5/5`

The soft categorical auxiliary materially improves V126, but the 2/5 joint result does not authorize fresh confirmation. Run one bounded target-temperature test on the two failing development scenes.
