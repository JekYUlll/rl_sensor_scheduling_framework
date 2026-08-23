# V130 soft-target temperature 0.5

|        seed |   ordinary_margin_vs_static |   macro_margin_vs_static |   ordinary_margin_vs_best_dynamic |   always_on |   always_off |   mid_duty |   switches_per_step |   selected_update |   validation_ratio |
|------------:|----------------------------:|-------------------------:|----------------------------------:|------------:|-------------:|-----------:|--------------------:|------------------:|-------------------:|
| 1304.000000 |                    0.028408 |                 0.089846 |                          0.025889 |    0.000000 |     1.000000 |   4.000000 |            0.022362 |          5.000000 |           1.044097 |
| 1305.000000 |                    0.041570 |                 0.141514 |                          0.033984 |    0.000000 |     2.000000 |   2.000000 |            0.005934 |         20.000000 |           0.944253 |

Both scenes pass both prediction endpoints and conventional dynamics. Seed1305 fails the behavior gate with two effectively always-off channels; temperature0.75 is the final bounded interpolation.
