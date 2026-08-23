# V126 dense on-policy forecast-value five-seed summary

Frozen V125 configuration: forecast-value auxiliary coefficient 0.5, stride 16; seed 1303 is the V125 pilot and seeds 1301/1302/1304/1305 are the frozen V126 expansion.

## Aggregate

- seeds: `5`
- ordinary_static_wins: `2`
- macro_static_wins: `1`
- joint_static_wins: `1`
- dynamic_wins: `3`
- behavior_passes: `5`
- mean_ordinary_margin_vs_static: `0.001767680971857344`
- mean_macro_margin_vs_static: `-0.03631139500807303`
- mean_ordinary_margin_vs_best_dynamic: `0.021746110044962984`

## Per seed

|   seed |   pdppo_loss |   static_loss |   ordinary_margin_vs_static |   pdppo_macro |   static_macro |   macro_margin_vs_static | best_dynamic   |   best_dynamic_loss |   ordinary_margin_vs_best_dynamic |   always_on |   always_off |   mid_duty |   switches_per_step |   warmup_aborts | behavior_pass   |   selected_update |   selected_validation_ratio |
|-------:|-------------:|--------------:|----------------------------:|--------------:|---------------:|-------------------------:|:---------------|--------------------:|----------------------------------:|------------:|-------------:|-----------:|--------------------:|----------------:|:----------------|------------------:|----------------------------:|
|   1301 |     0.610642 |      0.593930 |                   -0.016712 |      1.156124 |       1.127822 |                -0.028302 | random         |            0.671105 |                          0.060463 |           0 |            0 |          6 |            0.035316 |               0 | True            |                10 |                    0.979882 |
|   1302 |     0.505151 |      0.532404 |                    0.027254 |      1.085987 |       1.135862 |                 0.049875 | aoi            |            0.535379 |                          0.030228 |           0 |            0 |          6 |            0.047909 |               0 | True            |                10 |                    1.015686 |
|   1303 |     0.539172 |      0.548455 |                    0.009283 |      0.963249 |       0.865771 |                -0.097478 | random         |            0.578302 |                          0.039130 |           0 |            0 |          5 |            0.040961 |               0 | True            |                15 |                    1.105692 |
|   1304 |     0.574800 |      0.566859 |                   -0.007941 |      1.449991 |       1.422887 |                -0.027104 | aoi            |            0.564340 |                         -0.010460 |           0 |            0 |          4 |            0.020408 |               0 | True            |                20 |                    1.010680 |
|   1305 |     0.449385 |      0.446340 |                   -0.003045 |      1.330095 |       1.251547 |                -0.078548 | aoi            |            0.438754 |                         -0.010631 |           1 |            1 |          1 |            0.005500 |               0 | True            |                15 |                    1.003558 |

## Gate decision

The frozen configuration does not pass the five-seed joint gate. Do not launch a confirmatory expansion before diagnosing endpoint-specific failures.
