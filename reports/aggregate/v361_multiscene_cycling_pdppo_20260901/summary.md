# v361_multiscene_cycling_pdppo_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           4              0           0             -0.046976          -0.116356
        dynamic           4              3           3              0.008118           0.025382
      full_open           4              1           1             -0.013675          -0.031763

Behavior rows with zero warm-up aborts and zero constant channels: 3/4.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6901         7481                          10        0.428221     0.815975               -0.034885            -0.034226                 0.039354              0.070501                   0.037791                0.080815                   0                       0                        0                      6           0.032783    1.407917            1.92
 6902         7482                          10        0.380540     1.135750               -0.030974            -0.100404                 0.015656              0.061518                  -0.013732               -0.073600                   0                       0                        0                      6           0.030974    1.465069            2.06
 6903         7483                           5        0.443347     0.994310               -0.085768            -0.170243                -0.035773             -0.053822                  -0.061010               -0.095096                   0                       0                        1                      5           0.012592    1.246094            1.76
 6904         7484                          40        0.434062     0.890062               -0.036278            -0.160553                 0.013236              0.023330                  -0.017749               -0.039171                   0                       0                        0                      6           0.054422    1.504340            2.06
