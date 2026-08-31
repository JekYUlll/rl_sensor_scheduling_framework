# v346_onpolicy_forecast_aux_direct_mask_dev_20260901 V346 aggregate

All supplied seeds are included. Positive margins mean lower custom loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              1           0             -0.016068          -0.067378
        dynamic           2              2           2              0.028746           0.055726
      full_open           2              2           2              0.040408           0.067278

Behavior-valid rows: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7381                           5        0.462433     1.003399               -0.041610            -0.088499                 0.008232              0.027791                   0.038881                0.060326                   0                       0                        0                      6           0.031987    1.311354            1.92
 6872         7382                          35        0.267175     0.473172                0.009474            -0.046257                 0.049260              0.083662                   0.041934                0.074230                   0                       0                        0                      5           0.040093    1.360816            2.00
