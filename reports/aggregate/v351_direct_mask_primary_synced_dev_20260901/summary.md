# v351_direct_mask_primary_synced_dev_20260901 V346 aggregate

All supplied seeds are included. Positive margins mean lower custom loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.031644          -0.101427
        dynamic           2              1           1              0.013170           0.021678
      full_open           2              2           2              0.024832           0.033229

Behavior-valid rows: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7381                          10        0.477985     1.053802               -0.057162            -0.138902                -0.007321             -0.022613                   0.023328                0.009923                   0                       0                        0                      6           0.030685    1.494844            2.06
 6872         7382                          20        0.282774     0.490867               -0.006126            -0.063951                 0.033661              0.065968                   0.026335                0.056535                   0                       0                        0                      6           0.031264    1.335434            2.00
