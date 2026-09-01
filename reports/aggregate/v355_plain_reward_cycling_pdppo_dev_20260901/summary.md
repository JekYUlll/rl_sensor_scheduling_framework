# v355_plain_reward_cycling_pdppo_dev_20260901 V346 aggregate

All supplied seeds are included. Positive margins mean lower custom loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              1           1              0.002428           0.015828
        dynamic           2              2           1              0.023168           0.054168
      full_open           2              1           1              0.011063           0.040587

Behavior-valid rows: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6891         7421                          10        0.293137     0.627157                0.021336             0.093178                 0.033824              0.120692                   0.038093                0.130225                   0                       0                        0                      5           0.040237    1.451849            1.84
 6892         7422                          20        0.421462     1.180531               -0.016481            -0.061522                 0.012512             -0.012356                  -0.015968               -0.049052                   0                       0                        0                      6           0.044145    1.415321            2.00
