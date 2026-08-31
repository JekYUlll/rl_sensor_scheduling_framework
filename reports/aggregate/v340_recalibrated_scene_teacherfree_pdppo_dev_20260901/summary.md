# v340_recalibrated_scene_teacherfree_pdppo_dev_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           2              0           0             -0.133373          -0.348183
        dynamic           2              0           0             -0.088559          -0.225078
      full_open           2              0           0             -0.076898          -0.213527

Behavior rows with zero warm-up aborts and zero constant channels: 2/2.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6871         7331                          30        0.634091     1.467492               -0.213267            -0.552591                -0.163426             -0.436302                  -0.132777               -0.403767                   0                       0                        0                      4           0.039803    1.477318            1.90
 6872         7332                          10        0.330127     0.570689               -0.053479            -0.143774                -0.013693             -0.013855                  -0.021018               -0.023287                   0                       0                        0                      6           0.055869    1.508976            2.06
