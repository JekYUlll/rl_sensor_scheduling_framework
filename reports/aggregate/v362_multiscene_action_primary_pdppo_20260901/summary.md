# v362_multiscene_action_primary_pdppo_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           4              0           0             -0.038161          -0.122810
        dynamic           4              3           3              0.016934           0.018928
      full_open           4              1           1             -0.004860          -0.038217

Behavior rows with zero warm-up aborts and zero constant channels: 3/4.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6901         7491                          10        0.427203     0.833478               -0.033868            -0.051729                 0.040372              0.052998                   0.038808                0.063312                   0                       0                        0                      4           0.021856    1.202934            2.06
 6902         7492                          35        0.383392     1.173356               -0.033826            -0.138010                 0.012804              0.023912                  -0.016584               -0.111206                   0                       0                        0                      6           0.060067    1.490851            2.06
 6903         7493                          20        0.385422     0.909135               -0.027843            -0.085068                 0.022152              0.031353                  -0.003085               -0.009921                   0                       0                        0                      6           0.028803    1.470851            1.92
 6904         7494                          35        0.454891     0.945944               -0.057107            -0.216434                -0.007593             -0.032551                  -0.038578               -0.095053                   0                       0                        1                      3           0.016066    1.395877            2.06
