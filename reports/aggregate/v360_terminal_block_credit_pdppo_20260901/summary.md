# v360_terminal_block_credit_pdppo_20260901 recalibrated-scene PD-PPO development aggregate

All completed seeds are included. Positive margins mean lower PD-PPO loss.

baseline_family  seed_count  ordinary_wins  macro_wins  ordinary_mean_margin  macro_mean_margin
         static           4              0           0             -0.044177          -0.115956
        dynamic           4              3           3              0.010918           0.025783
      full_open           4              1           1             -0.010876          -0.031362

Behavior rows with zero warm-up aborts and zero constant channels: 1/4.

 seed  policy_seed  selected_checkpoint_update  pdppo_ordinary  pdppo_macro  static_ordinary_margin  static_macro_margin  dynamic_ordinary_margin  dynamic_macro_margin  full_open_ordinary_margin  full_open_macro_margin  warmup_abort_count  always_on_sensor_count  always_off_sensor_count  mid_duty_sensor_count  switches_per_step  power_mean  peak_power_max
 6901         7471                          25        0.424923     0.837411               -0.031587            -0.055662                 0.042652              0.049065                   0.041088                0.059379                   0                       0                        2                      2           0.030685    1.376215            1.82
 6902         7472                          15        0.387997     1.193317               -0.038431            -0.157971                 0.008199              0.003951                  -0.021189               -0.131167                   0                       0                        1                      4           0.061659    1.303785            2.06
 6903         7473                           5        0.430755     0.950907               -0.073176            -0.126840                -0.023181             -0.010419                  -0.048418               -0.051693                   0                       0                        0                      6           0.048343    1.510252            2.00
 6904         7474                           5        0.431297     0.852858               -0.033513            -0.123349                 0.016001              0.060534                  -0.014985               -0.001967                   0                       0                        1                      4           0.035895    1.499158            1.90
