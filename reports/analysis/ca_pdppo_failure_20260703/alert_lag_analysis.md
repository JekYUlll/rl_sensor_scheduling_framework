# Alert Lag Analysis

Margins are `bandit - PD-PPO`; positive means CA-PD-PPO is better.
Lag bins use ±`12` simulation steps around thresholded alert onsets/offsets.

| bin           | num_windows | pdppo_loss | bandit_loss | margin    | pdppo_specialist_entropy | pdppo_action_confidence | bandit_action_entropy |
| ------------- | ----------- | ---------- | ----------- | --------- | ------------------------ | ----------------------- | --------------------- |
| pre_onset     | 8875        | 1.546427   | 1.543635    | -0.002792 | 1.141271                 |                         | 1.102975              |
| early_event   | 9630        | 1.246921   | 1.249341    | 0.002420  | 1.038600                 |                         | 1.132552              |
| mid_event     | 59505       | 1.577350   | 1.585027    | 0.007676  | 1.031415                 |                         | 0.984503              |
| late_event    | 2393        | 1.649984   | 1.647125    | -0.002859 | 1.017525                 |                         | 0.974150              |
| post_offset   | 521         | 0.701402   | 0.687422    | -0.013981 | 0.331519                 |                         | 0.007217              |
| outside_alert | 17380       | 1.137296   | 1.133178    | -0.004118 | 0.030183                 |                         | 0.000000              |

Per-seed details: `/home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/analysis/ca_pdppo_failure_20260703/alert_lag_analysis.csv`
