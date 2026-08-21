# Context Confidence Bin Analysis

Margins are `bandit - PD-PPO`; positive means CA-PD-PPO is better.

| bin         | num_windows | pdppo_loss | bandit_loss | margin    | pdppo_specialist_entropy | pdppo_action_confidence | bandit_action_entropy |
| ----------- | ----------- | ---------- | ----------- | --------- | ------------------------ | ----------------------- | --------------------- |
| [0.00,0.40) | 18420       | 1.112777   | 1.107992    | -0.004785 | 0.124335                 |                         | 0.000237              |
| [0.40,0.55) | 1619        | 0.885148   | 0.881953    | -0.003195 | 1.064811                 |                         | 0.574122              |
| [0.55,0.70) | 3558        | 0.948801   | 0.948065    | -0.000736 | 1.049366                 |                         | 1.030458              |
| [0.70,0.85) | 2692        | 0.971749   | 0.974214    | 0.002465  | 1.042967                 |                         | 1.132565              |
| [0.85,1.00] | 72015       | 1.606405   | 1.612813    | 0.006407  | 1.034257                 |                         | 0.987188              |

Per-seed details: `/home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/analysis/ca_pdppo_failure_20260703/context_confidence_bins.csv`
