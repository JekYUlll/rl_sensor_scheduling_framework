# Receding-action online learnability audit

```text
          feature_set                       model  seeds  mean_top1_accuracy  mean_top3_accuracy  mean_action_cost_regret  mean_gain_vs_static  mean_fraction_recovered  positive_gain_seeds
        alert_context extra_trees_cost_regression      5            0.175781                 NaN                 0.042971             0.009618                 0.185625                    5
        alert_context      hist_gradient_boosting      5            0.180122            0.357205                 0.044473             0.008117                 0.153644                    4
        alert_context        multinomial_logistic      5            0.183594            0.354861                 0.044366             0.008224                 0.154462                    5
        alert_context       ridge_cost_regression      5            0.164844                 NaN                 0.044687             0.007903                 0.152035                    5
complete_online_state extra_trees_cost_regression      5            0.143142                 NaN                 0.039984             0.012606                 0.231218                    5
complete_online_state      hist_gradient_boosting      5            0.182639            0.328819                 0.047790             0.004800                 0.088730                    4
complete_online_state        multinomial_logistic      5            0.130642            0.282813                 0.049522             0.003068                 0.044176                    2
complete_online_state       ridge_cost_regression      5            0.111198                 NaN                 0.047012             0.005578                 0.103157                    5
```
