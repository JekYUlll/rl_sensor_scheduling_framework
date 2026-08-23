# Receding-action online learnability audit

```text
         training_source           feature_set                       model  seeds  mean_top1_accuracy  mean_top3_accuracy  mean_action_cost_regret  mean_gain_vs_static  mean_fraction_recovered  positive_gain_seeds
           rl_train_only         alert_context extra_trees_cost_regression      5            0.149653                 NaN                 0.038543             0.011263                 0.233906                    5
           rl_train_only         alert_context      hist_gradient_boosting      5            0.202951            0.397135                 0.039690             0.010115                 0.211711                    5
           rl_train_only         alert_context        multinomial_logistic      5            0.205729            0.398003                 0.040348             0.009457                 0.199766                    5
           rl_train_only         alert_context       ridge_cost_regression      5            0.153559                 NaN                 0.040384             0.009421                 0.197883                    5
           rl_train_only complete_online_state extra_trees_cost_regression      5            0.163542                 NaN                 0.037121             0.012684                 0.255214                    5
           rl_train_only complete_online_state      hist_gradient_boosting      5            0.147483            0.314149                 0.047112             0.002694                 0.054873                    4
           rl_train_only complete_online_state        multinomial_logistic      5            0.119705            0.289844                 0.048130             0.001676                 0.033522                    2
           rl_train_only complete_online_state       ridge_cost_regression      5            0.095660                 NaN                 0.045532             0.004273                 0.083212                    4
rl_train_plus_validation         alert_context extra_trees_cost_regression      5            0.178385                 NaN                 0.037192             0.013662                 0.272940                    5
rl_train_plus_validation         alert_context      hist_gradient_boosting      5            0.210503            0.411545                 0.042336             0.008518                 0.173034                    5
rl_train_plus_validation         alert_context        multinomial_logistic      5            0.221094            0.415799                 0.042704             0.008150                 0.166297                    5
rl_train_plus_validation         alert_context       ridge_cost_regression      5            0.168229                 NaN                 0.040033             0.010821                 0.214345                    5
rl_train_plus_validation complete_online_state extra_trees_cost_regression      5            0.200694                 NaN                 0.034063             0.016791                 0.329974                    5
rl_train_plus_validation complete_online_state      hist_gradient_boosting      5            0.194618            0.363802                 0.043829             0.007026                 0.136070                    4
rl_train_plus_validation complete_online_state        multinomial_logistic      5            0.146007            0.339236                 0.044603             0.006252                 0.121778                    5
rl_train_plus_validation complete_online_state       ridge_cost_regression      5            0.146441                 NaN                 0.038699             0.012155                 0.236972                    5
         validation_only         alert_context extra_trees_cost_regression      5            0.175781                 NaN                 0.042971             0.009618                 0.185625                    5
         validation_only         alert_context      hist_gradient_boosting      5            0.180122            0.357205                 0.044473             0.008117                 0.153644                    4
         validation_only         alert_context        multinomial_logistic      5            0.183594            0.354861                 0.044366             0.008224                 0.154462                    5
         validation_only         alert_context       ridge_cost_regression      5            0.164844                 NaN                 0.044687             0.007903                 0.152035                    5
         validation_only complete_online_state extra_trees_cost_regression      5            0.143142                 NaN                 0.039984             0.012606                 0.231218                    5
         validation_only complete_online_state      hist_gradient_boosting      5            0.182639            0.328819                 0.047790             0.004800                 0.088730                    4
         validation_only complete_online_state        multinomial_logistic      5            0.130642            0.282813                 0.049522             0.003068                 0.044176                    2
         validation_only complete_online_state       ridge_cost_regression      5            0.111198                 NaN                 0.047012             0.005578                 0.103157                    5
```
