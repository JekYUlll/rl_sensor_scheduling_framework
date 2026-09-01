# V352 cycling-scene control

This is a pre-training structural screen. The receding reference is privileged and is not a deployable policy.

- Scene change: event subtype assignment `cycling`, cycle length 12 samples; all other V338 physical and protocol settings are unchanged.
- Receding versus best static: ordinary 2/2 wins, mean margin +0.069595; macro 2/2 wins, mean margin +0.183206.
- Training decision: authorize PPO only if both seeds show positive oracle margins and all behavior fields are valid.

 seed  event_coverage_actual  event_cluster_count  particle_rate  flux_rate  thermal_rate event_subtype_assignment  event_subtype_cycle_steps  static_best_ordinary  static_best_macro  receding_ordinary  receding_macro  receding_minus_static_ordinary  receding_minus_static_macro  receding_action_coverage  receding_always_on  receding_always_off  receding_mid_duty  receding_switches_per_step  receding_warmup_abort
 6891               0.558222                  314       0.186194   0.186861      0.185167                  cycling                         12              0.314474           0.720335           0.233727        0.512701                        0.080746                     0.207634                        22                   0                    0                  6                    0.056785                      0
 6892               0.622861                  106       0.206083   0.207667      0.209111                  cycling                         12              0.404981           1.119009           0.346538        0.960231                        0.058444                     0.158778                        22                   0                    0                  6                    0.060041                      0
