# Flexible-subset v4b forecast-greedy warm-start diagnostic

The matched seed-405 run used frozen-forecast greedy labels across all 24
feasible actions for BC warm start, then disabled continuing AWBC. BC accuracy
reached 0.271 and the final policy executed 17 masks.

The broader action coverage did not improve scheduling. PD-PPO lost to selected
static by 0.042258 in mean loss and 0.134527 in macro loss, and lost to the best
conventional dynamic reference by 0.062929 and 0.227376. One channel remained
inactive.

Training logs show mean value loss 103.96 while mean policy loss magnitude was
0.00836. Actor and critic networks are distinct, but whole-model gradient
clipping lets critic gradients consume the shared clipping budget. The next
matched diagnostic enables separate actor, critic, and auxiliary gradient
clipping; no reward, teacher, or architecture module is added.
