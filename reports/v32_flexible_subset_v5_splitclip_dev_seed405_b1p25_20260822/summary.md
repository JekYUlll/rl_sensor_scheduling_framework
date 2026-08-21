# Flexible-subset v5 grouped-gradient diagnostic

The matched seed-405 run changed only gradient clipping relative to the strong-BC
v3 configuration. Separate clipping reduced final policy entropy from about
3.11 to 1.99, confirming that whole-model clipping had suppressed actor updates.

PD-PPO beat selected static on both endpoints and beat the best conventional
dynamic reference by 0.002737 in mean loss, but its macro loss was 0.004705
worse than random. The deterministic policy collapsed exactly to the four
training prototypes, although all six channels retained intermediate duty.

Grouped clipping is retained as the optimizer correction. Strong BC is not:
the next matched run restores v3's weaker BC settings while keeping grouped
clipping, allowing forecast-reward PPO updates to shape non-prototype masks.
