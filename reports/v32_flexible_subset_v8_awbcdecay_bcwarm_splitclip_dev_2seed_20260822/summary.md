# Flexible-subset v8 decaying-AWBC diagnostic

V8 retained the v6 configuration and applied physical-prototype AWBC at 0.15,
linearly decaying it to zero over the first 10,000 PPO steps. On development
seeds 406 and 407, average margins were -0.020409/-0.083357 against the selected
static subset and -0.008042/-0.061643 against the strongest conventional dynamic
policy. Seed 407 collapsed to a low-switching met-plus-FC4 policy. Short-lived
AWBC therefore did not stabilize the broad-action solution and is rejected.
