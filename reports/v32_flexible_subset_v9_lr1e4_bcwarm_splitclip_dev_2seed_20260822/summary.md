# Flexible-subset v9 low-learning-rate diagnostic

V9 restored v6 and changed only the PPO learning rate from 3e-4 to 1e-4. On
development seeds 406 and 407, average margins were -0.012654/-0.004511 against
the selected static subset and -0.001211/-0.032193 against the strongest
conventional dynamic policy. Executed subset coverage increased to 13 and 16,
but forecast performance did not pass. Lower learning rate is rejected as a
standalone fix.
