# Flexible-subset v10 calibration-checkpoint diagnostic

V10 evaluated every fifth PPO update on the existing calibration/validation
starts and restored the checkpoint with minimum validation mean forecast loss.
Seeds 406 and 407 selected updates 5 and 30. Average margins were
-0.009463/+0.038832 against static and +0.000694/-0.001590 against the strongest
conventional dynamic policy. Checkpoint selection functioned correctly but did
not pass both endpoints, so final-update selection was not the main blocker.
