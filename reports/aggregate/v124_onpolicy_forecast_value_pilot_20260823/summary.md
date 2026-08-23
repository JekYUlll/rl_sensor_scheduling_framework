# V124 sparse on-policy forecast-value pilot

Seed 1303 used the frozen V120 scene and complete CA-PD-PPO configuration, with
an on-policy all-action forecast-value auxiliary at coefficient 0.05 and stride
64. The final policy had ordinary forecast loss 0.552732 and static-normalized
macro 0.951966. The validation-selected static reference had 0.548455 and
0.865771, giving PD-PPO margins -0.004276 and -0.086195.

All six channels had intermediate duty, switching was 0.025040 per step, and
there were no warm-up aborts. The policy beat AoI, random, and round-robin on
ordinary loss. Validation max-static ratio improved from 1.303862 at the BC
checkpoint to 1.204061 after 20,480 PPO steps, but the forecast-value label rate
was only 0.015625 and the auxiliary regression loss remained near one.

The direction is retained for one denser strength check. V125 changes only the
auxiliary coefficient to 0.5 and stride to 16.
