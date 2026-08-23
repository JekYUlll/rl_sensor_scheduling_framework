# V125 dense on-policy forecast-value pilot

Seed 1303 retained the V124 configuration and changed only the forecast-value
auxiliary coefficient from 0.05 to 0.5 and its stride from 64 to 16.

The validation selector improved from 1.204061 in V124 to 1.105692. Final
ordinary loss was 0.539172 against static 0.548455, for margin +0.009283. The
static-normalized macro was 0.963249 against static 0.865771, for margin
-0.097478. There were no always-on or always-off channels, five channels had
intermediate duty, switching was 0.040961 per step, and abort count was zero.

The configuration passes the bounded ordinary-loss and behavior check but not
the macro endpoint. It is frozen for a four-seed expansion before any further
method decision.
