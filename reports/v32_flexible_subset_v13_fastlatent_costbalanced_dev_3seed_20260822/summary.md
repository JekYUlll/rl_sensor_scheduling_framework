# Flexible-subset v13 fast-latent development result

V13 changed only subtype latent update alpha from 0.22 to 0.55. PD-PPO lost
both mean-loss comparisons in all three seeds; average margins were
-0.034975/-0.098079 against static and -0.020109/-0.065995 against conventional
dynamic policies. Online context and privileged exact-label diagnostics also
failed to beat static consistently. Physical prototype actions lost both
endpoints to static in all three seeds. Faster innovations reduce the value of
current specialist observations for future prediction and are rejected.
