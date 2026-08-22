# Flexible-subset v14 strong-latent development result

V14 restored latent alpha 0.22 and increased particle, flux, and thermal target
amplitudes. PD-PPO improved over v13 but beat static on both endpoints only in
seed 407 and did not beat the strongest conventional dynamic policy on both
endpoints in any seed. Average margins were -0.008709/-0.027998 against static
and -0.013058/-0.032495 against dynamic. Physical diagnostics localized the
largest remaining upper-bound deficit to the thermal `{radiometer, IR}` pair.
