# V127 masked soft forecast-value pilot

- Seed: `1303`
- Auxiliary loss: `soft_ce`, temperature `1.0`, coefficient `0.5`, stride `16`
- Ordinary margin versus selected static: `+0.035874`
- Static-normalized event-macro margin: `+0.014179`
- Ordinary margin versus best conventional dynamic (random): `+0.065721`
- Behavior: always-on `0`, always-off `1`, mid-duty `5`, switches/step `0.039369`, aborts `0`

## Decision

The one-factor pilot passes both held-out endpoints and the behavior gate. Its checkpoint was selected by the frozen validation rule, but no validation checkpoint beat static on ordinary loss. Run a bounded two-scene development expansion before deciding whether to complete the five-scene wave.
