# V345 Direct-Mask BC-Only Diagnostic

## Protocol

V345 used the frozen V338 recalibrated scenes (`6871`, `6872`) and the V344
direct state--raw-candidate-mask utility head. Forecast-loss reward, the
22-action feasible-subset geometry, and hard feasibility rules were unchanged.
PPO updates were disabled (`total_timesteps=0`); the run therefore tests
whether the BC mapping alone is sufficient. No bandit signal, comparator
action, or final-test event label was supplied.

## Results

| Endpoint / comparator | Mean baseline-minus-custom | Custom wins |
|---|---:|---:|
| ordinary loss / validation-selected static | -0.082032372 | 0/2 |
| ordinary loss / feasible static | -0.053851469 | 0/2 |
| ordinary loss / AoI | -0.029382279 | 1/2 |
| ordinary loss / round-robin | -0.036048905 | 0/2 |
| ordinary loss / full-open reference | -0.025556834 | 1/2 |
| macro loss / validation-selected static | -0.167354628 | 0/2 |
| macro loss / feasible static | -0.069305205 | 0/2 |
| macro loss / AoI | -0.036449943 | 1/2 |
| macro loss / round-robin | -0.044250253 | 1/2 |
| macro loss / full-open reference | -0.032698967 | 1/2 |

Positive values favor custom PD-PPO. The direct head retained high training
action accuracy (`0.829` and `0.830`) and all `22` candidate actions were
represented in both BC runs, but closed-loop prediction still lost to the
selected static schedule on both scenes and both endpoints.

## Behavior gate

Both runs had zero warm-up aborts and zero always-on channels. The runs had,
respectively, `1` and `3` always-off channels and `4` and `3` intermediate-duty
channels. Switching rates were `0.014257` and `0.001447` per step.

## Decision

V345 rejects the hypothesis that PPO updates alone explain V344's transfer
failure. Local supervised action fitting is adequate, but the resulting policy
does not optimize the long-horizon executed forecast return. V345 is a
diagnostic, not primary evidence.

Raw artifacts are stored in the paired
`reports/v345_direct_mask_action_bc_only_diag_seed*_b1p75_20260822/`
directories.
