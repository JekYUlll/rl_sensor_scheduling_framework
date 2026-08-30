# V299 corrected-scene balanced-start development screen

Seeds: `6811`, `6812`. Truth and validation source artifacts were reused from
V294; the policy seeds were fresh. This is a bounded development diagnostic,
not independent final evidence.

The only learning change from V298 was reducing the PPO event-centered start
probability from `1.0` to `0.35`. Ordinary-loss wins were validation static
`0/2`, feasible static `0/2`, full-open `0/2`, AoI `1/2`, random `1/2`, and
round-robin `1/2`. Mean margins were `-0.035692`, `-0.011556`, `-0.025326`,
`-0.012625`, `+0.004862`, and `+0.010206`, respectively.

Static-normalized macro wins were validation static `1/2`, feasible static
`2/2`, full-open `0/2`, AoI `0/2`, random `2/2`, and round-robin `1/2`.
Both seeds had zero warm-up aborts and zero always-on channels. Seed 6811 had
one always-off channel and five mid-duty channels; seed 6812 had no constant
channels and five mid-duty channels. Switching rates were `0.034665` and
`0.018165` per step.

Decision: reject V299 as a mainline improvement. More balanced training starts
did not improve ordinary-loss transfer to either static baseline and did not
remove the residual constant-channel behavior.
