# V66 linear-oracle information gate

- Seeds: `[901, 902, 903, 904, 905]`
- Full-open wins over the strongest static schedule: ordinary `0/5`, macro `0/5`, joint `0/5`.
- Mean margins (static loss minus full-open loss): ordinary `-0.176308`, macro `-0.177406`.
- Decision: reject the linear predictor diagnostic; do not launch online-context replay or PPO from V66.
- Interpretation: changing predictor family does not repair information ordering and therefore is not a justified mainline modification.
