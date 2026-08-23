# V113--V115 online-information gate

Positive margins mean lower loss than the validation-selected static action on
the frozen test windows.

| Gate | Ordinary wins | Mean ordinary margin | Macro wins | Mean macro margin |
| --- | ---: | ---: | ---: | ---: |
| V113 one-threshold context | 3/5 | +0.0384 | 5/5 | +0.1064 |
| V114 fixed 0.75 intensity bins | 3/5 | +0.0282 | 5/5 | +0.0745 |
| V115 validation-selected high threshold | 3/5 | +0.0280 | 3/5 | +0.0692 |

The exact eight-step receding diagnostic beats the strongest static action in
all five scenes, with ordinary-loss margins from +0.0570 to +0.1126. V114 shows
that continuous online alert magnitude supports useful event-balanced dynamic
decisions, although the hand-built two-bin rule does not win both endpoints in
all scenes. V115 shows that further threshold selection does not transfer
reliably and is closed.

The next authorized experiment is one clean learned scheduler on the frozen
V113 scene: temporal history encoding, online context encoding, forecast-value
behavioral initialization, forecast-loss PPO, and validation-only checkpoint
selection. No event label, context-policy imitation, or final-test feedback is
used by the learned policy.
