from __future__ import annotations


class LinearEpsilonScheduler:
    def __init__(self, eps_start: float, eps_end: float, decay_steps: int) -> None:
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        step = max(0, int(step))
        ratio = min(1.0, step / self.decay_steps)
        return self.eps_start + ratio * (self.eps_end - self.eps_start)
