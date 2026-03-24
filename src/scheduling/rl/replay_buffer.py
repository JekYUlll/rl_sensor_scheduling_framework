from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        actions = np.asarray(a)
        if actions.dtype == object:
            actions = np.stack([np.asarray(item) for item in a], axis=0)
        return (
            np.asarray(s, dtype=np.float32),
            actions,
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
