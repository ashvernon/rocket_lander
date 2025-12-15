"""
Experience replay buffer utilities.

Responsibilities:
- Store transitions: (s, a, r, s_next, done)
- Sample batches efficiently (numpy -> torch float32)
- Optional: prioritized replay later
"""
import random
from collections import deque
import numpy as np
import torch

from . import config as C


class ReplayBuffer:
    def __init__(self, capacity: int = C.MEMORY_SIZE):
        self.mem = deque(maxlen=capacity)

    def __len__(self):
        return len(self.mem)

    def push(self, s, a, r, ns, done):
        self.mem.append((s, a, r, ns, done))

    def sample(self, batch_size: int = C.BATCH_SIZE):
        batch = random.sample(self.mem, batch_size)
        s, a, r, ns, d = zip(*batch)

        # fast + correct dtype
        s = torch.from_numpy(np.asarray(s, dtype=np.float32))
        ns = torch.from_numpy(np.asarray(ns, dtype=np.float32))
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)
        return s, a, r, ns, d
