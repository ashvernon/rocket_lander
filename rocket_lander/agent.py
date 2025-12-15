"""
RL agent definitions and action selection.

Responsibilities:
- DQN model definition (torch nn.Module)
- epsilon-greedy action selection helpers
- optional: device placement and inference helpers
"""
import torch
import torch.nn as nn

from . import config as C


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C.STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, C.ACTION_SIZE),
        )

    def forward(self, x):
        return self.net(x)


def set_torch_stability():
    # Helps avoid rare pygame+torch multithreading oddities on Windows
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
