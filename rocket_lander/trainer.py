"""
Training loop utilities.

Responsibilities:
- train_step(): sample batch, compute targets, backprop, gradient clip, optimizer step
- target network sync schedule
- epsilon decay schedule
- (optional) evaluation/showcase episode runner
"""
import torch
import torch.nn as nn

from . import config as C


class Trainer:
    def __init__(self, policy_net, target_net, optimizer, replay_buffer):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.buffer = replay_buffer
        self.loss_fn = nn.MSELoss()

    def train_step(self):
        if len(self.buffer) < C.BATCH_SIZE:
            return None

        s, a, r, ns, d = self.buffer.sample(C.BATCH_SIZE)

        q = self.policy_net(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            target = r + C.GAMMA * next_q * (1 - d)

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
