import gymnasium as gym

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import math


class DQN(nn.Module):

    def __init__(self, env: gym.Env, hidden_layers: Sequence[int],
                 gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000) -> None:
        super(DQN, self).__init__()

        self.env = env

        self.action_count = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape[0]

        node_counts = [self.observation_size, *hidden_layers, self.action_count]
        self.layers = nn.ParameterList()

        for idx in range(len(node_counts) - 1):
            self.layers.append(nn.Linear(node_counts[idx], node_counts[idx + 1]))

        self.activation = F.relu

        self.steps_taken = 0

        self.eps_start = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        for layer in self.layers[:-1]:
            h = self.activation(layer(h))

        return self.layers[-1](h)

    def train(self, batch, target, optimizer, criterion):
        if batch is None:
            return

        non_final_mask = torch.tensor(batch.is_non_terminal, dtype=torch.bool)
        non_final_next_states = torch.cat([s for i, s in enumerate(batch.next_state) if non_final_mask[i]])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.forward(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(state_batch.shape[0])
        with torch.no_grad():
            next_state_values[non_final_mask] = target(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        optimizer.step()

    def select_action(self, state):
        """epsilon-greedy action selection. The epsilon threshold has exponential decay"""
        sample = random.random()

        threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_taken / self.eps_decay)
        self.steps_taken += 1

        if sample > threshold:
            with torch.no_grad():
                return self.forward(state).max(1).indices.view(1, 1)

        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)
