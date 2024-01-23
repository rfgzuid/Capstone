import torch
import torch.nn as nn

import random
import math

from .settings import Env


class DQN(nn.Module):

    def __init__(self, env: Env) -> None:
        super(DQN, self).__init__()

        self.env = env.env

        self.action_count = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape[0]

        hidden_layers = env.settings['DQN_layers']
        node_counts = [self.observation_size, *hidden_layers, self.action_count]

        self.layers = nn.ParameterList()
        self.activation = env.settings['DQN_activation']

        for idx in range(len(node_counts) - 1):
            self.layers.append(nn.Linear(node_counts[idx], node_counts[idx + 1]))

        self.criterion = env.settings['DQN_criterion']()
        self.optimizer = env.settings['DQN_optim'](self.parameters(),
                                                   lr=env.settings['DQN_lr'], amsgrad=True)

        self.gamma = env.settings['gamma']
        self.tau = env.settings['tau']

        self.steps_taken = 0
        self.eps_start = env.settings['eps_start']
        self.eps_end = env.settings['eps_end']
        self.eps_decay = env.settings['eps_decay']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        for layer in self.layers[:-1]:
            h = self.activation(layer(h))

        return self.layers[-1](h)

    def select_action(self, x, exploration=True):
        """epsilon-greedy action selection. The epsilon threshold has exponential decay"""
        sample = random.random()

        if exploration:
            threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_taken / self.eps_decay)
            self.steps_taken += 1
        else:
            threshold = -1.

        if sample > threshold:
            with torch.no_grad():
                return self.forward(x).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def update(self, batch, target):
        if batch is None:
            return

        non_terminal = tuple(not is_terminal for is_terminal in batch.is_terminal)
        non_final_mask = torch.tensor(non_terminal, dtype=torch.bool)
        non_final_next_states = torch.cat([s for i, s in enumerate(batch.next_state) if non_final_mask[i]])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.forward(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(state_batch.shape[0])
        with torch.no_grad():
            next_state_values[non_final_mask] = target(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()

    def soft_update(self, source: 'DQN') -> None:
        """Used by target network - use this function to receive a soft update from the policy network"""
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
