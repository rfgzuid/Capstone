import gymnasium as gym

from collections.abc import Sequence

import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, env: gym.Env, hidden_layers: Sequence[int]):
        super(Actor, self).__init__()

        self.env = env

        self.action_dim = self.env.action_space.shape[0]
        self.observation_size = self.env.observation_space.shape[0]

        # output cat[action_means, action_stds]
        node_counts = [self.observation_size, *hidden_layers, self.action_dim]
        self.layers = nn.ParameterList()

        for idx in range(len(node_counts) - 1):
            self.layers.append(nn.Linear(node_counts[idx], node_counts[idx + 1]))

        self.activation = F.tanh

    def forward(self, x):
        h = x

        for layer in self.layers[:-1]:
            h = self.activation(layer(h))

        return self.layers[-1](h)

    def select_action(self, x):
        # return the action mean
        return self.forward(x)


class Critic(nn.Module):
    def __init__(self, env: gym.Env, hidden_layers: Sequence[int]):
        super(Critic, self).__init__()

        self.env = env

        self.observation_size = self.env.observation_space.shape[0]

        node_counts = [self.observation_size, *hidden_layers, 1]
        self.layers = nn.ParameterList()

        for idx in range(len(node_counts) - 1):
            self.layers.append(nn.Linear(node_counts[idx], node_counts[idx + 1]))

        self.activation = F.tanh

    def forward(self, x):
        h = x

        for layer in self.layers[:-1]:
            h = self.activation(layer(h))

        return self.layers[-1](h)
