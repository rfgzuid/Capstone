import gymnasium as gym

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNDM(nn.Module):
    """Neural Network Dynamical Model
    input: tensor of size (M, N_state + N_action) of M samples that each contain a state + action taken
    output: model tries to predict the following state, output tensor (M, N_state)

    The model is predicting the change in state (delta_state) because we found this improves generalization"""

    def __init__(self, env: gym.Env, hidden_layers: Sequence[int], noise=0.01) -> None:
        super(NNDM, self).__init__()

        self.env = env

        self.action_size = 1 if env.action_space.shape == tuple() else env.action_space.shape[0]
        self.observation_size = env.observation_space.shape[0]
        print(self.observation_size)

        node_counts = [self.observation_size + self.action_size, *hidden_layers, self.observation_size]
        self.layers = nn.ParameterList()

        for idx in range(len(node_counts) - 1):
            self.layers.append(nn.Linear(node_counts[idx], node_counts[idx+1]))

        self.activation = F.tanh

        self.std = noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        for layer in self.layers[:-1]:
            h = self.activation(layer(h))

        return self.layers[-1](h) + x[:, :self.observation_size] + \
            torch.normal(mean=0., std=self.std, size=(self.observation_size,))

    def train(self, batch, optimizer, criterion):
        if batch is None:
            return

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)

        x_train = torch.cat([state_batch, action_batch], dim=1)
        y_train = torch.cat(batch.next_state)

        optimizer.zero_grad()

        y_pred = self.forward(x_train)

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        return loss.item()
