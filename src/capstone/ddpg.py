import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from .settings import Env


def fanin_init(size, eps=None):
    fanin = size[0]
    v = eps if eps is not None else 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class OUNoise:
    """
    Implement Ornstein-Uhlenbeck noise for the action to encourage exploration
    """

    def __init__(self, env: Env) -> None:
        self.action_dim = env.env.action_space.shape[0]

        self.mu = env.settings['OU_mu']
        self.theta = env.settings['OU_theta']
        self.sigma = env.settings['OU_sigma']

        self.noise = torch.ones(self.action_dim) * self.mu

    def reset(self) -> None:
        self.noise = torch.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.noise) + \
             self.sigma * torch.normal(0, 1, (self.action_dim,))
        self.noise = self.noise + dx

        return self.noise


class Actor(nn.Module):
    def __init__(self, env: Env):
        super(Actor, self).__init__()

        self.env = env.env

        self.action_size = self.env.action_space.shape[0]
        self.action_bound = env.settings['Action_bound']
        self.observation_size = self.env.observation_space.shape[0]

        hidden_layers = env.settings['Actor_layers']
        node_counts = [self.observation_size, *hidden_layers, self.action_size]

        self.layers = nn.ParameterList()

        for idx in range(len(node_counts) - 1):
            self.layers.append(nn.Linear(node_counts[idx], node_counts[idx + 1]))
            self.layers[idx].weight.data = fanin_init(self.layers[idx].weight.data.shape)
        self.layers[-1].weight.data = fanin_init(self.layers[-1].weight.data.shape, eps=0.003)

        self.activation = env.settings['Actor_activation']
        self.optimizer = env.settings['Actor_optim'](self.parameters(), lr=env.settings['Actor_lr'])

        self.tau = env.settings['tau']

        self.noise = OUNoise(env)

    def forward(self, x):
        h = x

        for layer in self.layers[:-1]:
            h = self.activation(layer(h))

        # output an action that complies with the env action bound
        action = F.tanh(self.layers[-1](h)) * self.action_bound

        return action

    def select_action(self, x, exploration=True) -> torch.tensor:
        if exploration:
            action = self.forward(x) + self.noise.sample() * self.action_bound
        else:
            action = self.forward(x)
        return action

    def update(self, batch, critic: 'Critic'):
        if batch is None:
            return

        state_batch = torch.cat(batch.state)

        action_values = self.forward(state_batch)
        q_values = critic(state_batch, action_values)

        loss = -torch.sum(q_values)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()

    def soft_update(self, source: 'Actor') -> None:
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


class Critic(nn.Module):
    """
    Input: state, action
    Output: scalar Q value
    Note the structure of the network; s and a have separate hidden layers, which are then concatenated and processed
    """

    def __init__(self, env: Env):
        super(Critic, self).__init__()

        self.env = env.env

        self.action_size = self.env.action_space.shape[0]
        self.observation_size = self.env.observation_space.shape[0]

        hidden_layers = env.settings['Critic_layers']
        state_nodes = [self.observation_size, *hidden_layers['s']]
        action_nodes = [self.action_size, *hidden_layers['a']]
        concat_nodes = [hidden_layers['s'][-1] + hidden_layers['a'][-1], *hidden_layers['concat'], 1]

        self.s_layers = nn.ParameterList()
        self.a_layers = nn.ParameterList()
        self.concat_layers = nn.ParameterList()

        for idx in range(len(state_nodes) - 1):
            self.s_layers.append(nn.Linear(state_nodes[idx], state_nodes[idx + 1]))
            self.s_layers[idx].weight.data = fanin_init(self.s_layers[idx].weight.data.size())

        for idx in range(len(action_nodes) - 1):
            self.a_layers.append(nn.Linear(action_nodes[idx], action_nodes[idx + 1]))
            self.a_layers[idx].weight.data = fanin_init(self.a_layers[idx].weight.data.size())

        for idx in range(len(concat_nodes) - 1):
            self.concat_layers.append(nn.Linear(concat_nodes[idx], concat_nodes[idx + 1]))
            self.concat_layers[idx].weight.data = fanin_init(self.concat_layers[idx].weight.data.size())

        self.concat_layers[-1].weight.data = fanin_init(self.concat_layers[-1].weight.data.shape, eps=0.003)

        self.activation = env.settings['Critic_activation']

        self.criterion = env.settings['Critic_criterion']()
        self.optimizer = env.settings['Critic_optim'](self.parameters(), lr=env.settings['Critic_lr'])

        self.tau = env.settings['tau']
        self.gamma = env.settings['gamma']

    def forward(self, state, action):
        h_s = state
        h_a = action

        for layer in self.s_layers:
            h_s = self.activation(layer(h_s))
        for layer in self.a_layers:
            h_a = self.activation(layer(h_a))

        h = torch.concat((h_s, h_a), dim=1)

        for layer in self.concat_layers[:-1]:
            h = self.activation(layer(h))

        return self.concat_layers[-1](h)

    def update(self, batch, critic_target: 'Critic', actor_target: 'Actor'):
        if batch is None:
            return

        non_terminal = tuple(not is_terminal for is_terminal in batch.is_terminal)
        non_final_mask = torch.tensor(non_terminal, dtype=torch.bool)
        non_final_next_states = torch.cat([s for i, s in enumerate(batch.next_state) if non_final_mask[i]])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.forward(state_batch, action_batch)

        next_state_values = torch.zeros(state_batch.shape[0])
        with torch.no_grad():
            actions = actor_target(non_final_next_states)
            next_state_values[non_final_mask] = critic_target(non_final_next_states, actions).squeeze()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()

    def soft_update(self, source: 'Critic') -> None:
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
