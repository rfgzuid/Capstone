"""
This script stores only the settings that are used for following three OpenAI gymnasium environments:
- Discrete Cartpole
- Continuous Lunar Lander
- Continuous Bipedal Walker
The scripts only allow for action_space of type Discrete() or Box()
"""

import gymnasium as gym
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from typing import Any

import torch
import numpy as np

import math
from bound_propagation.polynomial import Pow
from bound_propagation.linear import FixedLinear


class Env(ABC):
    """
    Abstract Base Class. Inheriting classes contain the following attributes:
    - env: the gymnasium environment that is created
    - is_discrete: is the action space discrete or continuous? Used to choose either the DDQN or DDPG algorithm
    - settings: contain all hyperparameters used to train the RL agent
    """

    env: gym.Env
    is_discrete: bool
    settings: dict[Any]
    h_function: nn.Sequential


class Cartpole(Env):

    def __init__(self) -> None:

        self.env = gym.make("CartPole-v1")
        self.is_discrete = True

        self.settings = {
            'replay_size': 10_000,
            'batch_size': 128,
            'num_episodes': 200,
            'max_frames': 500,

            'gamma': 0.99,
            'tau': 0.005,

            'NNDM_layers': (64,),
            'NNDM_activation': nn.Tanh,
            'NNDM_criterion': nn.MSELoss,
            'NNDM_optim': optim.Adam,
            'NNDM_lr': 1e-3,

            'DQN_layers': (64,),
            'DQN_activation': F.tanh,
            'DQN_criterion': nn.SmoothL1Loss,
            'DQN_optim': optim.AdamW,
            'DQN_lr': 1e-3,

            'eps_start': 0.9,
            'eps_end': 0.05,
            'eps_decay': 1000
        }

        self.h_function = nn.Sequential(
            Pow(2),
            FixedLinear(
                torch.tensor([
                    [-1 / 2.4 ** 2, 0, 0, 0],
                    [0, 0, -1 / math.radians(12.) ** 2, 0]
                ]),
                torch.tensor([1., 1.])
            )
        )


class DiscreteLunarLander(Env):

    def __init__(self) -> None:

        self.env = gym.make("LunarLander-v2")
        self.is_discrete = True

        self.settings = {
            'replay_size': 10_000,
            'batch_size': 128,
            'num_episodes': 200,
            'max_frames': 1000,

            'gamma': 0.99,
            'tau': 0.005,

            'NNDM_layers': (64,),
            'NNDM_activation': nn.Tanh,
            'NNDM_criterion': nn.MSELoss,
            'NNDM_optim': optim.Adam,
            'NNDM_lr': 1e-3,

            'DQN_layers': (64,),
            'DQN_activation': F.tanh,
            'DQN_criterion': nn.SmoothL1Loss,
            'DQN_optim': optim.AdamW,
            'DQN_lr': 1e-3,

            'eps_start': 0.9,
            'eps_end': 0.05,
            'eps_decay': 1000
        }

        self.h_function = nn.Sequential(
            Pow(2),
            FixedLinear(
                torch.tensor([
                    [-1 / 1. ** 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1/math.radians(90.), 0, 0, 0]
                ]),
                torch.tensor([1., 1.])
            )
        )


class ContinuousLunarLander(Env):

    def __init__(self) -> None:

        self.env = gym.make("LunarLander-v2", continuous=True)
        self.is_discrete = False

        self.settings = {
            'replay_size': 1_000_000,
            'batch_size': 128,
            'num_episodes': 200,
            'max_frames': 1000,  # so that the lander prioritizes landing quick

            'gamma': 0.99,
            'tau': 0.001,

            'NNDM_layers': (64,),
            'NNDM_activation': nn.Tanh,
            'NNDM_criterion': nn.MSELoss,
            'NNDM_optim': optim.Adam,
            'NNDM_lr': 1e-3,

            'Actor_layers': (256, 128, 64),
            'Actor_activation': F.relu,
            'Actor_optim': optim.Adam,
            'Actor_lr': 1e-4,
            'Action_bound': 1.,  # action space is bounded to [-1, 1] - see gymnasium docs

            'Critic_layers': {'s': (256, 128), 'a': (128,), 'concat': (128,)},
            'Critic_activation': F.relu,
            'Critic_criterion': nn.SmoothL1Loss,
            'Critic_optim': optim.Adam,
            'Critic_lr': 1e-3,

            'OU_mu': 0,
            'OU_theta': 0.15,
            'OU_sigma': 0.2
        }

        self.h_function = nn.Sequential(
            Pow(2),
            FixedLinear(
                torch.tensor([
                    [-1 / 1. ** 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1 / math.radians(90.), 0, 0, 0]
                ]),
                torch.tensor([1., 1.])
            )
        )


class BipedalHull(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Adjust the shape of the observation space to include the additional state
        obs_space = env.observation_space
        low = np.append(obs_space.low.flatten(), -np.inf)
        high = np.append(obs_space.high.flatten(), np.inf)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        hull_position = np.array([self.env.unwrapped.hull.position[0]])
        # Flatten the state and add the x-coordinate of the hull's position
        extended_state = np.concatenate((state.flatten(), hull_position))
        return extended_state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        hull_position = np.array([self.env.unwrapped.hull.position[0]])
        # Flatten the state and add the x-coordinate of the hull's position
        extended_state = np.concatenate((state[0].flatten(), hull_position))
        return extended_state, 'info'


class BipedalWalker(Env):

    def __init__(self) -> None:

        env = gym.make("BipedalWalker-v3")
        self.env = BipedalHull(env)
        self.is_discrete = False

        self.settings = {
            'replay_size': 1_000_000,
            'batch_size': 128,
            'num_episodes': 1000,
            'max_frames': 1000,

            'gamma': 0.99,
            'tau': 0.001,

            'NNDM_layers': (128, 128),
            'NNDM_activation': nn.Tanh,
            'NNDM_criterion': nn.MSELoss,
            'NNDM_optim': optim.Adam,
            'NNDM_lr': 1e-3,

            'Actor_layers': (256, 128, 64),
            'Actor_activation': F.relu,
            'Actor_optim': optim.Adam,
            'Actor_lr': 1e-4,
            'Action_bound': 1.,

            'Critic_layers': {'s': (256, 128), 'a': (128,), 'concat': (128,)},
            'Critic_activation': F.relu,
            'Critic_criterion': nn.SmoothL1Loss,
            'Critic_optim': optim.Adam,
            'Critic_lr': 1e-3,

            'OU_mu': 0,
            'OU_theta': 0.15,
            'OU_sigma': 0.2
        }

        self.h_function = nn.Sequential(
            Pow(2),
            FixedLinear(
                torch.tensor([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/7.**2],
                ]),
                torch.tensor([1.])
            )
        )
