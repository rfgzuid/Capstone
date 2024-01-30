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

import math
from .noise import CartPoleNoise, LunarLanderNoise

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
    settings: dict[str, Any]
    h_function: nn.Sequential

    h_ids: list[float]
    std: list[float]


class Cartpole(Env):

    def __init__(self, noise: list[float]) -> None:
        if len(noise) != 4:
            raise ValueError(f'4 noise values expected, got {len(noise)}')

        env = gym.make("CartPole-v1")
        self.is_discrete = True

        self.settings = {
            'noise': {
                'x': noise[0],
                'theta': noise[1],
                'v_x': noise[2],
                'v_theta': noise[3]
            },

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

        # 1 - x{0}^2 / 1^2
        # 1 - x{2}^2 / rad(10)^2
        self.h_function = nn.Sequential(
            FixedLinear(
                torch.tensor([
                    [1., 0, 0, 0],
                    [0, 0, 1., 0]
                ]),
                torch.tensor([0., 0.])
            ),
            Pow(2),
            FixedLinear(
                torch.tensor([
                    [-1 / 1. ** 2, 0],
                    [0, -1 / math.radians(10.) ** 2]
                ]),
                torch.tensor([1., 1.])
            )
        )

        self.h_ids = [0, 2]
        self.std = [noise[i] for i in self.h_ids]
        self.env = CartPoleNoise(env, self.settings['noise'])


class DiscreteLunarLander(Env):

    def __init__(self, noise: list[float]) -> None:
        if len(noise) != 6:
            raise ValueError(f'6 noise values expected, got {len(noise)}')

        env = gym.make("LunarLander-v2")
        self.is_discrete = True

        self.settings = {
            'noise': {
                'x': noise[0],
                'y': noise[1],
                'theta': noise[2],
                'v_x': noise[3],
                'v_y': noise[4],
                'v_theta': noise[5]
            },

            'replay_size': 10_000,
            'batch_size': 128,
            'num_episodes': 1000,
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

        # 1 - x{0}^2 / 0.2^2
        # 1 - x{4}^2/ rad(20)^2
        self.h_function = nn.Sequential(
            FixedLinear(
                torch.tensor([
                    [1., 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1., 0, 0, 0]
                ]),
                torch.tensor([0., 0.])
            ),
            Pow(2),
            FixedLinear(
                torch.tensor([
                    [-1 / 0.2 ** 2, 0],
                    [0, -1 / math.radians(20.) ** 2]
                ]),
                torch.tensor([1., 1.])
            )
        )

        self.h_ids = [0, 4]
        self.std = [noise[i] for i in self.h_ids]
        self.env = LunarLanderNoise(env, self.settings['noise'])


class ContinuousLunarLander(Env):

    def __init__(self, noise: list[float]) -> None:
        if len(noise) != 6:
            raise ValueError(f'6 noise values expected, got {len(noise)}')

        env = gym.make("LunarLander-v2", continuous=True)
        self.is_discrete = False

        self.settings = {
            'noise': {
                'x': noise[0],
                'y': noise[1],
                'theta': noise[2],
                'v_x': noise[3],
                'v_y': noise[4],
                'v_theta': noise[5]
            },

            'replay_size': 1_000_000,
            'batch_size': 128,
            'num_episodes': 3000,
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
            'Actor_optim': optim.AdamW,
            'Actor_lr': 1e-4,
            'Action_bound': 1.,  # action space is bounded to [-1, 1] - see gymnasium docs

            'Critic_layers': {'s': (256, 128), 'a': (128,), 'concat': (128,)},
            'Critic_activation': F.relu,
            'Critic_criterion': nn.SmoothL1Loss,
            'Critic_optim': optim.AdamW,
            'Critic_lr': 1e-3,

            'OU_mu': 0,
            'OU_theta': 0.15,
            'OU_sigma': 0.2
        }

        # 1 - x{0}^2 / 0.2^2
        # 1 - x{4}^2 / rad(20)^2
        self.h_function = nn.Sequential(
            FixedLinear(
                torch.tensor([
                    [1., 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1., 0, 0, 0]
                ]),
                torch.tensor([0., 0.])
            ),
            Pow(2),
            FixedLinear(
                torch.tensor([
                    [-1 / 0.2 ** 2, 0],
                    [0, -1 / math.radians(20.) ** 2]
                ]),
                torch.tensor([1., 1.])
            )
        )

        self.h_ids = [0, 4]
        self.std = [noise[i] for i in self.h_ids]
        self.env = LunarLanderNoise(env, self.settings['noise'])
