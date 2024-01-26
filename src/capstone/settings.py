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
from collections.abc import Iterable

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
    h_name: list[str]


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

        # 1 - x{0}^2 / 2.4^2
        # 1 - x{4}^2 / rad(12)^2
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
                    [-1 / 2.4 ** 2, 0],
                    [0, -1 / math.radians(12.) ** 2]
                ]),
                torch.tensor([1., 1.])
            )
        )

        self.h_name = ['X Position [-2.4, 2.4]',
                       'Angle [-12, 12] deg']


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

        # 1 - x{0}^2 / 1^2
        # 1 - x{4}^2/ rad(90)^2
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
                    [-1 / 1. ** 2, 0],
                    [0, -1 / math.radians(90.) ** 2]
                ]),
                torch.tensor([1., 1.])
            )
        )

        self.h_name = ['X Position [-1, 1]',
                       'Angle [-90, 90] deg']


class ContinuousLunarLander(Env):

    def __init__(self, noise=False) -> None:

        env = gym.make("LunarLander-v2", continuous=True)
        self.is_discrete = False

        self.settings = {
            'noise': {
                'x': 10,
                'y': 10,
                'theta': 0.01,
                'v_x': 0.1,
                'v_y': 0.1,
                'v_theta': 0.1
            },

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

        # 1 - x{0}^2 / 1^2
        # 1 - x{4}^2 / rad(90)^2
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
                    [-1 / 1. ** 2, 0],
                    [0, -1 / math.radians(90.) ** 2]
                ]),
                torch.tensor([1., 1.])
            )
        )

        self.h_name = ['X Position [-1, 1]',
                       'Angle [-90, 90] deg']

        if noise:
            self.env = NoisyLanderWrapper(env, self.settings['noise'])
            print(self.env.spec)
        else:
            self.env = env


class NoisyLanderWrapper(gym.Wrapper):
    def __init__(self, env, noise: dict[str, float]):
        super().__init__(env)
        self.noise = noise

    def step(self, action):
        print('a')
        state, reward, terminated, truncated, _ = self.env.step(action)

        pos = np.array(self.env.unwrapped.lander.position)
        pos[0] = pos[0] + np.random.normal(0., self.noise['x'])
        pos[1] = pos[1] + np.random.normal(0., self.noise['y'])
        self.env.unwrapped.lander.position = tuple(pos)

        angle = np.array([self.env.unwrapped.lander.angle])
        angle = angle + np.random.normal(0., self.noise['theta'])
        self.env.unwrapped.lander.angle = angle[0]

        pos_vel = np.array(self.env.unwrapped.lander.linearVelocity)
        pos_vel[0] = pos_vel[0] + np.random.normal(0., self.noise['v_x'])
        pos_vel[1] = pos_vel[1] + np.random.normal(0., self.noise['v_y'])
        self.env.unwrapped.lander.linearVelocity = tuple(pos_vel)

        ang_vel = np.array([self.env.unwrapped.lander.angularVelocity])
        ang_vel = ang_vel + np.random.normal(0., self.noise['v_theta'])
        self.env.unwrapped.lander.angularVelocity = ang_vel[0]

        return state, reward, terminated, truncated, None
