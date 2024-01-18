import gymnasium as gym
import torch.optim as optim
import torch.nn as nn

import numpy as np

from nndm import NNDM
from dqn import DQN
from buffer import ReplayMemory
from training import Trainer
from a2c import Actor, Critic

# env = gym.make("LunarLander-v2", continuous=True)
# env = gym.make("CartPole-v1")
env = gym.make("Pendulum-v1")
is_discrete = isinstance(env.action_space, gym.spaces.discrete.Discrete)


class CustomTerminateWrapper(gym.Wrapper):
    def __init__(self, env, terminate_fn):
        super().__init__(env)
        self.terminate_fn = terminate_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.terminate_fn(obs):
            return obs, reward, True, False, info

        return obs, reward, terminated, truncated, info


def terminate_pendulum(obs):
    sin_angle = obs[1]
    terminate_angle = np.pi / 15  # 12 degrees

    return sin_angle > terminate_angle


env = CustomTerminateWrapper(env, terminate_pendulum)

buffer = ReplayMemory(10000)
nndm = NNDM(env, (64,))

# hyperparameters settings specified by the user
settings = {'batch_size': 128,
            'num_episodes': 10000,
            'tau': 0.005,
            'a2c_gamma': 0.95,
            'NNDM_optim': optim.Adam(nndm.parameters(), lr=1e-3),
            'NNDM_criterion': nn.MSELoss(),
            }

# depending on if the env is discrete, allow the user to specify DDQN or A2C networks
if is_discrete:
    policy = DQN(env, (64,), gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000)
    target = None

    settings['DQN_optim'] = optim.AdamW(policy.parameters(), lr=1e-3, amsgrad=True)
    settings['DQN_criterion'] = nn.SmoothL1Loss()
else:
    policy = Actor(env, (256, 256))
    target = Critic(env, (256, 256))

    settings['Actor_optim'] = optim.Adam(policy.parameters(), lr=1e-4)
    settings['Critic_optim'] = optim.Adam(target.parameters(), lr=5e-4)
    settings['Critic_criterion'] = nn.MSELoss()


pipeline = Trainer(env, nndm, policy, buffer, settings, target=target)
trained_policy = pipeline.train()
pipeline.play(trained_policy)

# termination_frames = pipeline.evaluate(trained_policy)
