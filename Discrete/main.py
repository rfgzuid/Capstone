import gymnasium as gym
import torch.optim as optim
import torch.nn as nn

from Architectures import NNDM
from DQN import DQN
from Buffer import ReplayMemory
from Training import Trainer

env = gym.make("CartPole-v1")
# env = gym.make("Acrobot-v1")
# env = gym.make("LunarLander-v2")
buffer = ReplayMemory(10000)

nndm = NNDM(env, (64,))
policy = DQN(env, (64,), gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000)

settings = {'batch_size': 128,
            'num_episodes': 100,
            'tau': 0.005,
            'DQN_optim': optim.AdamW(policy.parameters(), lr=1e-3, amsgrad=True),
            'DQN_criterion': nn.SmoothL1Loss(),
            'NNDM_optim': optim.Adam(nndm.parameters(), lr=1e-3),
            'NNDM_criterion': nn.MSELoss()
            }

pipeline = Trainer(env, nndm, policy, buffer, settings)
pipeline.train()
pipeline.play()
