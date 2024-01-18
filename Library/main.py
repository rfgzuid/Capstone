import gymnasium as gym
import torch.optim as optim
import torch.nn as nn

from Architectures import NNDM
from DQN import DQN
from Buffer import ReplayMemory
from Training import Trainer
from a2c import Actor, Critic

# env = gym.make("LunarLander-v2", continuous=True)
# env = gym.make("CartPole-v1")
env = gym.make("Pendulum-v1")
is_discrete = isinstance(env.action_space, gym.spaces.discrete.Discrete)

buffer = ReplayMemory(10000)
nndm = NNDM(env, (64,))

# hyperparameters settings specified by the user
settings = {'batch_size': 128,
            'num_episodes': 200,
            'tau': 0.005,
            'a2c_gamma': 0.99,
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
    policy = Actor(env, (64, 64))
    target = Critic(env, (64, 64))

    settings['Actor_optim'] = optim.AdamW(policy.parameters(), lr=1e-3, amsgrad=True)
    settings['Critic_optim'] = optim.AdamW(target.parameters(), lr=1e-3, amsgrad=True)
    settings['Critic_criterion'] = nn.MSELoss()


pipeline = Trainer(env, nndm, policy, buffer, settings, target=target)
trained_policy = pipeline.train()
pipeline.play(trained_policy)

# termination_frames = pipeline.evaluate(trained_policy)
