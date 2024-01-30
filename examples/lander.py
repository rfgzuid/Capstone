import torch

from src.capstone.settings import DiscreteLunarLander
from src.capstone.evaluation import Evaluator

from src.capstone.barriers import NNDM_H
from src.capstone.cbf import CBF

from src.capstone.nndm import NNDM
from src.capstone.dqn import DQN


env = DiscreteLunarLander([0.015, 0.015, 0.031415, 0.05, 0.05, 0.05])

policy = DQN(env)
policy_params = torch.load(f'../models/Agents/{type(env).__name__}')
policy.load_state_dict(policy_params)

nndm = NNDM(env)
nndm_params = torch.load(f'../models/NNDMs/{type(env).__name__}')
nndm.load_state_dict(nndm_params)

h = NNDM_H(env, nndm)
cbf = CBF(env, nndm, policy,
          alpha=[0.9, 0.8],
          delta=[0., 0.],
          no_action_partitions=64,
          no_noise_partitions=4,
          stochastic=False)

evaluator = Evaluator(env, cbf)

evaluator.play(policy, cbf=True, gif=True)
evaluator.plot(policy, 100)
