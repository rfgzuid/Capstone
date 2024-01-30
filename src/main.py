"""
Library TODO:
- Add docstrings to files, classes & functions, and add extra (line) comments where necessary
- Add type hints (only where it is really necessary for code understanding)
- Add code sources (for both DDQN and DDPG) + articles on which the code is based
  These sources are also going to be cited when we have to justify agent architectures (e.g. cartpole & bipedal walker)
  Also: Frederik gave a lot of RL architecture/training tips that we could briefly mention we used
"""

import torch

from capstone.settings import Cartpole, DiscreteLunarLander, ContinuousLunarLander
from capstone.training import Trainer
from capstone.evaluation import Evaluator

from capstone.barriers import NNDM_H
from capstone.cbf import CBF

from capstone.nndm import NNDM
from capstone.dqn import DQN
from capstone.ddpg import Actor

train = False

env = Cartpole([0.001, 0.001, 0.01, 0.01])
# env = DiscreteLunarLander([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
# env = ContinuousLunarLander([0.01, 0.01, 0.05, 0.05, 0.05, 0.05])


if train:
    pipeline = Trainer(env)
    policy, nndm = pipeline.train()

    torch.save(policy.state_dict(), f'../models/Agents/{type(env).__name__}')
    torch.save(nndm.state_dict(), f'../models/NNDMs/{type(env).__name__}')\

else:
    policy = DQN(env) if env.is_discrete else Actor(env)
    policy_params = torch.load(f'../models/Agents/{type(env).__name__}')
    policy.load_state_dict(policy_params)

    nndm = NNDM(env)
    nndm_params = torch.load(f'../models/NNDMs/{type(env).__name__}')
    nndm.load_state_dict(nndm_params)

    h = NNDM_H(env, nndm)
    cbf = CBF(env, h, policy,
              alpha=[0.9, 0.8],
              delta=[0., 0.],
              action_partitions=8,
              noise_partitions=8)

    evaluator = Evaluator(env, cbf)

    # evaluator.play(policy, True, cbf)
    evaluator.plot(policy, 100)
