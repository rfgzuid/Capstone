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

# env = Cartpole()
# env = DiscreteLunarLander()
env = ContinuousLunarLander()


if train:
    pipeline = Trainer(env)
    policy, nndm = pipeline.train()

    torch.save(policy.state_dict(), f'../models/Agents/{type(env).__name__}')
    torch.save(nndm.state_dict(), f'../models/NNDMs/{type(env).__name__}')
else:
    policy = DQN(env) if env.is_discrete else Actor(env)
    policy_params = torch.load(f'../models/Agents/{type(env).__name__}')
    policy.load_state_dict(policy_params)

    nndm = NNDM(env)
    nndm_params = torch.load(f'../models/NNDMs/{type(env).__name__}')
    nndm.load_state_dict(nndm_params)

    evaluator = Evaluator(env)

    h = NNDM_H(env, nndm)
    cbf = CBF(env, h, policy, alpha=0.9)

    # evaluator.play(policy)
    evaluator.plot(policy, 0.9, 0.099, 10)
    evaluator.plot(policy, 0.9, 0.099, 10, cbf=cbf)
