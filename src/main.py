"""
Library TODO:
- More metric monitoring/plotting in the training pipeline
- Implement the (S)CBF code for both discrete and continuous action environments
- Implement the evaluation metrics for CBF in evaluations.py

- Add docstrings to files, classes & functions, and add extra (line) comments where necessary
- Add type hints (only where it is really necessary for code understanding)
- Add code sources (for both DDQN and DDPG) + articles on which the code is based
  These sources are also going to be cited when we have to justify agent architectures (e.g. cartpole & bipedal walker)
  Also: Frederik gave a lot of RL architecture/training tips that we could briefly mention we used
"""

import torch

from capstone.settings import Cartpole, DiscreteLunarLander, ContinuousLunarLander, BipedalWalker
from capstone.training import Trainer
from capstone.evaluation import Evaluator

from capstone.dqn import DQN
from capstone.ddpg import Actor

train = True

env = Cartpole()
# env = DiscreteLunarLander()
# env = ContinuousLunarLander()
# env = BipedalWalker()

if train:
    pipeline = Trainer(env)
    policy, nndm = pipeline.train()

    torch.save(policy.state_dict(), f'../Agents/{type(env).__name__}')
    torch.save(nndm.state_dict(), f'../NNDMs/{type(env).__name__}')
else:
    policy = DQN(env) if env.is_discrete else Actor(env)
    trained_params = torch.load(f'../Agents/{type(env).__name__}')
    policy.load_state_dict(trained_params)

evaluator = Evaluator(env)
evaluator.play(policy)
