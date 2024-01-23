"""
Library TODO:
- More metric monitoring/plotting in the training pipeline
- Implement the (S)CBF code for both discrete and continuous action environments
- Implement the evaluation metrics for CBF in evaluations.py

- Add docstrings to files, classes & functions, and add extra (line) comments where necessary
- Add type hints (is it needed? Most variables originate from the env_settings file)
- Add code sources (for both DDQN and DDPG) + articles on which the code is based
  These sources are also going to be cited when we have to justify agent architectures (e.g. cartpole & bipedal walker)
  Also: Frederik gave a lot of RL architecture/training tips that we could briefly mention we used
- Correctly format the library on github
(requirements.txt, readme.txt, .gitignore to save files we won't update - like model weights, __init__.py script)
"""

import torch

from settings import Cartpole
from training import Trainer
from evaluation import Evaluator

from src.capstone.ddpg import Actor
from dqn import DQN

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

# termination_frames = evaluator.mc_simulate(policy)
# print(termination_frames)
