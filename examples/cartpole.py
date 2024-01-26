# Import the packages and modules created in source files
import torch

from src.capstone.settings import Cartpole, DiscreteLunarLander, ContinuousLunarLander
from src.capstone.training import Trainer
from src.capstone.evaluation import Evaluator

from src.capstone.barriers import NNDM_H
from src.capstone.cbf import CBF

from src.capstone.nndm import NNDM
from src.capstone.dqn import DQN
from src.capstone.ddpg import Actor

# Initialize the train parameter as True for this example
train = True

# Initialize the process with CBFs
with_CBF = True

# Number of repeated experiments for Monte Carlo Simulation
N = 10

# Create the environment of Cartpole
env = Cartpole()

# main
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

    evaluator.play(policy)

    # Evaluation metrics plot to show
    if with_CBF:
        evaluator.plot(policy, 0.9, 0, N, 500, cbf)
    else:
        evaluator.plot(policy, 0.9, 0, N, 500, None)