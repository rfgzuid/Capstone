import torch

from capstone.settings import Cartpole, DiscreteLunarLander, ContinuousLunarLander
from capstone.training import Trainer
from capstone.evaluation import Evaluator

from capstone.cbf import CBF

from capstone.nndm import NNDM
from capstone.dqn import DQN
from capstone.ddpg import Actor

# env = Cartpole([0.001, 0.001, 0.01, 0.01])
# env = DiscreteLunarLander([0.015, 0.015, 0.031415, 0.05, 0.05, 0.05])
env = ContinuousLunarLander([0.015, 0.015, 0.031415, 0.05, 0.05, 0.05])


def train(env):
    pipeline = Trainer(env)
    policy, nndm = pipeline.train()

    torch.save(policy.state_dict(), f'../models/Agents/{type(env).__name__}')
    torch.save(nndm.state_dict(), f'../models/NNDMs/{type(env).__name__}')\

def evaluate(env):
    policy = DQN(env) if env.is_discrete else Actor(env)
    policy_params = torch.load(f'../models/Agents/{type(env).__name__}')
    policy.load_state_dict(policy_params)

    nndm = NNDM(env)
    nndm_params = torch.load(f'../models/NNDMs/{type(env).__name__}')
    nndm.load_state_dict(nndm_params)

    cbf = CBF(env, nndm, policy,
              alpha=[0.9, 0.9],
              delta=[0., 0.],
              no_action_partitions=2,
              no_noise_partitions=2,
              stochastic=True)

    evaluator = Evaluator(env, cbf)

    # evaluator.play(policy, cbf=True, gif=False)
    evaluator.plot(policy, n=1000)


if __name__ == '__main__':
    print("\n#####################################\n"
          "#  TI3165TU Capstone project, 2024  #\n"
          "#####################################\n")

    # train(env)
    evaluate(env)
