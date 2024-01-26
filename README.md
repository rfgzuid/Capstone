# Capstone
Safe reinforcement learning with stochastic control barrier functions (TI3165TU Capstone project 2023)

Reinforcement learning compatible with OpenAI gymnasium. Currently implemented:
- Cartpole v1
- Lunar lander v2 (discrete & continuous)

The library support training of DDQN (discrete actions) and DDPG (continuous actions) agents. The code is based on [DDPG SOURCE] [DDQN SOURCE].

Also a Neural Network Dynamical Model (NNDM) is trained parallel to the agents, using the same Replay Memory. Both the NNDM and Agent parameters for each environment are saved in the 'Models' folder. These can directly be loaded and evaluated using the library.


## Installation
The use this library it is recommended to install the packages from the requirements.txt file. This can be done by running the following command in the terminal:

```pip install -r requirements.txt```

## Usage
The library can be used to train agents for the different environments. The training can be done by running the following command in the terminal: ```python main.py```. In the main.py file, the environment can be changed by uncommenting the desired environment. If the train bool is set to True either a DDQN or DDPG agent is trained, if it is set to False a simulation of the environment is run.

In the settings.py file all hyperparameters for the environment wrappers and agents can be changed. 

In the ddpg.py and the ddqn.py files, the neural network architectures and updater steps are setup.

## Example
Below is a short example of how to use the library to setup the cartpole environment:

```python
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
if not train:
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

## MOSCOW requirements
1.Must show understanding of RL by implementing model-based RL for a safety critical system.
    
    - Must show understanding of RL by implementing model free RL for a safety critical 
    system.
    
    - Used scripts: main.py, ddpg.py, ddqn.py, settings.py, noise.py

2.Should evaluate the method on 1 standard reinforcement learning benchmark (cartpole).
    
    - Should evaluate discrete CBF on 1 standard reinforcement learning benchmark (cartpole).
    
    - Used scripts: main.py, ddqn.py, settings.py, noise.py, evaluation.py, nndm.py, 

3.Could evaluate the method on 2 more advanced reinforcement learning benchmarks (lunar lander and bipedal walker).
    
    - Could evaluate discrete CBF on 1 advanced reinforcement learning benchmark (lunar lander discrete)
    
    - Could evaluate continuous CBF and SCBF on 1 advanced reinforcement learning benchmark (lunar lander continuous with added stochasticity)
    
    - Used scripts:

## (Stochastic) control barrier functions

Together with the h functions and a controller we can apply a filter on the controller in order to make the controller safer. Safety is defined through the h function. When you are in an environment you can use a control barrier function, discrete or continuous. If you want to account for stochasticity you can use a stochastic control barrier function but keep in mind this is only implemented for continuous controls. Takes a state as input and outputs a control.


## h functions

For the environments, concave h functions are used so that we convert the CBF constraint from ED to CED [SCBF SOURCE]. The output of the h functions are vectors; each element in the vector corresponds to a safety-critical state element for which we define bounds. The parabolas are defined to have maximum 1, and have their roots at the bounds of unsafety. 

Cartpole
- Angle [-12 deg, 12 deg]
- Position [-2.4 m, 2.4 m]

Lunar Lander
- Angle [-30 deg, 30 deg]
- X position [-1, 1]

We recognize that crafting a concave h function this way is very limited, as state elements are only considered independent of one another. More complex functions could be specified in the settings.py file; for example h constraints that consider a combination of position & velocity.

## References
- [DDPG SOURCE], (https://github.com/vy007vikas/PyTorch-ActorCriticRL)
- [DDQN SOURCE], (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [BOUNDPROPAGATION SOURCE], (https://github.com/Zinoex/bound_propagation/blob/main/README.md)


[DDPG SOURCE]: https://github.com/vy007vikas/PyTorch-ActorCriticRL
[DDQN SOURCE]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
[BOUNDPROPAGATION SOURCE]: https://github.com/Zinoex/bound_propagation/blob/main/README.md
