# Capstone
Safe reinforcement learning with stochastic control barrier functions (TI3165TU Capstone project 2023). Library that implements SCBF in to Reinforcement Learning, based on [SCBF SOURCE].

Reinforcement learning compatible with OpenAI gymnasium. Currently implemented:
- Cartpole v1
- Lunar lander v2 (discrete & continuous)

The library support training of DDQN (discrete actions) and DDPG (continuous actions) agents. The code is based on [DDPG SOURCE] and [DQN SOURCE].

Also a Neural Network Dynamical Model (NNDM) is trained parallel to the agents, using the same Replay Memory. Both the NNDM and Agent parameters for each environment are saved in the `Models/` folder. These can directly be loaded and evaluated using the library.

## Project description
In this project we implemented both discrete and continuous Control Barrier Functions (CBFs) on top of a Reinforcement learning agent in order to have some theoretical probability of being unsafe ($P_u$) in a certain time window of length K, this probability is tied to the initial location of the agent, and the environment it is in. The model is trained, and evaluated and plots of this theoretical and empirical probability are plotted.

## Installation
The use this library it is recommended to install the packages from the `requirements.txt` file. This can be done by running the following command in the terminal:

```pip install -r requirements.txt```

## Usage
The library can be used to train agents for the different environments. The training can be done by running the following command in the terminal: ```python main.py```. In the main.py file, the environment can be changed by uncommenting the desired environment. If the train bool variable is set to True either a DDQN or DDPG agent is trained, if it is set to False a simulation of the environment is run.

- `settings.py`: all hyperparameters for the environment wrappers and agents can be changed.

- `ddpg.py, ddqn.py`: the neural network architectures and updater steps for the agents are setup.

- `evaluation.py`: the evaluation metrics are calculated and plotted.

- `nndm.py`: the neural newtwork dynamical model (NNDM) is setup and trained. This neural network is used to predict the next state given the current state and action.

- `barriers.py`: the h functions are setup, these functions are used for the (S)CBFs.

- `cbf.py`: the CBFs(control barrier functions) are setup.

- `noise.py`: the noise functions are setup and custom wrappers for the environments are setup. We do this to make the environment stochastic in order to use the SCBFs.


## Example
Below is a short example of how to use the library to setup and train the cartpole environment:

```python
# Import the packages and modules created in source files
import torch

from capstone.settings import Cartpole, DiscreteLunarLander, ContinuousLunarLander
from capstone.training import Trainer
from capstone.evaluation import Evaluator

from capstone.cbf import CBF

from capstone.nndm import NNDM
from capstone.dqn import DQN
from capstone.ddpg import Actor

env = Cartpole([0.001, 0.001, 0.01, 0.01])

pipeline = Trainer(env)
policy, nndm = pipeline.train()

torch.save(policy.state_dict(), f'../models/Agents/{type(env).__name__}')
torch.save(nndm.state_dict(), f'../models/NNDMs/{type(env).__name__}')\
```

Example scripts using Control Barrier Functions are shown in the `examples/` folder.

## MOSCOW requirements
We managed to implement all the MoSCoW requirements. The deliverables for each requirement can be found in the following files:

1. **Must** show understanding of RL by implementing model free RL for a safety critical system.
    - Used scripts: ddpg.py, ddqn.py, settings.py, training.py
2. **Should** implement and train a Neural Network Dynamical Model for model identification.
    - Used scripts: main.py, nndm.py, buffer.py, training.py
3. **Should** implement Stochastic Control Barrier Functions (SCBF) to ensure safe reinforcement learning. 
    - Used scripts: barriers.py, settings.py, cbf.py, noise.py
4. **Should/could** evaluate the effectiveness of the RL system (with CBFs) by counting the number of safety violations and plotting the reward over episodes. 
    1. **Should** choose an appropriate evaluation metric. 
        - Used scripts: evaluation.py
    2. **Should** evaluate discrete CBF on 1 standard reinforcement learning benchmark (cartpole).
        - Used scripts: barriers.py, cbf.py, ddqn.py, settings.py, evaluation.py, nndm.py
    3. **Could** evaluate discrete CBF on 1 advanced reinforcement learning benchmark (lunar lander discrete)
        - Used scripts: barriers.py, cbf.py, ddqn.py, settings.py, evaluation.py, nndm.py
    4. **Could** evaluate continuous CBF and SCBF on 1 advanced reinforcement learning benchmark (lunar lander continuous with added stochasticity)
        - Used scripts: barriers.py, cbf.py, noise.py, probaility.py, ddpg.py, settings.py, evaluation.py, nndm.py
7. **Could** write a one pager that extends the theory of SCBF to NNDMs via linear bound propagation. 
    - Used scripts: onepager.tex

## (Stochastic) control barrier functions

Together with the h functions and a controller we can apply a filter on the controller in order to make the controller safer. Safety is defined through the h function. When you are in an environment you can use a control barrier function, discrete or continuous. If you want to account for stochasticity you can use a stochastic control barrier function but keep in mind this is only implemented for continuous controls. Takes a state as input and outputs a control.


## h functions

For the environments, concave h functions are used so that we convert the CBF constraint from ED to CED ([SCBF SOURCE]). The output of the h functions are vectors; each element in the vector corresponds to a safety-critical state element for which we define bounds. The parabolas are defined to have maximum 1, and have their roots at the bounds of unsafety. 

Cartpole
- Angle [-7 deg, 7 deg]
- Position [-1 m, 1 m]

Lunar Lander
- Angle [-20 deg, 20 deg]
- X position [-0.5, 0.5]

We recognize that crafting a concave h function this way is very limited, as state elements are only considered independent of one another. More complex functions could be specified in the settings.py file; for example h constraints that consider a combination of position & velocity.

Apart from the h function that influences how succesful a CBF can be, careful consideration should be placed in choosing the alpha hyperparameter that determines the tightness of the CBF constraint.
 
## References
- [SCBF SOURCE], (https://arxiv.org/abs/2302.07469)
- [DDPG SOURCE], (https://github.com/vy007vikas/PyTorch-ActorCriticRL)
- [DQN SOURCE], (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [BOUNDPROPAGATION SOURCE], (https://github.com/Zinoex/bound_propagation/blob/main/README.md)

[SCBF SOURCE]: https://arxiv.org/abs/2302.07469
[DDPG SOURCE]: https://github.com/vy007vikas/PyTorch-ActorCriticRL
[DQN SOURCE]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
[BOUNDPROPAGATION SOURCE]: https://github.com/Zinoex/bound_propagation/blob/main/README.md
