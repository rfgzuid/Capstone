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


## MOSCOW requirements
We managed to implement all the MoSCoW requirements. The deliverables for each requirement can be found in the following files:

1. Must show understanding of RL by implementing model free RL for a safety critical system.
    - Used scripts: ddpg.py, ddqn.py, settings.py, training.py
2. Should implement and train a Neural Network Dynamical Model for model identification.
    - Used scripts: main.py, nndm.py, buffer.py, training.py
3. Should implement Stochastic Control Barrier Functions (SCBF) to ensure safe reinforcement learning. 
    - Used scripts: barriers.py, settings.py, cbf.py, noise.py
4. Should/could evaluate the effectiveness of the RL system (with CBFs) by counting the number of safety violations and plotting the reward over episodes. 
    - Used scripts: evaluation.py
    1. Should choose an appropriate evaluation metric. 
        - Used scripts: evaluation.py
    2. Should evaluate discrete CBF on 1 standard reinforcement learning benchmark (cartpole).
        - Used scripts: barriers.py, cbf.py, ddqn.py, settings.py, evaluation.py, nndm.py
    3. Could evaluate discrete CBF on 1 advanced reinforcement learning benchmark (lunar lander discrete)
        - Used scripts: barriers.py, cbf.py, ddqn.py, settings.py, evaluation.py, nndm.py
    4. Could evaluate continuous CBF and SCBF on 1 advanced reinforcement learning benchmark (lunar lander continuous with added stochasticity)
        - Used scripts: barriers.py, cbf.py, noise.py, probaility.py, ddpg.py, settings.py, evaluation.py, nndm.py
7. Could write a one pager that extends the theory of SCBF to NNDMs via linear bound propagation. 
    - Used scripts: onepager.tex

## (Stochastic) control barrier functions

Together with the h functions and a controller we can apply a filter on the controller in order to make the controller safer. Safety is defined through the h function. When you are in an environment you can use a control barrier function, discrete or continuous. If you want to account for stochasticity you can use a stochastic control barrier function but keep in mind this is only implemented for continuous controls. Takes a state as input and outputs a control.


## h functions

For the environments, concave h functions are used so that we convert the CBF constraint from ED to CED [SCBF SOURCE]. The output of the h functions are vectors; each element in the vector corresponds to a safety-critical state element for which we define bounds. The parabolas are defined to have maximum 1, and have their roots at the bounds of unsafety. 

Cartpole
- Angle [-12 deg, 12 deg]
- Position [-2.4 m, 2.4 m]

Lunar Lander
- Angle []
- X position []

We recognize that crafting a concave h function this way is very limited, as state elements are only considered independent of one another. More complex functions could be specified in the settings.py file; for example h constraints that consider a combination of position & velocity.

## References
- 


[DDPG SOURCE]: https://github.com/vy007vikas/PyTorch-ActorCriticRL
[DDQN SOURCE]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
