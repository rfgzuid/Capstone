# Capstone
Safe reinforcement learning with stochastic control barrier functions (TI3165TU Capstone project 2023)

Reinforcement learning compatible with OpenAI gymnasium. Currently implemented:
- Cartpole v1
- Lunar lander v2 (discrete & continuous)
- Bipedal walker v3

The library support training of DDQN (discrete actions) and DDPG (continuous actions) agents. The code is based on [SOURCES].

Also a Neural Network Dynamical Model (NNDM) is trained parallel to the agents, using the same Replay Memory. Both the NNDM and Agent parameters for each environment are saved in the 'Models' folder. These can directly be loaded and evaluated using the library.

The CBFs we demonstrate in this notebook are of the form
$$\mathbf{h} = $$
