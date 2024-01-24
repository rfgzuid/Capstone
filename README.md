# Capstone
Safe reinforcement learning with stochastic control barrier functions (TI3165TU Capstone project 2023)

Reinforcement learning compatible with OpenAI gymnasium. Currently implemented:
- Cartpole v1
- Lunar lander v2 (discrete & continuous)
- Bipedal walker v3

The library support training of DDQN (discrete actions) and DDPG (continuous actions) agents. The code is based on [SOURCES].

Also a Neural Network Dynamical Model (NNDM) is trained parallel to the agents, using the same Replay Memory. Both the NNDM and Agent parameters for each environment are saved in the 'Models' folder. These can directly be loaded and evaluated using the library.

## h functions

For the environments, concave h functions are used so that we convert the CBF constraint from ED to CED [SCBF SOURCE]. The output of the h functions are vectors; each element in the vector corresponds to a safety-critical state element for which we define bounds. The parabolae defined to have maximum 1, and have their roots at the bounds of unsafety. 

Cartpole
- Angle [-12 deg, 12 deg]
- Position [-2.4 m, 2.4 m]

Lunar Lander
- Angle []
- Position []

Bipedal walker
- Hull/head height []

## Citations
citations
