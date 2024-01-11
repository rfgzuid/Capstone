"""DATA GENERATOR
generate dynamics (s[t], a[t], s[t+1]) data with a policy that is randomly choosing actions
save the data as a .npy file for the model.py script to train the NNDM

To do: normalize observations using a custom gym wrapper"""

import gymnasium as gym
from tqdm import tqdm
import numpy as np

# render the environment using pygame (note: not compatible with notebooks)
render_simulation = False  # disable this for super fast simulations!
env = gym.make("CartPole-v1", render_mode='human' if render_simulation else 'none')

observation, _ = env.reset(seed=42)  # initialize the environment with a set seed

N = 100_000  # total amount of samples we want to collect
M = 100  # maximum amount of iterations after which to reset the simulation

prev_state = observation  # save the initial state as the current s[t]
data: list[tuple[float, ...]] = []  # save the (s[t], a[t], s[t+1]) tuples for each iteration

for i in tqdm(range(N)):  # tqdm shows a progress bar in the terminal
    action = env.action_space.sample()  # select a random action
    state, reward, terminated, truncated, _ = env.step(action)  # get new state information

    data.append((*prev_state, action, *state))  # save data tuple

    prev_state = state  # for the next iteration set s[t+1] -> s[t]

    if i % M == M-1 or truncated:
        observation, _ = env.reset()
        prev_state = observation

env.close()  # close the simulation environment
np.save('Cartpole_data.npy', data)  # save the data in the folder as a .npy file