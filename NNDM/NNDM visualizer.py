'''Visualize the learnt dynamics by comparing it to the simulation'''

import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# load NNDM architecture
class NNDM(nn.Module):
    def __init__(self):
        super(NNDM, self).__init__()

        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 4)

        self.activation = nn.functional.tanh

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))
        h = self.activation(self.fc4(h))
        h = self.activation(self.fc5(h))
        return self.fc6(h) + x[:4]


model = NNDM()

# the saved model from 'model.py' will be loaded in and used in evaluation mode
model.load_state_dict(torch.load('NNDM.pt'))
model.eval()

render_simulation = False
env = gym.make("CartPole-v1", render_mode='human' if render_simulation else 'none')
observation, _ = env.reset()  # initialize the environment randomly

N = 100  # how long to run the simulation for
frames: list[int] = list(range(N+1))  # list of frames for plotting

# save the start information and create lists to save simulation data
true_position = [observation[0]]
true_angle = [observation[2]]
actions = []

for i in range(N):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)

    # append new state information to lists
    actions.append(action)
    true_position.append(state[0])
    true_angle.append(state[2])

    if truncated:
        state, _ = env.reset()

env.close()  # close the simulation environment


M = 10  # amount of monte carlo simulations
pred_positions = np.zeros([M, N+1])
pred_angles = np.zeros([M, N+1])

'''for each MC simulation (loop of length M):
- create a noisy initial observation
- let the NNDM predict the path of the cart by iteratively feeding states (loop of length N)
- save the predicted path in the numpy array of shape (M, N+1)
'''
with torch.no_grad():
    for j in range(M):
        noisy_observation = torch.tensor(observation) + torch.randn(4) * 0.03
        pred_position = [noisy_observation[0]]
        pred_angle = [noisy_observation[2]]

        input = torch.tensor((*noisy_observation, actions[0])) # initial state will be noisy for monte carlo simulation

        for i in range(N):
            output = model(input)

            pred_position.append(output[0])
            pred_angle.append(output[2])

            input = torch.zeros(5)
            input[:4] = output  # s[t+1] gets set to input s[t] for the next step
            input[4] = actions[i]

        pred_positions[j, :] = np.array(pred_position)
        pred_angles[j, :] = np.array(pred_angle)

# for plotting, we will show the average predicted path, as well as the std as a shaded area
avg_position = np.mean(pred_positions, axis=0)
avg_angle = np.mean(pred_angles, axis=0)

std_position = np.std(pred_positions, axis=0)
std_angle = np.std(pred_angles, axis=0)

# plotting
fig, ax = plt.subplots(1, 2)

ax[0].fill_between(frames, avg_position-std_position, avg_position+std_position, alpha=0.3, color='b')
ax[0].plot(frames, avg_position, c='b', label='Average MC')
ax[0].plot(frames, true_position, c='r', label='True')
ax[0].set_title('Cart position')
ax[0].set_xlabel('Frame number')
ax[0].legend()

ax[1].fill_between(frames, avg_angle-std_angle, avg_angle+std_angle, alpha=0.3, color='b')
ax[1].plot(frames, avg_angle, c='b', label='Average MC')
ax[1].plot(frames, true_angle, c='r', label='True')
ax[1].set_title('Pole angle')
ax[1].set_xlabel('Frame number')
ax[1].legend()

plt.show()