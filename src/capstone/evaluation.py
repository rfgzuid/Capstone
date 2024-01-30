import gymnasium as gym

import torch

from .settings import Env
from .noise import CartPoleNoise, LunarLanderNoise
from .cbf import CBF, InfeasibilityError

import matplotlib.pyplot as plt
import numpy as np
import time
import statistics

from tqdm import tqdm

from PIL import Image
import imageio


class Evaluator:
    def __init__(self, env: Env, cbf: CBF) -> None:
        self.env = env.env
        self.is_discrete = env.is_discrete

        self.max_frames = env.settings['max_frames']
        self.noise = env.settings['noise']

        self.cbf = cbf
        self.h_function = env.h_function

    def play(self, agent, GIF, cbf: CBF = None):
        specs = self.env.spec
        if not GIF:
            specs.kwargs['render_mode'] = 'human'
        else:
            specs.kwargs['render_mode'] = 'rgb_array'
        specs.additional_wrappers = tuple()

        play_env = gym.make(specs)

        if self.env.spec.id == 'LunarLander-v2':
            play_env = LunarLanderNoise(play_env, self.noise)
        elif self.env.spec.id == 'CartPole-v1':
            play_env = CartPoleNoise(play_env, self.noise)

        state, _ = play_env.reset()
        images = []

        for frame in range(self.max_frames):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            if cbf is None:
                action = agent.select_action(state, exploration=False)
            else:
                try:
                    action = cbf.safe_action(state)
                    nominal_action = agent.select_action(state, exploration=False)

                    if GIF:
                        rgb_array_large = play_env.render()
                        x_offset = 30
                        y_offset = 30

                        if torch.all(torch.eq(action, nominal_action)):
                            small_img = imageio.imread("off.png")
                            # image = Image.fromarray(small_img)
                            # image.show()
                        else:
                            small_img = imageio.imread("on.png")
                        small_img = Image.fromarray(small_img[:, :, 0:3]).resize((55,55))
                        small_img = np.array(small_img)
                        x_end = x_offset + small_img.shape[1]
                        y_end = y_offset + small_img.shape[0]

                        rgb_array_large[y_offset:y_end, x_offset:x_end] = small_img
                        images.append(Image.fromarray(rgb_array_large))

                except InfeasibilityError:
                    terminated = True

                if self.is_discrete:
                    state, reward, terminated, _, _ = play_env.step(action.item())
                else:
                    state, reward, terminated, _, _ = play_env.step(action.squeeze().detach().numpy())
            if terminated:
                break

        imageio.mimsave('movie.gif', images)
        play_env.close()  # close the simulation environment

    def mc_simulate(self, agent, num_agents, cbf: CBF = None):
        """
        Run a Monte Carlo simulation of [num_agents] agents
         - Returns a list of all the termination/truncation frames
        This allows to numerically estimate the Exit Probability
        """
        h_values = []
        unsafe_frames = []

        for _ in tqdm(range(num_agents)):
            h_list = []
            state, _ = self.env.reset(seed=42)
            state = torch.tensor(state).unsqueeze(0)

            current_frame = 0
            done = False

            while not done:
                h_tensor = self.h_function(state)
                h_list.append(h_tensor.squeeze().numpy())

                # try cbf action - if cbf disabled or no safe actions available, just follow agent policy
                try:
                    action = cbf.safe_action(state)
                except (AttributeError, InfeasibilityError):
                    action = agent.select_action(state, exploration=False)

                state, reward, terminated, truncated, _ = self.env.step(action.squeeze().detach().numpy())
                state = torch.tensor(state).unsqueeze(0)

                current_frame += 1

                if torch.any(self.h_function(state.unsqueeze(0)) < 0).item():
                    unsafe_frames.append(current_frame)
                    terminated = True

                done = terminated or truncated

            h_values.append(np.array(h_list))
        return unsafe_frames, h_values

    def plot(self, agent, n: int):
        state, _ = self.env.reset(seed=42)  # set the initial state for all agents
        dimension_h = self.h_function(torch.tensor(state).unsqueeze(0)).shape[1]  # how many h_i do you have

        h_fig, h_ax = plt.subplots(dimension_h, 2)  # second column with cbf
        p_fig, p_ax = plt.subplots()

        print('Simulating agents without CBF')
        end_frames, h_values = self.mc_simulate(agent, n, cbf=None)
        print('Simulating agents with CBF')
        cbf_end_frames, cbf_h_values = self.mc_simulate(agent, n, cbf=self.cbf)

        P_u = []

        h0 = self.h_function(torch.tensor(state).unsqueeze(0))  # get the state to tensor
        M = 1

        # gamma = 0
        for i in range(dimension_h):
            # plot the exponential decay lower bound of h_i
            h_bound = [h0[0][i].item()]
            h_bound.extend([h0[0][i].item() * self.cbf.alpha[i].item() ** (t+1) + self.cbf.delta[i].item()
                            for t in range(self.max_frames)])

            P_bound = [1 - (h0[0][i].item() / M) *
                       ((M * self.cbf.alpha[i].item() + self.cbf.delta[i].item()) / M) ** t
                       for t in range(self.max_frames)]
            P_u.append(P_bound)

            for run in h_values:
                h_ax[i, 0].plot(run[:, i], color='r', alpha=0.1)

            # apply the same plot scaling to the CBF plots
            h_ax[i, 1].set_xlim(h_ax[i, 0].get_xlim())
            h_ax[i, 1].set_ylim(h_ax[i, 0].get_ylim())

            for run in cbf_h_values:
                h_ax[i, 1].plot(run[:, i], color='g', alpha=0.1)
            h_ax[i, 1].plot(h_bound, color='black', linestyle='dashed')

        P = []
        for t in range(self.max_frames):
            P_succeed = 1
            for q in range(dimension_h):
                P_succeed *= (1 - P_u[q][t])
            P.append(1 - P_succeed)

        terminal = np.zeros(self.max_frames)

        for f in end_frames:
            terminal[f] += 1 / n
        P_emp = np.cumsum(terminal)

        terminal_cbf = np.zeros(self.max_frames)

        for f in cbf_end_frames:
            terminal_cbf[f-1] += 1 / n
        cbf_P_emp = np.cumsum(terminal_cbf)

        p_ax.plot(cbf_P_emp, color='g')
        p_ax.plot(P_emp, color='r')
        p_ax.plot(P, color='black', linestyle='dashed')

        h_fig.tight_layout()
        p_fig.tight_layout()
        plt.show()
