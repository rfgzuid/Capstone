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



class Evaluator:
    def __init__(self, env: Env) -> None:
        self.env = env.env
        self.is_discrete = env.is_discrete

        self.max_frames = env.settings['max_frames']
        self.noise = env.settings['noise']

        self.h_function = env.h_function

        self.titles = env.h_name
        self.image = type(env).__name__

    def play(self, agent, cbf: CBF = None):
        specs = self.env.spec
        specs.kwargs['render_mode'] = 'human'
        specs.additional_wrappers = tuple()

        play_env = gym.make(specs)

        if self.env.spec.id == 'LunarLander-v2':
            play_env = LunarLanderNoise(play_env, self.noise)
        elif self.env.spec.id == 'CartPole-v1':
            play_env = CartPoleNoise(play_env, self.noise)

        state, _ = play_env.reset()

        for frame in range(self.max_frames):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            if cbf is None:
                action = agent.select_action(state, exploration=False)
            else:
                action = cbf.safe_action(state)

            if self.is_discrete:
                state, reward, terminated, _, _ = play_env.step(action.item())
            else:
                state, reward, terminated, _, _ = play_env.step(action.squeeze().detach().numpy())

            if terminated:
                break

        play_env.close()  # close the simulation environment

    def mc_simulate(self, agent, num_agents, seed=42, cbf: CBF = None):
        """
        Run a Monte Carlo simulation of [num_agents] agents
         - Returns a list of all the termination/truncation frames
        This allows to numerically estimate the Exit Probability
        """
        h_values_all_runs = []
        end_frames = []
        agent_filter_times = []

        for i in tqdm(range(num_agents)):
            h_values = []
            state, _ = self.env.reset(seed=seed)

            agent_filter_time = 0
            current_frame = 0
            done = False

            while not done:
                state = torch.tensor(state).unsqueeze(0)

                h_tensor = self.h_function(state)
                h_values.append(h_tensor.squeeze().numpy())

                # try cbf action - if cbf disabled, just follow agent policy
                try:
                    # start_time = time.time()

                    cbf_action = cbf.safe_action(state.squeeze())

                    if self.is_discrete:
                        action = agent.select_action(state, exploration=False)
                    else:
                        action = agent.select_action(state.squeeze(), exploration=False)
                        action = action.detach()

                    if not torch.all(action == cbf_action):
                        print(action, cbf_action)

                    state, reward, terminated, truncated, _ = self.env.step(cbf_action.detach().numpy())

                    # end_time = time.time()
                    # agent_filter_time += end_time - start_time
                except InfeasibilityError:
                    terminated = True
                except AttributeError:
                    # no cbf enabled, use agent action
                    if self.is_discrete:
                        action = agent.select_action(state, exploration=False)
                        state, reward, terminated, truncated, _ = self.env.step(action.item())
                    else:
                        action = agent.select_action(state.squeeze(), exploration=False)
                        state, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())

                current_frame += 1
                done = terminated or truncated

            if terminated:
                end_frames.append(current_frame)

            agent_filter_times.append(agent_filter_time)

            h_values_all_runs.append(np.array(h_values))

        return end_frames, h_values_all_runs, agent_filter_times

    def plot(self, agent, cbf: CBF, N: int):  # N is the number of agents or experiments
        state, _ = self.env.reset(seed=42)  # this is the initial state
        dimension_h = self.h_function(torch.tensor(state).unsqueeze(0)).shape[1]  # how many h_i do you have

        h_fig, h_ax = plt.subplots(dimension_h, 2)  # second col with cbf
        p_fig, p_ax = plt.subplots()

        print('Simulating agents without CBF')
        end_frames, h_values, agent_filter_times = self.mc_simulate(agent, N, 42, cbf=None)
        mean_filter_time = statistics.mean(agent_filter_times)
        std = statistics.stdev(agent_filter_times)
        print(f'The mean CBF filter time for one agent is: {mean_filter_time:.2f}, the standard deviation is {std:.2f}')

        print('Simulating agents with CBF')
        cbf_end_frames, cbf_h_values, agent_filter_times = self.mc_simulate(agent, N, 42, cbf=cbf)
        mean_filter_time = statistics.mean(agent_filter_times)
        std = statistics.stdev(agent_filter_times)
        print(f'The mean CBF filter time for one agent is: {mean_filter_time:.2f}, the standard deviation is {std:.2f}')

        P_u = []

        h0 = self.h_function(torch.tensor(state).unsqueeze(0))  # get the state to tensor
        M = 1

        # gamma = 0
        for i in range(dimension_h):
            # plot the exponential decay lower bound of h_i
            h_bound = [h0[0][i].item()]
            h_bound.extend([h0[0][i].item() * cbf.alpha[i].item() ** (t+1) + cbf.delta[i].item()
                            for t in range(self.max_frames)])

            P_bound = [1 - (h0[0][i].item() / M) *
                     ((M * cbf.alpha[i].item() + cbf.delta[i].item()) / M) ** t
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
            terminal[f-1] += 1 / N
        P_emp = np.cumsum(terminal)

        terminal_cbf = np.zeros(self.max_frames)

        for f in cbf_end_frames:
            terminal_cbf[f-1] += 1 / N
        cbf_P_emp = np.cumsum(terminal_cbf)

        p_ax.plot(cbf_P_emp, color='g')
        p_ax.plot(P_emp, color='r')
        p_ax.plot(P, color='black', linestyle='dashed')

        h_fig.tight_layout()
        p_fig.tight_layout()
        plt.show()
