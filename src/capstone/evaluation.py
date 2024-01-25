import gymnasium as gym
import torch

from .settings import Env, BipedalHull
from .cbf import CBF

import matplotlib.pyplot as plt
import numpy as np


class Evaluator:
    def __init__(self, env: Env) -> None:
        self.env = env.env
        self.is_discrete = env.is_discrete
        self.max_frames = env.settings['max_frames']
        self.h_function = env.h_function

    def play(self, agent, cbf: CBF = None):
        specs = self.env.spec
        specs.kwargs['render_mode'] = 'human'
        specs.additional_wrappers = ()

        play_env = gym.make(specs)

        # use custom wrapper
        if play_env.spec.id == 'BipedalWalker-v3':
            play_env = BipedalHull(play_env)

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

    def mc_simulate(self, agent, num_agents, seed=42):
        """
        Run a Monte Carlo simulation of [num_agents] agents
         - Returns a list of all the termination/truncation frames
        This allows to numerically estimate the Exit Probability
        """
        h_values_all_runs = []

        for i in range(num_agents):

            h_values = []
            state, _ = self.env.reset(seed=seed)

            current_frame = 0
            done = False

            while not done:
                state = torch.tensor(state).unsqueeze(0)
                state = state + torch.normal(mean=0., std=0.04, size=state.shape)

                h_tensor = self.h_function(state)
                h_values.append(h_tensor.squeeze().numpy())

                action = agent.select_action(state, exploration=False)
                state, reward, terminated, truncated, _ = self.env.step(action.item())

                current_frame += 1
                done = truncated or terminated

            h_values_all_runs.append(np.array(h_values))

        return h_values_all_runs

    def nice_plots(self, agent, alpha, delta, N, K, M):
        s_knot, _ = self.env.reset(seed=42)
        all_h_values = self.mc_simulate(agent, N)

        fig, (ax1, ax2) = plt.subplots(2)

        for run in all_h_values:
            ax1.plot(run[:, 0], 'r', alpha=0.1)
            ax2.plot(run[:, 1], 'r', alpha=0.1)

        plt.show()

        """
            # Pu-t
            P_u_i_lst = []
            for t in range(K):
                P_u_i_lst.append(1 - (self.h_function(s_knot)[i] / M) * ((M * alpha + delta) / M) ** t)
            axs[i, 1].plot(x, P_u_i_lst)

        # at each timestep T: there is a failure probability P_u = 1 - PI(1-P_u_i)
        P_u_lst = []
        for t in range(K):
            P_succeed = 1
            for q in range(len(h_ind)):
                P_succeed *= 1 - P_u_i_lst[q][t]
            P_u_lst.append(1 - P_succeed)

        # TODO: MC simulation (how many percent of the experiments has failed up to time T)
        axs[0, 3].plot(x, P_u_lst)
        """
