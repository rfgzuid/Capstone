import gymnasium as gym

import torch

from .settings import Env
from .noise import CartPoleNoise, LunarLanderNoise
from .cbf import CBF, InfeasibilityError

import matplotlib.pyplot as plt
import numpy as np


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

        for i in range(num_agents):
            h_values = []
            state, _ = self.env.reset(seed=seed)

            current_frame = 0
            done = False

            while not done:
                state = torch.tensor(state).unsqueeze(0)

                h_tensor = self.h_function(state)
                h_values.append(h_tensor.squeeze().numpy())

                if cbf is None:
                    if self.is_discrete:
                        action = agent.select_action(state, exploration=False)
                        state, reward, terminated, truncated, _ = self.env.step(action.item())
                    else:
                        action = agent.select_action(state.squeeze(), exploration=False)
                        state, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())

                else:
                    try:
                        action = np.array(cbf.safe_action(state.squeeze()))
                        state, reward, terminated, truncated, _ = self.env.step(action)
                    except InfeasibilityError:
                        terminated = True

                current_frame += 1
                done = terminated or truncated

            end_frames.append(current_frame)

            h_values_all_runs.append(np.array(h_values))

        return end_frames, h_values_all_runs

    def plot(self, agent, alpha, delta, N, M, cbf: CBF = None):  # N is the number of experiments
        s_knot, _ = self.env.reset(seed=42)  # this is the initial state
        dimension_h = self.h_function(torch.tensor(s_knot).unsqueeze(0)).shape[1]  # how many h_i do you have

        # without cbf
        fig1, axs1 = plt.subplots(dimension_h, 2)  # first col for h_i, second col for p_ui
        fig2, axs2 = plt.subplots()  # combined P_u plot, theoretical and empirical (emp: orange)

        # with cbf
        fig3, axs3 = plt.subplots(dimension_h, 2)  # first col for h_i, second col for p_ui
        fig4, axs4 = plt.subplots()  # combined P_u plot, theoretical and empirical (emp: orange)

        for l in range(2):
            if l == 0:
                T = True
                title = "no CBF"
                col = "red"
                end_frames, all_h_values = self.mc_simulate(agent, N, 42)  # what you get from the simulation
            elif l == 1:
                T = False
                title = 'with CBF'
                col = "green"
                end_frames, all_h_values = self.mc_simulate(agent, N, 42,
                                                            cbf)  # what you get from the simulation

            P_u_lst = []

            for i in range(dimension_h):
                x = range(self.max_frames)
                h_s_knot = self.h_function(torch.tensor(s_knot).unsqueeze(0))  # get the state to tensor
                P_u_i_lst = []
                for t in range(self.max_frames):
                    P_u_i_lst.append(1 - (h_s_knot[0][i].item() / M) * ((M * alpha + delta) / M) ** t)
                P_u_lst.append(P_u_i_lst)

                for run in all_h_values:
                    # h_i_plot
                    if l == 0:
                        axs1[i, 0].plot(run[:, i], color=col, alpha=0.1)
                    if l == 1:
                        axs3[i, 0].plot(run[:, i], color=col, alpha=0.1)

                    # p_u_i plot

                if T:
                    axs1[i, 0].set_title("h_{}(t): Barier function, ".format(i) + title)
                    axs1[i, 1].plot(x, P_u_i_lst, color=col)
                else:
                    axs3[i, 0].set_title("h_{}(t): Barier function, ".format(i) + title)
                    axs3[i, 1].plot(x, P_u_i_lst, color=col)

                if T:
                    axs1[i, 1].set_title("P_u_{}(t): P unsafe, specific failure mode {} ".format(i, i) + title)
                else:
                    axs3[i, 1].set_title("P_u_{}(t): P unsafe, specific failure mode {} ".format(i, i) + title)

            P_u = []
            for t in range(self.max_frames):
                P_succeed = 1
                for q in range(dimension_h):
                    P_succeed *= (1 - P_u_lst[q][t])
                P_u.append(1 - P_succeed)

            end_frames.sort()

            P_u_emp = []

            for t in range(self.max_frames):
                counter = 0
                for frame in end_frames:
                    if frame <= t:
                        counter += 1
                counter = counter / N
                P_u_emp.append(counter)

            if T:
                axs2.plot(range(self.max_frames), P_u, label="Theoretical P_unsafe", color=col)
                axs2.plot(range(self.max_frames), P_u_emp, color='blue', label="Empirical P_unsafe")
                axs2.set_title("P unsafe combined" + title)
                axs2.legend(loc='lower right')
            else:
                axs4.plot(range(self.max_frames), P_u, label="Theoretical P_unsafe", color=col)
                axs4.plot(range(self.max_frames), P_u_emp, color='blue', label="Empirical P_unsafe")
                axs4.set_title("P unsafe combined" + title)
                axs4.legend(loc='lower right')

        plt.show()
