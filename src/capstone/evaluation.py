import gymnasium as gym

import torch

from .settings import Env
from .noise import CartPoleNoise, LunarLanderNoise
from .cbf import CBF, InfeasibilityError

import matplotlib.pyplot as plt
import numpy as np

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

        for i in tqdm(range(num_agents)):
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

    def plot(self, agent, cbf: CBF, N: int):  # N is the number of experiments
        state, _ = self.env.reset(seed=42)  # this is the initial state
        dimension_h = self.h_function(torch.tensor(state).unsqueeze(0)).shape[1]  # how many h_i do you have

        h_fig, h_ax = plt.subplots(dimension_h, 2)  # first col for h_i, second col for p_ui
        p_fig, p_ax = plt.subplots(dimension_h, 2)

        print('Simulating agents without CBF')
        end_frames, h_values = self.mc_simulate(agent, N, 42, cbf=None)  # what you get from the simulation

        print('Simulating agents with CBF')
        cbf_end_frames, cbf_h_values = self.mc_simulate(agent, N, 42, cbf=cbf)

        P_u = []

        h0 = self.h_function(torch.tensor(state).unsqueeze(0))  # get the state to tensor
        M = 1
        gamma = 0

        for i in range(dimension_h):
            # plot the exponential decay lower bound of h_i
            h_bound = [h0[0][i].item()]
            h_bound.extend([h0[0][i].item() * cbf.alpha[i].item() ** (t+1) + cbf.delta[i].item()
                            for t in range(self.max_frames)])

            P_u_i = [1 - (h0[0][i].item() / M) *
                     ((M * cbf.alpha[i].item() + cbf.delta[i].item()) / M) ** t
                     for t in range(self.max_frames)]
            P_u.append(P_u_i)

            for run in h_values:
                h_ax[i, 0].plot(run[:, i], color='r', alpha=0.1)
            for run in cbf_h_values:
                h_ax[i, 1].plot(run[:, i], color='g', alpha=0.1)

            print(h_ax[i, 1].get_xlim())
            h_ax[i, 1].plot(h_bound, scalex=False, scaley=False, color='black', linestyle='dashed')

            p_ax[i, 0].plot(P_u_i, color='g')

            h_ax[i, 0].set_title("h_{}(t): Barrier function, ".format(i) + 'M')
            h_ax[i, 1].set_title("P_u_{}(t): P unsafe, specific failure mode {} ".format(i, i) + 'M')

        P_u = []
        for t in range(self.max_frames):
            P_succeed = 1
            for q in range(dimension_h):
                P_succeed *= (1 - P_u[q][t])
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

        h_fig.tight_layout()
        plt.show()
