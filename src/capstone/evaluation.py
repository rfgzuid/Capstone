import gymnasium as gym
import torch

from .settings import Env


class Evaluator:
    def __init__(self, env: Env) -> None:
        self.env = env.env
        self.is_discrete = env.is_discrete
        self.max_frames = env.settings['max_frames']

    def play(self, agent):
        specs = self.env.spec
        specs.kwargs['render_mode'] = 'human'
        play_env = gym.make(specs)

        state, _ = play_env.reset()

        for frame in range(self.max_frames):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = agent.select_action(state, exploration=False).squeeze()

            if self.is_discrete:
                state, reward, terminated, _, _ = play_env.step(action.item())
            else:
                state, reward, terminated, _, _ = play_env.step(action.detach().numpy())

            if terminated:
                break

        play_env.close()  # close the simulation environment

    def mc_simulate(self, agent, num_agents=100, seed=42):
        """
        Run a Monte Carlo simulation of [num_agents] agents
         - Returns a list of all the termination/truncation frames
        This allows to numerically estimate the Exit Probability
        """
        end_frames = []

        for i in range(num_agents):
            # TODO: add a small variation (needed for MC)
            state, _ = self.env.reset(seed=seed)

            current_frame = 0
            done = False

            while not done:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action.detach().squeeze(dim=0).numpy())

                current_frame += 1
                done = truncated or terminated

            end_frames.append(current_frame)

        return end_frames
