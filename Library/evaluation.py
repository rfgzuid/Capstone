import gymnasium as gym
import torch

from env_settings import Env


class Evaluator:
    def __init__(self, env: Env) -> None:
        self.env = env.env

    def play(self, agent, frames: int = 500):

        specs = self.env.spec
        specs.kwargs['render_mode'] = 'human'
        play_env = gym.make(specs)

        state, _ = play_env.reset()

        for frame in range(frames):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = play_env.step(action.detach().squeeze(dim=0).numpy())

            if truncated or terminated:
                state, _ = play_env.reset()

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
