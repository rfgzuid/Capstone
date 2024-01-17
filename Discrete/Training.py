from Buffer import ReplayMemory, Transition
from Architectures import NNDM
from DQN import DQN

import torch
import matplotlib.pyplot as plt
import gymnasium as gym

from copy import deepcopy
from tqdm import tqdm


class Trainer:
    def __init__(self, env, nndm: NNDM, policy: DQN, buffer: ReplayMemory, settings: dict[str, float]) -> None:
        self.nndm = nndm
        self.env = env

        self.policy = policy
        self.target = deepcopy(self.policy)

        self.replay_memory = buffer
        self.settings = settings

        self.rewards = []
        self.episodes = []
        self.nndm_losses = []

    def train(self):
        for episode_num in tqdm(range(self.settings['num_episodes'])):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            done = False
            episode_reward = 0.
            nndm_loss = []

            while not done:
                action = self.policy.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward

                reward = torch.tensor([reward])
                done = terminated or truncated

                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                self.replay_memory.push(state, action, next_state, reward, int(not terminated))

                state = next_state

                if len(self.replay_memory) < self.settings['batch_size']:
                    batch = None
                else:
                    transitions = self.replay_memory.sample(self.settings['batch_size'])
                    batch = Transition(*zip(*transitions))

                loss = self.nndm.train(batch, self.settings['NNDM_optim'], self.settings['NNDM_criterion'])

                if loss is not None:
                    nndm_loss.append(loss)

                self.policy.train(batch, self.target, self.settings['DQN_optim'], self.settings['DQN_criterion'])

                target_net_state_dict = self.target.state_dict()
                policy_net_state_dict = self.policy.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.settings['tau'] + \
                                                 target_net_state_dict[key] * (1 - self.settings['tau'])
                self.target.load_state_dict(target_net_state_dict)

            avg_nndm_loss = sum(nndm_loss)/len(nndm_loss) if len(nndm_loss) != 0 else 0.

            self.episodes.append(episode_num)
            self.rewards.append(episode_reward)
            self.nndm_losses.append(avg_nndm_loss)

            self.train_plots()
        self.train_plots(is_result=True)

    def train_plots(self, avg_window=5, is_result=False):
        fig = plt.figure(1)
        _, _ = fig.subplots(1, 2)
        rewards = torch.tensor(self.rewards, dtype=torch.float)

        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Episodic reward')

        plt.plot(rewards)

        if len(rewards) >= avg_window:
            means = rewards.unfold(0, avg_window, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(avg_window-1), means))
            plt.plot(means.numpy())

        if not is_result:
            plt.title('Training...')
            plt.pause(0.001)
        else:
            plt.title('Result')
            plt.show()

            plt.plot(self.nndm_losses)
            plt.show()

    def play(self, frames=500):
        env_name = self.env.spec.id
        play_env = gym.make(env_name, render_mode='human')

        state, _ = play_env.reset()

        for frame in range(frames):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = self.policy.select_action(state)
            state, reward, terminated, truncated, _ = play_env.step(action.item())

            if truncated or terminated:
                state, _ = play_env.reset()

        play_env.close()  # close the simulation environment

    def evaluate(self):
        pass
