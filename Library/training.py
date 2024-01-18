from Buffer import ReplayMemory, Transition
from Architectures import NNDM
from DQN import DQN
from a2c import Actor, Critic

import torch
from torchrl.modules import TruncatedNormal
import matplotlib.pyplot as plt
import gymnasium as gym

import numpy as np

from copy import deepcopy
from tqdm import tqdm


class Trainer:
    def __init__(self, env, nndm: NNDM, policy: DQN|Actor, buffer: ReplayMemory,
                 settings: dict[str, float], target: Critic|None = None) -> None:
        self.nndm = nndm
        self.env = env

        self.is_discrete = True if target is None else False

        self.policy = policy  # is either a DQN network or Actor
        self.target = deepcopy(policy) if target is None else target  # set target as DQN copy or Critic

        self.replay_memory = buffer
        self.settings = settings

        self.rewards = []
        self.episodes = []
        self.nndm_losses = []

    def train(self):
        if self.is_discrete:
            return self.ddqn_train()
        else:
            return self.a2c_train()

    def ddqn_train(self):
        for episode_num in tqdm(range(self.settings['num_episodes'])):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            done = False
            episode_reward = 0.
            nndm_loss = []

            while not done:
                action = self.policy.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.detach().squeeze(dim=0).numpy())
                episode_reward += reward

                reward = torch.tensor([reward], dtype=torch.float32)
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
        return self.policy

    def a2c_train(self):
        for episode_num in tqdm(range(self.settings['num_episodes'])):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            done = False
            episode_reward = 0.

            while not done:
                mean_std = self.policy(state)
                action_mean, action_std = torch.chunk(mean_std, 2, dim=1)

                distribution = TruncatedNormal(action_mean, action_std)
                action = distribution.sample()

                next_state, reward, terminated, truncated, _ = self.env.step(action.detach().squeeze(dim=0).numpy())
                episode_reward += reward

                reward = torch.tensor([reward], dtype=torch.float32)
                done = terminated or truncated

                self.settings['Actor_optim'].zero_grad()
                self.settings['Critic_optim'].zero_grad()

                if not terminated:
                    td_target = reward + self.settings['a2c_gamma'] * \
                            self.target(torch.tensor(next_state)).detach()
                else:
                    td_target = reward

                delta_t = self.settings['Critic_criterion'](td_target, self.target(torch.tensor(state)).squeeze(dim=0))

                delta_t.backward()
                self.settings['Critic_optim'].step()
                self.settings['Critic_optim'].zero_grad()

                delta_t = td_target - self.target(torch.tensor(next_state))
                loss_actor = -distribution.log_prob(action) * delta_t
                loss_actor.backward()
                self.settings['Actor_optim'].step()

                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                self.replay_memory.push(state, action, next_state, reward, int(not terminated))

                state = next_state

            self.episodes.append(episode_num)
            self.rewards.append(episode_reward)

            self.train_plots()

        self.train_plots(is_result=True)
        return self.policy

    def train_plots(self, avg_window=10, is_result=False):
        plt.figure(1)
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

    def play(self, agent: DQN|Actor, frames: int = 500):

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

    def evaluate(self, agent: DQN|Actor, num_agents=100, seed=42):
        """Do a Monte Carlo simulation of num_agents agents
         - Returns a list of all the termination frames
        This allows to numerically estimate the Exit Probability"""
        termination_frames = []

        for i in range(num_agents):
            state, _ = self.env.reset(seed=seed)

            current_frame = 0
            done = False

            while not done:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action.detach().squeeze(dim=0).numpy())

                current_frame += 1
                done = truncated or terminated

            termination_frames.append(current_frame)

        return termination_frames
