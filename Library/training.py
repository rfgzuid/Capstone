from settings import Env
from buffer import ReplayMemory, Transition
from nndm import NNDM
from dqn import DQN
from ddpg import Actor, Critic

import torch
import matplotlib.pyplot as plt

import numpy as np

from copy import deepcopy
from tqdm import tqdm


class Trainer:
    def __init__(self, env: Env) -> None:
        self.env = env.env
        self.is_discrete = env.is_discrete
        self.settings = env.settings

        self.replay_memory = ReplayMemory(self.settings['replay_size'])
        self.max_frames = env.settings['max_frames']
        self.nndm = NNDM(env)

        if self.is_discrete:
            self.policy = DQN(env)
            self.target = deepcopy(self.policy)
        else:
            self.actor = Actor(env)
            self.actor_target = deepcopy(self.actor)

            self.critic = Critic(env)
            self.critic_target = deepcopy(self.critic)

        # metrics to track during training
        self.rewards = []
        self.episodes = []
        self.nndm_losses = []

    def train(self):
        if self.is_discrete:
            return self.ddqn_train()
        else:
            return self.ddpg_train()

    def ddqn_train(self):
        for episode_num in tqdm(range(self.settings['num_episodes'])):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            done = False
            episode_reward = 0.
            nndm_loss = []

            frame = 1

            while not done:
                action = self.policy.select_action(state)
                observation, reward, terminated, _, _ = self.env.step(action.item())
                truncated = (frame > self.max_frames)
                episode_reward += reward

                reward = torch.tensor([reward], dtype=torch.float32)
                done = terminated or truncated

                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                self.replay_memory.push(state, action, next_state, reward, int(terminated))

                state = next_state

                if len(self.replay_memory) < self.settings['batch_size']:
                    batch = None
                else:
                    transitions = self.replay_memory.sample(self.settings['batch_size'])
                    batch = Transition(*zip(*transitions))

                loss = self.nndm.update(batch)

                if loss is not None:
                    nndm_loss.append(loss)

                self.policy.update(batch, self.target)
                self.target.soft_update(self.policy)

                frame += 1

            avg_nndm_loss = sum(nndm_loss)/len(nndm_loss) if len(nndm_loss) != 0 else 0.

            self.episodes.append(episode_num)
            self.rewards.append(episode_reward)
            self.nndm_losses.append(avg_nndm_loss)

            self.train_plots()

        self.train_plots(is_result=True)

        return self.policy, self.nndm

    def ddpg_train(self):
        for episode_num in tqdm(range(self.settings['num_episodes'])):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            done = False
            episode_reward = 0.0
            nndm_loss = []

            frame = 1

            while not done:
                action = self.actor.select_action(state.squeeze())

                observation, reward, terminated, _, _ = self.env.step(action.detach().numpy())
                truncated = (frame > self.max_frames)
                episode_reward += reward

                action = action.clone().detach().unsqueeze(dim=0)
                reward = torch.tensor([reward], dtype=torch.float32)
                done = terminated or truncated

                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                self.replay_memory.push(state, action, next_state, reward, int(terminated))

                state = next_state

                if len(self.replay_memory) < self.settings['batch_size']:
                    batch = None
                else:
                    transitions = self.replay_memory.sample(self.settings['batch_size'])
                    batch = Transition(*zip(*transitions))

                loss = self.nndm.update(batch)

                if loss is not None:
                    nndm_loss.append(loss)

                self.critic.update(batch, self.critic_target, self.actor_target)
                self.actor.update(batch, self.critic)

                self.actor_target.soft_update(self.actor)
                self.critic_target.soft_update(self.critic)

                frame += 1

            avg_nndm_loss = sum(nndm_loss) / len(nndm_loss) if len(nndm_loss) != 0 else 0.

            self.episodes.append(episode_num)
            self.rewards.append(episode_reward)
            self.nndm_losses.append(avg_nndm_loss)

            self.train_plots()

        self.train_plots(is_result=True)

        return self.actor, self.nndm

    def train_plots(self, avg_window=10, is_result=False):
        # TODO: get a NNDM loss subplot that updates with the agent reward plot
        # maybe just use tensorboard or some advanced tool like that
        # also a good idea: save the model losses as an attribute in its class

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
