from buffer import ReplayMemory, Transition
from nndm import NNDM
from dqn import DQN
from a2c import Actor, Critic

import torch
from torch import nn
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
            # pi/60 = 3 degrees
            state, _ = self.env.reset(options={"x_init": np.pi / 60, "y_init": 0.01})

            done = False
            episode_reward = 0.0

            states = []
            log_probs = []
            rewards = []

            num_steps = 0

            while not done:
                action_mean = self.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)

                # Ensures that the mean stays within [-2, 2] without cutting off the gradient
                action_mean = action_mean.tanh() * 2

                distribution = TruncatedNormal(action_mean, torch.full_like(action_mean, 0.025), min=-2.0, max=2.0)
                action = distribution.sample()

                next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())
                episode_reward += reward

                done = terminated or truncated
                self.replay_memory.push(state, action, next_state, reward, int(not terminated))

                states.append(state)
                log_probs.append(distribution.log_prob(action))
                rewards.append(reward)

                state = next_state
                num_steps += 1

            self.settings['Actor_optim'].zero_grad()
            self.settings['Critic_optim'].zero_grad()

            if truncated:
                terminal_reward = self.target(torch.tensor(next_state)).squeeze().detach()
            else:
                # Penalize if it terminated early (only works if the reward is only negative).
                # If we let the terminal reward be 0, then that means we encourage it to terminate as
                # soon as possible (continuing would incur more costs).
                terminal_reward = torch.tensor(-200.0)

            total_reward = []
            acc_reward = terminal_reward

            # loop from end to beginning
            for t in reversed(range(len(states))):
                acc_reward = rewards[t] + self.settings['a2c_gamma'] * acc_reward
                total_reward.append(acc_reward)

            # Since we build total reward from the back, reverse.
            total_reward = torch.tensor(total_reward).flip(0)

            states = torch.tensor(np.array(states))  # It's faster to convert to numpy first, since each element is already a numpy array
            log_probs = torch.stack(log_probs)  # This is stack because log_probs is already a list of tensors and we don't want to loose gradient information

            pred_reward = self.target(states).squeeze(1)

            loss_critic = self.settings['Critic_criterion'](pred_reward, total_reward)
            loss_critic.backward()

            nn.utils.clip_grad_norm_(self.target.parameters(), 0.5)

            self.settings['Critic_optim'].step()
            self.settings['Critic_optim'].zero_grad()

            delta_t = total_reward - pred_reward
            loss_actor = (-log_probs * delta_t.detach()).mean()
            loss_actor.backward()

            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

            self.settings['Actor_optim'].step()

            # Statistics for the past 50-ish episodes might be more appropriate (filter out the noise)
            if episode_num % 10 == 0:
                print(f"Episode {episode_num:5}: {num_steps:3}/{total_reward[0].item():9.3f}/{loss_critic.item():10.3f}/{loss_actor.item():8.3f}")

            self.episodes.append(episode_num)
            self.rewards.append(episode_reward.item())

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

        state, _ = play_env.reset(options={"x_init": np.pi / 60, "y_init": 0.01})

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
            state, _ = self.env.reset(seed=seed, options={"x_init": np.pi / 60, "y_init": 0.01})

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

