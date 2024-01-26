from .settings import Env, NoiseWrapper
from .buffer import ReplayMemory, Transition
from .nndm import NNDM
from .dqn import DQN
from .ddpg import Actor, Critic

import torch
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm


class Trainer:
    def __init__(self, env: Env) -> None:
        self.env = NoiseWrapper(env.env)
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
        self.termination_frames = []
        self.nndm_losses = []
        self.actor_losses = []

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

            frame = 0

            while not done:
                action = self.policy.select_action(state)
                observation, reward, terminated, _, _ = self.env.step(action.item())
                truncated = (frame == self.max_frames)
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

                loss_nndm = self.nndm.update(batch)

                if loss_nndm is not None:
                    nndm_loss.append(loss_nndm)

                self.policy.update(batch, self.target)
                self.target.soft_update(self.policy)

                frame += 1

            avg_nndm_loss = sum(nndm_loss)/len(nndm_loss) if len(nndm_loss) != 0 else 0.

            self.rewards.append(episode_reward)
            self.termination_frames.append(frame)
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
            actor_loss = []

            frame = 0

            while not done:
                action = self.actor.select_action(state.squeeze())

                observation, reward, terminated, _, _ = self.env.step(action.detach().numpy())
                truncated = (frame == self.max_frames)
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

                loss_nndm = self.nndm.update(batch)
                loss_actor = self.actor.update(batch, self.critic)

                if loss_nndm is not None:
                    nndm_loss.append(loss_nndm)
                if loss_actor is not None:
                    actor_loss.append(loss_actor)

                self.critic.update(batch, self.critic_target, self.actor_target)
                self.actor.update(batch, self.critic)

                self.actor_target.soft_update(self.actor)
                self.critic_target.soft_update(self.critic)

                frame += 1
            avg_nndm_loss = sum(nndm_loss) / len(nndm_loss) if len(nndm_loss) != 0 else 0.
            avg_actor_loss = sum(actor_loss)/len(actor_loss) if len(actor_loss) != 0 else 0.

            self.rewards.append(episode_reward)
            self.termination_frames.append(frame)
            self.nndm_losses.append(avg_nndm_loss)
            self.actor_losses.append(avg_actor_loss)

            if episode_num % 50 == 0 and episode_num != 0:
                torch.save(self.actor.state_dict(), f'Saved_models/Episode:{episode_num}_reward:{round(episode_reward, 2)}')

            self.train_plots()

        self.train_plots(is_result=True)

        return self.actor, self.nndm

    def train_plots(self, avg_window=10, is_result=False):
        plt.figure(1)

        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Episodic reward')

        plt.plot(self.rewards, "-b")

        plt.legend(["Reward"], loc = "lower right")
        if not is_result:
            plt.title('Training...')
            plt.pause(0.001)
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            fig.subplots_adjust(wspace=1)
            fig.suptitle("Result")
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Episodic reward')
            ax1.plot(self.rewards)

            ax2.set_xlabel('Episode')
            ax2.set_ylabel('NNDM Loss')
            ax2.plot(self.nndm_losses)

            ax3.set_xlabel('Episode')
            ax3.set_ylabel('End frame')
            ax3.plot(self.termination_frames)

            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Actor loss')
            ax4.plot(self.actor_losses)

            plt.show()
