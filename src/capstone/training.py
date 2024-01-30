from .settings import Env
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
        self.termination_frames = []
        self.nndm_losses = []

    def train(self):
        for _ in tqdm(range(self.settings['num_episodes'])):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            done = False
            episode_reward = 0.
            nndm_loss = []

            frame = 0

            while not done:
                if self.is_discrete:
                    action = self.policy.select_action(state)
                else:
                    action = self.actor.select_action(state)
                observation, reward, terminated, _, _ = self.env.step(action.squeeze().detach().numpy())
                action = action.clone().detach()

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

                if self.is_discrete:
                    self.policy.update(batch, self.target)
                    self.target.soft_update(self.policy)

                else:
                    self.critic.update(batch, self.critic_target, self.actor_target)
                    self.actor.update(batch, self.critic)

                    self.actor_target.soft_update(self.actor)
                    self.critic_target.soft_update(self.critic)

                frame += 1

            avg_nndm_loss = sum(nndm_loss)/len(nndm_loss) if len(nndm_loss) != 0 else 0.

            self.rewards.append(episode_reward)
            self.termination_frames.append(frame)
            self.nndm_losses.append(avg_nndm_loss)

            self.train_plots()

        self.train_plots(is_result=True)

        return (self.policy, self.nndm) if self.is_discrete else (self.actor, self.nndm)

    def train_plots(self, is_result=False):
        plt.figure(1)

        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Episodic reward')

        plt.plot(self.rewards, "-b")

        if not is_result:
            plt.title('Training...')
            plt.pause(0.001)
        else:
            plt.xlabel('Episode')
            plt.ylabel('Episodic reward')
            plt.title(f"{self.env} Reward")
            plt.plot(self.rewards)
            plt.savefig(f"Plots_final_models/{self.env}_Reward")
            plt.show()

            plt.xlabel('Episode')
            plt.ylabel('NNDM Loss')
            plt.plot(self.nndm_losses)
            plt.title(f"{self.env} NNDM Loss")
            plt.savefig(f"Plots_final_models/{self.env}_NNDM_loss")
            plt.show()

            plt.xlabel('Episode')
            plt.ylabel('End frame')
            plt.plot(self.termination_frames)
            plt.title(f"{self.env} End frame")
            plt.savefig(f"Plots_final_models/{self.env}_EndFrame")
            plt.show()
