import gymnasium as gym
import numpy as np

from gymnasium import spaces, logger
from gymnasium.utils import seeding
from gymnasium.spaces import Box

from gymnasium.error import DependencyNotInstalled



class CartPoleNoise(gym.Wrapper):
    def __init__(self, env, noise: dict[str, float]):
        super().__init__(env)
        self.noise = noise

    def step(self, action):
        state, reward, terminated, truncated, _ = self.env.step(action)

        pos = np.array([self.env.unwrapped.state[0]])
        pos += np.random.normal(0., self.noise['x'])

        angle = np.array([self.env.unwrapped.state[1]])
        angle += np.random.normal(0., self.noise['theta'])

        pos_vel = np.array([self.env.unwrapped.state[2]])
        pos_vel += np.random.normal(0., self.noise['v_x'])

        ang_vel = np.array([self.env.unwrapped.state[3]])
        ang_vel += np.random.normal(0., self.noise['v_theta'])

        self.env.unwrapped.state = (pos[0], angle[0], pos_vel[0], ang_vel[0])

        return state, reward, terminated, truncated, None

    def __str__(self):
        return "Cartpole"


class LunarLanderNoise(gym.Wrapper):
    def __init__(self, env, noise: dict[str, float]):
        super().__init__(env)
        self.noise = noise

    def step(self, action):
        state, reward, terminated, truncated, _ = self.env.step(action)

        pos = np.array(self.env.unwrapped.lander.position)
        pos[0] += np.random.normal(0., self.noise['x'])
        pos[1] += np.random.normal(0., self.noise['y'])
        self.env.unwrapped.lander.position = tuple(pos)

        angle = np.array([self.env.unwrapped.lander.angle])
        angle += np.random.normal(0., self.noise['theta'])
        self.env.unwrapped.lander.angle = angle[0]

        pos_vel = np.array(self.env.unwrapped.lander.linearVelocity)
        pos_vel[0] += np.random.normal(0., self.noise['v_x'])
        pos_vel[1] += np.random.normal(0., self.noise['v_y'])
        self.env.unwrapped.lander.linearVelocity = tuple(pos_vel)

        ang_vel = np.array([self.env.unwrapped.lander.angularVelocity])
        ang_vel += np.random.normal(0., self.noise['v_theta'])
        self.env.unwrapped.lander.angularVelocity = ang_vel[0]

        return state, reward, terminated, truncated, None

    def __str__(self):
        if self.env.spec.__dict__["kwargs"]["continuous"]:
            return "Lunar Lander Continuous"
        else:
            return "Lunar Lander Discrete"
