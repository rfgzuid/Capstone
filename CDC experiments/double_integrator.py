import gymnasium as gym
from gymnasium.spaces import Box

import numpy as np


# Based on https://arxiv.org/abs/2302.07469
class DoubleIntegratorEnv(gym.Env):
    def __init__(self):
        super(DoubleIntegratorEnv, self).__init__()

        self.dt = 0.05
        self.render_mode = None

        max_distance = np.inf
        max_velocity = np.inf
        max_force = 100

        self.observation_space = Box(
            low=np.array([-max_distance, -max_distance, -max_velocity, -max_velocity]),
            high=np.array([max_distance, max_distance, max_velocity, max_velocity]),
            dtype=np.float32
        )

        self.action_space = Box(
            low=np.array([-max_force, -max_force]),
            high=np.array([max_force, max_force]),
            dtype=np.float32
        )

        self.state = None

    def step(self, action):
        """Action has to be numpy array of shape (2,)"""
        x, y, xdot, ydot = self.state.squeeze()
        Fx, Fy = action

        x = x + xdot * self.dt + Fx * 0.5 * self.dt ** 2
        y = y + ydot * self.dt + Fy * 0.5 * self.dt ** 2

        xdot = xdot + Fx * self.dt
        ydot = ydot + Fy * self.dt

        self.state = np.array([x, y, xdot, ydot], dtype=np.float32)
        return self.state, 0., False, False, {}

    def reset(self, seed=None):
        if seed is None:
            self.state = np.zeros((4,), dtype=np.float32)
        else:
            self.state = np.array(seed, dtype=np.float32)
        return self.state, {}
