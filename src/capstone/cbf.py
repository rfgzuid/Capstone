from .settings import Env
from .dqn import DQN
from .ddpg import Actor
from .barriers import H

import torch


class InfeasibilityError(Exception):
    """Exception raised if there are no actions that fulfill the safety criterions."""

    def __init__(self):
        super().__init__()
        self.message = "No safe action to take"


class CBF:
    def __init__(self, env: Env, h: H, policy: DQN|Actor, alpha: float):
        self.env = env.env
        self.state_size = self.env.observation_space.shape[0]

        self.is_discrete = env.is_discrete
        self.settings = env.settings
        self.h_func = env.h_function

        self.H = h
        self.policy = policy
        self.alpha = alpha

    def safe_action(self, state: torch.tensor):
        if self.is_discrete:
            return self.discrete_cbf(state)
        else:
            pass

    def discrete_cbf(self, state):
        # Discrete(n) has actions {0, 1, ..., n-1} - see gymnasium docs
        action_space = torch.arange(self.env.action_space.n)
        safe_actions = []

        h_cur = self.h_func(state)

        for action in action_space:
            h_input = torch.zeros((1, self.state_size + 1))
            h_input[:, :self.state_size] = state
            h_input[:, self.state_size] = action

            h_next = self.H(h_input)

            if torch.all(h_next >= self.alpha * h_cur).item():
                safe_actions.append(action)

        if safe_actions and len(safe_actions) > 1:
            q_values = self.policy(state).squeeze()
            mask = torch.zeros_like(q_values, dtype=torch.bool)

            for action in safe_actions:
                mask[action] = True

            safe_q_values = q_values.masked_fill(~mask, float('-inf'))
            best_action = torch.argmax(safe_q_values)

            return best_action.item()
        elif safe_actions:
            return safe_actions[0].item()
        else:
            raise InfeasibilityError()
