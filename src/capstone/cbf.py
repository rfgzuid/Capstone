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

        nominal_action = self.policy.select_action(state, exploration=False)
        best_action = nominal_action

        res = []

        for action in action_space:
            h = self.H(torch.cat(state, action.unsqueeze(0))).view(1, -1)
            h_prev = self.h_func(state)
            if torch.all(torch.ge(h, self.alpha * h_prev)):
                res += [(int(action != nominal_action), h, action)]
        best_action_tuple = min(res, key=lambda x: x[0])

        if sum(best_action_tuple[0] == action_tuple[0] for action_tuple in res) > 1:
            best_action_tuple = min([action_tuple for action_tuple in res if action_tuple[0] == best_action_tuple[0]],
                                    key=lambda x: x[1])
        best_action = best_action_tuple[2].view(1, 1)

        return best_action