from .settings import Env


class InfeasibilityError(Exception):
    """Exception raised if there are no actions that fulfill the safety criterions."""

    def __init__(self):
        super().__init__()
        self.message = "No safe action to take"


class CBF:
    def __init__(self, env: Env, alpha: float):
        self.env = env.env
        self.is_discrete = env.is_discrete
        self.settings = env.settings

        self.alpha = alpha

    def safe_action(self):
        if self.is_discrete:
            return self.discrete_cbf()
        else:
            pass

    def discrete_cbf(self):
        pass
