from .settings import Env
from .dqn import DQN
from .ddpg import Actor
from .barriers import NNDM_H

from bound_propagation import BoundModelFactory, HyperRectangle
from bound_propagation.bounds import LinearBounds
import torch
import cvxpy as cp


class InfeasibilityError(Exception):
    """Exception raised if there are no actions that fulfill the safety criterions."""

    def __init__(self):
        super().__init__()
        self.message = "No safe action to take"


class CBF:
    def __init__(self, env: Env, NNDM_H: NNDM_H, policy: DQN|Actor, alpha: float, partitions: int):
        self.env = env.env
        self.is_discrete = env.is_discrete
        self.settings = env.settings
        self.h_func = env.h_function

        self.NNDM_H = NNDM_H
        factory = BoundModelFactory()
        self.bounded_NNDM_H = factory.build(self.NNDM_H)
        self.policy = policy
        self.alpha = alpha

        if not self.is_discrete:
            self.action_partitions = self.create_action_partitions(partitions)
            self.bounded

    def safe_action(self, state: torch.tensor):
        if self.is_discrete:
            return self.discrete_cbf(state)
        else:
            return self.continuous_cbf(state)

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
    
    def create_action_partitions(self, partitions):
        action_space = self.env.action_space
        num_actions = action_space.shape[0]
        action_low = action_space.low
        action_high = action_space.high

        res = []

        def generate_partitions(dimensions, lower, upper, current_partition):
            if dimensions == num_actions:
                # If we've reached the number of dimensions, add the current partition
                res.append(HyperRectangle(torch.tensor(lower, dtype=torch.float32).unsqueeze(0), torch.tensor(upper, dtype=torch.float32).unsqueeze(0)))
            else:
                # Calculate the size of the partition for the current dimension
                partition_size = (action_high[dimensions] - action_low[dimensions]) / partitions

                for part in range(partitions):
                    # Determine the lower and upper bounds for the current dimension
                    dim_lower_bound = action_low[dimensions] + part * partition_size
                    dim_upper_bound = dim_lower_bound + partition_size

                    # Recursively generate partitions for the next dimension
                    generate_partitions(dimensions + 1, lower + [dim_lower_bound], upper + [dim_upper_bound], current_partition)

        generate_partitions(0, [], [], [])

        return res
        
    def get_lower_bound(self, state, action_partition):
        state_partition = HyperRectangle.from_eps(state.view(1, -1), 0.01)

        input_bounds = HyperRectangle(
            torch.cat((state_partition.lower, action_partition.lower), dim=1),
            torch.cat((state_partition.upper, action_partition.upper), dim=1)
        )
        crown_bounds = self.env.h.crown(input_bounds, bound_upper=False)

        crown_bounds.lower # (A, b)

        return crown_bounds.lower
    
    def create_bound_matrices(self, state):
        action_space = self.env.action_space
        action_dimensionality = action_space.shape[0]
        res = []
        for action_partition in self.action_partitions:
            print("partition", action_partition.lower.shape)
            print("state", state.shape)
            (A, b) = self.get_lower_bound(self.bounded_NNDM_H, state, action_partition, 0.01)
            h_action_dependent = A[:, :, -action_dimensionality:]
            # State input region is a hyperrectangle with "radius" 0.01
            state_input_bounds = HyperRectangle.from_eps(state, 0.01)
            # State dependent part of the A matrix
            state_A = A[:, :, :-action_dimensionality]
            # Make this into a (lower) linear bounds (\underbar{A}_x x + b \leq ...)
            state_linear_bounds = LinearBounds(state_input_bounds, (state_A, b), None)
            # Convert to lower interval bounds (b \leq ...)
            state_interval_bounds = state_linear_bounds.concretize()
            # Select the lower bound
            h_vec = state_interval_bounds.lower

            vecs = (action_partition, h_action_dependent.squeeze().detach().numpy(), h_vec.squeeze().detach().numpy())
            res.append(vecs)

        return res
    
    def continuous_cbf(self, state):
        nominal_action = self.policy(state)
        h_current = self.h_func(state)
        bound_matrices = self.create_bound_matrices(state)
        safe_actions = []
        for action_partition, h_action_dependent, h_vec in bound_matrices:
            num_actions = nominal_action.shape[0]
            action = cp.Variable(num_actions)

            # Constraints
            action_lower_bound = (action_partition).lower.reshape((-1,))
            action_upper_bound = (action_partition).upper.reshape((-1,))
            constraints = [action_lower_bound <= action, action <= action_upper_bound, h_action_dependent @ action + h_vec >= self.alpha * h_current]

            # Objective
            objective = cp.Minimize(cp.norm(action - nominal_action, 2))

            # Solve the problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status is cp.UNBOUNDED:
                print("something goes very wrong")
            elif problem.status is cp.INFEASIBLE:
                pass
            else:
                safe_actions.append((action.value, objective.value))

        if safe_actions and len(safe_actions) > 1:
            return min(safe_actions, key=lambda x: x[1], default=(None, None))[0]
        elif safe_actions:
            return safe_actions[0]
        else:
            raise InfeasibilityError()