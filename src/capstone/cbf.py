from .settings import Env
from .dqn import DQN
from .ddpg import Actor
from .barriers import NNDM_H, stochastic_NNDM_H

from bound_propagation import BoundModelFactory, HyperRectangle
from bound_propagation.bounds import LinearBounds
import torch
import cvxpy as cp
from collections.abc import Iterable
from itertools import product
import numpy as np
from scipy.stats import truncnorm
from scipy.special import erf, erfc
import math


class InfeasibilityError(Exception):
    """Exception raised if there are no actions that fulfill the safety criteria."""

    def __init__(self):
        super().__init__()
        self.message = "No safe action to take"


class CBF:
    def __init__(self, env: Env, nndm_h: NNDM_H, policy: DQN | Actor, alpha: float, action_partitions: int = 4, 
                 noise_partitions = 2, is_stochastic = False, stds = None, h_ids = None):
        self.env = env.env
        self.state_size = self.env.observation_space.shape[0]

        self.is_discrete = env.is_discrete
        self.settings = env.settings
        self.h_func = env.h_function

        self.is_stochastic = is_stochastic
        self.NNDM_H = nndm_h
        self.policy = policy
        self.alpha = alpha

        if not self.is_discrete:
            factory = BoundModelFactory()
            self.bounded_NNDM_H = factory.build(self.NNDM_H)
            self.action_partitions = self.create_action_partitions(action_partitions)
            if self.is_stochastic:
                assert isinstance(noise_partitions, int)
                assert isinstance(stds, Iterable)
                assert len(h_ids) == len(stds)
                self.h_ids = h_ids # TODO: need to be moved to settings
                self.noise_partitions = noise_partitions
                self.stds = stds # TODO: need to be moved need to be moved to settings


    def safe_action(self, state: torch.tensor):
        if self.is_discrete:
            return self.discrete_cbf(state)
        else:
            return self.continuous_cbf(state)

    def discrete_cbf(self, state):
        # Discrete(n) has actions {0, 1, ..., n-1} - see gymnasium docs
        action_space = torch.arange(self.env.action_space.n)
        safe_actions = []

        h_cur = self.h_func(state)

        for action in action_space:
            h_input = torch.zeros((1, self.state_size + 1))
            h_input[:, :self.state_size] = state
            h_input[:, self.state_size] = action

            h_next = self.NNDM_H(h_input)

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
    
    def create_action_partitions(self, partitions):
        num_actions = self.env.action_space.shape[0]
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high

        res = []

        def generate_partitions(dimensions, lower, upper, current_partition):
            if dimensions == num_actions:
                # If we've reached the number of dimensions, add the current partition
                res.append(HyperRectangle(torch.tensor(lower, dtype=torch.float32).unsqueeze(0),
                                          torch.tensor(upper, dtype=torch.float32).unsqueeze(0)))
            else:
                # Calculate the size of the partition for the current dimension
                partition_size = (action_high[dimensions] - action_low[dimensions]) / partitions

                for part in range(partitions):
                    # Determine the lower and upper bounds for the current dimension
                    dim_lower_bound = action_low[dimensions] + part * partition_size
                    dim_upper_bound = dim_lower_bound + partition_size

                    # Recursively generate partitions for the next dimension
                    generate_partitions(dimensions + 1, lower + [dim_lower_bound], upper + [dim_upper_bound],
                                        current_partition)

        generate_partitions(0, [], [], [])

        return res
        
    def get_lower_bound(self, state, action_partition):
        state_partition = HyperRectangle.from_eps(state.view(1, -1), 0.01)

        input_bounds = HyperRectangle(
            torch.cat((state_partition.lower, action_partition.lower), dim=1),
            torch.cat((state_partition.upper, action_partition.upper), dim=1)
        )

        crown_bounds = self.bounded_NNDM_H.crown(input_bounds, bound_upper=False)
        return crown_bounds.lower
    
    def create_bound_matrices(self, state):
        action_space = self.env.action_space
        action_dimensionality = action_space.shape[0]
        res = []
        for action_partition in self.action_partitions:
            (A, b) = self.get_lower_bound(state, action_partition)
            h_action_dependent = A[:, :, -action_dimensionality:]
            # State input region is a hyperrectangle with "radius" 0.01
            state_input_bounds = HyperRectangle.from_eps(state, 0.01)
            # State dependent part of the A matrix
            state_a = A[:, :, :-action_dimensionality]
            # Make this into a (lower) linear bounds (\underbar{A}_x x + b \leq ...)
            state_linear_bounds = LinearBounds(state_input_bounds, (state_a, b), None)
            # Convert to lower interval bounds (b \leq ...)
            state_interval_bounds = state_linear_bounds.concretize()
            # Select the lower bound
            h_vec = state_interval_bounds.lower

            vecs = (action_partition, h_action_dependent.squeeze().detach().numpy(), h_vec.squeeze().detach().numpy())
            res.append(vecs)

        return res
    
    def continuous_cbf(self, state):
        nominal_action = self.policy(state).squeeze(0).detach()
        h_current = self.h_func(state)
        bound_matrices = self.create_bound_matrices(state)
        safe_actions = []
        for action_partition, h_action_dependent, h_vec in bound_matrices:
            num_actions = nominal_action.shape[0]
            action = cp.Variable(num_actions)

            # Constraints
            action_lower_bound = action_partition.lower.reshape((-1,))
            action_upper_bound = action_partition.upper.reshape((-1,))
            constraints = [action_lower_bound <= action, action <= action_upper_bound,
                           h_action_dependent @ action + h_vec >= self.alpha * h_current]

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
            return torch.tensor(min(safe_actions, key=lambda x: x[1], default=(None, None))[0])
        elif safe_actions:
            return torch.tensor(safe_actions[0])
        else:
            raise InfeasibilityError()
        
    def create_noise_partitions(self):
        # Define the limits for partitioning
        partitions_lower = [-6 * std for std in self.stds]
        partitions_upper = [6 * std for std in self.stds]
        # Create the partition slices for each dimension in h_ids
        partition_slices = []
        for dim_num_slices, dim_min, dim_max in zip([self.noise_partitions] * len(self.h_ids), partitions_lower, partitions_upper):
            dim = torch.linspace(dim_min, dim_max, dim_num_slices + 1)
            centers = (dim[:-1] + dim[1:]) / 2
            half_widths = (dim[1:] - dim[:-1]) / 2
            partition_slices.append(list(zip(centers, half_widths)))

        # Create all combinations of partitions across the dimensions in h_ids
        hyperrectangles = []
        for combination in product(*partition_slices):
            lower_bounds = torch.zeros((1, self.state_size))
            upper_bounds = torch.zeros((1, self.state_size))

            for (center, half_width), h_id in zip(combination, self.h_ids):
                lower_bounds[0, h_id] = center - half_width
                upper_bounds[0, h_id] = center + half_width

            hyperrectangles.append(HyperRectangle(lower_bounds, upper_bounds))

        return hyperrectangles
    

    def weighted_noise_prob(self, HR):
        res = torch.tensor(len(self.h_ids))
        HR_prob = self.HR_probability(HR)
        for i in range(len(self.h_ids)):
            res[i] = HR_prob * truncated_normal_expectation(0, self.stds[i], HR.lower[i], HR.upper[i])
        return res

    def HR_probability(self, HR):
        lower_list = []
        upper_list = []
        len_vector = len(self.h_ids)
        for i in range(len_vector):
            lower_list += [HR.lower[i]/self.stds[i]]
            upper_list += [HR.upper[i]/self.stds[i]]
        prob = 0
        for j in range(len_vector):
            prob += (log_prob(upper_list[j], lower_list[j]))
        return np.exp(prob)

    def create_noise_bounds(self, state):
        action_dimensionality = self.env.action_space[0]
        x_inds = range(self.state_size)
        u_inds = range(self.state_size, self.state_size + action_dimensionality)
        w_inds = range(self.state_size + action_dimensionality, 2*self.state_size + action_dimensionality)
        # h_ids are the dimensions of the state that are used in h
        h_dim = len(self.h_ids)
        res = []
        for action_partition in self.action_partitions:
            # state input region is a hyperrectangle with "radius" 0.01
            state_input_bounds = HyperRectangle.from_eps(state.view(1, -1), 0.01)
            # initialise the part of the bound on h that is dependend on the action
            h_action_dependend = torch.zeros(1, h_dim, len(u_inds))
            # initialise the part of the bound on h that is INdependend on the action
            h_vec = torch.zeros(1, h_dim)
            for noise_partition in self.noise_partitions:
                # input region is a hyperrectangle with the state bounds and the noise + action partitions
                input_bounds = HyperRectangle(torch.cat((state_input_bounds.lower, action_partition.lower, noise_partition.lower), dim=1),
                                            torch.cat((state_input_bounds.upper, action_partition.upper, noise_partition.upper), dim=1))
                crown_bounds = self.bounded_NNDM_H.crown(input_bounds, bound_upper=False)

                # Get the lower bounds
                (A, b) = crown_bounds.lower

                # State, action, and noise dependent part of the A matrix
                state_A = A[:, :, x_inds]
                action_A = A[:, :, u_inds]
                noise_A = A[:, :, w_inds]

                # compute the probability of the the noise falling in the given partition of the noise space
                noise_prob = self.HR_probability(noise_partition)
                noise_prob = noise_prob.item()

                # Scale state_A and b corresponding to noise_prob
                state_A, b = noise_prob * state_A, noise_prob * b
                # Make this into a (lower) linear bounds (\underbar{A}_x x + b \leq ...)
                state_linear_bounds = LinearBounds(state_input_bounds, (state_A, b), None)
                # Convert to lower interval bounds (b \leq ...)
                state_interval_bounds = state_linear_bounds.concretize()
                # Select the lower bound
                h_vec_state = state_interval_bounds.lower.detach()

                # compute \int_{HR_{wi}} w \, p(w) \, dw
                # TODO: implement log exp trick
                std = 1 # TODO: This should be changed to use self.stds in the weighted noise_prob
                weighted_noise_prob = 1/(math.sqrt(2*math.pi) * std) * (torch.exp(torch.square(noise_partition.upper / std)/-2) - torch.exp(torch.square(noise_partition.lower / std)/-2))
                # The part of the bound on h that is dependend on the noise
                h_vec_noise = noise_A @ weighted_noise_prob.squeeze(0)
                # the part of the bound on h that is independend on the action
                h_vec +=  h_vec_state + h_vec_noise

                # the weighted part of the bound on h that is dependend on the action
                h_action_dependend += noise_prob * action_A
            res.append((action_partition, h_action_dependend.squeeze().detach().numpy(), h_vec.squeeze().detach().numpy()))
        return res
    
# TODO: Maybe move this function?
def log_prob(x, y):
    x, y = x/np.sqrt(2), y/np.sqrt(2)
    if abs(x) <= 1/np.sqrt(2) and abs(y) <= 1/np.sqrt(2):
        return np.log((erf(y) - erf(x))/2)
    elif x >= 0 and y >= 0:
        return np.log((erfc(x) - erfc(y))/2)
    elif x <= 0 and y <= 0:
        return np.log((erfc(-y) - erfc(-x))/2)
    else:
        return np.log((erf(y) - erf(x))/2)
    
# TODO: Maybe move this function?
def truncated_normal_expectation(mean, std_dev, lower_bound, upper_bound):
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
    return mean + std_dev * (truncnorm.expect(args=(a, b), loc=mean, scale=std_dev))