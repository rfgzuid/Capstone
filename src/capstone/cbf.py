from .settings import Env
from .dqn import DQN
from .ddpg import Actor
from .nndm import NNDM
from .barriers import NNDM_H, stochastic_NNDM_H
from .probability import HR_probability, weighted_noise_prob

from bound_propagation import BoundModelFactory, HyperRectangle
from bound_propagation.bounds import LinearBounds

import torch
import cvxpy as cp

from itertools import product


class InfeasibilityError(Exception):
    """Exception raised if there are no actions that fulfill the safety criteria."""

    def __init__(self):
        super().__init__()
        self.message = "No safe action to take"


class CBF:
    def __init__(self, env: Env, nndm: NNDM, policy: DQN | Actor,
                 alpha: list[float], delta: list[float],
                 no_action_partitions: int = 4, no_noise_partitions=2, stochastic=False):
        self.env = env.env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = 1 if env.is_discrete else self.env.action_space.shape[0]

        self.is_discrete = env.is_discrete
        self.settings = env.settings
        self.h_func = env.h_function

        self.is_stochastic = stochastic
        self.nndm = nndm

        if not self.is_stochastic:
            self.NNDM_H = NNDM_H(env, self.nndm)
        elif self.is_stochastic:
            self.x_inds = range(self.state_size)
            self.u_inds = range(self.state_size, self.state_size + self.action_size)
            self.w_inds = range(self.state_size + self.action_size, 2 * self.state_size + self.action_size)
            xu_inds = list(self.x_inds) + list(self.u_inds)
            self.NNDM_H = stochastic_NNDM_H(env, self.nndm, xu_inds, self.w_inds)

        self.policy = policy

        self.alpha = torch.tensor(alpha)
        self.delta = torch.tensor(delta)

        self.h_ids = env.h_ids
        self.stds = env.std

        factory = BoundModelFactory()
        self.bounded_NNDM_H = factory.build(self.NNDM_H)
        self.action_partitions = self.create_action_partitions(no_action_partitions)
        if self.is_stochastic:
            self.no_noise_partitions = no_noise_partitions
            self.noise_partitions = self.create_noise_partitions()

    def safe_action(self, state: torch.tensor):
        h_cur = self.h_func(state)
        nominal_action = self.policy.select_action(state, exploration=False)

        h_input = torch.zeros((1, self.state_size + self.action_size))
        h_input[:, :self.state_size] = state
        h_input[:, self.state_size:] = nominal_action

        if self.is_stochastic:
            h_input = torch.cat((h_input, torch.zeros(1, self.state_size)), dim=1)
        h_next = self.NNDM_H(h_input)

        if torch.all(h_next >= self.alpha * h_cur + self.delta).item():
            # nominal action is safe, no need to use cbf functions
            return nominal_action

        if self.is_discrete and not self.is_stochastic:
            return self.discrete_cbf(state)
        elif self.is_discrete and self.is_stochastic:
            return self.discrete_scbf(state)
        elif not self.is_discrete and not self.is_stochastic:
            return self.continuous_cbf(state)
        else:
            return self.continuous_scbf(state)

    def discrete_cbf(self, state):
        # Discrete(n) has actions {0, 1, ..., n-1} - see gymnasium docs
        action_space = torch.arange(self.env.action_space.n)
        safe_actions = []

        h_cur = self.h_func(state)

        for action in action_space:
            h_input = torch.zeros((1, self.state_size + self.action_size))
            h_input[:, :self.state_size] = state
            h_input[:, self.state_size] = action

            h_next = self.NNDM_H(h_input)

            if torch.all(h_next >= self.alpha * h_cur + self.delta).item():
                safe_actions.append(action)

        if safe_actions and len(safe_actions) > 1:
            q_values = self.policy(state).squeeze()
            mask = torch.zeros_like(q_values, dtype=torch.bool)

            for action in safe_actions:
                mask[action] = True

            safe_q_values = q_values.masked_fill(~mask, float('-inf'))
            best_action = torch.argmax(safe_q_values)

            return best_action.view(1, 1)
        elif safe_actions:
            return safe_actions[0].view(1, 1)
        else:
            raise InfeasibilityError()
        
    # def discrete_scbf(self, state):
    #     # Discrete(n) has actions {0, 1, ..., n-1} - see gymnasium docs
    #     action_space = list(range(self.env.action_space.n))
    #     safe_actions = []

    #     h_cur = self.h_func(state)

    #     for action in action_space:
    #         action_vec = torch.zeros(self.env.action_space.n)
    #         action_vec[action] = 1

    #         # h_input = torch.zeros((1, 2 * self.state_size + self.env.action_space.n))
    #         # h_input[:, :self.state_size] = state
    #         # h_input[:, self.state_size:self.state_size + self.env.action_space.n] = action_vec
    #         # hyperrec1 = HyperRectangle.from_eps(h_input, 0.01)
    #         state_input_bounds = HyperRectangle.from_eps(state, 0.01)
    #         action_input_bounds = HyperRectangle.from_eps(action_vec, 0.01)

    #         for noise_partition in self.noise_partitions:
    #             input_bounds = HyperRectangle(torch.cat([state_input_bounds.lower, action_input_bounds.lower, noise_partition.lower], dim=1,),
    #                                           torch.cat([state_input_bounds.upper, action_input_bounds.upper, noise_partition.upper], dim=1))
                
    #             crown_bounds = self.bounded_NNDM_H.crown(input_bounds, bound_upper=False)

    #             # Get the lower bounds
    #             (A, b) = crown_bounds.lower

    #             # State, action, and noise dependent part of the A matrix
    #             state_A = A[:, :, self.x_inds]
    #             action_A = A[:, :, self.u_inds]
    #             noise_A = A[:, :, self.w_inds]

    #             # compute the probability of the noise falling in the given partition of the noise space
    #             noise_prob = HR_probability(noise_partition, self.h_ids, self.stds)
    #             noise_prob = noise_prob.item()

    #             # Scale state_A and b corresponding to noise_prob
    #             state_A, b = noise_prob * state_A, noise_prob * b
    #             # Make this into a (lower) linear bounds (\underbar{A}_x x + b \leq ...)
    #             state_action_input_bounds = 
    #             state_linear_bounds = LinearBounds(state_input_bounds, (state_A, b), None)
    #             # Convert to lower interval bounds (b \leq ...)
    #             state_interval_bounds = state_linear_bounds.concretize()
    #             # Select the lower bound
    #             h_vec_state = state_interval_bounds.lower.detach()

    #             # compute \int_{HR_{wi}} w \, p(w) \, dw
    #             weighted_noise_proba = weighted_noise_prob(noise_partition, self.h_ids, self.stds)
    #             # The part of the bound on h that is dependent on the noise
    #             print("noise_A", noise_A.squeeze(0).shape)
    #             print("weighted_noise_proba.squeeze(0)", weighted_noise_proba.squeeze(0).shape)
    #             h_vec_noise = noise_A.squeeze(0) @ weighted_noise_proba.squeeze(0)
    #             # the part of the bound on h that is independent on the action
    #             h_vec += h_vec_state + h_vec_noise

    #             # the weighted part of the bound on h that is dependent on the action
    #             h_action_dependent += noise_prob * action_A

    #         res.append((action_partition, h_action_dependent.squeeze().detach().numpy(),
    #                     h_vec.squeeze().detach().numpy()))
                

    #         h_next = self.NNDM_H(h_input)

    #         if torch.all(h_next >= self.alpha * h_cur + self.delta).item():
    #             safe_actions.append(action)

    #     if safe_actions and len(safe_actions) > 1:
    #         q_values = self.policy(state).squeeze()
    #         mask = torch.zeros_like(q_values, dtype=torch.bool)

    #         for action in safe_actions:
    #             mask[action] = True

    #         safe_q_values = q_values.masked_fill(~mask, float('-inf'))
    #         best_action = torch.argmax(safe_q_values)

    #         return best_action.view(1, 1)
    #     elif safe_actions:
    #         return safe_actions[0].view(1, 1)
    #     else:
    #         raise InfeasibilityError()
    
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
                    generate_partitions(dimensions + 1, lower + [dim_lower_bound],
                                        upper + [dim_upper_bound], current_partition)

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
        return self.QP_solver(nominal_action, bound_matrices, h_current)
    
    def continuous_scbf(self, state):
        nominal_action = self.policy(state).squeeze(0).detach()
        h_current = self.h_func(state)
        bound_matrices = self.create_noise_bounds(state)
        return self.QP_solver(nominal_action, bound_matrices, h_current)

    def QP_solver(self, nominal_action, bound_matrices, h_current):
        safe_actions = []
        for action_partition, h_action_dependent, h_vec in bound_matrices:
            num_actions = nominal_action.shape[0]
            action = cp.Variable(num_actions)

            # Constraints
            action_lower_bound = action_partition.lower.reshape((-1,))
            action_upper_bound = action_partition.upper.reshape((-1,))
            constraints = [action_lower_bound <= action, action <= action_upper_bound,
                           (h_action_dependent @ action + h_vec).squeeze() >= self.alpha * h_current + self.delta]

            # Objective
            objective = cp.Minimize(cp.norm(action - nominal_action, 2))

            # Solve the problem, using ECOS as the default solver for small scale QP
            problem = cp.Problem(objective, constraints)
            problem.solve(solver='ECOS')

            if problem.status is cp.UNBOUNDED:
                print("something goes very wrong")
            elif problem.status is cp.INFEASIBLE:
                pass
            else:
                safe_actions.append((action.value, objective.value))

        if safe_actions and len(safe_actions) > 1:
            return torch.tensor(min(safe_actions, key=lambda x: x[1], default=(None, None))[0])
        elif safe_actions:
            return torch.tensor(safe_actions[0][0])
        else:
            raise InfeasibilityError()
        
    def create_noise_partitions(self):
        # Define the limits for partitioning
        partitions_lower = [-6 * std for std in self.stds]
        partitions_upper = [6 * std for std in self.stds]
        # Create the partition slices for each dimension in h_ids
        partition_slices = []
        for dim_num_slices, dim_min, dim_max in zip([self.no_noise_partitions] * len(self.h_ids),
                                                    partitions_lower, partitions_upper):
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

    def create_noise_bounds(self, state):
        # h_ids are the dimensions of the state that are used in h
        h_dim = len(self.h_ids)
        res = []
        for action_partition in self.action_partitions:
            # state input region is a hyperrectangle with "radius" 0.01
            state_input_bounds = HyperRectangle.from_eps(state.view(1, -1), 0.01)
            # initialise the part of the bound on h that is dependent on the action
            h_action_dependent = torch.zeros(1, h_dim, len(self.u_inds))
            # initialise the part of the bound on h that is independent on the action
            h_vec = torch.zeros(1, h_dim)
            for noise_partition in self.noise_partitions:
                # input region is a hyperrectangle with the state bounds and the noise + action partitions
                input_bounds = HyperRectangle(
                    torch.cat((state_input_bounds.lower, action_partition.lower, noise_partition.lower), dim=1),
                    torch.cat((state_input_bounds.upper, action_partition.upper, noise_partition.upper), dim=1)
                )
                crown_bounds = self.bounded_NNDM_H.crown(input_bounds, bound_upper=False)

                # Get the lower bounds
                (A, b) = crown_bounds.lower

                # State, action, and noise dependent part of the A matrix
                state_A = A[:, :, self.x_inds]
                action_A = A[:, :, self.u_inds]
                noise_A = A[:, :, self.w_inds]

                # compute the probability of the noise falling in the given partition of the noise space
                noise_prob = HR_probability(noise_partition, self.h_ids, self.stds)
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
                weighted_noise_proba = weighted_noise_prob(noise_partition, self.h_ids, self.stds)

                noise_vec = torch.zeros(self.state_size)
                for i, ind in enumerate(self.h_ids):
                    noise_vec[ind] = weighted_noise_proba[i]
                # The part of the bound on h that is dependent on the noise
                h_vec_noise = noise_A.squeeze(0) @ noise_vec
                # the part of the bound on h that is independent on the action
                h_vec += h_vec_state + h_vec_noise

                # the weighted part of the bound on h that is dependent on the action
                h_action_dependent += noise_prob * action_A

            res.append((action_partition, h_action_dependent.squeeze().detach().numpy(),
                        h_vec.squeeze().detach().numpy()))
            
        return res
