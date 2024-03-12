from capstone.evaluation import Evaluator
from capstone.cbf import CBF, InfeasibilityError
from capstone.settings import Env


from bound_propagation.parallel import Parallel
from bound_propagation.bivariate import Mul
from bound_propagation.reshape import Select
from bound_propagation.polynomial import UnivariateMonomial
from bound_propagation.linear import FixedLinear
from bound_propagation.activation import Sin

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

class PendulumNoise(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=np.array([float('-inf'), -8]),
                                     high=np.array([float('inf'), 8]),
                                     dtype=np.float32)
        
        self.dt = 0.01
        
    def step(self, action):
        action = np.array([action], dtype=np.float32)  # cannot be scalar like used in Discrete gym envs
        
        th, thdot = self.env.unwrapped.state
        _, _, _, truncated, _ = self.env.step(action)
        
        u = np.clip(action, -self.env.unwrapped.max_torque, 
                    self.env.unwrapped.max_torque)[0]
        
        newthdot = thdot + np.sin(th) * self.dt + u * self.dt + np.random.normal(0., 0.025)
        newth = th + thdot * self.dt + np.random.normal(0., 0.005)
        
        self.env.unwrapped.state = np.array([newth, newthdot], dtype=np.float32)
        
        return self.env.unwrapped.state, 0., False, truncated, {}
    
    def reset(self, seed: list[float, float] = None):
        # allow the user to set the initial state directly, otherwise random state
        self.env.reset()
        
        if seed is not None:
            self.env.unwrapped.state = np.array(seed, dtype=np.float32)
        return self.env.unwrapped.state, {}
    
class NNDM(nn.Sequential):
    # input [theta, theta_dot, u]
    
    def __init__(self):
        self.dt = 0.01
        
        super(NNDM, self).__init__(
            # outputs [theta{k+1}, theta{k}, theta_dot{k}, u{k}, sin() of previous 3 elements]
            Parallel(
                FixedLinear(
                    torch.tensor([
                        [1, self.dt, 0.]
                    ])
                ),
                
                Parallel(
                    FixedLinear(
                        torch.tensor([
                            [1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]
                        ])
                    ),
                    Sin(),
                )
            ),
            
            # now calculate theta_dot{k+1}, output [theta{k+1}, theta_dot{k+1}]
            Parallel(
                Select(0),
                FixedLinear(
                    torch.tensor([
                        [0., 0., 1., self.dt, self.dt, 0., 0.]
                    ])
                )
            )
        )

nndm = NNDM()

class Agent(nn.Module):
    # dummy agent of linear form a = c1 * s1 + c2 * s2 + c3, always outputs 0.
    
    def __init__(self):
        super(Agent, self).__init__()
        
        self.layer = nn.Linear(2, 1)
        
        # dummy output of u=0
        self.layer.weight = nn.Parameter(torch.tensor([[0., 0.]]))
        self.layer.bias = nn.Parameter(torch.tensor([0.]))
    
    def forward(self, x):
        return self.layer(x)
    
    def select_action(self, x, exploration=False):
        if exploration:
            raise ValueError('This model is not implemented for exploration')
        else:
            return self.forward(x)
        
policy = Agent()

class Pendulum(Env):
    def __init__(self):
        env = gym.make('Pendulum-v1')

        self.is_discrete = False
        
        self.settings = {
            'noise': [],
            'max_frames': 100
        }
        
        # h as defined in the paper
        self.h_function = nn.Sequential(
            Parallel(
                UnivariateMonomial([(0, 2)]),
                UnivariateMonomial([(1, 2)]),
                Mul(Select([0]), Select([1]))
            ),
            FixedLinear(
                torch.tensor([[1., 1., 2./(3.**0.5)]])
            ),
            FixedLinear(
                torch.tensor([[-36 / np.pi**2]]),
                torch.tensor([1.])
            )
        )
        
        self.h_ids = [0, 1]
        self.std = [0.005, 0.025]
        self.env = PendulumNoise(env)

env = Pendulum()

lambda_max = (1/np.pi**2) * (6912 + 3456 * 3 ** 0.5) ** 0.5
tr_cov = sum(noise ** 2 for noise in env.std)

psi = (lambda_max / 2) * tr_cov
alpha = 1 - psi
print(f'Alpha is {round(alpha, 3)}')

# what to do with delta? 
cbf = CBF(env, nndm, policy,
          alpha=[alpha],
          delta=[psi],
          no_action_partitions=8,
          no_noise_partitions=32,
          stochastic=True)

evaluator = Evaluator(env, cbf)
f, h = evaluator.mc_simulate(policy, 10, seed=[0.1, 0.])

def p_u_theoretical(xg, yg):
    res = np.zeros([xg.shape[1], yg.shape[0]])
    
    x_y = np.dstack((xg, yg))
    gridpoints = [point for row in x_y for point in row]
    h_values = [env.h_function(torch.tensor(point, dtype=torch.float32)) for point in gridpoints]
     
    idx = 0  # could have used enumerate(gridpoints), but this would disallow the tqdm progress bar
    for point in tqdm(gridpoints):
        p_u = 1 - (env.h_function(torch.tensor(point, dtype=torch.float32)).item() * alpha ** 100)
        
        i, j = idx % x_y.shape[1], idx // x_y.shape[0]
        res[j, i] = p_u
        
        idx += 1
    
    clipped_res = np.clip(res, 0., 1.)  
    return clipped_res

def p_u_experimental(xg, yg, num_agents=500):
    res = np.zeros([xg.shape[1], yg.shape[0]])

    x_y = np.dstack((xg, yg))
    gridpoints = [point for row in x_y for point in row]
    
    idx = 0
    for point in tqdm(gridpoints):
        end_frames, _ = evaluator.mc_simulate(policy, num_agents, cbf_enabled=True, seed=list(point), progress_bar=False)
        p_u = sum(f <= 100 for f in end_frames)/len(end_frames) if len(end_frames) > 0 else 0.
        
        i, j = idx % x_y.shape[1], idx // x_y.shape[0]
        res[i, j] = p_u
        
        idx += 1     
    return res

def state_space_plot(interp_resolution, num_agents=500):
    # define regions (left of the ellipse, ellipse and right of the ellipse)
    x_ellipse = np.linspace(-(3/2)**0.5 * np.pi/6 , (3/2)**0.5 * np.pi/6 , 1001)
    x_left = np.linspace(-0.75, -(3/2)**0.5 * np.pi/6, 101)
    x_right = np.linspace((3/2)**0.5 * np.pi/6, 0.75, 101)
    
    upper_ellipse = [-x/3**0.5 + (-2/3 * x**2 + np.pi**2 / 36)**0.5 for x in x_ellipse
                    if -(3/2)**0.5 * np.pi/6 <= x <= (3/2)**0.5 * np.pi/6]
    lower_ellipse = [-x/3**0.5 - (-2/3 * x**2 + np.pi**2 / 36)**0.5 for x in x_ellipse
                    if -(3/2)**0.5 * np.pi/6 <= x <= (3/2)**0.5 * np.pi/6]
        
    # points to evaluate simulation at
    x, y = np.linspace(-0.7, 0.7, 7), np.linspace(-0.7, 0.7, 15)
    xg, yg = np.meshgrid(x, y)
    
    experimental = p_u_experimental(xg, yg, num_agents)
    exp_interp = RegularGridInterpolator((x, y), experimental, method='cubic')

    # points to interpolate at
    xx, yy = np.linspace(x_ellipse[0], x_ellipse[-1], interp_resolution), np.linspace(-0.7, 0.7, interp_resolution)
    X, Y  = np.meshgrid(xx, yy)
    
    theoretical = p_u_theoretical(X, Y)
    
    
    fig = plt.figure(figsize=(8, 3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1/0.85])
    
    ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    
    ax[0].scatter(X, Y, c=exp_interp((X, Y)), cmap=plt.colormaps['coolwarm'], vmin=0, vmax=1)
    im = ax[1].scatter(X, Y, c=theoretical, cmap=plt.colormaps['coolwarm'], vmin=0, vmax=1)
    
    ax[0].plot(x_ellipse, upper_ellipse, color='black')
    ax[0].plot(x_ellipse, lower_ellipse, color='black') 
    ax[1].plot(x_ellipse, upper_ellipse, color='black')
    ax[1].plot(x_ellipse, lower_ellipse, color='black')  
    
    # Plots the color map here, some size scaling that worked 
    ax[0].scatter(xg, yg, s=4, color='black', label=r'$\mathbf{x}_0$')
    ax[1].scatter(xg, yg, s=4, color='black', label=r'$\mathbf{x}_0$')
    ax[0].legend(loc=1)
    
    # shade outer regions
    outer_color = plt.colormaps['coolwarm'](1.)[:3]
    ax[0].fill_between(x_ellipse, 0.75, upper_ellipse, color=outer_color, alpha=1)
    ax[0].fill_between(x_ellipse, lower_ellipse, -0.75, color=outer_color, alpha=1)
    ax[0].fill_between(x_left, -0.75, 0.75, color=outer_color, alpha=1)
    ax[0].fill_between(x_right, -0.75, 0.75, color=outer_color, alpha=1)
    ax[1].fill_between(x_ellipse, 0.75, upper_ellipse, color=outer_color, alpha=1)
    ax[1].fill_between(x_ellipse, lower_ellipse, -0.75, color=outer_color, alpha=1)
    ax[1].fill_between(x_left, -0.75, 0.75, color=outer_color, alpha=1)
    ax[1].fill_between(x_right, -0.75, 0.75, color=outer_color, alpha=1)
    
    ax[0].set_xlim(-0.7, 0.7)
    ax[0].set_ylim(-0.7, 0.7)
    ax[1].set_xlim(-0.7, 0.7)
    ax[1].set_ylim(-0.7, 0.7)
    
    ax[0].set_title(r'Estimated $P_u$')
    ax[1].set_title(r'$P_u$ bound')
    
    ax[0].set_xticks([-0.5, 0., 0.5], labels=['-0.5', r'$\theta$', '0.5'])
    ax[0].set_yticks([-0.5, 0., 0.5], labels=['-0.5', r'$\dot{\theta}$', '0.5'])
    ax[1].set_xticks([-0.5, 0., 0.5], labels=['-0.5', r'$\theta$', '0.5'])
    ax[1].set_yticks([])
    
    fig.tight_layout()
    fig.colorbar(im)
    
    plt.savefig("p_u_plot.png")
    plt.show()

state_space_plot(200, 1)

def plot_trajectories(num_agents, start_state, cbf_enabled=False):
    # plot the trajectories of the system, starting from the origin

    if start_state is None:
        start_state = [0., 0.]
    states = []

    for _ in tqdm(range(num_agents)):
        state_list = []
        state, _ = env.env.reset(seed=start_state)
        state = torch.tensor(state).unsqueeze(0)

        done = False

        while not done:   
            state_list.append(state.squeeze())

            # try cbf action - if cbf disabled or no safe actions available, just follow agent policy
            if cbf_enabled:
                try:
                    action = cbf.safe_action(state)
                except InfeasibilityError:
                    action = policy.select_action(state, exploration=False)
            else:
                action = policy.select_action(state, exploration=False)

            state, reward, terminated, truncated, _ = env.env.step(action.squeeze().detach().numpy())
            state = torch.tensor(state).unsqueeze(0)

            if torch.any(env.h_function(state.unsqueeze(0)) < 0).item():
                terminated = True

            done = terminated or truncated

        states.append(np.array(state_list))

    env.env.close()
    
    
    x_ellipse = np.linspace(-(3/2)**0.5 * np.pi/6 , (3/2)**0.5 * np.pi/6 , 1001)
    
    upper_ellipse = [-x/3**0.5 + (-2/3 * x**2 + np.pi**2 / 36)**0.5 for x in x_ellipse
                    if -(3/2)**0.5 * np.pi/6 <= x <= (3/2)**0.5 * np.pi/6]
    lower_ellipse = [-x/3**0.5 - (-2/3 * x**2 + np.pi**2 / 36)**0.5 for x in x_ellipse
                    if -(3/2)**0.5 * np.pi/6 <= x <= (3/2)**0.5 * np.pi/6]
    
    plt.plot(x_ellipse, upper_ellipse, color='black')
    plt.plot(x_ellipse, lower_ellipse, color='black') 
    plt.plot(x_ellipse, upper_ellipse, color='black')
    plt.plot(x_ellipse, lower_ellipse, color='black') 
    
    plt.fill_between(x_ellipse, lower_ellipse, upper_ellipse, color='g', alpha=0.3)
    
    for trajectory in states:
        # https://stackoverflow.com/questions/21519203/plotting-a-list-of-x-y-coordinates 
        x, y = zip(*trajectory)
        plt.plot(x, y, c='b', alpha=0.1)
        
    plt.savefig("trajectory_plot.png")
    plt.show()

# states_lst = plot_trajectories(500, [0., 0.], cbf_enabled=True)