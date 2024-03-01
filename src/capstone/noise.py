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



# Based on https://arxiv.org/abs/2302.07469

class DoubleIntegratorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, action_space = "continuous", render_mode="human", mass=1.0):
        super(DoubleIntegratorEnv, self).__init__()
        self.mass_square = mass #kg
        self.Ts = 0.05 #s 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.max_distance = 1

        self.A = np.array(
            [
                [1, 0, self.Ts, 0],
                [0, 1, 0, self.Ts],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        self.B = np.array(
            [
                [((self.Ts**2)/2), 0],
                [0, ((self.Ts**2)/2)],
                [self.Ts, 0],
                [0, self.Ts]
            ]
        )

        self.max_velocity = np.inf
        self.max_force = np.inf

        self.observation_space = Box(
            low = np.array([-self.max_distance, -self.max_distance, -self.max_velocity, -self.max_velocity]),
            high = np.array([self.max_distance, self.max_distance, self.max_velocity, self.max_velocity]),
            dtype=np.float32
        )
        self.continuous = None

        if action_space == "discrete" or action_space == "continuous":
            if action_space == "discrete":
                self.max_action = 2
                self.action_space = spaces.Discrete(1)
                self.continuous = False
            else:
                self.max_action = np.inf
                self.action_space = Box(
                    low = np.array([-self.max_action, -self.max_action]),
                    high = np.array([self.max_action, self.max_action]),
                    dtype=np.float32
                )
                self.continuous = True
        else:
            raise ValueError("The action space can either be discrete or continuous. Please give a valid action space: ")

        self.reward = 0.0

        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None

    
    def random_gaussian_noise(self):
        """Returns a sample of a random gaussian noise with zero mean"""
        Q = np.matmul(self.B, self.B.T)
        distribution = np.random.multivariate_normal(np.array([0,0,0,0]), Q)
        distribution = distribution.reshape((1,4))
        return distribution
    
    def stepPhysics(self, force):
        dk = self.random_gaussian_noise()
        next_state = np.matmul(self.state, self.A) + np.expand_dims(np.matmul(self.B, force),0) + dk
        return next_state


    def step(self, action):
        """Action has to be numpy array of shape 2x1"""
        force = action
        self.state = self.stepPhysics(force)
        x = self.state[0][0].item()
        y = self.state[0][1].item()
        x_dot = self.state[0][2].item()
        y_dot = self.state[0][3].item()

        self.state = np.array([[x, y, x_dot, y_dot]], dtype=np.float32)

        terminated = x < -self.max_distance/2 or x > self.max_distance/2 or y < -self.max_distance/2 or y > self.max_distance/2
        terminated = bool(terminated)

        truncated = False

        if not terminated:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
terminated or truncated = True. You should always call 'reset()' once you receive 'terminated or truncated = True'
Any further steps are undefined behavior.
            """)
            self.steps_beyond_done += 1

        return self.state, self.reward, terminated, truncated, {}

    def reset(self, seed=None):
        x0, y0, x_dot0, y_dot0 = (0.0, 0.0, 0.0, 0.0) 
        self.state = np.array([[x0, y0, x_dot0, y_dot0]], dtype=np.float32)
        self.steps_beyond_done = None
        return self.state, {}


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()


        world_width = (self.max_distance+4) * 2
        scale_x = self.screen_width / world_width
        scale_y = self.screen_height/world_width
        cartwidth = 1.0 * scale_x
        cartheight = 1.0 * scale_y

        origin_x = self.screen_width / 2.0
        origin_y = self.screen_height /2.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0][0] * scale_x + origin_x 
        carty = x[0][1] * scale_y + origin_y 
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]

        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (255, 0, 0))
        
        scb_x = 0.5*scale_x # scaling for boundry lines

        scb_y = 0.5*scale_y # scaling for boundry lines
        
        gfxdraw.aapolygon(
            self.surf,
            [(origin_x+scb_x, origin_y+scb_y), (origin_x+scb_x, origin_y-scb_y), (origin_x-scb_x, origin_y-scb_y), (origin_x-scb_x, origin_y+scb_y)],
            (0,0,0)
        )
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )



    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            


