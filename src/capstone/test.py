from settings import DiscreteLunarLander
import gymnasium as gym

specs = DiscreteLunarLander().env.spec
specs.kwargs['render_mode'] = 'human'

env = gym.make(specs)

def test(env):
    state, _ = env.reset(seed=42)
    initial_pos = [(8, 5), (5, 8), (8, 8), (10, 10)]

    for i in range(100):
        action = 0

        state, reward, _, _, _ = env.step(action)
        env.unwrapped.lander.position = initial_pos[i % 4]

test(env)
