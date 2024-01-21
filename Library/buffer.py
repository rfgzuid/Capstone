import random
from collections import namedtuple, deque


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_terminal'))


class ReplayMemory(object):
    """
    A memory of size [capacity] in which agent experiences are stored.
    The experiences are formatted as a Transition namedtuple - see above
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> Transition:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
