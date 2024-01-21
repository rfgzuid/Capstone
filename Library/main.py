"""
Library TODO:
- Implement DDPG for continuous actions from Wessels bipedal notebook
- Is GPU necessary? Matej said due to the sequential nature of env simulation (frame by frame)
  it might not help that much
  https://spinningup.openai.com/en/latest/algorithms/ddpg.html#deep-deterministic-policy-gradient
  "The Spinning Up implementation of DDPG does not support parallelization"
- Convert NNs to nn.Sequential to fit with linear bound propagation library
- More metric monitoring/plotting in the training pipeline
- Implement the (S)CBF code for both discrete and continuous action environments
- Implement the evaluation metrics for CBF in evaluations.py

- Add docstrings to files, classes & functions, and add extra (line) comments where necessary
- Add type hints (is it needed? Most variables originate from the env_settings file)
- Add code sources (for both DDQN and DDPG) + articles on which the code is based
  These sources are also going to be cited when we have to justify agent architectures (e.g. cartpole & bipedal walker)
  Also: Frederik gave a lot of RL architecture/training tips that we could briefly mention we used
- Correctly format the library on github
(requirements.txt, readme.txt, .gitignore to save files we won't update - like model weights, __init__.py script)
"""

import torch

from settings import Cartpole, LunarLander
from training import Trainer

# set a seed for reproducibility (does this influence the other files' seed as well?)
torch.manual_seed(42)

# env = Cartpole()
env = LunarLander()

pipeline = Trainer(env)
policy, nndm = pipeline.train()

# torch.save(policy.state_dict(), 'Capstone/');
# torch.save(nndm.state_dict(), 'Capstone/'))
# pipeline.play(trained_policy)

# termination_frames = pipeline.evaluate(trained_policy)
