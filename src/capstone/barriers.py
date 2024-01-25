from .settings import Env
from .nndm import NNDM

import torch.nn as nn


class NNDM_H(nn.Sequential):
    """
    Input: (s, a) pair
    Output: h vector of the next predicted state s'
    """
    def __init__(self, env: Env, nndm: NNDM):
        super(NNDM_H, self).__init__(
            nndm,
            env.h_function
        )
