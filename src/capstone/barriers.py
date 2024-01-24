from .settings import Env
from .nndm import NNDM

import torch.nn as nn
import torch


class Gaussian(nn.Module):
    def __init__(self, std=0.1):
        super(Gaussian, self).__init__()
        self.std = std

    def forward(self, x):
        return x + torch.normal(mean=0., std=self.std, size=x.shape)

class NNDM_H(nn.Sequential):
    """
    Input: (s, a) pair
    Output: h vector of the next predicted state s'
    """
    def __init__(self, env: Env, nndm: NNDM):
        super(H, self).__init__(
            nndm,
            Gaussian(std=0.1),
            env.h_function
        )
