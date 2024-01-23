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


class HHead(nn.Sequential):
    ...


class H(nn.Sequential):
    def __init__(self, nndm: NNDM):
        super(H, self).__init__(
            ('nndm', nndm),
            ('noise', Gaussian(std=0.1)),
            ('hhead', ...)
        )
