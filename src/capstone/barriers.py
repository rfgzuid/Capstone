from .settings import Env
from .nndm import NNDM

import torch.nn as nn
from bound_propagation.bivariate import Add
from bound_propagation.reshape import Select
from collections import OrderedDict


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

class stochastic_NNDM_H(nn.Sequential):
    def __init__(self, xu_inds, nndm, w_inds, h):
        super().__init__(OrderedDict([
            ('nndm(x, y) + w', Add(
                nn.Sequential(
                    Select(xu_inds),  # Select (x, u) from input hyperrectangle)
                    nndm
                ),
                Select(w_inds)  # Select w from input hyperrectangle
            )),  # The output of Add is nndm(x, u) + w
            ('h', h)  # Feed through h
        ]))