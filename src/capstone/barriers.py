from .settings import Env
from .nndm import NNDM

import torch.nn as nn


class H(nn.Sequential):
    def __init__(self, nndm: NNDM):
        super(H, self).__init__(
            ('nndm', nndm),
            ('hhead', ...)
        )
