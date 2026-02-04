import torch.nn as nn
from core.algebra import CliffordAlgebra

class CliffordModule(nn.Module):
    def __init__(self, algebra: CliffordAlgebra):
        super().__init__()
        self.algebra = algebra

    def forward(self, x):
        raise NotImplementedError
