import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.base import CliffordModule

class BladeSelector(CliffordModule):
    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """
        Learns to weight specific grades or basis blades.
        Acts as a soft projection or feature selection.
        """
        super().__init__(algebra)
        
        # Learn a weight for each basis element per channel
        # Weights: [Channels, Dim]
        self.weights = nn.Parameter(torch.Tensor(channels, algebra.dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Channels, Dim]
        """
        # Element-wise multiplication
        # w: [1, C, D]
        w = torch.sigmoid(self.weights).unsqueeze(0)
        return x * w
