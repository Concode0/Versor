import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.base import CliffordModule

class RotorLayer(CliffordModule):
    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """
        Learns a rotor R = exp(-B/2) per channel to rotate input multivectors.
        Preserves lengths and angles (and grades, mostly).
        """
        super().__init__(algebra)
        self.channels = channels
        
        # We learn the bivector B directly.
        # Bivector coefficients only.
        # We need to know which indices are bivectors.
        self.bivector_indices = self._get_bivector_indices()
        self.num_bivectors = len(self.bivector_indices)
        
        # Weights: [Channels, Num_Bivectors]
        self.bivector_weights = nn.Parameter(torch.Tensor(channels, self.num_bivectors))
        
        self.reset_parameters()
        
    def _get_bivector_indices(self):
        indices = []
        for i in range(self.algebra.dim):
            # count set bits
            cnt = 0
            temp = i
            while temp > 0:
                if temp & 1: cnt += 1
                temp >>= 1
            if cnt == 2:
                indices.append(i)
        return torch.tensor(indices, device=self.algebra.device, dtype=torch.long)

    def reset_parameters(self):
        # Initialize to small random rotations (near identity)
        nn.init.normal_(self.bivector_weights, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Channels, Dim]
        """
        # 1. Construct Bivector B
        B = torch.zeros(self.channels, self.algebra.dim, device=x.device, dtype=x.dtype)
        
        # Scatter weights into bivector components
        # We need to broadcast indices [Num_Bi] to [Channels, Num_Bi]
        indices = self.bivector_indices.unsqueeze(0).expand(self.channels, -1)
        B.scatter_(1, indices, self.bivector_weights)
        
        # 2. Compute Rotor R = exp(-B/2)
        # Using Taylor series approximation
        # OPTIMIZATION: If Euclidean, we could use cos/sin if B^2 is negative scalar.
        R = self.algebra.exp(-0.5 * B)
        
        # 3. Compute Reverse R_rev
        R_rev = self.algebra.reverse(R)
        
        # 4. Apply sandwich product: R * x * R_rev
        # Broadcasting: R [Channels, Dim], x [Batch, Channels, Dim]
        
        R_expanded = R.unsqueeze(0) # [1, C, D]
        R_rev_expanded = R_rev.unsqueeze(0) # [1, C, D]
        
        # Rx
        Rx = self.algebra.geometric_product(R_expanded, x)
        
        # (Rx)R_rev
        res = self.algebra.geometric_product(Rx, R_rev_expanded)
        
        return res
