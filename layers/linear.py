import torch
import torch.nn as nn
import math
from core.algebra import CliffordAlgebra
from layers.base import CliffordModule

class CliffordLinear(CliffordModule):
    def __init__(self, algebra: CliffordAlgebra, in_features: int, out_features: int, bias: bool = True):
        """
        A linear layer W x + B where W, x, B are multivectors.
        However, since 'in_features' and 'out_features' usually refer to channel dimensions 
        rather than GA dimensions in standard DL, we need to clarify.
        
        If x is [Batch, 2^n], this is just a single MV.
        If x is [Batch, C, 2^n], we have C channels of MVs.
        
        Standard "Clifford Linear" usually means mapping C_in MVs to C_out MVs via a matrix of MVs.
        Y_j = Sum_i (W_ji * X_i) + B_j
        
        W has shape [Out, In, 2^n]
        """
        super().__init__(algebra)
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight: [Out, In, Dim]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, algebra.dim))
        
        if bias:
            # Bias: [Out, Dim]
            self.bias = nn.Parameter(torch.Tensor(out_features, algebra.dim))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization is tricky for GA. 
        # Standard kaiming init on coefficients might be okay.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, In, Dim]
        Returns: [Batch, Out, Dim]
        """
        # We need to perform matrix multiplication where the scalar product is replaced by geometric product.
        # W: [O, I, D]
        # X: [B, I, D]
        
        # 1. Expand X to [B, 1, I, D] and W to [1, O, I, D]
        # 2. Geometric Product W * X -> [B, O, I, D] (element-wise on I)
        # 3. Sum over I -> [B, O, D]
        
        # This is expensive. W_ji * X_i
        
        # Loop implementation first for clarity/correctness, then optimize if needed.
        # Or utilize the algebra.geometric_product broadcasting.
        
        batch_size = x.size(0)
        
        # X: [B, 1, In, D]
        x_expanded = x.unsqueeze(1)
        
        # W: [1, Out, In, D]
        w_expanded = self.weight.unsqueeze(0)
        
        # Product: [B, Out, In, D]
        # We need to reshape to use algebra.geometric_product which expects [..., D]
        # but geometric_product supports broadcasting.
        
        prod = self.algebra.geometric_product(w_expanded, x_expanded)
        
        # Sum over In dimension
        y = prod.sum(dim=2) # [B, Out, D]
        
        if self.bias is not None:
            y = y + self.bias
            
        return y
import math
