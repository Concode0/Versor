import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricGELU(nn.Module):
    def __init__(self, algebra, channels: int = 1):
        """
        Magnitude-based activation. Preserves direction (orientation) of the multivector
        but scales its magnitude non-linearly using GELU.
        
        formula: x' = x * (GELU(|x|) / |x|)
        """
        super().__init__()
        self.algebra = algebra
        # Learnable bias for the magnitude?
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        # x: [Batch, Channels, Dim]
        
        # Compute magnitude (approximated by Euclidean norm of coefficients for now)
        # Strictly speaking, geometric magnitude is sqrt(x * x~). 
        # For Euclidean metric, this aligns with coefficient norm.
        # For mixed signature, coefficient norm is safer for stability.
        norm = x.norm(dim=-1, keepdim=True) # [Batch, C, 1]
        
        # Apply GELU to the biased norm
        # We want to scale x. 
        # scale = activation(norm + bias) / norm
        
        # Avoid div by zero
        eps = 1e-6
        scale = F.gelu(norm + self.bias.view(1, -1, 1)) / (norm + eps)
        
        return x * scale

class GradeSwish(nn.Module):
    def __init__(self, algebra, channels: int = 1):
        """
        Applies a learned gating factor per Grade.
        x_k' = x_k * Sigmoid(w_k * |x_k| + b_k)
        """
        super().__init__()
        self.algebra = algebra
        self.n_grades = algebra.n + 1
        
        # Parameters per grade
        self.grade_weights = nn.Parameter(torch.ones(self.n_grades))
        self.grade_biases = nn.Parameter(torch.zeros(self.n_grades))
        
        # Precompute grade masks
        self.register_buffer('grade_masks', self._build_masks())

    def _build_masks(self):
        # [Dim, 1] or similar to broadcast
        masks = torch.zeros(self.n_grades, self.algebra.dim, dtype=torch.bool)
        for i in range(self.algebra.dim):
            # count set bits
            grade = bin(i).count('1')
            masks[grade, i] = True
        return masks

    def forward(self, x: torch.Tensor):
        # x: [Batch, Channels, Dim]
        output = torch.zeros_like(x)
        
        for k in range(self.n_grades):
            mask = self.grade_masks[k] # [Dim]
            if not mask.any():
                continue
                
            # Extract k-vector part
            x_k = x[..., mask] # [Batch, C, Dim_k]
            
            # Compute norm of this grade
            norm_k = x_k.norm(dim=-1, keepdim=True) # [Batch, C, 1]
            
            # Gating factor
            # Swish-like: x * sigmoid(w*x + b)
            # Here: x_k * sigmoid(w * |x_k| + b)
            w = self.grade_weights[k]
            b = self.grade_biases[k]
            
            gate = torch.sigmoid(w * norm_k + b)
            
            # Broadcast gate back to x_k shape?
            # x_k is flattened subset.
            
            output[..., mask] = x_k * gate
            
        return output