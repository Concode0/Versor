# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricGELU(nn.Module):
    """Magnitude-based geometric activation function.

    Preserves the directional orientation of the multivector but scales its
    magnitude non-linearly using GELU. This ensures the operation is equivariant
    to global rotations.

    Formula: x' = x * (GELU(|x| + b) / |x|)

    Attributes:
        bias (nn.Parameter): Learnable scalar bias added to the magnitude.
    """

    def __init__(self, algebra, channels: int = 1):
        """Initializes Geometric GELU.

        Args:
            algebra: The algebra instance.
            channels (int, optional): Number of feature channels. Defaults to 1.
        """
        super().__init__()
        self.algebra = algebra
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        """Applies the activation.

        Args:
            x (torch.Tensor): Input multivectors.

        Returns:
            torch.Tensor: Scaled multivectors.
        """
        # Compute magnitude (approximated by Euclidean norm of coefficients)
        norm = x.norm(dim=-1, keepdim=True) # [Batch, C, 1]
        
        # Apply GELU to the biased norm
        # Avoid division by zero with eps
        eps = 1e-6
        scale = F.gelu(norm + self.bias.view(1, -1, 1)) / (norm + eps)
        
        return x * scale

class GradeSwish(nn.Module):
    """Grade-specific gating mechanism.

    Applies a learned gating factor per geometric grade (scalar, vector, bivector...).
    x_k' = x_k * Sigmoid(w_k * |x_k| + b_k)

    Attributes:
        grade_weights (nn.Parameter): Weights per grade.
        grade_biases (nn.Parameter): Biases per grade.
        grade_masks (torch.Tensor): Precomputed masks for grade extraction.
    """

    def __init__(self, algebra, channels: int = 1):
        """Initializes Grade Swish.

        Args:
            algebra: The algebra instance.
            channels (int, optional): Unused but kept for API consistency. Defaults to 1.
        """
        super().__init__()
        self.algebra = algebra
        self.n_grades = algebra.n + 1
        
        self.grade_weights = nn.Parameter(torch.ones(self.n_grades))
        self.grade_biases = nn.Parameter(torch.zeros(self.n_grades))
        
        self.register_buffer('grade_masks', self._build_masks())

    def _build_masks(self):
        """Precomputes boolean masks for each grade."""
        masks = torch.zeros(self.n_grades, self.algebra.dim, dtype=torch.bool)
        for i in range(self.algebra.dim):
            grade = bin(i).count('1')
            masks[grade, i] = True
        return masks

    def forward(self, x: torch.Tensor):
        """Applies grade-specific gating.

        Args:
            x (torch.Tensor): Input multivectors.

        Returns:
            torch.Tensor: Gated multivectors.
        """
        output = torch.zeros_like(x)
        
        for k in range(self.n_grades):
            mask = self.grade_masks[k]
            if not mask.any():
                continue
                
            # Extract k-vector part
            x_k = x[..., mask]
            
            # Compute norm of this grade
            norm_k = x_k.norm(dim=-1, keepdim=True)
            
            w = self.grade_weights[k]
            b = self.grade_biases[k]
            
            gate = torch.sigmoid(w * norm_k + b)
            
            output[..., mask] = x_k * gate
            
        return output