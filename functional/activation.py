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
    """Geometric GELU. Scales magnitude, keeps direction.

    Non-linearity without rotation.
    x' = x * (GELU(|x| + b) / |x|)
    """

    def __init__(self, algebra, channels: int = 1):
        """Sets up the activation."""
        super().__init__()
        self.algebra = algebra
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        """Activates."""
        norm = x.norm(dim=-1, keepdim=True)
        
        eps = 1e-6
        scale = F.gelu(norm + self.bias.view(1, -1, 1)) / (norm + eps)
        
        return x * scale

class GradeSwish(nn.Module):
    """Grade Swish. Gates by grade.

    Each grade gets its own gate.
    """

    def __init__(self, algebra, channels: int = 1):
        """Sets up the gates."""
        super().__init__()
        self.algebra = algebra
        self.n_grades = algebra.n + 1
        
        self.grade_weights = nn.Parameter(torch.ones(self.n_grades))
        self.grade_biases = nn.Parameter(torch.zeros(self.n_grades))
        
        self.register_buffer('grade_masks', self._build_masks())

    def _build_masks(self):
        """Precomputes masks."""
        masks = torch.zeros(self.n_grades, self.algebra.dim, dtype=torch.bool)
        for i in range(self.algebra.dim):
            grade = bin(i).count('1')
            masks[grade, i] = True
        return masks

    def forward(self, x: torch.Tensor):
        """Gates the grades."""
        output = torch.zeros_like(x)
        
        for k in range(self.n_grades):
            mask = self.grade_masks[k]
            if not mask.any():
                continue
                
            x_k = x[..., mask]
            norm_k = x_k.norm(dim=-1, keepdim=True)
            
            w = self.grade_weights[k]
            b = self.grade_biases[k]
            
            gate = torch.sigmoid(w * norm_k + b)
            
            output[..., mask] = x_k * gate
            
        return output