# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""SearchPlane + Algebraic Projection/Lift for hypothesis management in Cl(1,1).

Three classes:
- AlgebraicProjection (phi): Cl(3,0,1) -> Cl(1,1) via principled grade-norm decomposition
- AlgebraicLift (psi): Cl(1,1) -> grade-wise multiplicative modulation of Cl(3,0,1)
- SearchPlane: Active hypothesis management via Cl(1,1) hyperbolic rotors + FIM scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from core.metric import _hermitian_signs
from layers.primitives.base import CliffordModule


class AlgebraicProjection(nn.Module):
    """phi: Cl(3,0,1) world summary -> Cl(1,1) search state.

    Compresses 16D multivector into 4D via principled decomposition:
    - Scalar component preserved directly
    - Grade norms -> e+ and e- components (energy distribution)
    - Bivector coefficients -> e+e- component (relational phase)
    """

    def __init__(self, algebra_cpu: CliffordAlgebra):
        super().__init__()
        num_grades = algebra_cpu.num_grades  # 5 for Cl(3,0,1)
        num_bivectors = len(algebra_cpu._bv_indices)  # 6 for Cl(3,0,1)

        self.f_plus = nn.Linear(num_grades, 1)
        self.f_minus = nn.Linear(num_grades, 1)
        self.f_phase = nn.Linear(num_bivectors, 1)

        self._algebra_cpu = algebra_cpu

    def forward(self, world_summary: torch.Tensor) -> torch.Tensor:
        """Project world summary to search plane.

        Args:
            world_summary: Mean-pooled CPU state [B, 16].

        Returns:
            Search state [B, 4] in Cl(1,1).
        """
        self._algebra_cpu.ensure_device(world_summary.device)
        grade_norms = self._algebra_cpu.get_grade_norms(world_summary)  # [B, 5]
        bv_idx = self._algebra_cpu._bv_indices
        bv_coeffs = world_summary[:, bv_idx]  # [B, 6]

        return torch.stack([
            world_summary[:, 0],                # scalar preserved
            self.f_plus(grade_norms).squeeze(-1),   # e+: positive energy
            self.f_minus(grade_norms).squeeze(-1),  # e-: negative/null energy
            self.f_phase(bv_coeffs).squeeze(-1),    # e+e-: relational phase
        ], dim=-1)


class AlgebraicLift(nn.Module):
    """psi: Cl(1,1) hypotheses -> grade-wise modulation of Cl(3,0,1).

    Converts weighted hypothesis mean into per-grade multiplicative scales.
    Centered at 1.0 (no effect initially) via 1 + tanh(...), range [0, 2].
    """

    def __init__(self, algebra_cpu: CliffordAlgebra):
        super().__init__()
        num_grades = algebra_cpu.num_grades  # 5
        self.lift_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, num_grades),
        )
        # Precompute grade masks as float for broadcasting
        self._grade_masks = [m.float() for m in algebra_cpu.grade_masks]
        self._num_grades = num_grades
        self._dim = algebra_cpu.dim

    def forward(self, hypotheses: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """Compute grade-wise modulation from weighted hypotheses.

        Args:
            hypotheses: Hypothesis states [B, K, 4].
            weights: Attention weights [B, K].

        Returns:
            Modulation vector [B, D] (multiplicative, centered at 1.0).
        """
        # Conviction-weighted mean hypothesis
        weighted = torch.einsum('bk,bkd->bd', weights, hypotheses)  # [B, 4]
        grade_scales = 1.0 + torch.tanh(self.lift_mlp(weighted))  # [B, num_grades]

        B = hypotheses.shape[0]
        device = hypotheses.device
        modulation = torch.ones(B, self._dim, device=device, dtype=hypotheses.dtype)

        for g in range(self._num_grades):
            mask = self._grade_masks[g].to(device=device, dtype=hypotheses.dtype)
            modulation = modulation + (grade_scales[:, g:g+1] - 1.0) * mask.unsqueeze(0)

        return modulation


class SearchPlane(CliffordModule):
    """Active hypothesis management in Cl(1,1).

    Evolves K hypotheses via hyperbolic rotors, scores them using FIM,
    and computes soft attention weights. All operations are differentiable
    (no hard selection). Temperature controls exploration vs exploitation.
    """

    def __init__(self, algebra_ctrl: CliffordAlgebra,
                 num_hypotheses: int = 8,
                 evolve_hidden: int = 64):
        assert algebra_ctrl.p == 1 and algebra_ctrl.q == 1, \
            f"SearchPlane requires Cl(1,1), got Cl({algebra_ctrl.p},{algebra_ctrl.q})"
        super().__init__(algebra_ctrl)

        K = num_hypotheses
        self.num_hypotheses = K

        # Initial hypothesis states in Cl(1,1)
        self.hypothesis_init = nn.Parameter(torch.randn(K, 4) * 0.1)

        # Evolution network: context -> boost magnitude
        # Input: hypothesis (4) + world_summary projected to 16D -> concatenated
        self.evolve_net = nn.Sequential(
            nn.Linear(4 + 16, evolve_hidden),
            nn.ReLU(),
            nn.Linear(evolve_hidden, 1),
        )

        # Temperature buffer (annealed externally)
        self.register_buffer('_temperature', torch.tensor(1.0))

        # Precompute hermitian signs for Gram matrix
        self._ctrl_signs = None

    def set_temperature(self, tau: float):
        """Set softmax temperature for attention weights."""
        self._temperature.fill_(tau)

    def _get_signs(self, device: torch.device) -> torch.Tensor:
        """Get cached hermitian signs for Cl(1,1)."""
        if self._ctrl_signs is None or self._ctrl_signs.device != device:
            self._ctrl_signs = _hermitian_signs(self.algebra).to(device)
        return self._ctrl_signs

    def forward(self, hypotheses: torch.Tensor,
                world_summary: torch.Tensor,
                fim_values: torch.Tensor,
                fim_prev: torch.Tensor = None) -> dict:
        """Evolve hypotheses and compute attention weights.

        Args:
            hypotheses: Current hypothesis states [B, K, 4].
            world_summary: Mean-pooled CPU state [B, 16].
            fim_values: FIM scores for candidates [B, K].
            fim_prev: Previous FIM values [B, K] or None.

        Returns:
            dict with hypotheses, weights, fim_values, delta_info, gram, conviction.
        """
        B, K, _ = hypotheses.shape
        device = hypotheses.device
        self.algebra.ensure_device(device)

        world_exp = world_summary.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        h_flat = hypotheses.reshape(B * K, 4)
        ctx = torch.cat([h_flat, world_exp], dim=-1)  # [B*K, 20]
        raw_theta = self.evolve_net(ctx).reshape(B, K)
        # Smooth bounding via tanh: always has gradient, range [-3, 3]
        theta = torch.tanh(raw_theta) * 3.0

        # Build e+e- bivector (index 3 in Cl(1,1))
        bv = torch.zeros(B * K, 4, device=device, dtype=hypotheses.dtype)
        bv[:, 3] = theta.reshape(B * K)

        # Exponentiate and sandwich
        R = self.algebra.exp(-0.5 * bv)  # [B*K, 4]
        R_rev = self.algebra.reverse(R)
        evolved = self.algebra.geometric_product(
            self.algebra.geometric_product(R, h_flat), R_rev
        )
        # Symmlog: prevents unbounded drift, gradient = 1/(1+|x|), never zero
        evolved = torch.sign(evolved) * torch.log1p(evolved.abs())
        hypotheses = evolved.reshape(B, K, 4)

        delta_info = fim_values - fim_prev if fim_prev is not None else fim_values

        tau = self._temperature.clamp(min=0.01)
        weights = F.softmax(fim_values / tau, dim=-1)  # [B, K]

        gram = self.hermitian_gram(hypotheses)  # [B, K, K]

        conviction = weights.max(dim=-1).values  # [B]

        return {
            'hypotheses': hypotheses,
            'weights': weights,
            'fim_values': fim_values,
            'delta_info': delta_info,
            'gram': gram,
            'conviction': conviction,
        }

    def hermitian_gram(self, hypotheses: torch.Tensor) -> torch.Tensor:
        """Compute Hermitian Gram matrix for hypothesis orthogonality.

        Args:
            hypotheses: [B, K, 4].

        Returns:
            Gram matrix [B, K, K].
        """
        signs = self._get_signs(hypotheses.device).to(dtype=hypotheses.dtype)
        h_signed = hypotheses * signs  # [B, K, 4]
        return torch.einsum('bkd,bld->bkl', h_signed, hypotheses)

    @staticmethod
    def orthogonality_loss(gram: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality loss from Gram matrix.

        L_ortho = ||G_normalized - I||_F^2

        Args:
            gram: Gram matrix [B, K, K].

        Returns:
            Scalar loss.
        """
        K = gram.shape[1]
        device = gram.device
        # Normalize: G_norm[i,j] = G[i,j] / sqrt(|G[i,i]| * |G[j,j]|)
        diag = gram.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-8)  # [B, K]
        norm_factor = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))  # [B, K, K]
        gram_norm = gram / norm_factor
        eye = torch.eye(K, device=device, dtype=gram.dtype).unsqueeze(0)
        return ((gram_norm - eye) ** 2).mean()
