# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Spectral analysis of multivector data in a Clifford algebra.

Computes grade-energy spectrum, bivector-field decomposition, and
(optionally) the eigenvalue spectrum of the geometric-product operator.
"""

import torch
from typing import List, Optional

from core.algebra import CliffordAlgebra
from core.metric import hermitian_grade_spectrum
from core.decomposition import differentiable_invariant_decomposition

from ._types import CONSTANTS, SpectralResult


class SpectralAnalyzer:
    """Spectral analysis of multivector data.

    Three independent analyses are combined:

    1. **Grade energy spectrum** -- population-level distribution of
       Hermitian grade energy across all grades.
    2. **Bivector field spectrum** -- singular values from decomposing
       the mean bivector into simple components (rotation planes).
    3. **GP operator spectrum** -- eigenvalues of the left-multiplication
       operator :math:`L_x(y) = x \\cdot y` (only for small algebras).

    Args:
        algebra: :class:`CliffordAlgebra` instance.
        max_simple_components: Maximum number of simple components to
            extract from the mean bivector.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        max_simple_components: int = 5,
    ):
        self.algebra = algebra
        self.max_simple_components = max_simple_components

    def analyze(self, mv_data: torch.Tensor) -> SpectralResult:
        """Full spectral analysis.

        Args:
            mv_data: Multivector data.  Accepted shapes:

                * ``[N, dim]`` -- single-channel batch.
                * ``[N, C, dim]`` -- multi-channel batch (channels are
                  averaged before bivector / GP analysis).

        Returns:
            :class:`SpectralResult`.
        """
        if mv_data.ndim == 2:
            mv_data = mv_data.unsqueeze(1)  # [N, 1, dim]

        grade_energy = self.grade_energy_spectrum(mv_data)
        bv_spectrum, simple_comps = self.bivector_field_spectrum(mv_data)

        gp_eigs = None
        if self.algebra.n <= CONSTANTS.gp_spectrum_max_n:
            gp_eigs = self.gp_operator_spectrum(mv_data)

        return SpectralResult(
            grade_energy=grade_energy,
            bivector_spectrum=bv_spectrum,
            simple_components=simple_comps,
            gp_eigenvalues=gp_eigs,
        )

    def grade_energy_spectrum(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Mean Hermitian grade energy across the batch.

        Args:
            mv_data: ``[N, C, dim]`` multivector data.

        Returns:
            ``[n+1]`` tensor of mean grade energies.
        """
        # hermitian_grade_spectrum expects [..., dim]
        # Average over channels first to get [N, dim]
        flat = mv_data.mean(dim=1)  # [N, dim]
        spectrum = hermitian_grade_spectrum(self.algebra, flat)  # [N, n+1]
        return spectrum.mean(dim=0)  # [n+1]

    def bivector_field_spectrum(self, mv_data: torch.Tensor) -> tuple:
        """Decompose the mean bivector into simple components.

        Args:
            mv_data: ``[N, C, dim]`` multivector data.

        Returns:
            ``(singular_values, simple_components)`` where
            *singular_values* is a 1-D tensor of component norms and
            *simple_components* is a list of ``[dim]`` simple bivectors.
        """
        flat = mv_data.mean(dim=1)  # [N, dim]
        mean_mv = flat.mean(dim=0)  # [dim]

        # Guard: algebra needs at least grade 2 for bivectors
        if self.algebra.n < 2:
            return (
                torch.zeros(1, device=mv_data.device),
                [torch.zeros_like(mean_mv)],
            )

        # Extract grade-2 (bivector) part
        mean_bv = self.algebra.grade_projection(mean_mv, 2)

        bv_norm = mean_bv.norm()
        if bv_norm < self.algebra.eps_sq:
            return (
                torch.zeros(1, device=mv_data.device),
                [torch.zeros_like(mean_bv)],
            )

        decomp, _ = differentiable_invariant_decomposition(
            self.algebra,
            mean_bv.unsqueeze(0),  # [1, dim]
            k=self.max_simple_components,
        )

        # decomp is a list of [1, dim] tensors
        norms = []
        components = []
        for comp in decomp:
            c = comp.squeeze(0)  # [dim]
            n = c.norm()
            if n > self.algebra.eps_sq:
                norms.append(n)
                components.append(c)

        if not norms:
            return (
                torch.zeros(1, device=mv_data.device),
                [torch.zeros_like(mean_bv)],
            )

        sv = torch.stack(norms)
        # Sort descending
        order = sv.argsort(descending=True)
        sv = sv[order]
        components = [components[i] for i in order.tolist()]

        return sv, components

    def gp_operator_spectrum(self, mv_data: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        """Eigenvalue magnitudes of the left-multiplication operator.

        For a subsample of data points, constructs the explicit matrix
        representation of :math:`L_x(y) = x \\cdot y` and computes
        eigenvalues.

        Args:
            mv_data: ``[N, C, dim]`` multivector data.
            n_samples: Number of data points to sample.

        Returns:
            Sorted (descending) eigenvalue magnitudes.
        """
        if n_samples is None:
            n_samples = CONSTANTS.gp_spectrum_n_samples
        dim = self.algebra.dim
        flat = mv_data.mean(dim=1)  # [N, dim]
        N = flat.shape[0]

        k = min(N, n_samples)
        if k < N:
            idx = torch.randperm(N)[:k]
            flat = flat[idx]

        # Build GP matrix: L[:, j] = gp(mean_x, e_j)
        basis = torch.eye(dim, device=flat.device, dtype=flat.dtype)
        mean_x = flat.mean(dim=0)  # [dim]
        # Batched GP: expand mean_x to [dim, dim], basis is [dim, dim]
        # Result[j, :] = gp(mean_x, e_j) = L[:, j], so transpose
        L = self.algebra.geometric_product(mean_x.unsqueeze(0).expand(dim, -1), basis).T

        eigvals = torch.linalg.eigvals(L)  # complex
        magnitudes = eigvals.abs()
        return magnitudes.sort(descending=True).values
