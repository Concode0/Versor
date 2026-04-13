# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Symmetry and null-direction detection in Clifford algebras.

Detects null directions, grade-involution symmetry, reflection
symmetries, and continuous symmetries (via commutator analysis).
"""

import torch
from typing import Dict, List, Optional, Tuple

from core.algebra import CliffordAlgebra

from ._types import CONSTANTS, CommutatorResult, SymmetryResult


class SymmetryDetector:
    """Detect symmetries, null directions, and invariances.

    Args:
        algebra: :class:`CliffordAlgebra` instance.
        null_threshold: Energy threshold below which a direction is
            considered effectively null.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        null_threshold: float = 0.01,
    ):
        self.algebra = algebra
        self.null_threshold = null_threshold

    def analyze(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
    ) -> SymmetryResult:
        """Full symmetry analysis.

        Args:
            mv_data: Multivector data.  Accepted shapes:

                * ``[N, dim]`` -- single-channel batch.
                * ``[N, C, dim]`` -- multi-channel batch (channels
                  averaged for scalar-valued analyses).

            commutator_result: Pre-computed commutator results for
                continuous-symmetry refinement.

        Returns:
            :class:`SymmetryResult`.
        """
        if mv_data.ndim == 3:
            flat = mv_data.mean(dim=1)
        else:
            flat = mv_data  # [N, dim]

        null_dirs, null_scores = self.detect_null_directions(flat)
        inv_sym = self.detect_involution_symmetry(flat)
        refl_syms = self.detect_reflection_symmetries(flat)
        cont_dim = self.detect_continuous_symmetries(
            flat, commutator_result=commutator_result
        )

        return SymmetryResult(
            null_directions=null_dirs,
            null_scores=null_scores,
            involution_symmetry=inv_sym,
            reflection_symmetries=refl_syms,
            continuous_symmetry_dim=cont_dim,
        )

    def detect_null_directions(
        self, mv_data: torch.Tensor
    ) -> Tuple[List[int], torch.Tensor]:
        """Detect effectively null basis-vector directions.

        For each basis vector ``e_i``, computes the mean squared
        projection energy of the data onto that direction.  Directions
        below :attr:`null_threshold` are flagged.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            ``(null_indices, scores)`` where *scores* is ``[n]`` and
            *null_indices* lists those with score < threshold.
        """
        n = self.algebra.n
        g1_idx = (1 << torch.arange(n, device=mv_data.device)).long()

        # Energy on each grade-1 component
        g1_coeffs = mv_data[:, g1_idx]  # [N, n]
        scores = (g1_coeffs ** 2).mean(dim=0)  # [n]

        # Normalise so max is 1
        smax = scores.max()
        if smax > self.algebra.eps_sq:
            scores = scores / smax

        null_dirs = (scores < self.null_threshold).nonzero(as_tuple=True)[0]
        return null_dirs.tolist(), scores

    def detect_involution_symmetry(self, mv_data: torch.Tensor) -> float:
        """Measure grade-involution symmetry of the data distribution.

        Computes the fraction of total energy in odd-grade components:

            score = E[ ||x_odd||^2 / ||x||^2 ]

        where ``x_odd = (x - alpha(x)) / 2`` and ``alpha`` is the grade
        involution (flips odd-grade components).

        Returns:
            Score in ``[0, 1]``.  0 means the data lives entirely in
            the even sub-algebra; 1 means entirely odd grades.
        """
        alpha = self.algebra.grade_involution(mv_data)  # [N, dim]
        odd_part = (mv_data - alpha) / 2.0
        odd_energy = (odd_part ** 2).sum(dim=-1)
        total_energy = (mv_data ** 2).sum(dim=-1).clamp(min=self.algebra.eps_sq)
        return (odd_energy / total_energy).mean().item()

    def detect_reflection_symmetries(
        self, mv_data: torch.Tensor
    ) -> List[Dict]:
        """Test reflection symmetry along each basis-vector direction.

        For each basis vector ``e_i``, reflects the data
        ``x' = -e_i x e_i^{-1}`` and compares the reflected
        distribution to the original.

        Returns:
            List of dicts ``{"direction": i, "score": float}`` sorted
            by score ascending (lower = more symmetric).
        """
        n = self.algebra.n
        dim = self.algebra.dim
        N = mv_data.shape[0]

        # Build all n basis vectors: [n, dim]
        basis_vecs = torch.zeros(n, dim, device=mv_data.device, dtype=mv_data.dtype)
        blade_indices = (1 << torch.arange(n, device=mv_data.device)).long()
        basis_vecs[torch.arange(n, device=mv_data.device), blade_indices] = 1.0

        # Batch reflect: [n, N, dim]
        mv_exp = mv_data.unsqueeze(0).expand(n, N, dim)
        basis_exp = basis_vecs.unsqueeze(1).expand(n, N, dim)
        reflected = self.algebra.reflect(mv_exp, basis_exp)  # [n, N, dim]

        # Distributional distance for all directions at once
        orig_sorted = mv_data.sort(dim=0).values.unsqueeze(0).expand(n, N, dim)
        refl_sorted = reflected.sort(dim=1).values  # sort along N
        dist = ((orig_sorted - refl_sorted) ** 2).sum(dim=-1).mean(dim=-1)  # [n]
        norm = (mv_data ** 2).sum(dim=-1).mean().clamp(min=self.algebra.eps_sq)
        scores = dist / norm  # [n]

        results = [{"direction": i, "score": scores[i].item()} for i in range(n)]
        results.sort(key=lambda r: r["score"])
        return results

    def detect_continuous_symmetries(
        self,
        mv_data: torch.Tensor,
        commutator_result: Optional[CommutatorResult] = None,
        threshold: Optional[float] = None,
    ) -> int:
        """Estimate the dimension of the continuous symmetry group.

        A bivector ``B_j`` generates a continuous symmetry if
        ``E[||[B_j, x_i]||]`` is near zero for all data points.

        If a :class:`CommutatorResult` is provided, its exchange
        spectrum is used directly (eigenvalues near zero -> symmetry
        generators).  Otherwise the computation is done from scratch.

        Args:
            mv_data: ``[N, dim]`` multivector data.
            commutator_result: Pre-computed commutator analysis.
            threshold: Normalised commutator norm below which a bivector
                is considered a symmetry generator.

        Returns:
            Estimated dimension of the continuous symmetry group.
        """
        if threshold is None:
            threshold = CONSTANTS.continuous_symmetry_threshold

        if commutator_result is not None:
            spec = commutator_result.exchange_spectrum
            if spec.numel() > 0:
                max_val = spec.abs().max().clamp(min=self.algebra.eps_sq)
                return int((spec.abs() / max_val < threshold).sum().item())

        # Compute from scratch: test each basis bivector
        n = self.algebra.n
        dim = self.algebra.dim
        N = mv_data.shape[0]

        ii, jj = torch.triu_indices(n, n, offset=1)
        bv_indices = ((1 << ii) | (1 << jj)).tolist()

        if not bv_indices:
            return 0

        n_bv = len(bv_indices)
        # Build all bivector bases: [n_bv, dim]
        bv_bases = torch.zeros(n_bv, dim, device=mv_data.device, dtype=mv_data.dtype)
        bv_idx_tensor = torch.tensor(bv_indices, dtype=torch.long, device=mv_data.device)
        bv_bases[torch.arange(n_bv, device=mv_data.device), bv_idx_tensor] = 1.0

        # Batch commutator: [n_bv, N, dim]
        bv_exp = bv_bases.unsqueeze(1).expand(n_bv, N, dim)
        mv_exp = mv_data.unsqueeze(0).expand(n_bv, N, dim)
        comm = self.algebra.commutator(bv_exp, mv_exp)

        comm_norms = comm.norm(dim=-1).mean(dim=-1)  # [n_bv]
        data_norm = mv_data.norm(dim=-1).mean().clamp(min=self.algebra.eps_sq)

        return int((comm_norms / data_norm < threshold).sum().item())
