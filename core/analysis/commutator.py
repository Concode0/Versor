# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Commutator (Lie bracket) analysis of multivector data.

Computes pairwise commutativity indices, the exchange spectrum of the
adjoint operator, and tests for Lie-subalgebra closure among data
bivectors.
"""

import torch
from typing import Dict

from core.algebra import CliffordAlgebra

from ._types import CONSTANTS, CommutatorResult


class CommutatorAnalyzer:
    """Analyze algebraic exchange properties via commutators.

    The commutator ``[A, B] = AB - BA`` measures non-commutativity.
    In Clifford algebras, commutators of grade-1 elements yield grade-2
    elements (bivectors), directly related to rotation planes.

    Args:
        algebra: :class:`CliffordAlgebra` instance.
        max_bivectors: Maximum number of bivectors for Lie-bracket
            closure analysis (guards combinatorial cost).
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        max_bivectors: int = 15,
    ):
        self.algebra = algebra
        self.max_bivectors = max_bivectors

    def analyze(self, mv_data: torch.Tensor) -> CommutatorResult:
        """Full commutator analysis.

        Args:
            mv_data: Multivector data.  Accepted shapes:

                * ``[N, dim]`` -- single-channel batch.
                * ``[N, C, dim]`` -- multi-channel batch (channels
                  averaged).

        Returns:
            :class:`CommutatorResult`.
        """
        if mv_data.ndim == 3:
            flat = mv_data.mean(dim=1)
        else:
            flat = mv_data  # [N, dim]

        comm_matrix = self.commutativity_matrix(flat)
        ex_spectrum = self.exchange_spectrum(flat)
        mcn = self.mean_commutator_norm(flat)
        lie_struct = self.lie_bracket_closure(flat)

        return CommutatorResult(
            commutativity_matrix=comm_matrix,
            exchange_spectrum=ex_spectrum,
            mean_commutator_norm=mcn,
            lie_bracket_structure=lie_struct,
        )

    def commutativity_matrix(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Pairwise commutativity index for input dimensions.

        For each pair ``(i, j)`` of the *n* basis-vector directions,
        computes ``E[||[x_i, x_j]||]`` where ``x_i`` is the data
        projected onto ``e_i``.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            ``[n, n]`` symmetric matrix of commutativity indices.
        """
        n = self.algebra.n
        dim = self.algebra.dim
        device = mv_data.device
        dtype = mv_data.dtype

        # Build per-direction projections: keep only the e_i component
        g1_idx = (1 << torch.arange(n, device=device)).long()

        N = mv_data.shape[0]

        # Build all n projected multivectors: [n, N, dim]
        xi_all = torch.zeros(n, N, dim, device=device, dtype=dtype)
        coeffs = mv_data[:, g1_idx]  # [N, n]
        row_idx = torch.arange(n, device=device).unsqueeze(1).expand(n, N)
        batch_idx = torch.arange(N, device=device).unsqueeze(0).expand(n, N)
        col_idx = g1_idx.unsqueeze(1).expand(n, N)
        xi_all[row_idx, batch_idx, col_idx] = coeffs.T

        # All (i, j) pairs with i < j
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)

        # Batched commutator: [n_pairs, N, dim]
        comm = self.algebra.commutator(xi_all[i_idx], xi_all[j_idx])
        vals = comm.norm(dim=-1).mean(dim=-1)  # [n_pairs]

        matrix = torch.zeros(n, n, device=device, dtype=dtype)
        matrix[i_idx, j_idx] = vals
        matrix[j_idx, i_idx] = vals

        return matrix

    def exchange_spectrum(self, mv_data: torch.Tensor) -> torch.Tensor:
        """Eigenvalue magnitudes of the adjoint operator ``ad_mu``.

        Constructs the explicit matrix for ``ad_mu(x) = [mu, x]``
        where ``mu = E[x]`` and diagonalises it.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            Eigenvalue magnitudes sorted descending.  Returns an
            empty tensor if the algebra is too large.
        """
        n = self.algebra.n
        dim = self.algebra.dim
        device = mv_data.device
        dtype = mv_data.dtype

        if n > CONSTANTS.adjoint_max_n:
            return torch.tensor([], device=device, dtype=dtype)

        mu = mv_data.mean(dim=0)  # [dim]
        basis = torch.eye(dim, device=device, dtype=dtype)

        # Batched commutator: [dim, dim] x [dim, dim] -> [dim, dim], transpose
        ad_mu = self.algebra.commutator(mu.unsqueeze(0).expand(dim, -1), basis).T

        eigvals = torch.linalg.eigvals(ad_mu)  # complex
        magnitudes = eigvals.abs()
        return magnitudes.sort(descending=True).values

    def mean_commutator_norm(self, mv_data: torch.Tensor) -> float:
        """``E[||[x_i, mu]||_2]`` -- scalar non-commutativity summary.

        Generalises the *Geometric Uncertainty Index* from
        :func:`core.analysis.compute_uncertainty_and_alignment`.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            Mean commutator norm (float).
        """
        mu = mv_data.mean(dim=0, keepdim=True)  # [1, dim]
        comm = self.algebra.commutator(mv_data, mu.expand_as(mv_data))
        return comm.norm(dim=-1).mean().item()

    def lie_bracket_closure(self, mv_data: torch.Tensor) -> Dict:
        """Test whether the data bivectors close under the Lie bracket.

        Selects the top-*k* energetic bivectors from the batch,
        computes all pairwise brackets ``[B_i, B_j]``, and measures
        how well the results lie in the span of the original set.

        Args:
            mv_data: ``[N, dim]`` multivector data.

        Returns:
            Dict with ``"structure_constants"`` (``[k, k, k]``),
            ``"closure_error"`` (scalar), and ``"basis_indices"`` (list
            of multivector-coefficient indices of the chosen bivectors).
        """
        dim = self.algebra.dim
        n = self.algebra.n
        device = mv_data.device
        dtype = mv_data.dtype

        if n < 2:
            return {
                "structure_constants": torch.zeros(0, 0, 0, device=device),
                "closure_error": 0.0,
                "basis_indices": [],
            }

        # Extract grade-2 part of mean per-sample
        bv_data = self.algebra.grade_projection(mv_data, 2)  # [N, dim]
        mean_bv = bv_data.mean(dim=0)  # [dim]

        # Identify bivector basis indices
        ii, jj = torch.triu_indices(n, n, offset=1)
        bv_blade_indices = ((1 << ii) | (1 << jj)).tolist()

        if not bv_blade_indices:
            return {
                "structure_constants": torch.zeros(0, 0, 0, device=device),
                "closure_error": 0.0,
                "basis_indices": [],
            }

        # Pick top-k by energy in the mean bivector
        bv_idx_tensor = torch.tensor(bv_blade_indices, dtype=torch.long, device=device)
        energies = mean_bv[bv_idx_tensor].abs()
        k = min(self.max_bivectors, len(bv_blade_indices))
        topk_pos = energies.topk(k).indices.tolist()

        selected_indices = [bv_blade_indices[p] for p in topk_pos]

        # Build basis bivector multivectors
        B = torch.zeros(k, dim, device=device, dtype=dtype)
        sel_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
        B[torch.arange(k, device=device), sel_tensor] = 1.0

        # Compute structure constants c_{a,b,c} such that [B_a, B_b] ~= Sum_c c_{abc} B_c
        a_idx, b_idx = torch.triu_indices(k, k, offset=1, device=device)

        # Batched commutator and grade-2 projection
        brackets = self.algebra.commutator(B[a_idx], B[b_idx])  # [n_pairs, dim]
        brackets_bv = self.algebra.grade_projection(brackets, 2)  # [n_pairs, dim]

        # Project onto basis: coeffs[p, c] = <bracket_bv_p, B_c>
        coeffs = brackets_bv @ B.T  # [n_pairs, k]

        structure = torch.zeros(k, k, k, device=device, dtype=dtype)
        structure[a_idx, b_idx, :] = coeffs
        structure[b_idx, a_idx, :] = -coeffs  # antisymmetry

        # Closure errors: residual norm / bracket norm
        projected = coeffs @ B  # [n_pairs, dim]
        residuals = brackets_bv - projected
        res_norms = residuals.norm(dim=-1)  # [n_pairs]
        bracket_norms = brackets_bv.norm(dim=-1)  # [n_pairs]

        valid = bracket_norms > self.algebra.eps_sq
        if valid.any():
            closure_error = (res_norms[valid] / bracket_norms[valid]).mean().item()
        else:
            closure_error = 0.0

        return {
            "structure_constants": structure,
            "closure_error": closure_error,
            "basis_indices": selected_indices,
        }


def compute_uncertainty_and_alignment(algebra: CliffordAlgebra, data_tensor: torch.Tensor):
    """Compute Geometric Uncertainty Index (U) and Procrustes Alignment (V).

    Used by :class:`~layers.adapters.mother.MotherEmbedding` to initialise
    per-group / per-subject alignment rotors.

    Args:
        algebra: CliffordAlgebra instance.
        data_tensor: ``[N, D]`` tensor of raw features.

    Returns:
        Tuple ``(U, V)`` where *U* is a float (mean commutator norm) and
        *V* is a ``[D, D]`` Procrustes alignment matrix from SVD.
    """
    N, D = data_tensor.shape
    n = algebra.n

    # 1. Lift data to grade-1 for commutator analysis
    if D < n:
        pad = torch.zeros(N, n - D, device=data_tensor.device)
        x_n = torch.cat([data_tensor, pad], dim=-1)
    else:
        x_n = data_tensor[:, :n]

    x = algebra.embed_vector(x_n)  # [N, dim]

    # 2. Mean multivector and commutator [x_i, mu]
    mu = x.mean(dim=0, keepdim=True)  # [1, dim]
    comm = algebra.commutator(x, mu.expand_as(x))

    U = torch.norm(comm, p=2, dim=-1).mean().item()

    # 3. Procrustes alignment via SVD
    x_c = data_tensor - data_tensor.mean(dim=0, keepdim=True)
    try:
        x_cpu = x_c.cpu()
        _, _, V = torch.svd(x_cpu)
        V = V.to(data_tensor.device)
    except Exception:
        V = torch.eye(D, device=data_tensor.device)

    return U, V
