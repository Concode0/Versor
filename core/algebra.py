# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Differentiable Clifford Algebra core.

Implements the geometric product, grade projections, and rotor operations
for arbitrary signatures Cl(p, q, r).
"""

import torch
import math


class CliffordAlgebra:
    """Differentiable Clifford algebra kernel with memory-optimized blocked accumulation.

    Handles geometric product, grade projection, and rotor operations.

    Supports degenerate (null) dimensions via the ``r`` parameter:
    ``Cl(p, q, r)`` has ``p`` positive, ``q`` negative, and ``r`` null
    basis vectors (``e_i^2 = 0``).

    Attributes:
        p (int): Positive signature dimensions.
        q (int): Negative signature dimensions.
        r (int): Degenerate (null) dimensions.
        n (int): Total dimensions (p + q + r).
        dim (int): Total basis elements (2^n).
        device (str): Computation device.
    """
    _CACHED_TABLES = {}

    def __init__(self, p: int, q: int = 0, r: int = 0, device='cuda'):
        """Initialize the algebra and cache the Cayley table.

        Args:
            p (int): Positive dimensions (+1).
            q (int, optional): Negative dimensions (-1). Defaults to 0.
            r (int, optional): Degenerate dimensions (0). Defaults to 0.
            device (str, optional): The device on which computations are performed. Defaults to 'cuda'.
        """
        assert p >= 0, f"p must be non-negative, got {p}"
        assert q >= 0, f"q must be non-negative, got {q}"
        assert r >= 0, f"r must be non-negative, got {r}"
        assert p + q + r <= 12, f"p + q + r must be <= 12, got {p + q + r}"

        self.p, self.q, self.r = p, q, r
        self.n = p + q + r
        self.dim = 2 ** self.n
        self.device = device

        # Cache Cayley tables to avoid recomputation
        cache_key = (p, q, r, str(device))
        if cache_key not in CliffordAlgebra._CACHED_TABLES:
            CliffordAlgebra._CACHED_TABLES[cache_key] = self._generate_cayley_table()

        (
            self.cayley_indices,
            self.cayley_signs,
            self.gp_signs,
            self.grade_masks,
            self.rev_signs,
            self.bv_sq_scalar,
            self.wedge_gp_signs,
            self.inner_gp_signs,
            self.grade_index,
            self.rc_action,
        ) = CliffordAlgebra._CACHED_TABLES[cache_key]

    @property
    def num_grades(self) -> int:
        """Counts the number of grades (n + 1).

        Returns:
            int: Number of grades.
        """
        return self.n + 1

    def embed_vector(self, vectors: torch.Tensor) -> torch.Tensor:
        """Injects vectors into the Grade-1 subspace.

        Args:
            vectors (torch.Tensor): Raw vectors [..., n].

        Returns:
            torch.Tensor: Multivector coefficients [..., dim].
        """
        batch_shape = vectors.shape[:-1]
        mv = torch.zeros(*batch_shape, self.dim, device=vectors.device, dtype=vectors.dtype)
        for i in range(self.n):
            mv[..., 1 << i] = vectors[..., i]
        return mv

    def get_grade_norms(self, mv: torch.Tensor) -> torch.Tensor:
        """Calculates norms per grade. Useful for invariant features.

        Vectorized via scatter_add — no Python loops over grades.

        Args:
            mv (torch.Tensor): Input multivector [..., dim].

        Returns:
            torch.Tensor: Grade norms [..., num_grades].
        """
        gi = self.grade_index
        if gi.device != mv.device:
            gi = gi.to(mv.device)
        batch_shape = mv.shape[:-1]
        sq = mv.pow(2)
        flat = sq.reshape(-1, self.dim)
        idx = gi.unsqueeze(0).expand_as(flat)
        result = torch.zeros(flat.shape[0], self.num_grades, device=mv.device, dtype=mv.dtype)
        result.scatter_add_(1, idx, flat)
        return result.reshape(*batch_shape, self.num_grades).clamp(min=1e-6).sqrt()

    def _generate_cayley_table(self):
        """Precompute the Cayley table, grade masks, and reversion signs.

        Returns:
            tuple: Cached tensors for algebra operations.
        """
        indices = torch.arange(self.dim, device=self.device)

        # Result index = A XOR B
        cayley_indices = indices.unsqueeze(0) ^ indices.unsqueeze(1)
        cayley_signs = self._compute_signs(indices)

        # Precompute signs for geometric_product accumulation
        gp_signs = torch.gather(cayley_signs, 1, cayley_indices)

        # Grade masks: one bool tensor per grade (cached to avoid per-call Python loop)
        grade_masks = []
        for k in range(self.n + 1):
            mask = torch.tensor(
                [bin(i).count('1') == k for i in range(self.dim)],
                dtype=torch.bool, device=self.device,
            )
            grade_masks.append(mask)

        # Reverse signs: blade i gets sign (-1)^(k(k-1)/2) where k = grade(i)
        rev_signs = torch.zeros(self.dim, dtype=cayley_signs.dtype, device=self.device)
        for i in range(self.dim):
            k = bin(i).count('1')
            rev_signs[i] = (-1) ** (k * (k - 1) // 2)

        # Bivector squared scalars: for each basis bivector e_ab,
        # (e_ab)^2 = -s_a * s_b where s_i = +1 if i < p, -1 if p <= i < p+q, 0 if i >= p+q.
        # Used by closed-form exp for arbitrary signature.
        if self.n >= 2:
            bv_mask = grade_masks[2]
            bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)
            bv_sq_scalar = torch.zeros(len(bv_indices), dtype=cayley_signs.dtype,
                                       device=self.device)
            for idx_pos, blade_idx in enumerate(bv_indices.tolist()):
                bits = []
                for bit in range(self.n):
                    if blade_idx & (1 << bit):
                        bits.append(bit)
                if len(bits) == 2:
                    a, b = bits
                    s_a = 1.0 if a < self.p else (-1.0 if a < self.p + self.q else 0.0)
                    s_b = 1.0 if b < self.p else (-1.0 if b < self.p + self.q else 0.0)
                    bv_sq_scalar[idx_pos] = -s_a * s_b
        else:
            bv_sq_scalar = torch.zeros(0, dtype=cayley_signs.dtype,
                                       device=self.device)

        # Grade index: maps each basis element to its grade (popcount)
        grade_index = torch.tensor([bin(i).count('1') for i in range(self.dim)],
                                   dtype=torch.long, device=self.device)

        # Precomputed signs for single-pass wedge and inner product
        # wedge(A,B) = (AB - BA)/2 uses antisymmetric part of signs
        # inner(A,B) = (AB + BA)/2 uses symmetric part of signs
        wedge_cayley_signs = (cayley_signs - cayley_signs.T) / 2.0
        inner_cayley_signs = (cayley_signs + cayley_signs.T) / 2.0
        wedge_gp_signs = torch.gather(wedge_cayley_signs, 1, cayley_indices)
        inner_gp_signs = torch.gather(inner_cayley_signs, 1, cayley_indices)

        # Precomputed right-contraction action matrices for bivector-vector case
        # rc_action[b, i, j] encodes how basis bivector b acts on grade-1 vectors
        if self.n >= 2:
            bv_mask_idx = bv_indices.tolist()
            n = self.n
            rc_action = torch.zeros(len(bv_mask_idx), n, n,
                                    dtype=cayley_signs.dtype, device=self.device)
            for idx_pos, blade_idx in enumerate(bv_mask_idx):
                bits = []
                for bit in range(n):
                    if blade_idx & (1 << bit):
                        bits.append(bit)
                if len(bits) == 2:
                    a, b_bit = bits
                    s_a = 1.0 if a < self.p else (-1.0 if a < self.p + self.q else 0.0)
                    s_b = 1.0 if b_bit < self.p else (-1.0 if b_bit < self.p + self.q else 0.0)
                    # e_{ab} . e_b = s_b * e_a  (right contraction)
                    rc_action[idx_pos, a, b_bit] = s_b
                    # e_{ab} . e_a = -s_a * e_b
                    rc_action[idx_pos, b_bit, a] = -s_a
        else:
            rc_action = torch.zeros(0, self.n, self.n,
                                    dtype=cayley_signs.dtype, device=self.device)

        return (cayley_indices, cayley_signs, gp_signs, grade_masks,
                rev_signs, bv_sq_scalar, wedge_gp_signs, inner_gp_signs,
                grade_index, rc_action)

    def _compute_signs(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute the sign matrix from commutation parity and metric signature.

        Handles three signature types:
        - Positive (i < p): e_i^2 = +1
        - Negative (p <= i < p+q): e_i^2 = -1
        - Null (i >= p+q): e_i^2 = 0

        Args:
            indices (torch.Tensor): Basis indices.

        Returns:
            torch.Tensor: Sign matrix.
        """
        # 1. Commutation Sign: Count swaps needed to reorder basis vectors
        # A bit-wise comparison counts inversions
        A = indices.unsqueeze(1) # Row
        B = indices.unsqueeze(0) # Col

        swap_counts = torch.zeros((self.dim, self.dim), dtype=torch.long, device=self.device)
        for i in range(self.n):
            a_i = (A >> i) & 1
            # Count set bits in B strictly lower than bit i
            lower_mask = (1 << i) - 1
            b_lower = (B & lower_mask)

            # Count bits in b_lower
            b_lower_cnt = torch.zeros_like(B)
            temp_b = b_lower
            for _ in range(self.n):
                b_lower_cnt += (temp_b & 1)
                temp_b = temp_b >> 1

            swap_counts += a_i * b_lower_cnt

        commutator_sign = (-1) ** swap_counts

        # 2. Metric Sign: e_i^2 = -1 if p <= i < p+q, 0 if i >= p+q
        intersection = A & B

        # Mask for negative signature dimensions (p <= i < p+q)
        q_mask = 0
        for i in range(self.p, self.p + self.q):
            q_mask |= (1 << i)

        neg_intersection = intersection & q_mask

        # Count set bits in negative intersection
        neg_cnt = torch.zeros_like(neg_intersection)
        temp_neg = neg_intersection
        for _ in range(self.n):
            neg_cnt += (temp_neg & 1)
            temp_neg = temp_neg >> 1

        metric_sign = (-1) ** neg_cnt

        # 3. Null dimensions: if any null basis vector appears in the intersection
        # (i.e., e_i^2 = 0 for i >= p+q), the entire product is killed.
        if self.r > 0:
            r_mask = 0
            for i in range(self.p + self.q, self.n):
                r_mask |= (1 << i)
            null_intersection = intersection & r_mask
            # Any set bit in null_intersection means a null vector is squared -> 0
            has_null = (null_intersection != 0).to(metric_sign.dtype)
            metric_sign = metric_sign * (1 - has_null)

        return (commutator_sign * metric_sign).to(dtype=torch.float32)

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the Geometric Product.

        Uses vectorized gather + broadcast multiply + sum. No Python loops.

        Args:
            A (torch.Tensor): Left operand [..., Dim].
            B (torch.Tensor): Right operand [..., Dim].

        Returns:
            torch.Tensor: The product AB [..., Dim].
        """
        from core.validation import check_multivector
        check_multivector(A, self, "geometric_product(A)")
        check_multivector(B, self, "geometric_product(B)")

        # Gather B coefficients according to Cayley table: B_gathered[..., i, k] = B[..., cayley[i,k]]
        idx = self.cayley_indices  # [D, D]
        if idx.device != A.device:
            self.ensure_device(A.device)
            idx = self.cayley_indices

        # Expand B for gather: [..., D] -> [..., D, D] via advanced indexing
        B_gathered = B[..., idx]  # [..., D, D]

        # result[..., k] = sum_i A[..., i] * B[..., cayley[i,k]] * signs[i,k]
        # = sum_i (A[..., i, None] * B_gathered[..., i, :] * signs[i, :])  summed over i
        return (A.unsqueeze(-1) * B_gathered * self.gp_signs).sum(dim=-2)

    def ensure_device(self, device) -> None:
        """Move cached tables to the given device if not already there.

        Args:
            device (torch.device): Target device.
        """
        if self.cayley_indices.device == device:
            return
        self.cayley_indices = self.cayley_indices.to(device)
        self.cayley_signs = self.cayley_signs.to(device)
        self.gp_signs = self.gp_signs.to(device)
        self.grade_masks = [m.to(device) for m in self.grade_masks]
        self.rev_signs = self.rev_signs.to(device)
        self.bv_sq_scalar = self.bv_sq_scalar.to(device)
        self.wedge_gp_signs = self.wedge_gp_signs.to(device)
        self.inner_gp_signs = self.inner_gp_signs.to(device)
        self.grade_index = self.grade_index.to(device)
        self.rc_action = self.rc_action.to(device)

        cache_key = (self.p, self.q, self.r, str(self.device))
        CliffordAlgebra._CACHED_TABLES[cache_key] = (
            self.cayley_indices, self.cayley_signs, self.gp_signs,
            self.grade_masks, self.rev_signs, self.bv_sq_scalar,
            self.wedge_gp_signs, self.inner_gp_signs,
            self.grade_index, self.rc_action,
        )

    def grade_projection(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        """Isolates a specific grade.

        Args:
            mv (torch.Tensor): Multivector [..., Dim].
            grade (int): Target grade.

        Returns:
            torch.Tensor: Projected multivector [..., Dim].
        """
        mask = self.grade_masks[grade]
        if mask.device != mv.device:
            mask = mask.to(mv.device)
        result = torch.zeros_like(mv)
        result[..., mask] = mv[..., mask]
        return result

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        """Computes the reversion. The Clifford conjugate.

        Args:
            mv (torch.Tensor): Input multivector [..., Dim].

        Returns:
            torch.Tensor: Reversed multivector [..., Dim].
        """
        rev = self.rev_signs
        if rev.device != mv.device or rev.dtype != mv.dtype:
            rev = rev.to(dtype=mv.dtype, device=mv.device)
        return mv * rev

    def wedge(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the wedge (outer) product: A ^ B = (AB - BA)/2.

        Single-pass implementation using precomputed antisymmetric signs.

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
            from Irreducibles." arXiv:2507.11688v1 [cs.LG]

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Wedge product A ^ B [..., dim].
        """
        idx = self.cayley_indices
        if idx.device != A.device:
            self.ensure_device(A.device)
            idx = self.cayley_indices
        B_gathered = B[..., idx]
        return (A.unsqueeze(-1) * B_gathered * self.wedge_gp_signs).sum(dim=-2)

    def right_contraction(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the right contraction: A _| B.

        Fast path for bivector-vector case using precomputed skew-symmetric
        action matrices (avoids full geometric product + grade projection).

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
            from Irreducibles." arXiv:2507.11688v1 [cs.LG], Algorithm 2

        Args:
            A (torch.Tensor): Left operand (bivector) [..., dim].
            B (torch.Tensor): Right operand (vector) [..., dim].

        Returns:
            torch.Tensor: Right contraction A _| B [..., dim].
        """
        bv_mask = self.grade_masks[2]
        if bv_mask.device != A.device:
            self.ensure_device(A.device)
            bv_mask = self.grade_masks[2]

        bv_coeffs = A[..., bv_mask]                          # [..., num_bv]
        g1_idx = self.grade_masks[1].nonzero(as_tuple=False).squeeze(-1)
        v_coeffs = B[..., g1_idx]                            # [..., n]

        rc = self.rc_action
        if rc.device != A.device or rc.dtype != A.dtype:
            rc = rc.to(device=A.device, dtype=A.dtype)

        # M[..., i, j] = sum_b bv_coeffs[..., b] * rc_action[b, i, j]
        M = torch.einsum('...b, bij -> ...ij', bv_coeffs, rc)  # [..., n, n]
        result_v = torch.matmul(M, v_coeffs.unsqueeze(-1)).squeeze(-1)  # [..., n]

        result = torch.zeros_like(A)
        result[..., g1_idx] = result_v
        return result

    def inner_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the inner product: A . B = (AB + BA)/2.

        Single-pass implementation using precomputed symmetric signs.

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
            from Irreducibles." arXiv:2507.11688v1 [cs.LG]

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Inner product A . B [..., dim].
        """
        idx = self.cayley_indices
        if idx.device != A.device:
            self.ensure_device(A.device)
            idx = self.cayley_indices
        B_gathered = B[..., idx]
        return (A.unsqueeze(-1) * B_gathered * self.inner_gp_signs).sum(dim=-2)

    def exp(self, mv: torch.Tensor) -> torch.Tensor:
        """Exponentiates a bivector to produce a rotor via closed-form.

        Three signature regimes:
            - Elliptic  (B^2 < 0): exp(B) = cos(th) + sin(th)/th * B
            - Hyperbolic (B^2 > 0): exp(B) = cosh(th) + sinh(th)/th * B
            - Parabolic  (B^2 ~ 0): exp(B) ~ 1 + B

        For n <= 3 every bivector is simple, so the closed-form is exact.
        For n >= 4 the closed-form is exact for simple bivectors and a
        first-order approximation for non-simple ones.  Use
        ``exp_decomposed()`` when exact non-simple handling is needed
        (inference only — see ``core.decomposition``).

        Args:
            mv (torch.Tensor): Pure bivector [..., dim].

        Returns:
            torch.Tensor: Rotor exp(mv) [..., dim].
        """
        return self._exp_bivector_closed(mv)

    def _exp_bivector_closed(self, B: torch.Tensor) -> torch.Tensor:
        """Closed-form exponential for bivectors in arbitrary signature.

        For a bivector B, computes B^2 (scalar part) using the metric:
            B^2_scalar = Sum_k b_k^2 . (e_k)^2   where (e_ab)^2 = -s_a.s_b

        Three regimes:
            - B^2 < 0 (elliptic): exp(B) = cos(theta) + sin(theta)/theta . B,  theta = Sqrt(-B^2)
            - B^2 > 0 (hyperbolic): exp(B) = cosh(theta) + sinh(theta)/theta . B,  theta = Sqrt(B^2)
            - B^2 ~= 0 (parabolic): exp(B) ~= 1 + B

        Uses zero geometric products. Exact for simple bivectors in any
        Clifford algebra Cl(p,q,r).

        Args:
            B (torch.Tensor): Pure bivector [..., dim].

        Returns:
            torch.Tensor: Rotor exp(B) [..., dim].
        """
        bv_mask = self.grade_masks[2]
        if bv_mask.device != B.device:
            bv_mask = bv_mask.to(B.device)
        bv_coeffs = B[..., bv_mask]  # [..., num_bivectors]

        bv_sq = self.bv_sq_scalar
        if bv_sq.device != B.device:
            bv_sq = bv_sq.to(B.device)

        # Signed squared norm: alpha = Sum_k b_k^2 . (e_k)^2
        # alpha < 0 -> elliptic (Euclidean-like), alpha > 0 -> hyperbolic
        alpha = (bv_coeffs * bv_coeffs * bv_sq).sum(dim=-1, keepdim=True)

        abs_alpha = alpha.abs().clamp(min=1e-12)
        theta = torch.sqrt(abs_alpha)  # [..., 1]

        # Elliptic branch: cos(theta) and sin(theta)/theta
        cos_theta = torch.cos(theta)
        sinc_theta = torch.where(
            theta > 1e-7,
            torch.sin(theta) / theta,
            1.0 - abs_alpha / 6.0,
        )

        # Hyperbolic branch: cosh(theta) and sinh(theta)/theta
        cosh_theta = torch.cosh(theta)
        sinhc_theta = torch.where(
            theta > 1e-7,
            torch.sinh(theta) / theta,
            1.0 + abs_alpha / 6.0,
        )

        # Select branch based on sign of alpha
        is_elliptic = alpha < -1e-12
        is_hyperbolic = alpha > 1e-12
        # Parabolic (null) falls through: scalar=1, coeff=1

        scalar_part = torch.where(
            is_elliptic, cos_theta,
            torch.where(is_hyperbolic, cosh_theta, torch.ones_like(theta))
        )
        coeff_part = torch.where(
            is_elliptic, sinc_theta,
            torch.where(is_hyperbolic, sinhc_theta, torch.ones_like(theta))
        )

        result = coeff_part * B
        result[..., 0] = scalar_part.squeeze(-1)

        return result

    def _exp_taylor(self, mv: torch.Tensor, order: int = 8) -> torch.Tensor:
        """Taylor series exponential with scaling-and-squaring (fallback).

        Args:
            mv (torch.Tensor): General multivector [..., dim].
            order (int, optional): Taylor order. Defaults to 8.

        Returns:
            torch.Tensor: exp(mv) [..., dim].
        """
        norm = mv.norm(dim=-1, keepdim=True)
        k = torch.ceil(torch.log2(torch.clamp(norm, min=1.0))).int()

        max_k = k.max().item()
        if max_k > 0:
            mv_scaled = mv / (2.0 ** max_k)
        else:
            mv_scaled = mv

        res = torch.zeros_like(mv)
        res[..., 0] = 1.0

        term = torch.zeros_like(mv)
        term[..., 0] = 1.0

        for i in range(1, order + 1):
            term = self.geometric_product(term, mv_scaled)
            res = res + term / math.factorial(i)

        if max_k > 0:
            for _ in range(int(max_k)):
                res = self.geometric_product(res, res)

        return res

    def exp_decomposed(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Exponentiates a bivector via decomposition into simple components.

        Decomposes a general bivector into a sum of simple (blade) bivectors
        using GA power iteration, exponentiates each in closed form, then
        composes via geometric product.  Exact for all bivectors, but the
        power iteration loop is not differentiable through its normalization
        steps.  During training the loop is detached and gradients flow only
        through the final closed-form exp + composition.

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear
            Layers from Irreducibles." arXiv:2507.11688v1 [cs.LG]

        Args:
            mv (torch.Tensor): Pure bivector [..., dim].
            **kwargs: Forwarded to ``core.decomposition.exp_decomposed``.
                use_decomposition (bool): Enable decomposition. Default True.
                k (int | None): Number of simple components (auto if None).
                threshold (float): Convergence threshold. Default 1e-6.
                max_iterations (int): Power iteration cap. Default 100.

        Returns:
            torch.Tensor: Rotor exp(mv) [..., dim].
        """
        from core.decomposition import exp_decomposed
        if 'use_decomposition' not in kwargs:
            kwargs['use_decomposition'] = True
        return exp_decomposed(self, mv, **kwargs)
