# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Differentiable Clifford Algebra core.

Implements the geometric product, grade projections, and rotor operations
for arbitrary signatures Cl(p, q, r).
"""

import torch
import torch.nn as nn
import math
from core.validation import check_multivector


class CliffordAlgebra(nn.Module):
    """Differentiable Clifford algebra kernel with memory-optimized blocked accumulation.

    Extends ``nn.Module`` so that all Cayley tables are registered as
    non-persistent buffers. This means ``model.to(device)`` automatically
    moves tables

    Supports degenerate (null) dimensions via the ``r`` parameter:
    ``Cl(p, q, r)`` has ``p`` positive, ``q`` negative, and ``r`` null
    basis vectors (``e_i^2 = 0``).

    Attributes:
        p (int): Positive signature dimensions.
        q (int): Negative signature dimensions.
        r (int): Degenerate (null) dimensions.
        n (int): Total dimensions (p + q + r).
        dim (int): Total basis elements (2^n).
    """
    _CACHED_TABLES = {}

    def __init__(self, p: int, q: int = 0, r: int = 0, device='cuda',
                 dtype: torch.dtype = torch.float32,
                 exp_policy: str = 'auto'):
        """Initialize the algebra and cache the Cayley table.

        Args:
            p (int): Positive dimensions (+1).
            q (int, optional): Negative dimensions (-1). Defaults to 0.
            r (int, optional): Degenerate dimensions (0). Defaults to 0.
            device (str, optional): The device on which computations are performed. Defaults to 'cuda'.
            dtype (torch.dtype, optional): Floating-point dtype for algebra tables.
                Defaults to ``torch.float32``.  Pass ``torch.bfloat16`` or
                ``torch.float16`` when the model will be trained in a reduced
                precision (e.g. AMP on CUDA bfloat16 mode).
            exp_policy (str or ExpPolicy, optional): Bivector exp policy.
                ``'auto'`` (default), ``'fast'``, or ``'exact'``.
                See :class:`core.decomposition.ExpPolicy`.
        """
        super().__init__()

        assert p >= 0, f"p must be non-negative, got {p}"
        assert q >= 0, f"q must be non-negative, got {q}"
        assert r >= 0, f"r must be non-negative, got {r}"
        assert p + q + r <= 12, f"p + q + r must be <= 12, got {p + q + r}"

        self.p, self.q, self.r = p, q, r
        self.n = p + q + r
        self.dim = 2 ** self.n

        # Exp regime: dispatch at init
        if p == 0 or q == 0:
            self._exp_regime = 'elliptic'
        elif p == 1 and q == 1 and r == 0:
            self._exp_regime = 'hyperbolic'
        else:
            self._exp_regime = 'mixed'

        # Exp policy: controls decomposition strategy
        from core.decomposition import ExpPolicy
        if isinstance(exp_policy, str):
            self._exp_policy = ExpPolicy(exp_policy)
        else:
            self._exp_policy = exp_policy

        # Cache Cayley tables to avoid recomputation
        cache_key = (p, q, r, str(device), str(dtype))
        if cache_key not in CliffordAlgebra._CACHED_TABLES:
            CliffordAlgebra._CACHED_TABLES[cache_key] = self._generate_cayley_table(device, dtype)

        (
            cayley_indices, cayley_signs, gp_signs, grade_masks_list,
            rev_signs, bv_sq_scalar, wedge_gp_signs, inner_gp_signs,
            grade_index, rc_action, lc_gp_signs, conj_signs,
            comm_gp_signs, anti_comm_gp_signs,
        ) = CliffordAlgebra._CACHED_TABLES[cache_key]

        # Register all tables as non-persistent buffers so .to(device) moves them
        self.register_buffer('cayley_indices', cayley_indices, persistent=False)
        self.register_buffer('cayley_signs', cayley_signs, persistent=False)
        self.register_buffer('gp_signs', gp_signs, persistent=False)
        self.register_buffer('rev_signs', rev_signs, persistent=False)
        self.register_buffer('bv_sq_scalar', bv_sq_scalar, persistent=False)
        self.register_buffer('wedge_gp_signs', wedge_gp_signs, persistent=False)
        self.register_buffer('inner_gp_signs', inner_gp_signs, persistent=False)
        self.register_buffer('grade_index', grade_index, persistent=False)
        self.register_buffer('rc_action', rc_action, persistent=False)
        self.register_buffer('lc_gp_signs', lc_gp_signs, persistent=False)
        self.register_buffer('conj_signs', conj_signs, persistent=False)
        self.register_buffer('comm_gp_signs', comm_gp_signs, persistent=False)
        self.register_buffer('anti_comm_gp_signs', anti_comm_gp_signs, persistent=False)

        # Grade involution signs: (-1)^k per basis element
        inv_signs = ((-1.0) ** grade_index.float()).to(dtype=conj_signs.dtype)
        self.register_buffer('_involution_signs', inv_signs, persistent=False)

        # Stack grade masks: [n+1, dim] bool and float
        stacked = torch.stack(grade_masks_list)                 # [n+1, dim]
        self.register_buffer('_grade_masks', stacked, persistent=False)
        self.register_buffer('_grade_masks_float', stacked.to(dtype=cayley_signs.dtype), persistent=False)

        # Bivector indices
        if self.n >= 2:
            bv_idx = stacked[2].nonzero(as_tuple=False).squeeze(-1)
        else:
            bv_idx = torch.zeros(0, dtype=torch.long, device=device)
        self.register_buffer('_bv_indices', bv_idx, persistent=False)

        # Pre-initialize derived tables (sandwich_product / pseudoscalar_product)
        self._init_derived_tables()

        # Precomputed finfo-derived tolerances for dtype-aware numerical guards.
        # Plain floats for zero-overhead usage in clamp/where operations.
        _finfo = torch.finfo(self.cayley_signs.dtype)
        self.eps: float = float(_finfo.eps)
        self.eps_sq: float = float(_finfo.eps ** 2)

    @property
    def device(self):
        """Return the device of the algebra tables."""
        return self.cayley_indices.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype of the algebra tables.

        Reflects the current state — updated automatically when the algebra
        is moved via ``.to(dtype=...)``.
        """
        return self.cayley_signs.dtype

    def _apply(self, fn):
        """Propagate device/dtype moves and keep eps tolerances in sync."""
        result = super()._apply(fn)
        _finfo = torch.finfo(self.cayley_signs.dtype)
        self.eps = float(_finfo.eps)
        self.eps_sq = float(_finfo.eps ** 2)
        return result

    @property
    def grade_masks(self):
        """Grade masks indexed by grade: ``grade_masks[k]`` -> ``[dim]`` bool."""
        return self._grade_masks

    @property
    def grade_masks_float(self):
        """Float grade masks indexed by grade: ``grade_masks_float[k]`` -> ``[dim]`` float."""
        return self._grade_masks_float

    @property
    def exp_policy(self):
        """Active :class:`ExpPolicy` controlling ``exp()`` dispatch."""
        return self._exp_policy

    @exp_policy.setter
    def exp_policy(self, value):
        from core.decomposition import ExpPolicy
        if isinstance(value, str):
            value = ExpPolicy(value)
        self._exp_policy = value

    def _init_derived_tables(self):
        """Precompute derived tables for sandwich_product and pseudoscalar_product.

        Called once from ``__init__``. Tables move automatically via
        ``register_buffer`` when ``.to(device)`` is called.
        """
        D = self.dim
        k_range = torch.arange(D, device=self.cayley_indices.device)
        ci = self.cayley_indices

        # For sandwich_product: left-sign table
        ls = self.gp_signs[ci, k_range.unsqueeze(0).expand(D, D)]
        self.register_buffer('_left_sign_T', ls.T.contiguous(), persistent=False)

        # For pseudoscalar_product: permutation and signs
        self.register_buffer('_ps_source', k_range ^ (D - 1), persistent=False)
        self.register_buffer('_ps_signs', self.gp_signs[self._ps_source, k_range], persistent=False)

        # Diagonal of cayley_signs: sign of e_I^2 for each basis element
        cayley_diag = torch.diagonal(self.cayley_signs).clone()
        self.register_buffer('_cayley_diag', cayley_diag, persistent=False)

        # Pre-merged signs for norm_sq: rev_signs * cayley_diag
        self.register_buffer('_norm_sq_signs',
                             (self.rev_signs * cayley_diag).clone(), persistent=False)

        # Hermitian signs: conj_signs * cayley_diag
        # Equivalent to the full _hermitian_signs() computation but vectorized
        self.register_buffer('_hermitian_signs',
                             (self.conj_signs * cayley_diag).clone(), persistent=False)

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
        g1_idx = (1 << torch.arange(self.n, device=vectors.device)).long()
        mv = torch.zeros(*vectors.shape[:-1], self.dim,
                          device=vectors.device, dtype=vectors.dtype)
        mv.scatter_(-1, g1_idx.expand_as(vectors), vectors)
        return mv

    def get_grade_norms(self, mv: torch.Tensor) -> torch.Tensor:
        """Calculates norms per grade. Useful for invariant features.

        Vectorized via scatter_add.

        Args:
            mv (torch.Tensor): Input multivector [..., dim].

        Returns:
            torch.Tensor: Grade norms [..., num_grades].
        """
        gi = self.grade_index
        batch_shape = mv.shape[:-1]
        sq = mv.pow(2)
        flat = sq.reshape(-1, self.dim)
        idx = gi.unsqueeze(0).expand_as(flat)
        result = torch.zeros(flat.shape[0], self.num_grades, device=mv.device, dtype=mv.dtype)
        result.scatter_add_(1, idx, flat)
        return result.reshape(*batch_shape, self.num_grades).clamp(min=self.eps).sqrt()

    def _generate_cayley_table(self, device, dtype: torch.dtype = torch.float32):
        """Precompute the Cayley table, grade masks, and reversion signs.

        Args:
            device: The device to create tensors on.
            dtype: Floating-point dtype for sign tables.

        Returns:
            tuple: Cached tensors for algebra operations.
        """
        indices = torch.arange(self.dim, device=device)

        # Result index = A XOR B
        cayley_indices = indices.unsqueeze(0) ^ indices.unsqueeze(1)
        cayley_signs = self._compute_signs(indices, device, dtype)

        # Precompute signs for geometric_product accumulation
        gp_signs = torch.gather(cayley_signs, 1, cayley_indices)

        # Grade index: maps each basis element to its grade (popcount)
        # Compute once, derive grade_masks and rev_signs from it.
        grade_index = torch.tensor([bin(i).count('1') for i in range(self.dim)],
                                   dtype=torch.long, device=device)

        # Grade masks: one bool tensor per grade
        grade_masks = [grade_index == k for k in range(self.n + 1)]

        # Reverse signs: blade i gets sign (-1)^(k(k-1)/2) where k = grade(i)
        gk = grade_index
        rev_signs = ((-1.0) ** (gk * (gk - 1) // 2)).to(dtype=cayley_signs.dtype)

        # Bivector squared scalars: for each basis bivector e_ab,
        # (e_ab)^2 = -s_a * s_b where s_i = +1 if i < p, -1 if p <= i < p+q, 0 if i >= p+q.
        # Used by closed-form exp for arbitrary signature.
        if self.n >= 2:
            bv_mask = grade_masks[2]
            bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)
            bv_sq_scalar = torch.zeros(len(bv_indices), dtype=cayley_signs.dtype,
                                       device=device)
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
                                       device=device)

        # Precomputed signs for single-pass wedge and inner product
        # wedge(A,B) = (AB - BA)/2 uses antisymmetric part of signs
        # inner(A,B) = (AB + BA)/2 uses symmetric part of signs
        wedge_cayley_signs = (cayley_signs - cayley_signs.T) / 2.0
        inner_cayley_signs = (cayley_signs + cayley_signs.T) / 2.0
        wedge_gp_signs = torch.gather(wedge_cayley_signs, 1, cayley_indices)
        inner_gp_signs = torch.gather(inner_cayley_signs, 1, cayley_indices)

        # Precomputed signs for commutator [A,B] = AB - BA and
        # anti-commutator {A,B} = AB + BA (no 1/2 factor).
        comm_cayley_signs = cayley_signs - cayley_signs.T
        anti_comm_cayley_signs = cayley_signs + cayley_signs.T
        comm_gp_signs = torch.gather(comm_cayley_signs, 1, cayley_indices)
        anti_comm_gp_signs = torch.gather(anti_comm_cayley_signs, 1, cayley_indices)

        # Precomputed right-contraction action matrices for bivector-vector case
        # rc_action[b, i, j] encodes how basis bivector b acts on grade-1 vectors
        if self.n >= 2:
            bv_mask_idx = bv_indices.tolist()
            n = self.n
            rc_action = torch.zeros(len(bv_mask_idx), n, n,
                                    dtype=cayley_signs.dtype, device=device)
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
                                    dtype=cayley_signs.dtype, device=device)

        # Precomputed signs for left contraction: A _| B
        # In the [i, k] indexing (where j = i^k): valid when
        # grade(i) <= grade(j=i^k) and grade(k) = grade(j) - grade(i)
        gi = grade_index.unsqueeze(1)      # [D, 1] - grade of summation index i
        gj = grade_index[cayley_indices]   # [D, D] - grade of j = i^k
        gk = grade_index.unsqueeze(0)      # [1, D] - grade of result index k
        lc_valid = (gi <= gj) & (gk == gj - gi)
        lc_gp_signs = gp_signs * lc_valid.to(dtype=gp_signs.dtype)

        # Clifford conjugation signs: (-1)^k * (-1)^{k(k-1)/2}
        conj_signs = (((-1.0) ** grade_index) * rev_signs).to(
            dtype=cayley_signs.dtype)

        return (cayley_indices, cayley_signs, gp_signs, grade_masks,
                rev_signs, bv_sq_scalar, wedge_gp_signs, inner_gp_signs,
                grade_index, rc_action, lc_gp_signs, conj_signs,
                comm_gp_signs, anti_comm_gp_signs)

    def _compute_signs(self, indices: torch.Tensor, device,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Compute the sign matrix from commutation parity and metric signature.

        Handles three signature types:
        - Positive (i < p): e_i^2 = +1
        - Negative (p <= i < p+q): e_i^2 = -1
        - Null (i >= p+q): e_i^2 = 0

        Args:
            indices (torch.Tensor): Basis indices.
            device: The device to create tensors on.
            dtype: Floating-point dtype for the returned sign matrix.

        Returns:
            torch.Tensor: Sign matrix.
        """
        # 1. Commutation Sign: Count swaps needed to reorder basis vectors
        # A bit-wise comparison counts inversions
        A = indices.unsqueeze(1) # Row
        B = indices.unsqueeze(0) # Col

        swap_counts = torch.zeros((self.dim, self.dim), dtype=torch.long, device=device)
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

        return (commutator_sign * metric_sign).to(dtype=dtype)

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the Geometric Product.

        Uses vectorized gather + broadcast multiply + sum.

        Args:
            A (torch.Tensor): Left operand [..., Dim].
            B (torch.Tensor): Right operand [..., Dim].

        Returns:
            torch.Tensor: The product AB [..., Dim].
        """
        check_multivector(A, self, "geometric_product(A)")
        check_multivector(B, self, "geometric_product(B)")

        # Gather B coefficients according to Cayley table: B_gathered[..., i, k] = B[..., cayley[i,k]]
        B_gathered = B[..., self.cayley_indices]  # [..., D, D]

        # result[..., k] = sum_i A[..., i] * B[..., cayley[i,k]] * signs[i,k]
        return torch.matmul(A.unsqueeze(-2), B_gathered * self.gp_signs).squeeze(-2)

    def grade_projection(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        """Isolates a specific grade.

        Uses multiplicative masking (mv * float_mask) instead of boolean
        indexing to avoid ``nonzero`` calls that break ``torch.compile``.

        Args:
            mv (torch.Tensor): Multivector [..., Dim].
            grade (int): Target grade.

        Returns:
            torch.Tensor: Projected multivector [..., Dim].
        """
        mask = self.grade_masks_float[grade]
        if mask.dtype != mv.dtype:
            mask = mask.to(dtype=mv.dtype)
        return mv * mask

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        """Computes the reversion. The Clifford conjugate.

        Args:
            mv (torch.Tensor): Input multivector [..., Dim].

        Returns:
            torch.Tensor: Reversed multivector [..., Dim].
        """
        rev = self.rev_signs
        if rev.dtype != mv.dtype:
            rev = rev.to(dtype=mv.dtype)
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
        B_gathered = B[..., self.cayley_indices]
        return torch.matmul(A.unsqueeze(-2), B_gathered * self.wedge_gp_signs).squeeze(-2)

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
        # Use gather instead of boolean indexing (compile-friendly)
        bv_idx_exp = self._bv_indices.expand(*A.shape[:-1], -1)
        bv_coeffs = torch.gather(A, -1, bv_idx_exp)          # [..., num_bv]

        # Grade-1 indices: powers of 2 for basis vectors
        g1_idx = torch.arange(self.n, device=A.device)
        g1_idx = (1 << g1_idx).long()                         # [n]
        g1_idx_exp = g1_idx.expand(*B.shape[:-1], -1)
        v_coeffs = torch.gather(B, -1, g1_idx_exp)            # [..., n]

        rc = self.rc_action
        if rc.dtype != A.dtype:
            rc = rc.to(dtype=A.dtype)

        # M[..., i, j] = sum_b bv_coeffs[..., b] * rc_action[b, i, j]
        M = torch.einsum('...b, bij -> ...ij', bv_coeffs, rc)  # [..., n, n]
        result_v = torch.matmul(M, v_coeffs.unsqueeze(-1)).squeeze(-1)  # [..., n]

        result = torch.zeros_like(A)
        result.scatter_(-1, g1_idx_exp, result_v)
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
        B_gathered = B[..., self.cayley_indices]
        return torch.matmul(A.unsqueeze(-2), B_gathered * self.inner_gp_signs).squeeze(-2)

    def commutator(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the commutator (Lie bracket): [A, B] = AB - BA.

        Single-pass implementation using precomputed antisymmetric signs
        (same structure as :meth:`wedge` but without the 1/2 factor).

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Commutator [A, B] [..., dim].
        """
        B_gathered = B[..., self.cayley_indices]
        return torch.matmul(A.unsqueeze(-2), B_gathered * self.comm_gp_signs).squeeze(-2)

    def anti_commutator(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the anti-commutator: {A, B} = AB + BA.

        Single-pass implementation using precomputed symmetric signs
        (same structure as :meth:`inner_product` but without the 1/2 factor).

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Anti-commutator {A, B} [..., dim].
        """
        B_gathered = B[..., self.cayley_indices]
        return torch.matmul(A.unsqueeze(-2), B_gathered * self.anti_comm_gp_signs).squeeze(-2)

    def blade_inverse(self, blade: torch.Tensor) -> torch.Tensor:
        """Compute the inverse of a blade: B^{-1} = B_rev / <B * B_rev>_0.

        Works for any simple blade (non-degenerate). For null blades the
        scalar denominator is clamped to avoid division by zero.

        Args:
            blade (torch.Tensor): Blade multivector [..., dim].

        Returns:
            torch.Tensor: Inverse blade [..., dim].
        """
        blade_rev = self.reverse(blade)
        blade_sq = self.geometric_product(blade, blade_rev)
        scalar = blade_sq[..., 0:1].clamp(min=self.eps_sq)
        return blade_rev / scalar

    def sandwich_product(self, R: torch.Tensor, x: torch.Tensor,
                         R_rev: torch.Tensor = None) -> torch.Tensor:
        """Optimized sandwich product R x R~ via action matrix.

        Builds a [N, D, D] sandwich action matrix from the rotor, then applies
        it to all C channels via a single batched matmul.  This is much faster
        than two separate ``geometric_product`` calls when x has extra channel
        dimensions that R does not.

        Memory: O(N*D*D) where N = batch (without channels).
        Compare to naive: O(N*C*D*D) - a factor of C improvement.

        Args:
            R: Rotors [N, D] (2-D, batch-flattened).
            x: Multivectors [N, C, D] (3-D, C channels per rotor).
            R_rev: Optional precomputed reverse of R [N, D].

        Returns:
            Sandwiched result [N, C, D].
        """
        if R_rev is None:
            R_rev = self.reverse(R)

        ci = self.cayley_indices  # [D, D], ci[i, j] = i ^ j

        # Left-multiplication matrix L_R:  L_R[n, k, j] = R[n, j^k] * gp_signs[j^k, k]
        R_gathered = R[:, ci]  # [N, D(j), D(k)]
        L_R = R_gathered.permute(0, 2, 1) * self._left_sign_T.unsqueeze(0)

        # Right-multiplication matrix R_{R~}:  R_Rr[n, k, i] = R~[n, i^k] * gp_signs[i, k]
        gp_T = self.gp_signs.T
        Rr_gathered = R_rev[:, ci]  # [N, D(i), D(k)]
        R_Rr = Rr_gathered.permute(0, 2, 1) * gp_T.unsqueeze(0)

        # Sandwich matrix:  M = R_Rr @ L_R   ->   (R x R~)[k] = sum_j M[k, j] * x[j]
        M = torch.bmm(R_Rr, L_R)  # [N, D, D]

        # Apply to all channels:  result[n, c, k] = sum_j M[n, k, j] * x[n, c, j]
        return torch.matmul(x, M.transpose(-2, -1))

    def per_channel_sandwich(self, R: torch.Tensor, x: torch.Tensor,
                              R_rev: torch.Tensor = None) -> torch.Tensor:
        """Sandwich product with per-channel rotors via action matrices.

        Each channel c has its own rotor R[c]. Builds a [C, D, D] action matrix
        (one per channel), then applies to all batch elements in one matmul.

        Memory: O(C*D*D + B*C*D) vs naive two-GP: O(2*B*C*D*D).

        Args:
            R: Per-channel rotors [C, D].
            x: Batched multivectors [B, C, D].
            R_rev: Optional precomputed reverse of R [C, D].

        Returns:
            Sandwiched result [B, C, D].
        """
        if R_rev is None:
            R_rev = self.reverse(R)

        ci = self.cayley_indices  # [D, D]

        # Build per-channel action matrices M[c, k, j]
        R_gathered = R[:, ci]                                    # [C, D, D]
        L_R = R_gathered.permute(0, 2, 1) * self._left_sign_T.unsqueeze(0)

        gp_T = self.gp_signs.T
        Rr_gathered = R_rev[:, ci]                               # [C, D, D]
        R_Rr = Rr_gathered.permute(0, 2, 1) * gp_T.unsqueeze(0)

        M = torch.bmm(R_Rr, L_R)  # [C, D, D]

        # Apply: result[b, c, k] = sum_j M[c, j, k] * x[b, c, j]
        # x: [B, C, D], M.T: [C, D, D] -> einsum or matmul with broadcast
        return torch.einsum('bcd,cdk->bck', x, M.transpose(-2, -1))

    def multi_rotor_sandwich(self, R: torch.Tensor, x: torch.Tensor,
                              R_rev: torch.Tensor = None) -> torch.Tensor:
        """Sandwich product with K rotors applied to C-channel input.

        Builds K action matrices [K, D, D] once, then applies all K
        rotors to x in a single einsum.  This replaces the naive
        two-sequential-geometric-product approach used by MultiRotorLayer.

        Memory: O(K*D*D) setup + O(B*C*K*D) apply.
        Compare to naive two-GP: O(2*B*C*K*D*D).

        Args:
            R: Per-rotor versors [K, D].
            x: Batched multivectors [B, C, D].
            R_rev: Optional precomputed reverse/inverse of R [K, D].

        Returns:
            Per-rotor sandwiched result [B, C, K, D].
        """
        if R_rev is None:
            R_rev = self.reverse(R)

        ci = self.cayley_indices  # [D, D]

        R_gathered = R[:, ci]                                    # [K, D, D]
        L_R = R_gathered.permute(0, 2, 1) * self._left_sign_T.unsqueeze(0)

        gp_T = self.gp_signs.T
        Rr_gathered = R_rev[:, ci]                               # [K, D, D]
        R_Rr = Rr_gathered.permute(0, 2, 1) * gp_T.unsqueeze(0)

        M = torch.bmm(R_Rr, L_R)  # [K, D, D]

        # Apply all K rotors: x[B,C,D] @ M[K,D,D].T -> [B,C,K,D]
        return torch.einsum('bcd,kde->bcke', x, M.transpose(-2, -1))

    def pseudoscalar_product(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by the unit pseudoscalar: x * I.

        Maps grade-k to grade-(n-k) (Hodge dual).  Computed as a simple
        permutation with sign flips - no geometric product needed.

        Args:
            x: Multivector [..., D].

        Returns:
            Result [..., D].
        """
        ps_signs = self._ps_signs
        if ps_signs.dtype != x.dtype:
            ps_signs = ps_signs.to(dtype=x.dtype)

        return x[..., self._ps_source] * ps_signs

    def blade_project(self, mv: torch.Tensor, blade: torch.Tensor) -> torch.Tensor:
        """Project multivector onto blade subspace: (mv . B) B^{-1}.

        Args:
            mv (torch.Tensor): Multivector to project [..., dim].
            blade (torch.Tensor): Blade defining the subspace [..., dim].

        Returns:
            torch.Tensor: Projected multivector [..., dim].
        """
        inner = self.inner_product(mv, blade)
        return self.geometric_product(inner, self.blade_inverse(blade))

    def blade_reject(self, mv: torch.Tensor, blade: torch.Tensor) -> torch.Tensor:
        """Reject multivector from blade subspace: mv - proj_B(mv).

        The orthogonal complement of the projection onto blade.

        Args:
            mv (torch.Tensor): Multivector to reject [..., dim].
            blade (torch.Tensor): Blade defining the subspace [..., dim].

        Returns:
            torch.Tensor: Rejected multivector [..., dim].
        """
        return mv - self.blade_project(mv, blade)

    def grade_involution(self, mv: torch.Tensor) -> torch.Tensor:
        """Grade involution (main involution): x_hat = sum (-1)^k <x>_k.

        Flips sign of all odd-grade components, preserves even-grade.
        This is an algebra automorphism: (AB)^ = A_hat B_hat.

        Args:
            mv (torch.Tensor): Input multivector [..., dim].

        Returns:
            torch.Tensor: Involuted multivector [..., dim].
        """
        signs = self._involution_signs
        if signs.dtype != mv.dtype:
            signs = signs.to(dtype=mv.dtype)
        return mv * signs

    def clifford_conjugation(self, mv: torch.Tensor) -> torch.Tensor:
        """Clifford conjugation: bar{x} = grade_involution(reverse(x)).

        Combines reversion and grade involution. For a k-blade:
            bar{e_I} = (-1)^k * (-1)^{k(k-1)/2} * e_I

        This is an anti-automorphism: bar{AB} = bar{B} bar{A}.

        Args:
            mv (torch.Tensor): Input multivector [..., dim].

        Returns:
            torch.Tensor: Conjugated multivector [..., dim].
        """
        cs = self.conj_signs
        if cs.dtype != mv.dtype:
            cs = cs.to(dtype=mv.dtype)
        return mv * cs

    def norm_sq(self, mv: torch.Tensor) -> torch.Tensor:
        """Squared norm: <x * reverse(x)>_0.

        Returns the scalar (grade-0) part of the product of a multivector
        with its reverse. For blades, this equals the square of the
        magnitude with the appropriate sign from the metric.

        Optimized: the scalar component of A*~A is ``sum_i a_i^2 * rev_signs[i]
        * cayley_signs[i, i]``. No full geometric product needed.

        Args:
            mv (torch.Tensor): Input multivector [..., dim].

        Returns:
            torch.Tensor: Scalar norm squared [..., 1].
        """
        # <x ~x>_0 = sum_i x_i^2 * rev_signs[i] * diag_signs[i]
        signs = self._norm_sq_signs
        if signs.dtype != mv.dtype:
            signs = signs.to(dtype=mv.dtype)
        return (mv * mv * signs).sum(dim=-1, keepdim=True)

    def left_contraction(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Left contraction: A _| B.

        Selects components where grade(A) <= grade(B) and
        grade(result) = grade(B) - grade(A). This is the standard
        contraction used in GA for projection-like operations.

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Left contraction A _| B [..., dim].
        """
        B_gathered = B[..., self.cayley_indices]
        return torch.matmul(A.unsqueeze(-2), B_gathered * self.lc_gp_signs).squeeze(-2)

    def dual(self, mv: torch.Tensor) -> torch.Tensor:
        """Hodge dual: x* = x I^{-1}, maps grade-k to grade-(n-k).

        Equivalent to ``pseudoscalar_product`` but with conventional name.

        Args:
            mv (torch.Tensor): Input multivector [..., dim].

        Returns:
            torch.Tensor: Dual multivector [..., dim].
        """
        return self.pseudoscalar_product(mv)

    def reflect(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Reflect x through the hyperplane orthogonal to vector n.

        Implements the versor reflection: x' = -n x n^{-1}.

        Uses the sandwich product action-matrix when x has a channel
        dimension (3-D input), falling back to two sequential geometric
        products for general shapes.

        Args:
            x (torch.Tensor): Multivector to reflect [..., dim].
            n (torch.Tensor): Normal vector (grade-1) [..., dim].

        Returns:
            torch.Tensor: Reflected multivector [..., dim].
        """
        n_hat = self.grade_involution(n)   # -n for grade-1
        n_inv = self.blade_inverse(n)

        # Use sandwich machinery when shapes allow it
        if x.dim() == 3 and n.dim() == 2 and x.shape[0] != n.shape[0]:
            # n: [C, D], x: [B, C, D] -> per_channel_sandwich
            return self.per_channel_sandwich(n_hat, x, n_inv)
        if x.dim() == 3 and n.dim() == 2 and x.shape[0] == n.shape[0]:
            # n: [N, D], x: [N, C, D] -> sandwich_product
            return self.sandwich_product(n_hat, x, n_inv)

        return self.geometric_product(
            self.geometric_product(n_hat, x), n_inv
        )

    def versor_product(self, V: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """General versor transformation: V x V^{-1} or hat{V} x V^{-1}.

        For even versors (rotors), this is the sandwich product.
        For odd versors (reflections), the grade involution is applied.
        The parity is determined from the grade structure of V.

        In practice, this computes: grade_involution(V) * x * V^{-1},
        which correctly handles both even and odd versors.

        Args:
            V (torch.Tensor): Versor [..., dim].
            x (torch.Tensor): Multivector to transform [..., dim].

        Returns:
            torch.Tensor: Transformed multivector [..., dim].
        """
        V_inv = self.blade_inverse(V)
        V_hat = self.grade_involution(V)
        return self.geometric_product(
            self.geometric_product(V_hat, x), V_inv
        )

    def exp(self, mv: torch.Tensor) -> torch.Tensor:
        """Exponentiates a bivector to produce a rotor.

        Dispatches by :attr:`exp_policy`:

        - ``FAST``  -- closed-form only (fastest, approximate for
          non-simple bivectors in n >= 4).
        - ``AUTO``  (default) -- closed-form for n <= 3 (exact),
          compiled-safe decomposition for n >= 4.
        - ``EXACT`` -- always compiled-safe decomposition (exact for
          all bivectors, ``torch.compile``-safe).

        Three signature regimes handled in the closed-form path:
            - Elliptic  (B^2 < 0): exp(B) = cos(th) + sin(th)/th * B
            - Hyperbolic (B^2 > 0): exp(B) = cosh(th) + sinh(th)/th * B
            - Parabolic  (B^2 ~ 0): exp(B) ~ 1 + B

        Args:
            mv (torch.Tensor): Pure bivector [..., dim].

        Returns:
            torch.Tensor: Rotor exp(mv) [..., dim].
        """
        from core.decomposition import ExpPolicy

        policy = self._exp_policy

        if policy == ExpPolicy.FAST:
            return self._exp_bivector_closed(mv)

        if policy == ExpPolicy.AUTO:
            if self.n <= 3:
                return self._exp_bivector_closed(mv)
            return self._exp_compiled_safe(mv)

        # ExpPolicy.EXACT
        return self._exp_compiled_safe(mv)

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
        idx_expanded = self._bv_indices.expand(*B.shape[:-1], -1)
        bv_coeffs = torch.gather(B, -1, idx_expanded)  # [..., num_bivectors]

        # Signed squared norm: alpha = Sum_k b_k^2 . (e_k)^2
        # alpha < 0 -> elliptic (Euclidean-like), alpha > 0 -> hyperbolic
        alpha = (bv_coeffs * bv_coeffs * self.bv_sq_scalar).sum(dim=-1, keepdim=True)

        abs_alpha = alpha.abs().clamp(min=self.eps_sq)
        theta = torch.sqrt(abs_alpha)  # [..., 1]

        g0_mask = self.grade_masks_float[0]
        if g0_mask.dtype != B.dtype:
            g0_mask = g0_mask.to(dtype=B.dtype)

        # Dispatch by signature regime (Python branch, no graph break)
        if self._exp_regime == 'elliptic':
            # Pure Euclidean: alpha is always negative, only cos/sinc needed
            cos_theta = torch.cos(theta)
            sinc_theta = torch.where(
                theta > self.eps,
                torch.sin(theta) / theta,
                1.0 - abs_alpha / 6.0,
            )
            return cos_theta * g0_mask + sinc_theta * B

        if self._exp_regime == 'hyperbolic':
            # Pure negative: alpha is always positive, only cosh/sinhc needed
            cosh_theta = torch.cosh(theta)
            sinhc_theta = torch.where(
                theta > self.eps,
                torch.sinh(theta) / theta,
                1.0 + abs_alpha / 6.0,
            )
            return cosh_theta * g0_mask + sinhc_theta * B

        # Mixed signature: need both branches + runtime select
        cos_theta = torch.cos(theta)
        sinc_theta = torch.where(
            theta > self.eps,
            torch.sin(theta) / theta,
            1.0 - abs_alpha / 6.0,
        )
        cosh_theta = torch.cosh(theta)
        sinhc_theta = torch.where(
            theta > self.eps,
            torch.sinh(theta) / theta,
            1.0 + abs_alpha / 6.0,
        )

        is_elliptic = alpha < -self.eps_sq
        is_hyperbolic = alpha > self.eps_sq

        scalar_part = torch.where(
            is_elliptic, cos_theta,
            torch.where(is_hyperbolic, cosh_theta, torch.ones_like(theta))
        )
        coeff_part = torch.where(
            is_elliptic, sinc_theta,
            torch.where(is_hyperbolic, sinhc_theta, torch.ones_like(theta))
        )

        return scalar_part * g0_mask + coeff_part * B

    def _exp_compiled_safe(self, B: torch.Tensor) -> torch.Tensor:
        """Compiled-safe exponential: runs both closed-form and decomposed,
        selects per-element via ``torch.where`` based on simplicity.

        A bivector is simple iff ``B*B`` has no grade-4 component (i.e.
        ``B^2`` is purely scalar).  Both paths are computed unconditionally
        so there is no data-dependent branching.

        Args:
            B (torch.Tensor): Pure bivector [..., dim].

        Returns:
            torch.Tensor: Rotor exp(B) [..., dim].
        """
        from core.decomposition import compiled_safe_decomposed_exp

        R_closed = self._exp_bivector_closed(B)
        R_decomposed = compiled_safe_decomposed_exp(self, B)

        BB = self.geometric_product(B, B)
        # Subtract scalar part, check if residual is negligible
        scalar_part = self.grade_projection(BB, 0)
        non_scalar_energy = (BB - scalar_part).norm(dim=-1, keepdim=True)
        is_simple = non_scalar_energy < self.eps * 100

        return torch.where(is_simple, R_closed, R_decomposed)

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

        .. deprecated::
            Set :attr:`exp_policy` and call :meth:`exp` instead.

        Args:
            mv (torch.Tensor): Pure bivector [..., dim].
            **kwargs: Legacy kwargs (``use_decomposition``, ``k``, etc.).

        Returns:
            torch.Tensor: Rotor exp(mv) [..., dim].
        """
        import warnings
        warnings.warn(
            "exp_decomposed() is deprecated. Set algebra.exp_policy and "
            "use algebra.exp() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Legacy compat: use_decomposition=False means closed-form
        if kwargs.get('use_decomposition') is False:
            return self._exp_bivector_closed(mv)
        return self.exp(mv)
