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
import math

class CliffordAlgebra:
    """Differentiable Clifford algebra kernel with memory-optimized blocked accumulation.

    Handles geometric product, grade projection, and rotor operations.

    Attributes:
        p (int): Positive signature dimensions.
        q (int): Negative signature dimensions.
        n (int): Total dimensions (p + q).
        dim (int): Total basis elements (2^n).
        device (str): Computation device.
    """
    _CACHED_TABLES = {}

    def __init__(self, p: int, q: int = 0, device='cuda'):
        """Initialize the algebra and cache the Cayley table.

        Args:
            p (int): Positive dimensions (+1).
            q (int, optional): Negative dimensions (-1). Defaults to 0.
            device (str, optional): The device on which computations are performed. Defaults to 'cuda'.
        """
        self.p, self.q = p, q
        self.n = p + q
        self.dim = 2 ** self.n
        self.device = device
        
        # Cache Cayley tables to avoid recomputation
        cache_key = (p, q, str(device))
        if cache_key not in CliffordAlgebra._CACHED_TABLES:
            CliffordAlgebra._CACHED_TABLES[cache_key] = self._generate_cayley_table()

        (
            self.cayley_indices,
            self.cayley_signs,
            self.gp_signs,
            self.grade_masks,
            self.rev_signs,
        ) = CliffordAlgebra._CACHED_TABLES[cache_key]

    @property
    def num_grades(self) -> int:
        """Counts the number of grades (n + 1)."""
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

        Args:
            mv (torch.Tensor): Input multivector [..., dim].

        Returns:
            torch.Tensor: Grade norms [..., num_grades].
        """
        batch_shape = mv.shape[:-1]
        res = torch.zeros(*batch_shape, self.num_grades, device=mv.device, dtype=mv.dtype)
        for k in range(self.num_grades):
            mv_k = self.grade_projection(mv, k)
            res[..., k] = mv_k.norm(dim=-1)
        return res

    def _generate_cayley_table(self):
        """Precompute the Cayley table, grade masks, and reversion signs."""
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

        return cayley_indices, cayley_signs, gp_signs, grade_masks, rev_signs

    def _compute_signs(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute the sign matrix from commutation parity and metric signature.

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
        
        # 2. Metric Sign: e_i^2 = -1 if i >= p
        intersection = A & B
        
        # Mask for negative signature dimensions
        q_mask = 0
        for i in range(self.p, self.n):
            q_mask |= (1 << i)
            
        neg_intersection = intersection & q_mask
        
        # Count set bits in negative intersection
        neg_cnt = torch.zeros_like(neg_intersection)
        temp_neg = neg_intersection
        for _ in range(self.n):
            neg_cnt += (temp_neg & 1)
            temp_neg = temp_neg >> 1
            
        metric_sign = (-1) ** neg_cnt
        
        return commutator_sign * metric_sign

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the Geometric Product.

        Uses blocked accumulation to optimize VRAM usage.

        Args:
            A (torch.Tensor): Left operand [..., Dim].
            B (torch.Tensor): Right operand [..., Dim].

        Returns:
            torch.Tensor: The product AB [..., Dim].
        """
        try:
            batch_shape = torch.broadcast_shapes(A.shape[:-1], B.shape[:-1])
        except RuntimeError:
            raise RuntimeError(f"Shapes {A.shape} and {B.shape} are not broadcastable")

        D = self.dim
        result = torch.zeros(*batch_shape, D, device=A.device, dtype=A.dtype)
        A_expanded = A.unsqueeze(-1) # [..., D, 1]
        
        # Block size for vectorization vs memory trade-off
        BLOCK_SIZE = 32
        
        for k_start in range(0, D, BLOCK_SIZE):
            k_end = min(k_start + BLOCK_SIZE, D)
            
            # Gather relevant columns from B based on Cayley table
            idx_chunk = self.cayley_indices[:, k_start:k_end]
            B_gathered = B[..., idx_chunk]
            
            signs_chunk = self.gp_signs[:, k_start:k_end] # [D, Chunk]
            
            # Compute term: A_i * B_j * Sign
            term = A_expanded * B_gathered * signs_chunk
            
            # Sum over inner dimension to get result coefficients
            chunk_res = term.sum(dim=-2)
            result[..., k_start:k_end] = chunk_res
            
        return result
        
    def grade_projection(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        """Isolates a specific grade.

        Args:
            mv (torch.Tensor): Multivector.
            grade (int): Target grade.

        Returns:
            torch.Tensor: Projected multivector.
        """
        mask = self.grade_masks[grade].to(mv.device)
        result = torch.zeros_like(mv)
        result[..., mask] = mv[..., mask]
        return result

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        """Computes the reversion. The Clifford conjugate.

        Args:
            mv (torch.Tensor): Input multivector.

        Returns:
            torch.Tensor: Reversed multivector.
        """
        return mv * self.rev_signs.to(dtype=mv.dtype, device=mv.device)

    def wedge(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the wedge (outer) product: A ∧ B = (AB - BA)/2.

        The wedge product is antisymmetric and grade-raising.

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
            from Irreducibles." arXiv:2507.11688v1 [cs.LG]

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Wedge product A ∧ B [..., dim].
        """
        AB = self.geometric_product(A, B)
        BA = self.geometric_product(B, A)
        return (AB - BA) / 2.0

    def right_contraction(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the right contraction: A ⌋ B.

        For a bivector b and vector v, this extracts the grade-1 component
        of the geometric product. This is the core operation in GA power iteration.

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
            from Irreducibles." arXiv:2507.11688v1 [cs.LG], Algorithm 2

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Right contraction A ⌋ B [..., dim].
        """
        # Right contraction of A into B
        # For bivector-vector contraction, we extract grade-1 component
        AB = self.geometric_product(A, B)
        return self.grade_projection(AB, 1)

    def inner_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes the inner product: A · B = (AB + BA)/2.

        The inner product is symmetric and grade-lowering. Useful for computing
        norms and scalar parts of multivectors.

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
            from Irreducibles." arXiv:2507.11688v1 [cs.LG]

        Args:
            A (torch.Tensor): Left operand [..., dim].
            B (torch.Tensor): Right operand [..., dim].

        Returns:
            torch.Tensor: Inner product A · B [..., dim].
        """
        AB = self.geometric_product(A, B)
        BA = self.geometric_product(B, A)
        return (AB + BA) / 2.0

    def exp(self, mv: torch.Tensor, order: int = 12) -> torch.Tensor:
        """Exponentiates a bivector to generate a rotor.

        Uses scaling and squaring to ensure numerical stability.

        Args:
            mv (torch.Tensor): Input bivector.
            order (int, optional): Taylor order. Defaults to 12.

        Returns:
            torch.Tensor: exp(A).
        """
        # 1. Scale down: Find k such that norm(A)/2^k <= 1.0
        norm = mv.norm(dim=-1, keepdim=True)
        k = torch.ceil(torch.log2(torch.clamp(norm, min=1.0))).int()
        
        # Use max k over batch for uniform tensor operations
        max_k = k.max().item()
        if max_k > 0:
            scale_factor_global = 2.0 ** max_k
            mv_scaled = mv / scale_factor_global
        else:
            mv_scaled = mv

        # 2. Taylor Series Approximation on scaled input
        res = torch.zeros_like(mv)
        res[..., 0] = 1.0 # Scalar 1
        
        term = torch.zeros_like(mv)
        term[..., 0] = 1.0
        
        for i in range(1, order + 1):
            term = self.geometric_product(term, mv_scaled)
            res = res + term / math.factorial(i)

        # 3. Square up max_k times to recover original scale
        if max_k > 0:
            for _ in range(int(max_k)):
                res = self.geometric_product(res, res)

        return res

    def exp_decomposed(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Exponentiates a bivector using optional decomposition.

        This method provides an alternative to the standard exp() that decomposes
        the bivector into simple components before exponentiating. This can be
        more parameter-efficient and interpretable for certain applications.

        Reference:
            Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
            from Irreducibles." arXiv:2507.11688v1 [cs.LG]

        Args:
            mv (torch.Tensor): Input bivector [..., dim].
            **kwargs: Additional arguments passed to core.decomposition.exp_decomposed.
                use_decomposition (bool): If True, use decomposition. Default True.
                k (int, optional): Number of simple components.
                threshold (float): Convergence threshold. Default 1e-6.
                max_iterations (int): Max iterations. Default 100.

        Returns:
            torch.Tensor: Rotor exp(mv) [..., dim].
        """
        from core.decomposition import exp_decomposed
        # Set default to actually use decomposition
        if 'use_decomposition' not in kwargs:
            kwargs['use_decomposition'] = True
        return exp_decomposed(self, mv, **kwargs)
