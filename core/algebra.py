import torch
import math

class CliffordAlgebra:
    _CACHED_TABLES = {}

    def __init__(self, p: int, q: int = 0, device='cuda'):
        """
        p: Positive Measure Dimension
        q: Negative Measure Dimension
        n = p + q: Total Dimension
        dim = 2^n: Multi-vector's Depth
        """
        self.p, self.q = p, q
        self.n = p + q
        self.dim = 2 ** self.n
        self.device = device
        
        # Check cache
        cache_key = (p, q, str(device))
        if cache_key not in CliffordAlgebra._CACHED_TABLES:
            CliffordAlgebra._CACHED_TABLES[cache_key] = self._generate_cayley_table()
            
        self.cayley_indices, self.cayley_signs, self.gp_signs = CliffordAlgebra._CACHED_TABLES[cache_key]

    @property
    def is_euclidean(self):
        return self.q == 0

    def _generate_cayley_table(self):
        # For all Basis - 0 ~ 2^n-1
        indices = torch.arange(self.dim, device=self.device)
        
        # Result_Index = Index_A XOR Index_B
        # cayley_indices[i, j] = i ^ j
        cayley_indices = indices.unsqueeze(0) ^ indices.unsqueeze(1)
        
        # cayley_signs[i, j] = sign(e_i * e_j)
        cayley_signs = self._compute_signs(indices) 
        
        # Precompute signs for geometric_product accumulation
        # gp_signs[i, k] = sign(e_i * e_{i^k})
        # This aligns the signs so that the k-th column corresponds to the k-th result basis vector
        # We need to gather from cayley_signs.
        # Target indices in cayley_signs: rows=i, cols=i^k (which is cayley_indices[i, k])
        gp_signs = torch.gather(cayley_signs, 1, cayley_indices)
        
        return cayley_indices, cayley_signs, gp_signs

    def _compute_signs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the sign table for the geometric product.
        Sign depends on:
        1. Number of swaps required to reorder basis vectors (anti-commutativity).
        2. Metric signature of the basis vectors that square to scalar (contraction).
        """
        # 1. Commutation Sign (swaps)
        # Count number of set bits (grade)
        grades = torch.zeros_like(indices)
        for i in range(self.n):
            grades += (indices >> i) & 1
            
        # grade_A * grade_B is not enough. We need specific bit interactions.
        # Canonical reordering sign for e_A e_B -> e_{A^B}
        # Based on "Efficient Implementation of Geometric Algebra"
        
        # We can compute signs by iterating or bit-logic. 
        # For small dimensions (n < 10), iteration is acceptable for __init__.
        # But let's try a vectorized bitwise approach.
        
        # Sign from swaps: s = (-1)^(Sum_{i>j} a_i b_j) where a_i is i-th bit of A.
        # This counts how many bits in A are "higher" than bits in B, which is related to swaps.
        
        # Expand indices for broadcasting
        # shape: [dim, dim]
        A = indices.unsqueeze(1) # Row
        B = indices.unsqueeze(0) # Col
        
        # Calculate swap sign
        swap_counts = torch.zeros((self.dim, self.dim), dtype=torch.long, device=self.device)
        for i in range(self.n):
            # Bits of A at position i
            a_i = (A >> i) & 1
            # Bits of B at positions < i
            # Create mask for bits less than i
            lower_mask = (1 << i) - 1
            b_lower = (B & lower_mask)
            # Count set bits in b_lower
            # (There are faster bit count hacks, but loop is clear for now)
            b_lower_cnt = torch.zeros_like(B)
            temp_b = b_lower
            for _ in range(self.n): # iterate max bits
                b_lower_cnt += (temp_b & 1)
                temp_b = temp_b >> 1
            
            swap_counts += a_i * b_lower_cnt
            
        commutator_sign = (-1) ** swap_counts
        
        # 2. Metric Sign (contraction)
        # e_i * e_i = +1 for i < p, -1 for i >= p (negative signature)
        # When A and B share bits, those basis vectors "cancel" out.
        # If the shared basis vector has negative signature, we get a -1.
        
        # Intersection: bits present in both A and B
        intersection = A & B
        
        # We need to know which bits in intersection correspond to negative signature.
        # Negative signature indices are from p to p+q-1.
        # Create a mask for q dimensions
        q_mask = 0
        for i in range(self.p, self.n):
            q_mask |= (1 << i)
            
        neg_intersection = intersection & q_mask
        
        # Count set bits in neg_intersection
        neg_cnt = torch.zeros_like(neg_intersection)
        temp_neg = neg_intersection
        for _ in range(self.n):
            neg_cnt += (temp_neg & 1)
            temp_neg = temp_neg >> 1
            
        metric_sign = (-1) ** neg_cnt
        
        return commutator_sign * metric_sign

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Geometric Product
        A, B Shape: [Batch, 2^n] or [Batch, ..., 2^n]
        Optimized to reduce memory usage by avoiding large intermediate tensor (Batch, Dim, Dim).
        """
        # A: [..., D]
        # B: [..., D]
        # Determine broadcasted batch shape
        try:
            batch_shape = torch.broadcast_shapes(A.shape[:-1], B.shape[:-1])
        except RuntimeError:
            raise RuntimeError(f"Shapes {A.shape} and {B.shape} are not broadcastable")

        D = self.dim
        
        # Result accumulator
        result = torch.zeros(*batch_shape, D, device=A.device, dtype=A.dtype)
        
        # Expand A for broadcasting in the inner loop: [..., D, 1]
        A_expanded = A.unsqueeze(-1)
        
        # Flatten batch dimensions of B to simplify gather? 
        # No, gather works on specific dimension.
        # We want to gather along the last dimension of B.
        # B: [..., D]
        
        # Block size for vectorization (trade-off between memory and loop overhead)
        # 32 is a reasonable default for typical GA dimensions (8, 16, 32, 64, 128, 256)
        BLOCK_SIZE = 32
        
        for k_start in range(0, D, BLOCK_SIZE):
            k_end = min(k_start + BLOCK_SIZE, D)
            # chunk of result indices
            
            # 1. Gather B values
            # Indices for B: cayley_indices[i, k] = i ^ k
            # We want indices[i, k] for k in [k_start, k_end]
            # shape: [D, Chunk]
            idx_chunk = self.cayley_indices[:, k_start:k_end]
            
            # We need to gather B along its last dim using these indices.
            # B is [..., D]. idx_chunk is [D, Chunk].
            # We want B_gathered [..., D, Chunk]
            # B[..., i] corresponds to A[..., i]
            # B_gathered[..., i, c] = B[..., idx_chunk[i, c]]
            
            # PyTorch gather requires index to match dimensions.
            # B: [Batch..., D]
            # idx: [D, Chunk] -> Broadcast to [Batch..., D, Chunk]
            
            # Expanding idx to match batch dims might be expensive memory-wise if Batch is large.
            # Alternative: B[..., idx] works in numpy style advanced indexing?
            # B[..., idx_chunk] where idx_chunk is tensor.
            # If B is [B, D] and idx is [D, C], B[:, idx] -> [B, D, C].
            # This works directly in PyTorch!
            B_gathered = B[..., idx_chunk]
            
            # 2. Gather Signs
            # gp_signs[i, k]
            signs_chunk = self.gp_signs[:, k_start:k_end] # [D, Chunk]
            
            # 3. Compute Term
            # A_expanded: [..., D, 1]
            # B_gathered: [..., D, Chunk]
            # signs_chunk: [D, Chunk] (broadcasts to batch)
            
            term = A_expanded * B_gathered * signs_chunk
            
            # 4. Sum over i (dim -2) to get C_k
            # [..., Chunk]
            chunk_res = term.sum(dim=-2)
            
            # 5. Store
            result[..., k_start:k_end] = chunk_res
            
        return result
        
    def grade_projection(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        """
        Project multivector to specific grade
        """
        # Determine which indices correspond to the requested grade
        # This could be pre-computed in __init__
        mask = torch.zeros(self.dim, device=self.device, dtype=torch.bool)
        for i in range(self.dim):
            # count set bits
            cnt = 0
            temp = i
            while temp > 0:
                if temp & 1: cnt += 1
                temp >>= 1
            if cnt == grade:
                mask[i] = True
                
        result = torch.zeros_like(mv)
        result[..., mask] = mv[..., mask]
        return result

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        """
        Reverse (Reversion) of a multivector.
        The reverse of a k-vector A is (-1)^(k(k-1)/2) A.
        """
        result = mv.clone()
        for i in range(self.dim):
            # Calculate grade k
            k = 0
            temp = i
            while temp > 0:
                if temp & 1: k += 1
                temp >>= 1
            
            sign = (-1) ** (k * (k - 1) // 2)
            if sign == -1:
                result[..., i] = -result[..., i]
        return result

    def exp(self, mv: torch.Tensor, order: int = 12) -> torch.Tensor:
        """
        Exponential of a multivector using Taylor series.
        exp(A) = 1 + A + A^2/2! + ...
        """
        res = torch.zeros_like(mv)
        res[..., 0] = 1.0 # scalar 1
        
        term = torch.zeros_like(mv)
        term[..., 0] = 1.0
        
        for i in range(1, order + 1):
            term = self.geometric_product(term, mv)
            res = res + term / math.factorial(i)
            
        return res