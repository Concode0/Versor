import torch
import math

class CliffordAlgebra:
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
        
        self.cayley_indices, self.cayley_signs = self._generate_cayley_table()

    @property
    def is_euclidean(self):
        return self.q == 0

    def _generate_cayley_table(self):
        # For all Basis - 0 ~ 2^n-1
        indices = torch.arange(self.dim, device=self.device)
        
        # Result_Index = Index_A XOR Index_B
        cayley_indices = indices.unsqueeze(0) ^ indices.unsqueeze(1)
        
        cayley_signs = self._compute_signs(indices) 
        
        return cayley_indices, cayley_signs

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
        """
        # Ensure input shapes are compatible for broadcasting if needed, 
        # but strictly for this method assuming last dim is 2^n
        
        # A: [..., I, 1] -> [..., I, 1]
        # B: [..., 1, J] -> [..., 1, J]
        # We need to perform outer product on the last dimension and then reduce.
        
        # Let's handle generic batch shapes.
        # Flatten batch dimensions for operation, then restore? 
        # Or use einsum with ellipsis.
        
        # torch.einsum('...i, ...j -> ...ij', A, B)
        prod = torch.einsum('...i, ...j -> ...ij', A, B)
        
        # Apply signs
        weighted_prod = prod * self.cayley_signs
        
        # Flatten the last two dimensions to scatter add
        # Target indices are in cayley_indices [dim, dim]
        # We need to map ...ij to ...k where k = indices[i,j]
        
        # This scatter operation is tricky with arbitrary batch dims in pure pytorch without loop.
        # For fixed batch size, it's easier.
        
        # We use prod shape to determine the actual batch shape after broadcasting
        batch_shape = prod.shape[:-2]
        flat_batch_dim = prod.shape[:-2].numel()
        
        weighted_prod_flat = weighted_prod.reshape(flat_batch_dim, self.dim * self.dim) # [Batch, Dim*Dim]
        
        # Flatten cayley indices: row-major 
        # map (i, j) -> i*dim + j. 
        # cayley_indices stores the *target* basis index k.
        # We want to sum all (i, j) that map to k.
        
        # We can simply iterate over the resulting basis vectors (0 to dim-1)
        # and sum the relevant entries. This is O(dim) iterations but vectorized over batch.
        
        result = torch.zeros(*batch_shape, self.dim, device=A.device, dtype=A.dtype)
        
        # Optimized: pre-compute masks for each k? 
        # Or just use the scatter_add logic if we can linearize correctly.
        
        # Let's try to linearize the target indices for the flat batch.
        # target_indices for a single item: cayley_indices.view(-1) -> [Dim*Dim]
        
        flat_indices = self.cayley_indices.view(-1).repeat(flat_batch_dim, 1) # [Batch, Dim*Dim]
        
        result_flat = result.view(flat_batch_dim, self.dim)
        result_flat.scatter_add_(1, flat_indices, weighted_prod_flat)
        
        return result_flat.view(*batch_shape, self.dim)
        
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