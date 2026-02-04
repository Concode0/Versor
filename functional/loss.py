import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricMSELoss(nn.Module):
    def __init__(self, algebra=None):
        """
        Standard MSE on Multivector Coefficients.
        Equivalent to squared Euclidean distance in the embedding space.
        """
        super().__init__()
        self.algebra = algebra

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred, target: [Batch, ..., Dim]
        # Sum over Dim (coefficients), Mean over Batch
        return F.mse_loss(pred, target, reduction='mean')

class SubspaceLoss(nn.Module):
    def __init__(self, algebra, target_indices: list = None, exclude_indices: list = None):
        """
        Penalizes energy (magnitude squared) in specific basis components.
        Used for Manifold Regularization (forcing data to lie on a subspace).
        
        Args:
            algebra: CliffordAlgebra instance.
            target_indices: List of indices that are ALLOWED (penalty=0). All others penalized.
            exclude_indices: List of indices to PENALIZE. (Alternative to target_indices).
        """
        super().__init__()
        self.algebra = algebra
        
        if target_indices is not None:
            # Create a mask of indices to PENALIZE (inverse of target)
            mask = torch.ones(algebra.dim, dtype=torch.bool)
            mask[target_indices] = False
        elif exclude_indices is not None:
            mask = torch.zeros(algebra.dim, dtype=torch.bool)
            mask[exclude_indices] = True
        else:
            raise ValueError("Must provide target_indices or exclude_indices")
            
        self.register_buffer('penalty_mask', mask)

    def forward(self, x: torch.Tensor):
        # x: [Batch, ..., Dim]
        # Select penalized components
        penalty_components = x[..., self.penalty_mask]
        
        # Mean squared magnitude of these components
        loss = (penalty_components ** 2).sum(dim=-1).mean()
        return loss

class IsometryLoss(nn.Module):
    def __init__(self, algebra):
        """
        Ensures that the transformation preserves the metric norm (quadratic form).
        Important for Rotors (which should preserve lengths).
        Loss = MSE( Q(pred), Q(target) ) where Q(v) is the quadratic form.
        """
        super().__init__()
        self.algebra = algebra
        
        # Metric signature diagonal: [1, 1, -1, ...]
        # We need to construct this from algebra.p and algebra.q
        # But for general multivectors, "norm" is more complex.
        # For vectors v = c_i e_i: v^2 = sum (c_i^2 * e_i^2)
        
        # Let's compute the diagonal metric signature for the basis vectors.
        # e_i^2 can be computed.
        # This is essentially the diagonal of the Cayley table where indices match?
        # No, Cayley table gives the INDEX of the result. We need the SIGN.
        # We want to know s_k such that e_k * e_k = s_k (scalar part).
        
        # Actually, for vectors, Isometry means v^2 is preserved.
        # For general multivectors, R M R~ preserves grades and magnitudes.
        
        self.metric_diag = self._compute_metric_diagonal()

    def _compute_metric_diagonal(self):
        # Compute e_k * e_k for all k=0..Dim-1
        # The result of e_k * e_k is +/- 1 (scalar).
        # We can extract this from algebra logic or simply test it.
        diag = torch.zeros(self.algebra.dim, device=self.algebra.device)
        
        # We can use the algebra to compute e_k * e_k
        # Create a batch of all basis vectors
        basis = torch.eye(self.algebra.dim, device=self.algebra.device)
        
        # Square them: basis * basis
        # Use geometric product
        sq = self.algebra.geometric_product(basis, basis)
        
        # The scalar part (index 0) is what we want.
        diag = sq[:, 0]
        return diag

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # Compute Quadratic Form Q(v) = Sum( coeff_i^2 * metric_i )
        # This approximates v^2 scalar part for orthogonal basis.
        
        # Note: This is strictly valid for Grade-1 vectors or simple blades.
        # For general multivectors, the "squared norm" definition varies.
        # But assuming we want to preserve the "metric magnitude":
        
        pred_sq = (pred ** 2) * self.metric_diag
        target_sq = (target ** 2) * self.metric_diag
        
        pred_norm = pred_sq.sum(dim=-1)
        target_norm = target_sq.sum(dim=-1)
        
        return F.mse_loss(pred_norm, target_norm)

class BivectorRegularization(nn.Module):
    def __init__(self, algebra, grade=2):
        super().__init__()
        self.algebra = algebra
        self.grade = grade

    def forward(self, x: torch.Tensor):
        """
        Penalize components that are NOT of the target grade.
        """
        # Project to target grade
        target_part = self.algebra.grade_projection(x, self.grade)
        # Residual is the "noise"
        residual = x - target_part
        return (residual ** 2).sum(dim=-1).mean()