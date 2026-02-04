import torch
from core.algebra import CliffordAlgebra
from core.metric import Metric

class MultiVector:
    def __init__(self, data: torch.Tensor, algebra: CliffordAlgebra):
        """
        data: Tensor of shape [..., 2^n]
        algebra: Instance of CliffordAlgebra
        """
        self.data = data
        self.algebra = algebra

    @classmethod
    def from_tensor(cls, data: torch.Tensor, metric: Metric, device='cpu'):
        algebra = CliffordAlgebra(metric.p, metric.q, device=device)
        return cls(data, algebra)

    def __add__(self, other):
        if isinstance(other, MultiVector):
            assert self.algebra.n == other.algebra.n
            return MultiVector(self.data + other.data, self.algebra)
        return MultiVector(self.data + other, self.algebra)

    def __sub__(self, other):
        if isinstance(other, MultiVector):
            assert self.algebra.n == other.algebra.n
            return MultiVector(self.data - other.data, self.algebra)
        return MultiVector(self.data - other, self.algebra)

    def __mul__(self, other):
        """Geometric Product"""
        if isinstance(other, MultiVector):
            assert self.algebra.n == other.algebra.n
            res = self.algebra.geometric_product(self.data, other.data)
            return MultiVector(res, self.algebra)
        # Scalar multiplication
        return MultiVector(self.data * other, self.algebra)

    def __repr__(self):
        return f"MultiVector(shape={self.data.shape}, metric={self.algebra.p},{self.algebra.q})"

    def project(self, grade: int):
        """Project to specific grade"""
        res = self.algebra.grade_projection(self.data, grade)
        return MultiVector(res, self.algebra)
    
    def grade(self, k: int):
        return self.project(k)
        
    def __getitem__(self, key):
        return MultiVector(self.data[key], self.algebra)
