import torch
import unittest
from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer

class TestLayers(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.alg = CliffordAlgebra(3, 0, device=self.device) # 3D Euclidean
        
    def test_linear_shape(self):
        # Batch=4, In=2 channels, Out=3 channels
        # x: [4, 2, 8]
        x = torch.randn(4, 2, 8)
        layer = CliffordLinear(self.alg, 2, 3)
        y = layer(x)
        self.assertEqual(y.shape, (4, 3, 8))
        
    def test_rotor_shape(self):
        # Batch=4, Channels=5
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(self.alg, 5)
        y = layer(x)
        self.assertEqual(y.shape, (4, 5, 8))
        
        # Test equivariance (norm preservation for vector part)
        # Vector part is indices 1,2,4 (for 3D basis 1, e1, e2, e3... indices are bitmasks)
        # 1=001, 2=010, 4=100
        vec_indices = [1, 2, 4]
        
        # Create pure vector input
        x_vec = torch.zeros(4, 5, 8)
        x_vec[..., vec_indices] = torch.randn(4, 5, 3)
        
        y_vec = layer(x_vec)
        
        # Norm should be preserved
        x_norm = x_vec.norm(dim=-1)
        y_norm = y_vec.norm(dim=-1)
        
        # Note: Rotor preserves magnitude of the multivector, 
        # and specifically rotates k-vectors to k-vectors.
        # So the norm of the whole multivector should be preserved exactly.
        
        self.assertTrue(torch.allclose(x_norm, y_norm, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
