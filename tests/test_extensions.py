# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

import torch
import unittest
from core.cga import ConformalAlgebra
from layers.gnn import CliffordGraphConv
from core.algebra import CliffordAlgebra

class TestExtensions(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'

    def test_cga_null_property(self):
        """
        Test that points embedded in CGA are null vectors (P * P = 0).
        """
        cga = ConformalAlgebra(euclidean_dim=3, device=self.device)
        
        # Random Euclidean points
        x = torch.randn(5, 3, device=self.device)
        
        # Embed
        P = cga.to_cga(x)
        
        # Geometric Product P * P
        # Should be scalar 0
        sq = cga.algebra.geometric_product(P, P)
        
        # Check norm of the result (should be 0)
        # Note: Precision might require loose tolerance
        self.assertTrue(torch.allclose(sq, torch.zeros_like(sq), atol=1e-5),
                        f"P^2 should be 0, got {sq[0, :5]}...")
        
        # Check reconstruction
        x_recon = cga.from_cga(P)
        self.assertTrue(torch.allclose(x, x_recon, atol=1e-5),
                        "Reconstructed x should match original")

    def test_gnn_layer(self):
        """
        Test Clifford GCN forward pass.
        """
        # Algebra (e.g., 2D)
        algebra = CliffordAlgebra(p=2, q=0, device='cpu')
        gnn = CliffordGraphConv(algebra, in_channels=2, out_channels=4)
        
        # 3 Nodes, 2 Channels, 2^2=4 Dim
        x = torch.randn(3, 2, 4)
        
        # Adjacency (3x3) - e.g., line graph 0-1-2
        adj = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        # Normalize (simplified)
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
        
        out = gnn(x, adj)
        
        # Check shape: [3, 4, 4]
        self.assertEqual(out.shape, (3, 4, 4))
        # Check values are not NaN
        self.assertFalse(torch.isnan(out).any())

if __name__ == '__main__':
    unittest.main()
