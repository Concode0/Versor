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
from core.algebra import CliffordAlgebra
from core.search import MetricSearch
from layers.rotor import RotorLayer

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'

    def test_metric_search_euclidean(self):
        """Test that MetricSearch prefers Euclidean signature for Euclidean data."""
        # Create random Euclidean data in 2D
        # x coordinates
        data = torch.randn(20, 2)
        
        searcher = MetricSearch(device=self.device)
        
        # We expect (2, 0) to be the best for random flat data if we measure
        # preservation of "flatness" or basic distances.
        
        best_p, best_q = searcher.search(data)
        
        # It should prefer p=2, q=0
        self.assertEqual((best_p, best_q), (2, 0))

    def test_rotor_pruning(self):
        """Test that RotorLayer correctly prunes small bivector weights."""
        algebra = CliffordAlgebra(p=3, q=0, device='cpu')
        layer = RotorLayer(algebra, channels=1)
        
        # Manually set weights: one large, one small
        with torch.no_grad():
            layer.bivector_weights.fill_(0.0)
            layer.bivector_weights[0, 0] = 1.0 # Large
            layer.bivector_weights[0, 1] = 1e-5 # Small
            
        # Prune
        num_pruned = layer.prune_bivectors(threshold=1e-3)
        
        # p=3 has 3 bivectors (e12, e13, e23).
        # We set index 0 to 1.0, index 1 to 1e-5. Index 2 is 0.0.
        # Both index 1 and 2 are < 1e-3, so they are pruned.
        self.assertEqual(num_pruned, 2)
        self.assertEqual(layer.bivector_weights[0, 0], 1.0)
        self.assertEqual(layer.bivector_weights[0, 1], 0.0)

    def test_sparsity_loss(self):
        """Test that sparsity loss returns L1 norm."""
        algebra = CliffordAlgebra(p=2, q=0, device='cpu')
        layer = RotorLayer(algebra, channels=1)
        
        with torch.no_grad():
            layer.bivector_weights.fill_(0.5)
            
        loss = layer.sparsity_loss()
        # Num bivectors in 2D (e12) is 1. 
        # Weights shape [1, 1], value 0.5 -> L1 = 0.5
        
        expected = torch.tensor(0.5)
        self.assertTrue(torch.isclose(loss, expected))

if __name__ == '__main__':
    unittest.main()