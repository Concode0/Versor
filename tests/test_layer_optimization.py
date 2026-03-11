# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import pytest
from core.algebra import CliffordAlgebra
from layers import RotorLayer

pytestmark = pytest.mark.unit


class TestOptimization:
    def test_rotor_pruning(self):
        """Test that RotorLayer correctly prunes small bivector weights."""
        algebra = CliffordAlgebra(p=3, q=0, device='cpu')
        layer = RotorLayer(algebra, channels=1)

        # Manually set weights: one large, one small
        with torch.no_grad():
            layer.bivector_weights.fill_(0.0)
            layer.bivector_weights[0, 0] = 1.0  # Large
            layer.bivector_weights[0, 1] = 1e-5  # Small

        # Prune
        num_pruned = layer.prune_bivectors(threshold=1e-3)

        # p=3 has 3 bivectors (e12, e13, e23).
        # We set index 0 to 1.0, index 1 to 1e-5. Index 2 is 0.0.
        # Both index 1 and 2 are < 1e-3, so they are pruned.
        assert num_pruned == 2
        assert layer.bivector_weights[0, 0] == 1.0
        assert layer.bivector_weights[0, 1] == 0.0

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
        assert torch.isclose(loss, expected)
