# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import math
import pytest
import torch
from core.algebra import CliffordAlgebra

pytestmark = pytest.mark.unit


class TestCliffordAlgebra:
    def test_euclidean_2d_cayley(self):
        # E2: e1*e1=1, e2*e2=1
        # Basis: 1, e1, e2, e12
        # e1*e2 = e12
        # e2*e1 = -e12
        alg = CliffordAlgebra(p=2, q=0, device='cpu')

        # indices: 0(1), 1(e1), 2(e2), 3(e12)
        # 1 * 2 (e1 * e2) -> 3 (e12), sign +
        target_idx = alg.cayley_indices[1, 2]
        sign = alg.cayley_signs[1, 2]
        assert target_idx.item() == 3
        assert sign.item() == 1.0

        # 2 * 1 (e2 * e1) -> 3 (e12), sign -
        target_idx = alg.cayley_indices[2, 1]
        sign = alg.cayley_signs[2, 1]
        assert target_idx.item() == 3
        assert sign.item() == -1.0

    def test_geometric_product_simple(self):
        # E2
        alg = CliffordAlgebra(p=2, q=0, device='cpu')

        # A = 2*e1
        A = torch.zeros(1, 4)
        A[0, 1] = 2.0

        # B = 3*e2
        B = torch.zeros(1, 4)
        B[0, 2] = 3.0

        # C = A*B = 6*e12
        C = alg.geometric_product(A, B)
        assert C[0, 3].item() == 6.0

    def test_rotor_exp(self):
        # Rotation in 2D plane by 90 degrees
        # R = exp(-theta/2 * e12)
        # theta = pi/2 -> -pi/4 * e12
        alg = CliffordAlgebra(p=2, q=0, device='cpu')

        B = torch.zeros(1, 4)
        B[0, 3] = 1.0  # unit bivector

        theta = math.pi / 2
        R = alg.exp(-0.5 * theta * B)

        # R = cos(pi/4) - sin(pi/4)e12
        val = math.cos(math.pi / 4)
        assert abs(R[0, 0].item() - val) < 1e-5
        assert abs(R[0, 3].item() - (-val)) < 1e-5

        # Rotate e1 -> e2
        # v = e1
        v = torch.zeros(1, 4)
        v[0, 1] = 1.0

        R_rev = alg.reverse(R)

        # Rv
        Rv = alg.geometric_product(R, v)
        # v' = RvR~
        v_prime = alg.geometric_product(Rv, R_rev)

        # Expected e2
        assert abs(v_prime[0, 1].item() - 0.0) < 1e-5
        assert abs(v_prime[0, 2].item() - 1.0) < 1e-5
