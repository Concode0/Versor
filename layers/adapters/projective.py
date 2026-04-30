# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch

from core.algebra import CliffordAlgebra
from core.module import CliffordModule


class ProjectiveEmbedding(CliffordModule):
    """Projective Geometric Algebra (PGA) embedding layer.

    Maps Euclidean R^d points to grade-1 elements of Cl(d, 0, 1) and back.

    In PGA, the degenerate basis vector e_0 (where e_0^2 = 0) represents
    the origin. A Euclidean point x is embedded as the 1-vector:

        P(x) = x_1 e_1 + x_2 e_2 + ... + x_d e_d + e_0

    The e_0 component acts as a homogeneous coordinate. Directions (ideal
    points) have e_0 = 0, while finite points have e_0 != 0.

    Extraction normalizes so that the e_0 coefficient equals 1, then
    reads the Euclidean basis coefficients.

    Requires an algebra with signature Cl(d, 0, >=1).

    Attributes:
        euclidean_dim (int): Physical dimension d.
    """

    def __init__(self, algebra: CliffordAlgebra, euclidean_dim: int):
        """Sets up the projective embedding.

        Args:
            algebra: Clifford algebra instance with signature Cl(d, 0, >=1).
            euclidean_dim: Physical dimension d.
        """
        super().__init__(algebra)
        d = euclidean_dim
        assert algebra.p >= d and algebra.r >= 1, (
            f"Projective embedding needs Cl(>={d}, 0, >=1), got Cl({algebra.p},{algebra.q},{algebra.r})"
        )
        self.euclidean_dim = d

        # Grade-1 indices for the Euclidean basis vectors e_1..e_d
        # In binary: e_i has index 2^(i-1), so for i=0..d-1: index = 1 << i
        g1_idx = (1 << torch.arange(d)).long()
        self.register_buffer("_g1_idx", g1_idx)

        # Index of the degenerate basis vector e_0
        # It's the (p+q)-th basis vector, so index = 1 << (p + q)
        idx_e0 = 1 << (algebra.p + algebra.q)
        self.register_buffer("_idx_e0", torch.tensor(idx_e0, dtype=torch.long))

        # Precomputed e_0 multivector for additive embedding
        e0 = torch.zeros(algebra.dim)
        e0[idx_e0] = 1.0
        self.register_buffer("_e0", e0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed Euclidean points into PGA as grade-1 elements.

        Args:
            x: Euclidean points [..., d].

        Returns:
            PGA multivectors [..., dim].
        """
        x_mv = torch.zeros(*x.shape[:-1], self.algebra.dim, device=x.device, dtype=x.dtype)
        x_mv.scatter_(-1, self._g1_idx.expand_as(x), x)
        return x_mv + self._e0

    def embed_direction(self, v: torch.Tensor) -> torch.Tensor:
        """Embed Euclidean directions (ideal points) into PGA.

        Directions have e_0 = 0, representing points at infinity.

        Args:
            v: Direction vectors [..., d].

        Returns:
            PGA multivectors [..., dim] with zero e_0 component.
        """
        x_mv = torch.zeros(*v.shape[:-1], self.algebra.dim, device=v.device, dtype=v.dtype)
        x_mv.scatter_(-1, self._g1_idx.expand_as(v), v)
        return x_mv

    def extract(self, P: torch.Tensor) -> torch.Tensor:
        """Project PGA points back to Euclidean space.

        Normalizes by the e_0 coefficient, then extracts the
        Euclidean basis components.

        Args:
            P: PGA multivectors [..., dim].

        Returns:
            Euclidean coordinates [..., d].
        """
        d = self.euclidean_dim
        # Normalize by e_0 coefficient (homogeneous coordinate)
        e0_coeff = P[..., self._idx_e0 : self._idx_e0 + 1].clamp(min=1e-6)
        P_norm = P / e0_coeff
        return torch.gather(P_norm, -1, self._g1_idx.expand(*P.shape[:-1], d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: embed Euclidean points.

        Args:
            x: Euclidean points [..., d].

        Returns:
            PGA multivectors [..., dim].
        """
        return self.embed(x)
