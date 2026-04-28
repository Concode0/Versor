# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
from core.algebra import CliffordAlgebra
from ..primitives.base import CliffordModule


class ConformalEmbedding(CliffordModule):
    """Conformal Geometric Algebra embedding layer.

    Maps Euclidean R^d vectors to the null cone of Cl(d+1, 1) and back.

    The standard conformal embedding is:
        P(x) = x + 0.5 * ||x||^2 * e_inf + e_o

    where e_inf = e_- + e_+ (point at infinity) and e_o = 0.5*(e_- - e_+)
    (origin). Basis vectors e_inf and e_o are precomputed as buffers.

    Requires an algebra with signature Cl(d+1, 1, ...).

    Attributes:
        euclidean_dim (int): Physical dimension d.
    """

    def __init__(self, algebra: CliffordAlgebra, euclidean_dim: int):
        """Sets up the conformal embedding.

        Args:
            algebra: Clifford algebra instance with signature Cl(d+1, 1, ...).
            euclidean_dim: Physical dimension d.
        """
        super().__init__(algebra)
        d = euclidean_dim
        assert algebra.p >= d + 1 and algebra.q >= 1, (
            f"Conformal embedding needs Cl(>={d + 1}, >=1), got Cl({algebra.p},{algebra.q},{algebra.r})"
        )
        self.euclidean_dim = d

        # Precompute basis indices as buffers
        idx_ep = 1 << d
        idx_em = 1 << (d + 1)
        g1_idx = (1 << torch.arange(d)).long()
        self.register_buffer('_g1_idx', g1_idx)

        e_inf = torch.zeros(algebra.dim)
        e_inf[idx_em] = 1.0
        e_inf[idx_ep] = 1.0
        self.register_buffer('_e_inf', e_inf)

        e_o = torch.zeros(algebra.dim)
        e_o[idx_em] = 0.5
        e_o[idx_ep] = -0.5
        self.register_buffer('_e_o', e_o)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed Euclidean points into the conformal null cone.

        Args:
            x: Euclidean points [..., d].

        Returns:
            Conformal multivectors [..., dim].
        """
        d = self.euclidean_dim
        x_mv = torch.zeros(*x.shape[:-1], self.algebra.dim, device=x.device, dtype=x.dtype)
        x_mv.scatter_(-1, self._g1_idx.expand_as(x), x)

        x_sq = (x * x).sum(dim=-1, keepdim=True)  # [..., 1]
        return x_mv + 0.5 * x_sq * self._e_inf + self._e_o

    def extract(self, P: torch.Tensor) -> torch.Tensor:
        """Project conformal points back to Euclidean space.

        Normalizes so that P . e_inf = -1, then extracts vector part.

        Args:
            P: Conformal multivectors [..., dim].

        Returns:
            Euclidean coordinates [..., d].
        """
        d = self.euclidean_dim
        # Scale: -(P . e_inf)_0
        P_einf = self.algebra.geometric_product(P, self._e_inf.expand_as(P))
        scale = (-P_einf[..., 0:1]).clamp(min=1e-6)
        P_norm = P / scale

        return torch.gather(P_norm, -1, self._g1_idx.expand(*P.shape[:-1], d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: embed Euclidean points.

        Args:
            x: Euclidean points [..., d].

        Returns:
            Conformal multivectors [..., dim].
        """
        return self.embed(x)
