"""Benchmark models for GDO experiments.

Three curated showcases covering the primitives -> blocks -> transformer
hierarchy:

- ``SmallGBNModel``: primitives stack -- ``CliffordLayerNorm`` +
  ``RotorLayer`` + ``GeometricGELU`` + ``CliffordLinear`` over Cl(3, 0).
- ``MultiRotorRegistrationModel``: rotor-bank pattern -- ``MultiRotorLayer``
  (K=3) fitted to a clustered point-cloud alignment in Cl(3, 0).
- ``GeometricTransformerToyModel``: full attention block --
  ``MultivectorEmbedding`` + ``RotaryBivectorPE`` +
  ``GeometricTransformerBlock`` (``GeometricProductAttention`` +
  ``MultiRotorFFN``) on a synthetic regression target.

Removed in this curation (restorable from git history):

* Analytic objectives (no GA structure):
  ``RosenbrockModel``, ``RastriginModel``, ``AckleyModel``, ``StyblinskiTangModel``.
* Single-rotor registration variants (subsumed by SmallGBN + MultiRotor):
  ``RotorRegistrationModel`` Cl(3, 0), ``MinkowskiRotorModel`` Cl(2, 1),
  ``ConformalRegistrationModel`` Cl(4, 1).
* GBN size variants (one is enough for the primitive showcase):
  ``MediumGBNModel`` Cl(3, 0)/16ch/3-layer, ``MultiSigGBNModel`` Cl(2, 1)/8ch/2-layer.
* Manifold task (niche to the geodesic integrator):
  ``SO3InterpolationModel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.module import CliffordModule
from experiments._lib import setup_algebra
from functional.activation import GeometricGELU
from layers import (
    BladeSelector,
    CliffordLayerNorm,
    CliffordLinear,
    GeometricTransformerBlock,
    MultiRotorLayer,
    MultivectorEmbedding,
    RotaryBivectorPE,
    RotorLayer,
)

# --- Primitives showcase ---------------------------------------------------


class SmallGBNModel(CliffordModule):
    """Canonical primitive stack on Cl(3, 0).

    ``CliffordLayerNorm`` -> ``RotorLayer`` -> ``GeometricGELU`` ->
    ``CliffordLinear``. Demonstrates how the four primitive building blocks
    compose; intentionally tiny so the optimizer can converge in ~200 steps.
    """

    def __init__(self, p: int = 3, q: int = 0, channels: int = 4, device: str = "cpu"):
        algebra = setup_algebra(p, q, device=device)
        super().__init__(algebra)
        self.norm = CliffordLayerNorm(self.algebra, channels)
        self.rotor = RotorLayer(self.algebra, channels)
        self.linear = CliffordLinear(self.algebra, channels, channels)
        self.act = GeometricGELU(self.algebra)
        self._channels = channels
        self._dim = self.algebra.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.rotor(x)
        x = self.act(x)
        x = self.linear(x)
        return x


# --- Rotor-bank showcase ---------------------------------------------------


class MultiRotorRegistrationModel(CliffordModule):
    """Rotor-bank fit: align ``n_clusters`` rotated point clusters via ``MultiRotorLayer``.

    Each cluster carries a different ground-truth rotation; a ``MultiRotorLayer``
    with ``num_rotors=n_clusters`` learns a weighted superposition that aligns
    the source to the target. Exercises commutator scheduling and multi-modal
    optimisation.
    """

    def __init__(self, n_clusters: int = 3, points_per_cluster: int = 20, device: str = "cpu"):
        algebra = setup_algebra(3, 0, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim

        torch.manual_seed(42)
        sources = []
        targets = []

        for c in range(n_clusters):
            center = torch.randn(3, device=device)
            pts = center + 0.2 * torch.randn(points_per_cluster, 3, device=device)
            sources.append(pts)

            angle = 0.5 + c * 1.0
            axis = F.normalize(torch.randn(3, device=device), dim=0)
            bv = torch.zeros(dim, device=device)
            bv[3] = angle * axis[2]
            bv[5] = -angle * axis[1]
            bv[6] = angle * axis[0]
            rotor = self.algebra.exp(-0.5 * bv.unsqueeze(0))
            pts_mv = self.algebra.embed_vector(pts)
            rotated = self.algebra.sandwich_product(
                rotor.expand(points_per_cluster, -1),
                pts_mv.unsqueeze(1),
            ).squeeze(1)
            tgt_pts = torch.stack([rotated[..., 1], rotated[..., 2], rotated[..., 4]], dim=-1)
            targets.append(tgt_pts + 0.03 * torch.randn_like(tgt_pts))

        self.register_buffer("source", torch.cat(sources))
        self.register_buffer("target", torch.cat(targets))

        self.multi_rotor = MultiRotorLayer(self.algebra, channels=1, num_rotors=n_clusters)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source).unsqueeze(1)
        rotated_mv = self.multi_rotor(source_mv).squeeze(1)
        pred = torch.stack([rotated_mv[..., 1], rotated_mv[..., 2], rotated_mv[..., 4]], dim=-1)
        return F.mse_loss(pred, self.target)


# --- Transformer-block showcase --------------------------------------------


class GeometricTransformerToyModel(CliffordModule):
    """Tiny ``GeometricTransformerBlock`` demo on a synthetic regression task.

    Stack: ``MultivectorEmbedding`` -> ``RotaryBivectorPE`` ->
    ``GeometricTransformerBlock`` (``GeometricProductAttention`` +
    ``MultiRotorFFN``) -> ``CliffordLayerNorm`` -> ``BladeSelector`` ->
    ``nn.Linear`` readout. Cl(3, 0), <3K params, fixed input batch with a
    grade-0-mean regression target so it converges in the standard 200-step
    budget. The point is to demonstrate the layers/blocks/transformer combo,
    not to beat any baseline.

    Attributes:
        vocab_size (int): Surface for the synthetic input.
        seq_len (int): Sequence length of the fixed sample batch.
    """

    def __init__(
        self,
        p: int = 3,
        q: int = 0,
        channels: int = 4,
        num_heads: int = 1,
        num_rotors: int = 2,
        vocab_size: int = 16,
        seq_len: int = 8,
        batch_size: int = 4,
        device: str = "cpu",
    ):
        algebra = setup_algebra(p, q, device=device)
        super().__init__(algebra)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self._channels = channels
        self._dim = self.algebra.dim

        self.embed = MultivectorEmbedding(self.algebra, vocab_size, channels)
        self.pos = RotaryBivectorPE(self.algebra, channels, max_seq_len=seq_len)
        self.block = GeometricTransformerBlock(
            self.algebra,
            channels=channels,
            num_heads=num_heads,
            num_rotors=num_rotors,
            dropout=0.0,
        )
        self.out_norm = CliffordLayerNorm(self.algebra, channels)
        self.gate = BladeSelector(self.algebra, channels)
        self.readout = nn.Linear(channels * self.algebra.dim, 1)

        torch.manual_seed(42)
        self.register_buffer(
            "sample_input",
            torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        )
        with torch.no_grad():
            init_mv = self.embed(self.sample_input)  # [B, L, C, D]
            target = init_mv[..., 0].mean(dim=(1, 2))  # [B]
        self.register_buffer("target", target)

    def _encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)  # [B, L, C, D]
        x = self.pos(x)  # [B, L, C, D]
        x = self.block(x)  # [B, L, C, D]
        B, L, C, D = x.shape
        x = self.out_norm(x.reshape(B * L, C, D)).reshape(B, L, C, D)
        x = self.gate(x.reshape(B * L, C, D)).reshape(B, L, C, D)
        return x

    def forward(self, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids is None:
            x = self._encode(self.sample_input)
            B, L, C, D = x.shape
            pooled = x.mean(dim=1).reshape(B, C * D)
            pred = self.readout(pooled).squeeze(-1)
            return F.mse_loss(pred, self.target)
        return self._encode(token_ids)
