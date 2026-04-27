# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""
==============================================================================
VERSOR EXPERIMENT: IDEA INCUBATOR (SPIN-OFF CONCEPT)
==============================================================================

This script serves as an early-stage proof-of-concept for radical, non-Euclidean
architectures. The concepts demonstrated here are strongly driven by geometric
intuition and may currently reside ahead of established academic literature.

Please understand that rigorous mathematical proofs or comprehensive citations
might be incomplete at this stage. If this geometric hypothesis proves
structurally sound, it is planned to be spun off into a dedicated, independent
repository for detailed research.

==============================================================================

Lattice Morphing via Invertible Geometric Algebra Transformations.

Hypothesis
  Learnable lattice deformations should be expressible as invertible
  Clifford-algebra transformations in ``Cl(p, q)``. The morph stack combines
  a global rotor, per-basis twist rotors, and dynamic scaling factors, and
  every stage has an exact algebraic inverse, so the full pipeline remains
  reversible while still learning meaningful basis deformations.

Execute Command
  uv run python -m experiments.inc_lattice_morph
  uv run python -m experiments.inc_lattice_morph --mode compound
  uv run python -m experiments.inc_lattice_morph --mode minkowski
"""

import os
import sys
import argparse
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments._lib import (
    build_visualization_metadata, ensure_output_dir, make_experiment_parser,
    save_experiment_figure, section_header, set_seed,
    signature_metadata,
)
from core.algebra import CliffordAlgebra
from core.decomposition import ExpPolicy
from core.metric import induced_norm
from layers.primitives.base import CliffordModule
from optimizers.riemannian import RiemannianAdam


_DTYPE_MAP = {
    'float32': torch.float32,
    'float64': torch.float64,
}


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _grade1_indices(n: int) -> list:
    """Multivector indices of grade-1 basis vectors e_i (i=0..n-1)."""
    return [1 << i for i in range(n)]


def _make_skew_basis(n: int, skew_factor: float, device,
                     dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Build a skewed basis matrix: identity + noise then upper-triangular skew.

    Args:
        n: Lattice dimension.
        skew_factor: Magnitude of off-diagonal skew (0 = orthogonal).
        device: Tensor device.
        dtype: Floating-point dtype for the generated basis.
    Returns:
        [n, n] basis matrix.
    """
    base = torch.eye(n, device=device, dtype=dtype)
    noise = torch.randn(n, n, device=device, dtype=dtype) * 0.1
    vecs = base + noise
    if skew_factor != 0:
        skew = torch.eye(n, device=device, dtype=dtype)
        upper = torch.triu(torch.randn(n, n, device=device, dtype=dtype),
                           diagonal=1)
        skew = skew + skew_factor * upper
        vecs = vecs @ skew
    return vecs


class MorphMode(str, Enum):
    """Operation mode for harder lattice morph experiments."""
    BASIC = 'basic'        # Original behaviour
    COMPOUND = 'compound'  # Non-simple bivector morphs (n >= 4)
    SKEW = 'skew'          # Anisotropic skew + gram regularization
    MINKOWSKI = 'minkowski'  # Signature-preserving Minkowski morphs


# ---------------------------------------------------------------------------
# Structure Tracker — lattice invariant computation
# ---------------------------------------------------------------------------

class StructureTracker:
    """Computes lattice geometric invariants from basis multivectors."""

    def __init__(self, algebra: CliffordAlgebra):
        self.algebra = algebra

    def compute_volume(self, basis_mvs: torch.Tensor) -> torch.Tensor:
        """det(L) = ||b_1 ^ b_2 ^ ... ^ b_n|| via iterative outer product.

        Uses grade_projection(GP(blade, b_i), k+1) instead of wedge(),
        because wedge() is (AB-BA)/2 which only works for two vectors.

        Args:
            basis_mvs: [n, D] grade-1 multivectors.
        Returns:
            Scalar tensor.
        """
        blade = basis_mvs[0]
        current_grade = 1
        for i in range(1, basis_mvs.shape[0]):
            product = self.algebra.geometric_product(blade, basis_mvs[i])
            current_grade += 1
            blade = self.algebra.grade_projection(product, current_grade)
        return induced_norm(self.algebra, blade).squeeze(-1)

    def compute_gram_matrix(self, basis_mvs: torch.Tensor) -> torch.Tensor:
        """G[i,j] = <b_i b_j>_0  (scalar part of geometric product).

        Args:
            basis_mvs: [n, D].
        Returns:
            [n, n] tensor.
        """
        # Vectorized: [n,1,D] x [1,n,D] -> [n,n,D], extract scalar
        A = basis_mvs.unsqueeze(1)  # [n, 1, D]
        B = basis_mvs.unsqueeze(0)  # [1, n, D]
        products = self.algebra.geometric_product(A, B)  # [n, n, D]
        return products[..., 0]  # scalar part -> [n, n]

    def compute_norms(self, basis_mvs: torch.Tensor) -> torch.Tensor:
        """||b_i|| for each basis vector.

        Args:
            basis_mvs: [n, D].
        Returns:
            [n] tensor.
        """
        return induced_norm(self.algebra, basis_mvs).squeeze(-1)

    def compute_angles(self, basis_mvs: torch.Tensor) -> torch.Tensor:
        """cos(theta_ij) = G[i,j] / (||b_i|| ||b_j||).

        Args:
            basis_mvs: [n, D].
        Returns:
            [n, n] tensor of cosines.
        """
        gram = self.compute_gram_matrix(basis_mvs)
        norms = self.compute_norms(basis_mvs)
        outer = norms.unsqueeze(1) * norms.unsqueeze(0)  # [n, n]
        return gram / outer.clamp(min=1e-8)

    def compute_minkowski_invariant(self,
                                    basis_mvs: torch.Tensor) -> torch.Tensor:
        """M[i,j] = <b_i b_j>_0 — the indefinite scalar product matrix.

        Identical to ``compute_gram_matrix`` (the scalar part of the geometric
        product *is* the indefinite inner product for any signature). Provided
        as a named alias so Minkowski-mode losses read naturally.

        Args:
            basis_mvs: [n, D].
        Returns:
            [n, n] tensor whose signature matches the algebra's.
        """
        return self.compute_gram_matrix(basis_mvs)

    def snapshot(self, basis_mvs: torch.Tensor) -> dict:
        """Compute all invariants."""
        return {
            'volume': self.compute_volume(basis_mvs),
            'gram': self.compute_gram_matrix(basis_mvs),
            'angles': self.compute_angles(basis_mvs),
            'norms': self.compute_norms(basis_mvs),
        }


# ---------------------------------------------------------------------------
# MorphStage — single invertible transformation
# ---------------------------------------------------------------------------

class MorphStage(CliffordModule):
    """One invertible morph: GlobalRotor -> RelativeTwist -> DynamicScale.

    When ``compound_blades >= 2`` and ``n >= 4`` the global and twist rotors
    learn a *sum* of independent simple bivectors per slot, producing
    non-simple bivectors that are handled by ``algebra.exp()`` via the active
    :class:`~core.decomposition.ExpPolicy`.
    """

    def __init__(self, algebra: CliffordAlgebra, n: int,
                 compound_blades: int = 1):
        """
        Args:
            algebra: Clifford algebra instance.
            n: Number of basis vectors (lattice dimension).
            compound_blades: Number of summed simple bivectors per rotor slot.
                Default 1 reproduces the original behaviour.
        """
        super().__init__(algebra)
        self.n = n
        self.compound_blades = compound_blades

        # Bivector mask and indices
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)
        self.register_buffer('bv_indices', bv_indices)
        num_bv = len(bv_indices)

        # (a) Global rotation — single bivector (or compound sum)
        if compound_blades == 1:
            self.global_bv = nn.Parameter(torch.randn(num_bv) * 0.01)
        else:
            self.global_bv = nn.Parameter(
                torch.randn(compound_blades, num_bv) * 0.01)
        self.global_bv._manifold = 'spin'

        # (b) Relative twist — per-basis bivectors (or per-basis compound sums)
        if compound_blades == 1:
            self.twist_bvs = nn.Parameter(torch.randn(n, num_bv) * 0.01)
        else:
            self.twist_bvs = nn.Parameter(
                torch.randn(n, compound_blades, num_bv) * 0.01)
        self.twist_bvs._manifold = 'spin'

        # (c) Dynamic scale — per-basis log-scale (Euclidean)
        self.log_scale = nn.Parameter(torch.zeros(n))

        # Rotor cache populated by forward(); inverse() reuses these so that
        # forward and inverse share the same approximate decomposition for
        # n>=4 (otherwise the stochastic power-iteration init drifts between
        # calls and breaks algebraic invertibility).
        self._cached_R = None
        self._cached_R_rev = None
        self._cached_T = None
        self._cached_T_rev = None

    def _scatter_bv(self, weights: torch.Tensor) -> torch.Tensor:
        """Scatter sparse bivector weights into full multivector.

        Args:
            weights: [..., num_bv] sparse bivector coefficients.
        Returns:
            [..., D] full multivector with bivector components filled.
        """
        shape = weights.shape[:-1] + (self.algebra.dim,)
        full = torch.zeros(shape, device=weights.device, dtype=weights.dtype)
        idx = self.bv_indices.expand_as(weights)
        full.scatter_(-1, idx, weights)
        return full

    def _make_rotor(self, B: torch.Tensor) -> tuple:
        """Exponentiate bivector to a unit rotor.

        ``self.algebra.exp_policy`` controls the dispatch. The experiment
        sets it to ``ExpPolicy.AUTO`` so simple bivectors use the closed-form
        path while non-simple bivectors fall back to the compiled-safe
        decomposition path.

        Args:
            B: Full bivector multivector [..., D].
        Returns:
            (R, R_rev) both [..., D].
        """
        half_B = -0.5 * B
        R = self.algebra.exp(half_B)
        R_rev = self.algebra.reverse(R)
        return R, R_rev

    def _apply_sandwich(self, R: torch.Tensor, x: torch.Tensor,
                        R_rev: torch.Tensor) -> torch.Tensor:
        """Sandwich product R x R~ via two geometric products.

        Args:
            R: Rotors [..., D].
            x: Multivectors [..., D] (same or broadcastable batch).
            R_rev: Reverse of R [..., D].
        Returns:
            [..., D] sandwiched result.
        """
        return self.algebra.geometric_product(
            self.algebra.geometric_product(R, x), R_rev
        )

    def _global_bivector(self) -> torch.Tensor:
        """Build the effective global bivector multivector [D]."""
        if self.compound_blades == 1:
            return self._scatter_bv(self.global_bv)            # [D]
        # [compound, num_bv] -> scatter -> [compound, D] -> sum -> [D]
        return self._scatter_bv(self.global_bv).sum(dim=0)

    def _twist_bivector(self) -> torch.Tensor:
        """Build the effective per-basis twist bivector multivector [n, D]."""
        if self.compound_blades == 1:
            return self._scatter_bv(self.twist_bvs)            # [n, D]
        # [n, compound, num_bv] -> [n, compound, D] -> sum dim=1 -> [n, D]
        return self._scatter_bv(self.twist_bvs).sum(dim=1)

    def forward(self, basis_mvs: torch.Tensor) -> torch.Tensor:
        """Apply GlobalRotor -> RelativeTwist -> DynamicScale.

        Caches the freshly built rotors on the module so ``inverse()`` can
        reuse them. This guarantees forward/inverse pairing even when the
        underlying ``algebra.exp`` takes the approximate decomposition path
        (n>=4), which would otherwise produce slightly different rotors per
        call due to stochastic power-iteration initialization.

        Args:
            basis_mvs: [n, D] grade-1 multivectors.
        Returns:
            [n, D] morphed multivectors.
        """
        # --- Global rotation ---
        B_global = self._global_bivector()                     # [D]
        R, R_rev = self._make_rotor(B_global.unsqueeze(0))     # [1, D]
        self._cached_R, self._cached_R_rev = R, R_rev
        x = self._apply_sandwich(R, basis_mvs, R_rev)          # [n, D]

        # --- Relative twist ---
        B_twist = self._twist_bivector()                       # [n, D]
        T, T_rev = self._make_rotor(B_twist)                   # [n, D]
        self._cached_T, self._cached_T_rev = T, T_rev
        x = self._apply_sandwich(T, x, T_rev)                  # [n, D]

        # --- Dynamic scale ---
        scale = torch.exp(self.log_scale.clamp(-3.0, 3.0))    # [n]
        x = x * scale.unsqueeze(-1)                            # [n, D]

        return x

    def inverse(self, basis_mvs: torch.Tensor) -> torch.Tensor:
        """Exact inverse: Scale^{-1} -> Twist^{-1} -> Rotation^{-1}.

        Reuses the rotors cached by the most recent ``forward()`` call so
        the inverse is bit-identical to undoing that forward, regardless of
        whether ``algebra.exp`` used the closed-form or the approximate
        decomposition path.

        Args:
            basis_mvs: [n, D] morphed multivectors.
        Returns:
            [n, D] reconstructed multivectors.
        """
        if self._cached_R is None or self._cached_T is None:
            raise RuntimeError(
                "MorphStage.inverse() requires a prior forward() call to "
                "populate the rotor cache."
            )

        # --- Inverse scale ---
        inv_scale = torch.exp(-self.log_scale.clamp(-3.0, 3.0))
        x = basis_mvs * inv_scale.unsqueeze(-1)

        # --- Inverse twist (swap T and T_rev) ---
        x = self._apply_sandwich(self._cached_T_rev, x, self._cached_T)

        # --- Inverse global rotation (swap R and R_rev) ---
        x = self._apply_sandwich(self._cached_R_rev, x, self._cached_R)

        return x


# ---------------------------------------------------------------------------
# MorphPipeline — sequential stage composition
# ---------------------------------------------------------------------------

class MorphPipeline(nn.Module):
    """Sequential composition of MorphStages with intermediate tracking."""

    def __init__(self, algebra: CliffordAlgebra, n: int, num_stages: int = 3,
                 compound_blades: int = 1):
        super().__init__()
        self.compound_blades = compound_blades
        self.stages = nn.ModuleList([
            MorphStage(algebra, n, compound_blades=compound_blades)
            for _ in range(num_stages)
        ])

    def forward(self, basis_mvs: torch.Tensor):
        """Apply all stages, recording intermediates.

        Args:
            basis_mvs: [n, D].
        Returns:
            (morphed [n, D], intermediates: list of [n, D] after each stage).
        """
        intermediates = []
        x = basis_mvs
        for stage in self.stages:
            x = stage(x)
            intermediates.append(x)
        return x, intermediates

    def inverse(self, basis_mvs: torch.Tensor) -> torch.Tensor:
        """Apply inverse of all stages in reverse order.

        Args:
            basis_mvs: [n, D] morphed.
        Returns:
            [n, D] reconstructed.
        """
        x = basis_mvs
        for stage in reversed(self.stages):
            x = stage.inverse(x)
        return x


# ---------------------------------------------------------------------------
# LatticeMorpher — main experiment class
# ---------------------------------------------------------------------------

class LatticeMorpher:
    """Lattice morphing experiment with invertible GA transformations."""

    def __init__(self, n: int = 3, signature: str = 'euclidean',
                 num_stages: int = 3, seed: int = 42, device: str = 'cpu',
                 compound_blades: int = 1, mode: MorphMode = MorphMode.BASIC,
                 lambda_gram: float = 0.1, lambda_minkowski: float = 0.5,
                 dtype: torch.dtype = torch.float64):
        self.n = n
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.compound_blades = compound_blades
        self.lambda_gram = lambda_gram
        self.lambda_minkowski = lambda_minkowski
        set_seed(seed)

        if signature == 'minkowski':
            p, q = n - 1, 1
        else:
            p, q = n, 0
        self.algebra = CliffordAlgebra(
            p=p, q=q, device=device, dtype=dtype).to(device)
        self.algebra.exp_policy = ExpPolicy.AUTO
        self.signature_q = q
        self.tracker = StructureTracker(self.algebra)
        self.pipeline = MorphPipeline(
            self.algebra, n, num_stages,
            compound_blades=compound_blades,
        ).to(device=device, dtype=dtype)
        self.signature_name = f"Cl({p},{q})"

        if mode == MorphMode.MINKOWSKI and q == 0:
            print("  WARNING: --ops minkowski has no effect on Cl(p,0); "
                  "use --signature minkowski for an indefinite metric.")
        if mode == MorphMode.COMPOUND and (n < 4 or compound_blades < 2):
            print(f"  WARNING: --ops compound is a no-op when n<4 or "
                  f"compound_blades<2 (got n={n}, compound_blades="
                  f"{compound_blades}).")

        print(f"Initialized LatticeMorpher: dim={n}, {self.signature_name}, "
              f"stages={num_stages}, device={device}, "
              f"mode={mode.value}, compound_blades={compound_blades}")

    def create_lattice(self, basis_matrix: torch.Tensor = None,
                       skew_factor: float = 0.0) -> torch.Tensor:
        """Create a lattice from basis matrix or random.

        Args:
            basis_matrix: [n, n] or None for random (identity + noise + skew).
            skew_factor: Skew for random generation.
        Returns:
            [n, D] grade-1 multivectors.
        """
        if basis_matrix is not None:
            vecs = basis_matrix.to(self.device, dtype=self.dtype)
        else:
            vecs = _make_skew_basis(self.n, skew_factor, self.device,
                                    dtype=self.dtype)

        return self.algebra.embed_vector(vecs)  # [n, D]

    def generate_lattice_points(self, basis_mvs: torch.Tensor,
                                grid_range: int = 3) -> tuple:
        """Vectorized lattice point generation.

        Args:
            basis_mvs: [n, D].
            grid_range: Integer range per axis.
        Returns:
            (points [N, D], coeffs [N, n]).
        """
        coords_1d = torch.arange(-grid_range, grid_range + 1,
                                  dtype=basis_mvs.dtype, device=basis_mvs.device)
        grids = [coords_1d] * self.n
        coeffs = torch.cartesian_prod(*grids)  # [(2r+1)^n, n]
        if self.n == 1:
            coeffs = coeffs.unsqueeze(-1)
        points = torch.matmul(coeffs, basis_mvs)  # [N, D]
        return points, coeffs

    def _print_invariants(self, label: str, basis_mvs: torch.Tensor):
        """Print lattice invariants."""
        with torch.no_grad():
            snap = self.tracker.snapshot(basis_mvs)
            print(f"\n  [{label}]")
            print(f"    Volume:  {snap['volume'].item():.6f}")
            print(f"    Norms:   {snap['norms'].cpu().numpy()}")
            print(f"    Angles (cos):")
            ang = snap['angles'].cpu().numpy()
            for i in range(self.n):
                row = "  ".join(f"{ang[i,j]:+.4f}" for j in range(self.n))
                print(f"      [{row}]")

    def _make_optimizer(self, lr: float):
        """Create the optimizer used by all morphing runs."""
        return RiemannianAdam.from_model(
            self.pipeline, lr=lr, algebra=self.algebra
        )

    def _mode_extra_loss(self, morphed: torch.Tensor,
                         reference_gram: torch.Tensor = None) -> torch.Tensor:
        """Mode-specific regularization added on top of the base objective.

        - SKEW: penalty on off-diagonal gram entries (drives orthogonality
          even when the base loss does not directly target it).
        - MINKOWSKI: Frobenius distance between the morphed indefinite
          metric and a reference metric (preserves causality structure).
        Otherwise returns zero.
        """
        zero = morphed.new_zeros(())
        extra = zero
        if self.mode == MorphMode.SKEW:
            gram = self.tracker.compute_gram_matrix(morphed)
            mask = 1.0 - torch.eye(self.n, device=self.device)
            extra = extra + self.lambda_gram * (gram * mask).pow(2).sum()
        elif self.mode == MorphMode.MINKOWSKI and self.signature_q >= 1:
            if reference_gram is None:
                return extra
            M = self.tracker.compute_minkowski_invariant(morphed)
            extra = extra + self.lambda_minkowski * (M - reference_gram).pow(2).sum()
        return extra

    def _optimization_step(self, optimizer, loss_fn):
        """Execute one optimization step."""

        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        return loss.item()

    def _run_morphing_loop(self, source: torch.Tensor, optimizer, loss_fn,
                           steps: int, title: str, final_frame=None) -> dict:
        """Shared optimization loop for target and free morphing."""
        loss_history = []
        invariant_history = []
        recon_errors = []

        print(title)
        for step in range(steps):
            loss_val = self._optimization_step(optimizer, loss_fn)

            with torch.no_grad():
                morphed, _ = self.pipeline(source)
                snap = self.tracker.snapshot(morphed)
                reconstructed = self.pipeline.inverse(morphed)
                recon_err = (source - reconstructed).pow(2).sum().sqrt().item()

                loss_history.append(loss_val)
                invariant_history.append({
                    k: v.detach().cpu() for k, v in snap.items()
                })
                recon_errors.append(recon_err)

            if step % 50 == 0 or step == steps - 1:
                print(f"    step {step:4d}: loss={loss_val:.6f}, "
                      f"vol={snap['volume'].item():.4f}, "
                      f"recon_err={recon_err:.2e}")

        with torch.no_grad():
            _, final_intermediates = self.pipeline(source)

        intermediates = [source] + final_intermediates
        if final_frame is not None:
            intermediates.append(final_frame)

        return {
            'loss_history': loss_history,
            'invariant_history': invariant_history,
            'recon_errors': recon_errors,
            'intermediates': intermediates,
        }

    def run_target_morphing(self, target_basis: torch.Tensor,
                            lr: float = 0.01, steps: int = 300) -> dict:
        """Optimize pipeline to morph source -> target basis.

        Args:
            target_basis: [n, D] target lattice basis.
            lr: Learning rate.
            steps: Optimization steps.
        Returns:
            Dict with loss_history, invariant_history, recon_errors, intermediates.
        """
        source = self.source_basis.detach()
        target = target_basis.detach()
        optimizer = self._make_optimizer(lr)
        # Reference Minkowski metric: target's indefinite inner product matrix.
        with torch.no_grad():
            ref_gram = self.tracker.compute_gram_matrix(target)

        def loss_fn():
            morphed, _ = self.pipeline(source)
            base = (morphed - target).pow(2).sum()
            return base + self._mode_extra_loss(morphed, reference_gram=ref_gram)

        title = (f"\n  Target Morphing: {steps} steps, lr={lr}"
              f"{' [ops=' + self.mode.value + ']' if self.mode != MorphMode.BASIC else ''}")
        results = self._run_morphing_loop(
            source, optimizer, loss_fn, steps, title, final_frame=target
        )
        return results

    def run_free_morphing(self, lr: float = 0.01, steps: int = 200,
                          objective: str = 'orthogonalize') -> dict:
        """Optimize pipeline toward a geometric objective.

        Args:
            lr: Learning rate.
            steps: Optimization steps.
            objective: 'orthogonalize', 'equalize_norms', or 'minimize_volume'.
        Returns:
            Dict with loss_history, invariant_history, recon_errors, intermediates.
        """
        source = self.source_basis.detach()
        optimizer = self._make_optimizer(lr)
        # Reference Minkowski metric for MINKOWSKI mode: source's metric — the
        # morph should reshape the basis while preserving the causal structure.
        with torch.no_grad():
            ref_gram = self.tracker.compute_gram_matrix(source)

        def loss_fn():
            morphed, _ = self.pipeline(source)
            if objective == 'orthogonalize':
                gram = self.tracker.compute_gram_matrix(morphed)
                mask = 1.0 - torch.eye(self.n, device=self.device)
                base = (gram * mask).pow(2).sum()
                norms = self.tracker.compute_norms(morphed)
                base = base + 0.1 * (norms - 1.0).pow(2).sum()
            elif objective == 'equalize_norms':
                base = self.tracker.compute_norms(morphed).var()
            elif objective == 'minimize_volume':
                base = self.tracker.compute_volume(morphed)
            else:
                raise ValueError(f"Unknown objective: {objective}")
            return base + self._mode_extra_loss(morphed, reference_gram=ref_gram)

        title = (f"\n  Free Morphing ({objective}): {steps} steps, lr={lr}"
              f"{' [ops=' + self.mode.value + ']' if self.mode != MorphMode.BASIC else ''}")
        return self._run_morphing_loop(source, optimizer, loss_fn, steps, title)

    def verify_reconstruction(self, tolerance: float = 1e-5) -> dict:
        """Verify pipeline invertibility.

        Returns:
            Dict with per-basis errors and pass/fail.
        """
        with torch.no_grad():
            morphed, _ = self.pipeline(self.source_basis)
            reconstructed = self.pipeline.inverse(morphed)
            per_basis_err = (self.source_basis - reconstructed).pow(2).sum(
                dim=-1).sqrt()
            max_err = per_basis_err.max().item()
            passed = max_err < tolerance

        print(f"\n  Reconstruction Verification (tol={tolerance:.0e}):")
        for i in range(self.n):
            print(f"    b_{i+1} error: {per_basis_err[i].item():.2e}")
        print(f"    Max error: {max_err:.2e} -> {'PASS' if passed else 'FAIL'}")

        return {
            'per_basis_errors': per_basis_err.cpu(),
            'max_error': max_err,
            'passed': passed,
        }


# ---------------------------------------------------------------------------
# MorphVisualizer
# ---------------------------------------------------------------------------

class MorphVisualizer:
    """Visualization for lattice morphing."""

    def __init__(self, algebra: CliffordAlgebra, n: int, output_dir: str,
                 metadata: str, args: argparse.Namespace):
        self.algebra = algebra
        self.n = n
        self.output_dir = ensure_output_dir(output_dir)
        self.metadata = metadata
        self.args = args
        self.g1_indices = _grade1_indices(n)

    def _extract_coords(self, mvs: torch.Tensor) -> np.ndarray:
        """Extract grade-1 coordinates from multivectors.

        Args:
            mvs: [..., D] multivectors.
        Returns:
            [..., n] numpy array of vector coordinates.
        """
        coords = torch.stack([mvs[..., idx] for idx in self.g1_indices], dim=-1)
        return coords.detach().cpu().numpy()

    def plot_morph_sequence(self, intermediates: list, grid_range: int = 3,
                            labels: list = None):
        """Side-by-side lattice plots at each morph stage."""
        n_plots = len(intermediates)
        if labels is None:
            labels = ['Source'] + [f'Stage {i+1}' for i in range(n_plots - 1)]

        if self.n == 2:
            self._plot_sequence_2d(intermediates, grid_range, labels)
        elif self.n == 3:
            self._plot_sequence_3d(intermediates, grid_range, labels)

    def _generate_points(self, basis_mvs: torch.Tensor, grid_range: int):
        """Vectorized lattice point generation for plotting."""
        coords_1d = torch.arange(-grid_range, grid_range + 1,
                                  dtype=basis_mvs.dtype, device=basis_mvs.device)
        grids = [coords_1d] * self.n
        coeffs = torch.cartesian_prod(*grids)
        if self.n == 1:
            coeffs = coeffs.unsqueeze(-1)
        return torch.matmul(coeffs, basis_mvs)

    def _plot_sequence_2d(self, intermediates: list, grid_range: int,
                          labels: list):
        n_plots = len(intermediates)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_plots))

        for idx, (basis_mvs, ax, label) in enumerate(
                zip(intermediates, axes, labels)):
            pts = self._generate_points(basis_mvs, grid_range)
            xy = self._extract_coords(pts)
            ax.scatter(xy[:, 0], xy[:, 1], c=[colors[idx]], alpha=0.4, s=20)

            # Draw basis vectors as arrows
            basis_xy = self._extract_coords(basis_mvs)
            for i in range(self.n):
                ax.annotate('', xy=basis_xy[i], xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='black',
                                            lw=2))
                ax.text(basis_xy[i, 0] * 1.15, basis_xy[i, 1] * 1.15,
                        f'b{i+1}', fontsize=9, ha='center')

            # Unit cell parallelogram
            cell = np.array([
                [0, 0],
                basis_xy[0],
                basis_xy[0] + basis_xy[1],
                basis_xy[1],
                [0, 0],
            ])
            ax.plot(cell[:, 0], cell[:, 1], 'k--', alpha=0.5, lw=1)

            ax.set_title(label, fontsize=11)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvline(0, color='grey', lw=0.5)

        plt.tight_layout()
        path = save_experiment_figure(
            fig,
            output_dir=self.output_dir,
            experiment_name='inc_lattice_morph',
            metadata=self.metadata,
            plot_name='morph_sequence_2d',
            args=self.args,
            module=__name__,
            dpi=150,
        )
        print(f"  Saved: {path}")

    def _plot_sequence_3d(self, intermediates: list, grid_range: int,
                          labels: list):
        n_plots = len(intermediates)
        fig = plt.figure(figsize=(6 * n_plots, 6))
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_plots))

        for idx, (basis_mvs, label) in enumerate(zip(intermediates, labels)):
            ax = fig.add_subplot(1, n_plots, idx + 1, projection='3d')
            pts = self._generate_points(basis_mvs, grid_range)
            xyz = self._extract_coords(pts)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       c=[colors[idx]], alpha=0.2, s=10)

            # Basis arrows
            basis_xyz = self._extract_coords(basis_mvs)
            for i in range(self.n):
                ax.plot([0, basis_xyz[i, 0]], [0, basis_xyz[i, 1]],
                        [0, basis_xyz[i, 2]], color='black', lw=2)
                ax.text(basis_xyz[i, 0] * 1.15, basis_xyz[i, 1] * 1.15,
                        basis_xyz[i, 2] * 1.15, f'b{i+1}', fontsize=9)

            ax.scatter([0], [0], [0], color='black', s=60, marker='*')
            ax.set_title(label, fontsize=11)

        plt.tight_layout()
        path = save_experiment_figure(
            fig,
            output_dir=self.output_dir,
            experiment_name='inc_lattice_morph',
            metadata=self.metadata,
            plot_name='morph_sequence_3d',
            args=self.args,
            module=__name__,
            dpi=150,
        )
        print(f"  Saved: {path}")

    def plot_invariant_evolution(self, invariant_history: list,
                                recon_errors: list, loss_history: list):
        """Plot volume, norms, angles, recon error over optimization."""
        steps = range(len(loss_history))
        volumes = [h['volume'].item() for h in invariant_history]
        norms_all = torch.stack([h['norms'] for h in invariant_history]).numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axes[0, 0].semilogy(steps, loss_history)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].grid(True, alpha=0.3)

        # Volume
        axes[0, 1].plot(steps, volumes)
        axes[0, 1].set_title('Lattice Volume')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].grid(True, alpha=0.3)

        # Per-basis norms
        for i in range(norms_all.shape[1]):
            axes[1, 0].plot(steps, norms_all[:, i], label=f'b{i+1}')
        axes[1, 0].set_title('Basis Vector Norms')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Reconstruction error
        axes[1, 1].semilogy(steps, recon_errors)
        axes[1, 1].set_title('Reconstruction Error')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_experiment_figure(
            fig,
            output_dir=self.output_dir,
            experiment_name='inc_lattice_morph',
            metadata=self.metadata,
            plot_name='invariant_evolution',
            args=self.args,
            module=__name__,
            dpi=150,
        )
        print(f"  Saved: {path}")

    def plot_mode_diagnostics(self, mode: 'MorphMode',
                              invariant_history: list,
                              pipeline: 'MorphPipeline' = None,
                              reference_gram: torch.Tensor = None):
        """Mode-specific diagnostic panel.

        SKEW       — off-diagonal gram Frobenius norm over training.
        MINKOWSKI  — ||gram_morphed - reference_gram||_F over training.
        COMPOUND   — per-stage compound-blade L2 norms (final state).

        No-op for BASIC (returns without saving).
        """
        if mode == MorphMode.BASIC:
            return

        if mode == MorphMode.SKEW:
            mask = 1.0 - torch.eye(self.n)
            offdiag = [(h['gram'] * mask).pow(2).sum().sqrt().item()
                       for h in invariant_history]
            steps = range(len(offdiag))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(steps, offdiag)
            ax.set_xlabel('Step')
            ax.set_ylabel('||off-diag(G)||_F')
            ax.set_title('Off-diagonal Gram norm (SKEW mode)')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = save_experiment_figure(
                fig,
                output_dir=self.output_dir,
                experiment_name='inc_lattice_morph',
                metadata=self.metadata,
                plot_name='mode_skew',
                args=self.args,
                module=__name__,
                dpi=150,
            )
            print(f"  Saved: {path}")
            return

        if mode == MorphMode.MINKOWSKI:
            if reference_gram is None:
                print("  Skipping Minkowski plot: no reference gram provided.")
                return
            ref = reference_gram.detach().cpu()
            dev = [(h['gram'] - ref).pow(2).sum().sqrt().item()
                   for h in invariant_history]
            steps = range(len(dev))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(steps, dev)
            ax.set_xlabel('Step')
            ax.set_ylabel('||M_morphed - M_ref||_F')
            ax.set_title('Minkowski invariant deviation (MINKOWSKI mode)')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = save_experiment_figure(
                fig,
                output_dir=self.output_dir,
                experiment_name='inc_lattice_morph',
                metadata=self.metadata,
                plot_name='mode_minkowski',
                args=self.args,
                module=__name__,
                dpi=150,
            )
            print(f"  Saved: {path}")
            return

        if mode == MorphMode.COMPOUND:
            if pipeline is None or pipeline.compound_blades < 2:
                print("  Skipping compound plot: pipeline has compound_blades<2.")
                return
            with torch.no_grad():
                stage_norms = []
                for stage in pipeline.stages:
                    # global_bv is [compound, num_bv]
                    g_norms = stage.global_bv.norm(dim=-1).cpu().numpy()
                    stage_norms.append(g_norms)
            stage_norms = np.array(stage_norms)  # [num_stages, compound]
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(stage_norms, aspect='auto', cmap='viridis')
            ax.set_xlabel('Compound blade index')
            ax.set_ylabel('Stage')
            ax.set_xticks(range(stage_norms.shape[1]))
            ax.set_yticks(range(stage_norms.shape[0]))
            ax.set_title('Per-stage compound bivector L2 norms (COMPOUND mode)')
            plt.colorbar(im, ax=ax)
            fig.tight_layout()
            path = save_experiment_figure(
                fig,
                output_dir=self.output_dir,
                experiment_name='inc_lattice_morph',
                metadata=self.metadata,
                plot_name='mode_compound',
                args=self.args,
                module=__name__,
                dpi=150,
            )
            print(f"  Saved: {path}")
            return


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    title = (f" Lattice Morphing Experiment (dim={args.dim}, "
             f"sig={args.signature}, ops={args.ops})")
    print('\n' + section_header(title))

    # Auto-detect device
    device = args.device
    ops_mode = MorphMode(args.ops)
    dtype = _DTYPE_MAP[args.dtype]

    morpher = LatticeMorpher(
        n=args.dim, signature=args.signature,
        num_stages=args.num_stages, seed=args.seed, device=device,
        compound_blades=args.compound_blades,
        mode=ops_mode,
        lambda_gram=args.lambda_gram,
        lambda_minkowski=args.lambda_minkowski,
        dtype=dtype,
    )

    # Create source lattice
    morpher.source_basis = morpher.create_lattice(
        skew_factor=args.skew_factor)
    morpher._print_invariants("Source Lattice", morpher.source_basis)

    # Print basis vectors
    with torch.no_grad():
        vecs = morpher.source_basis.cpu().numpy()
        g1 = _grade1_indices(args.dim)
        for i in range(args.dim):
            coords = [vecs[i, idx] for idx in g1]
            print(f"    b_{i+1}: {np.array(coords)}")

    # Run morphing
    target_basis = None
    if args.mode == 'target':
        # Target: orthonormal lattice
        target_matrix = torch.eye(args.dim, device=device, dtype=dtype)
        target_basis = morpher.create_lattice(target_matrix)
        morpher._print_invariants("Target Lattice", target_basis)
        results = morpher.run_target_morphing(
            target_basis, lr=args.lr, steps=args.steps)
    else:
        results = morpher.run_free_morphing(
            lr=args.lr, steps=args.steps, objective=args.objective)

    # Final state
    with torch.no_grad():
        final_morphed, _ = morpher.pipeline(morpher.source_basis)
    morpher._print_invariants("Final Morphed Lattice", final_morphed)

    # Reconstruction verification
    morpher.verify_reconstruction()

    # Visualization
    p, q = (args.dim - 1, 1) if args.signature == 'minkowski' else (args.dim, 0)
    plot_metadata = build_visualization_metadata(
        signature_metadata(p, q),
        dim=args.dim,
        ops=args.ops,
        seed=args.seed,
    )
    if args.dim in [2, 3]:
        print("\nGenerating visualizations...")
        viz = MorphVisualizer(
            morpher.algebra, morpher.n, args.output_dir, plot_metadata, args,
        )

        # Build labels
        n_inter = len(results['intermediates'])
        if args.mode == 'target':
            labels = (['Source'] +
                       [f'Stage {i+1}' for i in range(n_inter - 2)] +
                       ['Target'])
        else:
            labels = (['Source'] +
                       [f'Stage {i+1}' for i in range(n_inter - 1)])

        viz.plot_morph_sequence(results['intermediates'],
                                grid_range=3, labels=labels)
        viz.plot_invariant_evolution(
            results['invariant_history'],
            results['recon_errors'],
            results['loss_history'],
        )

    if ops_mode != MorphMode.BASIC:
        viz = MorphVisualizer(
            morpher.algebra, morpher.n, args.output_dir, plot_metadata, args,
        )
        with torch.no_grad():
            ref_basis = target_basis if target_basis is not None else morpher.source_basis
            ref_gram = morpher.tracker.compute_gram_matrix(ref_basis)
        viz.plot_mode_diagnostics(
            ops_mode, results['invariant_history'],
            pipeline=morpher.pipeline,
            reference_gram=ref_gram,
        )

    print(f"\nExperiment complete.\n")


def parse_args() -> argparse.Namespace:
    parser = make_experiment_parser(
        "Lattice Morphing via Geometric Algebra",
        include=('seed', 'device', 'lr', 'output_dir'),
        defaults={'lr': 0.01, 'output_dir': 'lattice_morph_plots'},
    )
    parser.add_argument('--dim', type=int, default=3,
                        help='Lattice dimension')
    parser.add_argument('--signature', choices=['euclidean', 'minkowski'],
                        default='euclidean', help='Algebra signature')
    parser.add_argument('--num-stages', type=int, default=3,
                        help='Number of morph stages')
    parser.add_argument('--mode', choices=['free', 'target'],
                        default='target', help='Morphing mode')
    parser.add_argument('--objective',
                        choices=['orthogonalize', 'equalize_norms',
                                 'minimize_volume'],
                        default='orthogonalize',
                        help='Free morphing objective')
    parser.add_argument('--steps', type=int, default=300,
                        help='Optimization steps')
    parser.add_argument('--skew-factor', type=float, default=0.5,
                        help='Skew factor for source lattice')
    parser.add_argument('--ops', choices=[m.value for m in MorphMode],
                        default=MorphMode.BASIC.value,
                        help='Operation mode: basic | compound | skew | minkowski')
    parser.add_argument('--compound-blades', type=int, default=1,
                        help='Number of summed simple bivectors per rotor slot '
                             '(>=2 enables non-simple bivectors; needs n>=4)')
    parser.add_argument('--lambda-gram', type=float, default=0.1,
                        help='Off-diagonal gram penalty weight (ops=skew)')
    parser.add_argument('--lambda-minkowski', type=float, default=0.5,
                        help='Minkowski metric preservation weight (ops=minkowski)')
    parser.add_argument('--dtype', choices=list(_DTYPE_MAP.keys()),
                        default='float64',
                        help='Floating-point precision for algebra and tensors')
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
