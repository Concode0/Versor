# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Automated metric search for optimal geometric signatures.

Algorithms to discover the geometric signature that best preserves manifold topology.

Also provides geodesic flow analysis and dimension lifting tests — tools to
probe whether causal structure is visible in a dataset's multivector geometry.
"""

import torch
import itertools
from typing import Tuple, List, Optional, Dict
from core.algebra import CliffordAlgebra
from core.metric import induced_norm


class MetricSearch:
    """Finds the optimal signature (p, q).

    It calculates distortion. If the distortion is low, the manifold is preserved.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def _compute_pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances as a reference baseline."""
        # x: [N, D]
        # dist: [N, N]
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        return torch.norm(diff, dim=-1)

    def evaluate_signature(self, data: torch.Tensor, p: int, q: int) -> float:
        """Measures the geometric distortion introduced by the algebra.

        Stress = || D_euc - D_geo(p,q) ||_F / || D_euc ||_F.

        Args:
            data (torch.Tensor): Input data [Batch, Dim].
            p (int): Positive dimensions.
            q (int): Negative dimensions.

        Returns:
            float: Distortion score. Lower is better.
        """
        # Move data to the correct device
        data = data.to(self.device)

        N, D = data.shape
        if p + q != D:
            raise ValueError(f"Signature (p={p}, q={q}) must sum to data dimension {D}")

        # 1. Compute target pairwise distances (Euclidean/Intrinsic reference)
        algebra = CliffordAlgebra(p, q, device=self.device)

        # Embed data as vectors
        # Calculate pairwise geometric distances
        diff_coeffs = data.unsqueeze(1) - data.unsqueeze(0) # [N, N, D]

        # Map coefficients to multivector basis indices
        diff_flat = diff_coeffs.view(-1, D)

        mv = torch.zeros(diff_flat.shape[0], algebra.dim, device=self.device)
        for i in range(D):
            mv[:, 1 << i] = diff_flat[:, i]

        # Compute induced norms
        dists_geo = induced_norm(algebra, mv) # [N*N, 1]
        dists_geo = dists_geo.view(N, N)

        # Reference: Euclidean Distance
        dists_euc = self._compute_pairwise_distances(data)

        # Stress
        stress = torch.norm(dists_euc - dists_geo) / (torch.norm(dists_euc) + 1e-8)
        return stress.item()

    def search(self, data: torch.Tensor, limit_combinations: int = 10) -> Tuple[int, int]:
        """Find the optimal signature by exhaustive evaluation.

        Args:
            data (torch.Tensor): Input data [N, D].
            limit_combinations (int): Unused. All combinations are evaluated.

        Returns:
            Tuple[int, int]: The optimal (p, q).
        """
        N, D = data.shape
        best_pq = (D, 0) # Default to Euclidean
        best_score = float('inf')

        # Iterate over all p+q = D
        candidates = []
        for q in range(D + 1):
            p = D - q
            candidates.append((p, q))

        results = []
        for p, q in candidates:
            score = self.evaluate_signature(data, p, q)
            results.append(((p, q), score))

            if score < best_score:
                best_score = score
                best_pq = (p, q)

        return best_pq


# ---------------------------------------------------------------------------
# Geodesic Flow
# ---------------------------------------------------------------------------

class GeodesicFlow:
    """Geodesic flow analysis in Clifford algebra.

    Interprets data points as grade-1 multivectors and computes the *flow
    field* — a bivector at each point that encodes the direction of shortest
    algebraic paths to its k-nearest neighbours.

    The flow is computed as the mean of *connection bivectors*:

        B_ij = <x_i · x̃_j>₂   (grade-2 part of the geometric product)

    This bivector encodes the rotational "turn" needed to map x_i toward
    x_j, analogous to the parallel transport connection on a Lie group.

    The coherence and curvature of this field reveal whether the data has
    causal (directional) structure:

    - **High coherence, low curvature** → the flow is smooth and aligned in
      one direction.  Causality is visible.
    - **Low coherence, high curvature** → the flow is fragmented and
      collides with itself.  The signal is dominated by noise.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        k (int): Number of nearest neighbours for the flow field.
    """

    def __init__(self, algebra: CliffordAlgebra, k: int = 8):
        self.algebra = algebra
        self.k = k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, data: torch.Tensor) -> torch.Tensor:
        """Embeds raw vectors into the grade-1 multivector subspace.

        Args:
            data (torch.Tensor): ``[N, d]`` where d ≤ algebra.n.

        Returns:
            torch.Tensor: ``[N, algebra.dim]`` grade-1 multivectors.
        """
        N, d = data.shape
        n = self.algebra.n
        if d > n:
            raise ValueError(
                f"Data dimension {d} exceeds algebra dimension {n}. "
                f"Use DimensionLifter to lift data before flow analysis."
            )
        # Pad to algebra dimension with zeros if d < n
        if d < n:
            pad = torch.zeros(N, n - d, device=data.device, dtype=data.dtype)
            data = torch.cat([data, pad], dim=-1)
        return self.algebra.embed_vector(data)

    def _knn(self, mv: torch.Tensor) -> torch.Tensor:
        """Returns k-nearest neighbour indices in multivector coefficient space.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: ``[N, k]`` neighbour indices.
        """
        N = mv.shape[0]
        k = min(self.k, N - 1)
        diff = mv.unsqueeze(1) - mv.unsqueeze(0)     # [N, N, dim]
        dists = diff.norm(dim=-1)                    # [N, N]
        dists.fill_diagonal_(float('inf'))
        _, idx = dists.topk(k, dim=-1, largest=False)
        return idx                                    # [N, k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _connection_bivectors(self, mv: torch.Tensor) -> torch.Tensor:
        """Computes unit connection bivectors for all (point, neighbour) pairs.

        The connection bivector from x_i to x_j encodes the rotational "turn"
        in the algebra needed to map one vector toward the other:

            B_ij = unit( <x_i · x̃_j>₂ )

        Args:
            mv (torch.Tensor): ``[N, dim]`` grade-1 multivectors.

        Returns:
            torch.Tensor: ``[N, k, dim]`` unit connection bivectors.
        """
        N, D = mv.shape
        k = min(self.k, N - 1)
        nn_idx = self._knn(mv)

        neighbors = mv[nn_idx]                          # [N, k, dim]
        xi = mv.unsqueeze(1).expand(N, k, D).reshape(N * k, D)
        xj_rev = self.algebra.reverse(neighbors.reshape(N * k, D))

        prod = self.algebra.geometric_product(xi, xj_rev)        # [N*k, dim]
        bv_raw = self.algebra.grade_projection(prod, 2)           # [N*k, dim]
        bv_norm = bv_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (bv_raw / bv_norm).reshape(N, k, D)               # [N, k, dim]

    def flow_bivectors(self, mv: torch.Tensor) -> torch.Tensor:
        """Computes the mean flow bivector at each data point.

        For each point x_i, aggregates the unit connection bivectors to its
        k-nearest neighbours:

            B_i = mean_j { unit( <x_i · x̃_j>₂ ) }

        .. note::
            For perfectly symmetric data (e.g. a closed circle) the mean
            cancels to zero — which is geometrically correct since there is
            no preferred flow direction.  Use :meth:`coherence` to measure
            structure without this cancellation.

        Args:
            mv (torch.Tensor): ``[N, dim]`` grade-1 multivectors.

        Returns:
            torch.Tensor: ``[N, dim]`` mean flow bivectors.
        """
        bv = self._connection_bivectors(mv)   # [N, k, dim]
        return bv.mean(dim=1)                 # [N, dim]

    def coherence(self, mv: torch.Tensor) -> float:
        """Measures concentration of connection bivectors within each neighbourhood.

        For each point, computes the mean **absolute** cosine similarity between
        all pairs of its k connection bivectors.  This captures how consistently
        the neighbourhood connections lie on the same rotation plane.

        - **1.0**: all connections at every point are parallel or anti-parallel
          (maximally structured).
        - **1/num_bivectors** (≈ baseline): connections point in random directions.

        .. note::
            In Cl(2,0) the grade-2 space is 1-dimensional (only e₁₂), so
            coherence is trivially 1.0 for any data — use at least Cl(3,0)
            for meaningful discrimination.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            float: Coherence score in [0, 1].
        """
        bv = self._connection_bivectors(mv)   # [N, k, dim]
        N, k, D = bv.shape

        # Pairwise absolute cosine similarity within each neighbourhood
        bi = bv.unsqueeze(2)                  # [N, k, 1, dim]
        bj = bv.unsqueeze(1)                  # [N, 1, k, dim]
        abs_cos = (bi * bj).sum(dim=-1).abs() # [N, k, k]

        # Exclude self-pairs (diagonal)
        mask = ~torch.eye(k, dtype=torch.bool, device=mv.device)  # [k, k]
        off_diag = abs_cos[:, mask]           # [N, k*(k-1)]
        return off_diag.mean().item()

    def curvature(self, mv: torch.Tensor) -> float:
        """Measures how much connection structure changes across the manifold.

        Computes the mean **dissimilarity** of connection bivectors between
        neighbouring pairs of points:

            dissimilarity(i, j) = 1 − mean_abs_cos( {B_ia}, {B_jb} )

        where {B_ia} is the set of k unit connection bivectors at point i and
        {B_jb} at point j, and mean_abs_cos is the cross-set absolute cosine
        similarity.

        - **0.0**: all neighbouring points have the same connection structure
          (flat geodesics, smooth manifold).
        - **High**: the connection direction changes rapidly between neighbours
          (high curvature, fragmented flow).

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            float: Curvature score in [0, 1].
        """
        bv = self._connection_bivectors(mv)   # [N, k, dim]
        N, k, D = bv.shape
        nn_idx = self._knn(mv)                # [N, k_nn]

        # For each neighbouring pair (i, j): cross-set mean |cos|
        bi = bv.unsqueeze(2)                             # [N, k, 1, dim]
        bj_all = bv[nn_idx]                              # [N, k_nn, k, dim]

        # Flatten for efficient computation: compare point i's set against
        # point j's set for the first knn neighbour only (most representative)
        # Use first neighbour for a stable, fast estimate
        bj = bj_all[:, 0]                                # [N, k, dim] — first neighbour's set
        bj = bj.unsqueeze(1)                             # [N, 1, k, dim]

        cross_cos = (bi * bj).sum(dim=-1).abs()          # [N, k, k]
        alignment = cross_cos.mean(dim=(-1, -2))         # [N]

        return (1.0 - alignment.mean()).item()

    def interpolate(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """Interpolates along the geodesic from ``a`` to ``b``.

        Uses the *Lie group exponential map* on the transition element:

            T = a⁻¹ · b
            log(T) ≈ <T - 1>₂     (grade-2 approximation for small angles)
            γ(t) = a · exp(t · log(T))

        Exact when a and b are close; a first-order approximation otherwise.

        Args:
            a (torch.Tensor): Start multivector ``[dim]``.
            b (torch.Tensor): End multivector ``[dim]``.
            steps (int): Number of interpolation steps (including endpoints).

        Returns:
            torch.Tensor: ``[steps, dim]`` sequence of multivectors.
        """
        a = a.unsqueeze(0)  # [1, dim]
        b = b.unsqueeze(0)  # [1, dim]

        # a⁻¹ = ã / <a · ã>₀
        a_rev = self.algebra.reverse(a)
        a_sq = (a * a_rev)[..., 0:1].clamp(min=1e-8)   # grade-0 scalar
        a_inv = a_rev / a_sq                            # [1, dim]

        # Transition element T = a⁻¹ · b
        T = self.algebra.geometric_product(a_inv, b)    # [1, dim]

        # Log approximation: grade-2 part of (T − 1)
        T_shift = T.clone()
        T_shift[..., 0] -= 1.0
        log_T = self.algebra.grade_projection(T_shift, 2)  # [1, dim]

        # Sample t ∈ [0, 1]
        ts = torch.linspace(0.0, 1.0, steps, device=a.device, dtype=a.dtype)

        frames = []
        for t in ts:
            exp_tlogT = self.algebra.exp(t * log_T)       # [1, dim]
            frame = self.algebra.geometric_product(a, exp_tlogT)  # [1, dim]
            frames.append(frame)

        return torch.cat(frames, dim=0)  # [steps, dim]

    def causal_report(self, data: torch.Tensor) -> Dict:
        """Full geodesic flow analysis with a causal interpretation.

        Embeds data, computes coherence and curvature, and returns a
        human-readable verdict:

        - **Causal**: coherence > 0.5 and curvature < 0.5
          → flow is smooth and aligned in one direction.
        - **Noisy**: otherwise
          → flow is fragmented and collides with itself.

        Args:
            data (torch.Tensor): ``[N, d]`` raw data.

        Returns:
            dict with keys: ``coherence``, ``curvature``, ``causal``,
            ``label``.
        """
        mv = self._embed(data)
        coh = self.coherence(mv)
        curv = self.curvature(mv)
        is_causal = (coh > 0.5) and (curv < 0.5)
        return {
            'coherence': coh,
            'curvature': curv,
            'causal': is_causal,
            'label': (
                'Causal — smooth, aligned flow (low curvature)'
                if is_causal else
                'Noisy — fragmented, colliding flow (high curvature)'
            ),
        }


# ---------------------------------------------------------------------------
# Dimension Lifting
# ---------------------------------------------------------------------------

class DimensionLifter:
    """Tests whether lifting data to a higher-dimensional algebra reveals structure.

    The hypothesis: data living on an n-dimensional manifold may possess
    latent structure that only becomes visible in Cl(n+1, q) or Cl(n, q+1).

    Lifting appends extra coordinates to the grade-1 embedding:

    - **Positive lift** ``Cl(p, q) → Cl(p+1, q)``: adds a spacelike dimension.
      The extra coordinate is set to 1 (projective / homogeneous lift).
    - **Null lift** ``Cl(p, q) → Cl(p, q+1)``: adds a timelike dimension.
      The extra coordinate is set to 0 (null vector lift for conformal-like embeddings).

    After lifting, geodesic flow coherence is re-measured.  An improvement
    indicates that the extra dimension captures hidden geometric structure.

    Args:
        device (str): Computation device.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def lift(
        self,
        data: torch.Tensor,
        target_algebra: CliffordAlgebra,
        fill: float = 1.0,
    ) -> torch.Tensor:
        """Lifts data into the grade-1 subspace of a higher-dimensional algebra.

        Pads each data vector with ``fill`` values in the new dimensions,
        then embeds as a grade-1 multivector.

        Args:
            data (torch.Tensor): ``[N, d]`` source data.
            target_algebra (CliffordAlgebra): Target algebra with n ≥ d.
            fill (float): Coordinate value for the new dimensions.
                Use 1.0 for a projective (homogeneous) lift,
                0.0 for a null-vector (conformal) lift.

        Returns:
            torch.Tensor: ``[N, target_algebra.dim]`` grade-1 multivectors.
        """
        N, d = data.shape
        n = target_algebra.n
        if n < d:
            raise ValueError(f"Target algebra dim {n} < source data dim {d}.")
        if n == d:
            return target_algebra.embed_vector(data.to(self.device))

        pad = torch.full(
            (N, n - d), fill,
            device=self.device, dtype=data.dtype
        )
        lifted = torch.cat([data.to(self.device), pad], dim=-1)  # [N, n]
        return target_algebra.embed_vector(lifted)

    def test(
        self,
        data: torch.Tensor,
        p: int,
        q: int,
        k: int = 8,
    ) -> Dict:
        """Compares geodesic flow coherence before and after dimension lifting.

        Tests three algebras in parallel:

        1. **Original** Cl(p, q): baseline coherence and curvature.
        2. **Positive lift** Cl(p+1, q): spacelike extra dimension, fill=1.
        3. **Null lift** Cl(p, q+1): timelike extra dimension, fill=0.

        Args:
            data (torch.Tensor): ``[N, d]`` data where d = p + q.
            p (int): Original positive signature.
            q (int): Original negative signature.
            k (int): Nearest neighbours for geodesic flow.

        Returns:
            dict with keys ``original``, ``lift_positive``, ``lift_null``,
            and ``best`` (name of the algebra with highest coherence).
        """
        data = data.to(self.device)
        N, d = data.shape
        results: Dict = {}

        def _measure(alg: CliffordAlgebra, mv: torch.Tensor) -> Dict:
            gf = GeodesicFlow(alg, k=k)
            coh = gf.coherence(mv)
            curv = gf.curvature(mv)
            return {
                'signature': (alg.p, alg.q),
                'coherence': coh,
                'curvature': curv,
                'causal': (coh > 0.5) and (curv < 0.5),
            }

        # 1. Original
        alg_orig = CliffordAlgebra(p, q, device=self.device)
        mv_orig = alg_orig.embed_vector(data[..., :alg_orig.n])
        results['original'] = _measure(alg_orig, mv_orig)

        # 2. Positive lift: Cl(p+1, q)
        alg_pos = CliffordAlgebra(p + 1, q, device=self.device)
        mv_pos = self.lift(data, alg_pos, fill=1.0)
        results['lift_positive'] = _measure(alg_pos, mv_pos)

        # 3. Null lift: Cl(p, q+1)
        alg_null = CliffordAlgebra(p, q + 1, device=self.device)
        mv_null = self.lift(data, alg_null, fill=0.0)
        results['lift_null'] = _measure(alg_null, mv_null)

        # Which lifting gave the highest coherence?
        best = max(
            ('original', 'lift_positive', 'lift_null'),
            key=lambda key: results[key]['coherence'],
        )
        results['best'] = best

        return results

    def format_report(self, results: Dict) -> str:
        """Renders a lifting test result as a human-readable string.

        Args:
            results (dict): Output of :meth:`test`.

        Returns:
            str: Multi-line report.
        """
        lines = ['Dimension Lifting Report', '=' * 40]
        for key in ('original', 'lift_positive', 'lift_null'):
            r = results[key]
            p, q = r['signature']
            coh = r['coherence']
            curv = r['curvature']
            causal = '✓ Causal' if r['causal'] else '✗ Noisy'
            lines.append(
                f"  Cl({p},{q})  coherence={coh:+.3f}  curvature={curv:.3f}  {causal}"
            )
        lines.append(f"\n  Best algebra: {results['best']}")
        return '\n'.join(lines)
