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

Also provides geodesic flow analysis and dimension lifting tests - tools to
probe whether causal structure is visible in a dataset's multivector geometry.
"""

import torch
import torch.nn as nn
import warnings
import copy
import concurrent.futures
from typing import Tuple, List, Optional, Dict
from core.algebra import CliffordAlgebra
from core.metric import induced_norm
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.projection import BladeSelector


class _SignatureProbe(nn.Module):
    """Minimal single-rotor probe for bivector energy analysis.

    Architecture: CliffordLinear(1, C) -> RotorLayer(C) -> BladeSelector(C).
    Only one linear layer for channel expansion; the rotor bivector energy
    is the primary signal for signature discovery.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int = 4):
        """Initialize Signature Probe.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Channel count.
        """
        super().__init__()
        self.algebra = algebra
        self.linear_in = CliffordLinear(algebra, 1, channels)
        self.rotor = RotorLayer(algebra, channels)
        self.linear_out = CliffordLinear(algebra, channels, 1)
        self.selector = BladeSelector(algebra, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): [N, 1, dim] input tokens.

        Returns:
            torch.Tensor: [N, 1, dim] output tokens.
        """
        x = self.linear_in(x)   # [N, C, dim]
        x = self.rotor(x)       # [N, C, dim]
        x = self.linear_out(x)  # [N, 1, dim]
        x = self.selector(x)
        return x

    def get_rotor_layers(self) -> List[RotorLayer]:
        """Returns all RotorLayer modules for post-training analysis.

        Returns:
            List[RotorLayer]: List of rotor layers in the probe.
        """
        return [m for m in self.modules() if isinstance(m, RotorLayer)]


def _apply_biased_init(probe: _SignatureProbe, algebra: CliffordAlgebra,
                       bias_type: str = 'random') -> None:
    """Biases RotorLayer bivector weights based on signature type.

    Uses ``algebra.bv_sq_scalar`` to classify each basis bivector:
    - bv_sq = -1: elliptic (positive-signature base vectors)
    - bv_sq = +1: hyperbolic (mixed-signature base vectors)
    - bv_sq =  0: null (degenerate base vectors)

    Args:
        probe (_SignatureProbe): The probe model to initialize.
        algebra (CliffordAlgebra): The algebra for signature classification.
        bias_type (str): One of 'euclidean', 'minkowski', 'projective', 'random'.
    """
    bv_sq = algebra.bv_sq_scalar  # [num_bivectors]
    for rotor in probe.get_rotor_layers():
        with torch.no_grad():
            if bias_type == 'euclidean':
                # Heavy weight on elliptic bivectors (bv_sq = -1)
                weights = torch.where(bv_sq < -0.5, torch.tensor(1.0), torch.tensor(0.1))
                rotor.bivector_weights.copy_(
                    weights.unsqueeze(0).expand_as(rotor.bivector_weights)
                    + torch.randn_like(rotor.bivector_weights) * 0.05
                )
            elif bias_type == 'minkowski':
                # Mixed elliptic + hyperbolic
                weights = torch.where(
                    bv_sq.abs() > 0.5, torch.tensor(1.0), torch.tensor(0.1)
                )
                rotor.bivector_weights.copy_(
                    weights.unsqueeze(0).expand_as(rotor.bivector_weights)
                    + torch.randn_like(rotor.bivector_weights) * 0.05
                )
            elif bias_type == 'projective':
                # Uniform across all types including null
                nn.init.uniform_(rotor.bivector_weights, -0.5, 0.5)
            else:  # 'random'
                nn.init.normal_(rotor.bivector_weights, 0.0, 0.3)


class MetricSearch:
    """Learns optimal (p, q, r) signature via GBN probe training and bivector
    energy analysis.

    Trains small single-rotor GBN probes on conformally-lifted data using
    coherence + curvature as the loss. After training, reads the learned
    bivector energy distribution to infer the optimal signature.

    Multiple probes with biased initialization combat local minima.
    """

    def __init__(
        self,
        device: str = 'cpu',
        num_probes: int = 6,
        probe_epochs: int = 80,
        probe_lr: float = 0.005,
        probe_channels: int = 4,
        k: int = 8,
        energy_threshold: float = 0.05,
        curvature_weight: float = 0.3,
        sparsity_weight: float = 0.01,
        max_workers: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        early_stop_patience: int = 0,
    ):
        """Initialize Metric Search.

        Args:
            device (str): Computation device.
            num_probes (int): Number of parallel probes to train.
            probe_epochs (int): Epochs per probe training.
            probe_lr (float): Learning rate for probes.
            probe_channels (int): Channels in probe models.
            k (int): Nearest neighbors for flow analysis.
            energy_threshold (float): Energy cutoff for active base vectors.
            curvature_weight (float): Curvature penalty in probe loss.
            sparsity_weight (float): Sparsity penalty in probe loss.
            max_workers (int, optional): Max threads for parallel training.
            micro_batch_size (int, optional): If set, train probes on random
                micro-batches of this size each epoch instead of full data.
            early_stop_patience (int): Stop probe training if best loss has
                not improved for this many epochs. 0 disables early stopping.
        """
        self.device = device
        self.num_probes = num_probes
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.probe_channels = probe_channels
        self.k = k
        self.energy_threshold = energy_threshold
        self.curvature_weight = curvature_weight
        self.sparsity_weight = sparsity_weight
        self.max_workers = max_workers
        self.micro_batch_size = micro_batch_size
        self.early_stop_patience = early_stop_patience

    def _lift_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, CliffordAlgebra]:
        """Lifts [N, X] data into Cl(X+1, 1, 0) via CGA-style embedding.

        Adds 2 extra dimensions (one positive, one negative):
            lifted = [x_1..x_X, 0.5*||x||^2, 1.0]

        Args:
            data (torch.Tensor): Input data [N, X].

        Returns:
            Tuple[torch.Tensor, CliffordAlgebra]: (mv_data [N, 1, dim], algebra).
        """
        data = data.to(self.device)
        N, X = data.shape

        if X + 2 > 8:
            warnings.warn(
                f"Data dimension {X} yields algebra dim 2^{X+2}={2**(X+2)}. "
                f"Consider PCA pre-reduction to X <= 6 for tractable computation."
            )

        # CGA-style: [x, 0.5*||x||^2, 1.0]
        norm_sq = 0.5 * (data ** 2).sum(dim=-1, keepdim=True)
        ones = torch.ones(N, 1, device=self.device, dtype=data.dtype)
        lifted = torch.cat([data, norm_sq, ones], dim=-1)  # [N, X+2]

        algebra = CliffordAlgebra(X + 1, 1, 0, device=self.device)
        mv = algebra.embed_vector(lifted)  # [N, 2^(X+2)]
        mv = mv.unsqueeze(1)  # [N, 1, dim]
        return mv, algebra

    def _train_probe(
        self,
        mv_data: torch.Tensor,
        algebra: CliffordAlgebra,
        bias_type: str = 'random',
    ) -> Dict:
        """Trains a single probe and returns results.

        Args:
            mv_data (torch.Tensor): Conformally lifted data [N, 1, dim].
            algebra (CliffordAlgebra): The conformal algebra.
            bias_type (str): Initialization bias type.

        Returns:
            Dict: Results with 'loss', 'coherence', 'curvature', 'probe'.
        """
        probe = _SignatureProbe(algebra, channels=self.probe_channels)
        probe.to(self.device)
        _apply_biased_init(probe, algebra, bias_type)

        gf = GeodesicFlow(algebra, k=self.k)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        N = mv_data.shape[0]

        for _ in range(self.probe_epochs):
            # Micro-batch: sample a random subset each epoch
            if self.micro_batch_size and self.micro_batch_size < N:
                idx = torch.randperm(N, device=mv_data.device)[:self.micro_batch_size]
                batch = mv_data[idx]
            else:
                batch = mv_data

            optimizer.zero_grad()
            output = probe(batch)  # [B, 1, dim]
            output_flat = output.squeeze(1)  # [B, dim]

            coherence_t = gf._coherence_tensor(output_flat)
            curvature_t = gf._curvature_tensor(output_flat)

            sparsity = sum(
                r.sparsity_loss() for r in probe.get_rotor_layers()
            )

            loss = (
                -coherence_t
                + self.curvature_weight * curvature_t
                + self.sparsity_weight * sparsity
            )

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = copy.deepcopy(probe.state_dict())
                patience_counter = 0
            elif self.early_stop_patience > 0:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break

        # Restore best
        if best_state is not None:
            probe.load_state_dict(best_state)

        # Final metrics
        with torch.no_grad():
            output = probe(mv_data).squeeze(1)
            coh = gf.coherence(output)
            curv = gf.curvature(output)

        return {
            'loss': best_loss,
            'coherence': coh,
            'curvature': curv,
            'probe': probe,
        }

    def _analyze_bivector_energy(
        self,
        probe: _SignatureProbe,
        algebra: CliffordAlgebra,
        original_dim: int,
    ) -> Tuple[Tuple[int, int, int], Dict]:
        """Maps learned bivector energy to (p, q, r) signature.

        Algorithm:
        1. Collect bivector_weights from all RotorLayers -> energy = weight^2
        2. Average across channels
        3. For each basis bivector e_ab, look up bv_sq_scalar:
           -1 -> elliptic, +1 -> hyperbolic, 0 -> null
        4. Collect active base vector indices (energy > threshold) by type
        5. Subtract 2 conformal dimensions from counts -> (p, q, r)

        Args:
            probe (_SignatureProbe): Trained probe model.
            algebra (CliffordAlgebra): The conformal algebra.
            original_dim (int): Original data dimension X (before lifting).

        Returns:
            Tuple[Tuple[int, int, int], Dict]: ((p, q, r), energy_breakdown dict).
        """
        bv_sq = algebra.bv_sq_scalar  # [num_bivectors]
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        # Collect energy from all rotor layers
        total_energy = torch.zeros(len(bv_indices), device=self.device)
        n_layers = 0
        for rotor in probe.get_rotor_layers():
            with torch.no_grad():
                # [channels, num_bivectors] -> energy = weight^2, mean across channels
                energy = (rotor.bivector_weights ** 2).mean(dim=0)
                total_energy += energy
                n_layers += 1

        if n_layers > 0:
            total_energy /= n_layers

        # Normalize
        max_energy = total_energy.max().clamp(min=1e-8)
        normalized_energy = total_energy / max_energy

        # Classify base vectors by their bivector participation
        n = algebra.n
        base_type = {}  # base_vector_index -> set of signature types it participates in
        base_active = {}  # base_vector_index -> max energy

        for bv_idx_pos, blade_idx in enumerate(bv_indices.tolist()):
            energy_val = normalized_energy[bv_idx_pos].item()
            # Decode which base vectors form this bivector
            bits = []
            for bit in range(n):
                if blade_idx & (1 << bit):
                    bits.append(bit)
            if len(bits) != 2:
                continue

            sq_val = bv_sq[bv_idx_pos].item()
            if sq_val < -0.5:
                sig_type = 'elliptic'
            elif sq_val > 0.5:
                sig_type = 'hyperbolic'
            else:
                sig_type = 'null'

            for b in bits:
                if b not in base_type:
                    base_type[b] = set()
                    base_active[b] = 0.0
                base_type[b].add(sig_type)
                base_active[b] = max(base_active[b], energy_val)

        # Count active base vectors by dominant type
        active_positive = 0  # base vectors mostly in elliptic bivectors
        active_negative = 0  # base vectors in hyperbolic bivectors
        active_null = 0  # base vectors in null bivectors

        for b_idx in range(n):
            if b_idx not in base_active or base_active[b_idx] < self.energy_threshold:
                continue
            types = base_type.get(b_idx, set())
            if 'null' in types and 'elliptic' not in types and 'hyperbolic' not in types:
                active_null += 1
            elif 'hyperbolic' in types:
                active_negative += 1
            else:
                active_positive += 1

        # Subtract 2 conformal dimensions (1 positive + 1 negative from CGA lift)
        p = max(0, active_positive - 1)
        q = max(0, active_negative - 1)
        r = active_null

        # Clamp to original data dimension
        total = p + q + r
        if total > original_dim:
            # Scale down proportionally
            scale = original_dim / max(total, 1)
            p = max(1, round(p * scale))
            q = round(q * scale)
            r = round(r * scale)
            # Ensure they sum to <= original_dim
            while p + q + r > original_dim:
                if r > 0:
                    r -= 1
                elif q > 0:
                    q -= 1
                else:
                    p -= 1
        elif total == 0:
            p = original_dim  # Default to Euclidean

        energy_breakdown = {
            'per_bivector_energy': normalized_energy.tolist(),
            'active_positive': active_positive,
            'active_negative': active_negative,
            'active_null': active_null,
            'bv_sq_scalar': bv_sq.tolist(),
        }

        return (p, q, r), energy_breakdown

    def search(self, data: torch.Tensor) -> Tuple[int, int, int]:
        """Returns optimal (p, q, r) signature for the data.

        Args:
            data (torch.Tensor): Input data [N, D].

        Returns:
            Tuple[int, int, int]: Optimal signature (p, q, r).
        """
        result = self.search_detailed(data)
        return result['signature']

    def search_detailed(self, data: torch.Tensor) -> Dict:
        """Returns signature and full diagnostics.

        Args:
            data (torch.Tensor): Input data [N, D].

        Returns:
            Dict: Diagnostics with 'signature', 'coherence', 'curvature',
                'energy_breakdown', 'per_probe_results'.
        """
        data = data.to(self.device)
        N, X = data.shape

        # Conformal lift
        mv_data, algebra = self._lift_data(data)

        # Bias schedule: first 3 are deterministic, rest random
        bias_types = ['euclidean', 'minkowski', 'projective']
        while len(bias_types) < self.num_probes:
            bias_types.append('random')
        bias_types = bias_types[:self.num_probes]

        # Train probes
        def _run_probe(bias_type):
            return self._train_probe(mv_data, algebra, bias_type)

        if self.num_probes <= 2:
            # Sequential for small probe counts
            probe_results = [_run_probe(bt) for bt in bias_types]
        else:
            # Parallel with threads (avoids pickle issues, PyTorch releases GIL)
            max_w = self.max_workers or min(self.num_probes, 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = [pool.submit(_run_probe, bt) for bt in bias_types]
                probe_results = [f.result() for f in futures]

        # Select best probe by loss
        best_idx = min(range(len(probe_results)), key=lambda i: probe_results[i]['loss'])
        best = probe_results[best_idx]

        # Analyze bivector energy
        signature, energy_breakdown = self._analyze_bivector_energy(
            best['probe'], algebra, X
        )

        return {
            'signature': signature,
            'coherence': best['coherence'],
            'curvature': best['curvature'],
            'energy_breakdown': energy_breakdown,
            'per_probe_results': [
                {'loss': r['loss'], 'coherence': r['coherence'], 'curvature': r['curvature']}
                for r in probe_results
            ],
        }


class GeodesicFlow:
    """Geodesic flow analysis in Clifford algebra.

    Interprets data points as grade-1 multivectors and computes the *flow
    field* - a bivector at each point that encodes the direction of shortest
    algebraic paths to its k-nearest neighbours.

    The flow is computed as the mean of *connection bivectors*:

        B_ij = <x_i . ~x_j>_2   (grade-2 part of the geometric product)

    This bivector encodes the rotational "turn" needed to map x_i toward
    x_j, analogous to the parallel transport connection on a Lie group.

    The coherence and curvature of this field reveal whether the data has
    causal (directional) structure:

    - **High coherence, low curvature** -> the flow is smooth and aligned in
      one direction.  Causality is visible.
    - **Low coherence, high curvature** -> the flow is fragmented and
      collides with itself.  The signal is dominated by noise.
    """

    def __init__(self, algebra: CliffordAlgebra, k: int = 8):
        """Initialize Geodesic Flow.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            k (int): Number of nearest neighbours for the flow field.
        """
        self.algebra = algebra
        self.k = k

    def _embed(self, data: torch.Tensor) -> torch.Tensor:
        """Embeds raw vectors into the grade-1 multivector subspace.

        Args:
            data (torch.Tensor): ``[N, d]`` where d <= algebra.n.

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

    def _connection_bivectors(self, mv: torch.Tensor) -> torch.Tensor:
        """Computes unit connection bivectors for all (point, neighbour) pairs.

        The connection bivector from x_i to x_j encodes the rotational "turn"
        in the algebra needed to map one vector toward the other:

            B_ij = unit( <x_i . ~x_j>_2 )

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

            B_i = mean_j { unit( <x_i . ~x_j>_2 ) }

        .. note::
            For perfectly symmetric data (e.g. a closed circle) the mean
            cancels to zero - which is geometrically correct since there is
            no preferred flow direction.  Use :meth:`coherence` to measure
            structure without this cancellation.

        Args:
            mv (torch.Tensor): ``[N, dim]`` grade-1 multivectors.

        Returns:
            torch.Tensor: ``[N, dim]`` mean flow bivectors.
        """
        bv = self._connection_bivectors(mv)   # [N, k, dim]
        return bv.mean(dim=1)                 # [N, dim]

    def _coherence_tensor(self, mv: torch.Tensor) -> torch.Tensor:
        """Differentiable coherence — returns a scalar tensor with grad_fn.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: Scalar coherence in [0, 1].
        """
        bv = self._connection_bivectors(mv)   # [N, k, dim]
        N, k, D = bv.shape

        bi = bv.unsqueeze(2)                  # [N, k, 1, dim]
        bj = bv.unsqueeze(1)                  # [N, 1, k, dim]
        abs_cos = (bi * bj).sum(dim=-1).abs() # [N, k, k]

        mask = ~torch.eye(k, dtype=torch.bool, device=mv.device)  # [k, k]
        off_diag = abs_cos[:, mask]           # [N, k*(k-1)]
        return off_diag.mean()

    def coherence(self, mv: torch.Tensor) -> float:
        """Measures concentration of connection bivectors within each neighbourhood.

        For each point, computes the mean **absolute** cosine similarity between
        all pairs of its k connection bivectors.  This captures how consistently
        the neighbourhood connections lie on the same rotation plane.

        - **1.0**: all connections at every point are parallel or anti-parallel
          (maximally structured).
        - **1/num_bivectors** (~= baseline): connections point in random directions.

        .. note::
            In Cl(2,0) the grade-2 space is 1-dimensional (only e_12), so
            coherence is trivially 1.0 for any data - use at least Cl(3,0)
            for meaningful discrimination.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            float: Coherence score in [0, 1].
        """
        return self._coherence_tensor(mv).item()

    def _curvature_tensor(self, mv: torch.Tensor) -> torch.Tensor:
        """Differentiable curvature — returns a scalar tensor with grad_fn.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: Scalar curvature in [0, 1].
        """
        bv = self._connection_bivectors(mv)   # [N, k, dim]
        N, k, D = bv.shape
        nn_idx = self._knn(mv)                # [N, k_nn]

        bi = bv.unsqueeze(2)                             # [N, k, 1, dim]
        bj_all = bv[nn_idx]                              # [N, k_nn, k, dim]

        bj = bj_all[:, 0]                                # [N, k, dim]
        bj = bj.unsqueeze(1)                             # [N, 1, k, dim]

        cross_cos = (bi * bj).sum(dim=-1).abs()          # [N, k, k]
        alignment = cross_cos.mean(dim=(-1, -2))         # [N]

        return 1.0 - alignment.mean()

    def curvature(self, mv: torch.Tensor) -> float:
        """Measures how much connection structure changes across the manifold.

        Computes the mean **dissimilarity** of connection bivectors between
        neighbouring pairs of points:

            dissimilarity(i, j) = 1 - mean_abs_cos( {B_ia}, {B_jb} )

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
        return self._curvature_tensor(mv).item()

    def interpolate(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """Interpolates along the geodesic from ``a`` to ``b``.

        Uses the *Lie group exponential map* on the transition element:

            T = a_inv . b
            log(T) ~= <T - 1>_2     (grade-2 approximation for small angles)
            gamma(t) = a . exp(t . log(T))

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

        # a_inv = ~a / <a . ~a>_0
        a_rev = self.algebra.reverse(a)
        a_sq = (a * a_rev)[..., 0:1].clamp(min=1e-8)   # grade-0 scalar
        a_inv = a_rev / a_sq                            # [1, dim]

        # Transition element T = a_inv . b
        T = self.algebra.geometric_product(a_inv, b)    # [1, dim]

        # Log approximation: grade-2 part of (T - 1)
        T_shift = T.clone()
        T_shift[..., 0] -= 1.0
        log_T = self.algebra.grade_projection(T_shift, 2)  # [1, dim]

        # Sample t in [0, 1]
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
          -> flow is smooth and aligned in one direction.
        - **Noisy**: otherwise
          -> flow is fragmented and collides with itself.

        Args:
            data (torch.Tensor): ``[N, d]`` raw data.

        Returns:
            Dict: report with keys ``coherence``, ``curvature``, ``causal``, ``label``.
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
                'Causal - smooth, aligned flow (low curvature)'
                if is_causal else
                'Noisy - fragmented, colliding flow (high curvature)'
            ),
        }


class DimensionLifter:
    """Tests whether lifting data to a higher-dimensional algebra reveals structure.

    The hypothesis: data living on an n-dimensional manifold may possess
    latent structure that only becomes visible in Cl(n+1, q) or Cl(n, q+1).

    Lifting appends extra coordinates to the grade-1 embedding:

    - **Positive lift** ``Cl(p, q) -> Cl(p+1, q)``: adds a spacelike dimension.
      The extra coordinate is set to 1 (projective / homogeneous lift).
    - **Null lift** ``Cl(p, q) -> Cl(p, q+1)``: adds a timelike dimension.
      The extra coordinate is set to 0 (null vector lift for conformal-like embeddings).

    After lifting, geodesic flow coherence is re-measured.  An improvement
    indicates that the extra dimension captures hidden geometric structure.
    """

    def __init__(self, device: str = 'cpu'):
        """Initialize Dimension Lifter.

        Args:
            device (str): Computation device.
        """
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
            target_algebra (CliffordAlgebra): Target algebra with n >= d.
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
            Dict: results with keys ``original``, ``lift_positive``, ``lift_null``,
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
            results (Dict): Output of :meth:`test`.

        Returns:
            str: Multi-line report.
        """
        lines = ['Dimension Lifting Report', '=' * 40]
        for key in ('original', 'lift_positive', 'lift_null'):
            r = results[key]
            p, q = r['signature']
            coh = r['coherence']
            curv = r['curvature']
            causal = 'O Causal' if r['causal'] else 'X Noisy'
            lines.append(
                f"  Cl({p},{q})  coherence={coh:+.3f}  curvature={curv:.3f}  {causal}"
            )
        lines.append(f"\n  Best algebra: {results['best']}")
        return '\n'.join(lines)
