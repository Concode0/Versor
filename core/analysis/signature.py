# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Automated metric-signature search for optimal geometric signatures.

Provides :class:`MetricSearch` (probe-based signature discovery) and
:class:`SignatureSearchAnalyzer` (higher-level wrapper with dimension
reduction and bootstrap confidence intervals).
"""

import concurrent.futures
import copy
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.config import make_algebra
from core.module import AlgebraLike
from layers import BladeSelector, CliffordLinear, RotorLayer

from ._types import CONSTANTS, DimensionResult, SamplingConfig, SignatureResult
from .geodesic import GeodesicFlow


class _SignatureProbe(nn.Module):
    """Minimal single-rotor probe for bivector energy analysis.

    Architecture: CliffordLinear(1, C) -> RotorLayer(C) -> BladeSelector(C).
    Only one linear layer for channel expansion; the rotor bivector energy
    is the primary signal for signature discovery.
    """

    def __init__(self, algebra: AlgebraLike, channels: int = 4):
        super().__init__()
        self.algebra = algebra
        self.linear_in = CliffordLinear(algebra, 1, channels)
        self.rotor = RotorLayer(algebra, channels)
        self.linear_out = CliffordLinear(algebra, channels, 1)
        self.selector = BladeSelector(algebra, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.rotor(x)
        x = self.linear_out(x)
        x = self.selector(x)
        return x

    def get_rotor_layers(self) -> List[RotorLayer]:
        return [m for m in self.modules() if isinstance(m, RotorLayer)]


def _apply_biased_init(
    probe: _SignatureProbe,
    algebra: AlgebraLike,
    bias_type: str = "random",
) -> None:
    """Biases RotorLayer bivector weights based on signature type.

    Uses ``algebra.bv_sq_scalar`` to classify each basis bivector:
    - bv_sq = -1: elliptic (positive-signature base vectors)
    - bv_sq = +1: hyperbolic (mixed-signature base vectors)
    - bv_sq =  0: null (degenerate base vectors)
    """
    bv_sq = algebra.bv_sq_scalar
    ell = CONSTANTS.bv_sq_elliptic_bound
    hyp = CONSTANTS.bv_sq_hyperbolic_bound
    for rotor in probe.get_rotor_layers():
        with torch.no_grad():
            if bias_type == "euclidean":
                weights = torch.where(bv_sq < ell, torch.tensor(1.0), torch.tensor(0.1))
                rotor.bivector_weights.copy_(
                    weights.unsqueeze(0).expand_as(rotor.bivector_weights)
                    + torch.randn_like(rotor.bivector_weights) * 0.05
                )
            elif bias_type == "minkowski":
                weights = torch.where(bv_sq.abs() > hyp, torch.tensor(1.0), torch.tensor(0.1))
                rotor.bivector_weights.copy_(
                    weights.unsqueeze(0).expand_as(rotor.bivector_weights)
                    + torch.randn_like(rotor.bivector_weights) * 0.05
                )
            elif bias_type == "projective":
                nn.init.uniform_(rotor.bivector_weights, -0.5, 0.5)
            else:  # 'random'
                nn.init.normal_(rotor.bivector_weights, 0.0, 0.3)


class MetricSearch:
    """Learns optimal (p, q, r) signature via GBN probe training and bivector
    energy analysis.

    Trains small single-rotor GBN probes on conformally-lifted data using
    coherence + curvature as the loss.  After training, reads the learned
    bivector energy distribution to infer the optimal signature.

    Multiple probes with biased initialization combat local minima.
    """

    def __init__(
        self,
        device: str = "cpu",
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

    def _lift_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, AlgebraLike]:
        """Lifts [N, X] data into Cl(X+1, 1, 0) via CGA-style embedding."""
        data = data.to(self.device)
        N, X = data.shape

        if X + 2 > 12:
            warnings.warn(
                f"Data dimension {X} yields algebra dim 2^{X + 2}={2 ** (X + 2)}. "
                f"Consider PCA pre-reduction to X <= 10 for tractable computation."
            )

        norm_sq = 0.5 * (data**2).sum(dim=-1, keepdim=True)
        ones = torch.ones(N, 1, device=self.device, dtype=data.dtype)
        lifted = torch.cat([data, norm_sq, ones], dim=-1)

        algebra = make_algebra(X + 1, 1, 0, device=self.device)
        mv = algebra.embed_vector(lifted)
        mv = mv.unsqueeze(1)
        return mv, algebra

    def _train_probe(
        self,
        mv_data: torch.Tensor,
        algebra: AlgebraLike,
        bias_type: str = "random",
    ) -> Dict:
        """Trains a single probe and returns results."""
        probe = _SignatureProbe(algebra, channels=self.probe_channels)
        probe.to(self.device)
        _apply_biased_init(probe, algebra, bias_type)

        gf = GeodesicFlow(algebra, k=self.k)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        N = mv_data.shape[0]

        for _ in range(self.probe_epochs):
            if self.micro_batch_size and self.micro_batch_size < N:
                idx = torch.randperm(N, device=mv_data.device)[: self.micro_batch_size]
                batch = mv_data[idx]
            else:
                batch = mv_data

            optimizer.zero_grad()
            output = probe(batch)
            output_flat = output.squeeze(1)

            coherence_t = gf._coherence_tensor(output_flat)
            curvature_t = gf._curvature_tensor(output_flat)

            sparsity = sum(r.sparsity_loss() for r in probe.get_rotor_layers())

            loss = -coherence_t + self.curvature_weight * curvature_t + self.sparsity_weight * sparsity

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

        if best_state is not None:
            probe.load_state_dict(best_state)

        with torch.no_grad():
            output = probe(mv_data).squeeze(1)
            coh = gf.coherence(output)
            curv = gf.curvature(output)

        return {
            "loss": best_loss,
            "coherence": coh,
            "curvature": curv,
            "probe": probe,
        }

    def _analyze_bivector_energy(
        self,
        probe: _SignatureProbe,
        algebra: AlgebraLike,
        original_dim: int,
    ) -> Tuple[Tuple[int, int, int], Dict]:
        """Maps learned bivector energy to (p, q, r) signature."""
        bv_sq = algebra.bv_sq_scalar
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        total_energy = torch.zeros(len(bv_indices), device=self.device)
        n_layers = 0
        for rotor in probe.get_rotor_layers():
            with torch.no_grad():
                energy = (rotor.bivector_weights**2).mean(dim=0)
                total_energy += energy
                n_layers += 1

        if n_layers > 0:
            total_energy /= n_layers

        max_energy = total_energy.max().clamp(min=algebra.eps)
        normalized_energy = total_energy / max_energy

        n = algebra.n
        base_type_energy: dict = {}
        base_active: dict = {}

        for bv_idx_pos, blade_idx in enumerate(bv_indices.tolist()):
            energy_val = normalized_energy[bv_idx_pos].item()
            bits = []
            for bit in range(n):
                if blade_idx & (1 << bit):
                    bits.append(bit)
            if len(bits) != 2:
                continue

            sq_val = bv_sq[bv_idx_pos].item()
            if sq_val < CONSTANTS.bv_sq_elliptic_bound:
                sig_type = "elliptic"
            elif sq_val > CONSTANTS.bv_sq_hyperbolic_bound:
                sig_type = "hyperbolic"
            else:
                sig_type = "null"

            for b in bits:
                if b not in base_type_energy:
                    base_type_energy[b] = {
                        "elliptic": 0.0,
                        "hyperbolic": 0.0,
                        "null": 0.0,
                    }
                    base_active[b] = 0.0
                base_type_energy[b][sig_type] += energy_val
                base_active[b] = max(base_active[b], energy_val)

        active_positive = 0
        active_negative = 0
        active_null = 0

        for b_idx in range(n):
            if b_idx not in base_active or base_active[b_idx] < self.energy_threshold:
                continue
            type_energy = base_type_energy[b_idx]
            dominant = max(type_energy, key=type_energy.get)
            if dominant == "null":
                active_null += 1
            elif dominant == "hyperbolic":
                active_negative += 1
            else:
                active_positive += 1

        p = max(0, active_positive - 1)
        q = max(0, active_negative - 1)
        r = active_null

        total = p + q + r
        if total > original_dim:
            scale = original_dim / max(total, 1)
            p = max(1, round(p * scale))
            q = round(q * scale)
            r = round(r * scale)
            while p + q + r > original_dim:
                if r > 0:
                    r -= 1
                elif q > 0:
                    q -= 1
                else:
                    p -= 1
        elif total == 0:
            p = original_dim

        energy_breakdown = {
            "per_bivector_energy": normalized_energy.tolist(),
            "active_positive": active_positive,
            "active_negative": active_negative,
            "active_null": active_null,
            "bv_sq_scalar": bv_sq.tolist(),
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
        return result["signature"]

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

        mv_data, algebra = self._lift_data(data)

        bias_types = ["euclidean", "minkowski", "projective"]
        while len(bias_types) < self.num_probes:
            bias_types.append("random")
        bias_types = bias_types[: self.num_probes]

        def _run_probe(bias_type):
            return self._train_probe(mv_data, algebra, bias_type)

        if self.num_probes <= 2:
            probe_results = [_run_probe(bt) for bt in bias_types]
        else:
            max_w = self.max_workers or min(self.num_probes, 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = [pool.submit(_run_probe, bt) for bt in bias_types]
                probe_results = [f.result() for f in futures]

        best_idx = min(range(len(probe_results)), key=lambda i: probe_results[i]["loss"])
        best = probe_results[best_idx]

        signature, energy_breakdown = self._analyze_bivector_energy(best["probe"], algebra, X)

        return {
            "signature": signature,
            "coherence": best["coherence"],
            "curvature": best["curvature"],
            "energy_breakdown": energy_breakdown,
            "per_probe_results": [
                {
                    "loss": r["loss"],
                    "coherence": r["coherence"],
                    "curvature": r["curvature"],
                }
                for r in probe_results
            ],
        }


class SignatureSearchAnalyzer:
    """Discover optimal ``(p, q, r)`` with automatic dimension reduction.

    Wraps :class:`MetricSearch` and adds:

    * Automatic PCA reduction when the input exceeds the algebra
      tractability threshold.
    * Bootstrap confidence intervals on the signature estimate.

    Args:
        device: Torch device string.
        max_search_dim: Maximum data dimension before PCA reduction.
            CGA lift adds 2, so ``max_search_dim=10`` -> Cl(11,1) with
            2^12 = 4096-dim multivectors (the algebra ceiling).
            Defaults to 10.
        metric_search_kwargs: Extra keyword arguments forwarded to
            :class:`MetricSearch`.
    """

    def __init__(
        self,
        device: str = "cpu",
        max_search_dim: int = 10,
        metric_search_kwargs: Optional[Dict] = None,
    ):
        self.device = device
        self.max_search_dim = max_search_dim
        self._ms_kwargs = metric_search_kwargs or {}

    def analyze(
        self,
        data: torch.Tensor,
        dim_result: Optional[DimensionResult] = None,
    ) -> SignatureResult:
        """Run metric-signature search, optionally reducing dimensions.

        Args:
            data: ``[N, D]`` raw data.
            dim_result: Pre-computed dimension analysis.  When provided
                and ``dim_result.intrinsic_dim < D``, the data is
                PCA-reduced before searching.

        Returns:
            :class:`SignatureResult`.
        """
        from .dimension import EffectiveDimensionAnalyzer

        data = data.to(self.device).float()
        effective_dim_used = None

        target_dim = data.shape[1]
        if dim_result is not None and dim_result.intrinsic_dim < target_dim:
            target_dim = dim_result.intrinsic_dim

        if target_dim > self.max_search_dim:
            target_dim = self.max_search_dim

        if target_dim < data.shape[1]:
            reducer = EffectiveDimensionAnalyzer(device=self.device)
            data = reducer.reduce(data, target_dim)
            effective_dim_used = target_dim

        ms = MetricSearch(device=self.device, **self._ms_kwargs)
        result = ms.search_detailed(data)

        return SignatureResult(
            signature=result["signature"],
            coherence=result.get("coherence", 0.0),
            curvature=result.get("curvature", 0.0),
            energy_breakdown=result.get("energy_breakdown", {}),
            effective_dim_used=effective_dim_used,
        )

    def analyze_with_confidence(
        self,
        data: torch.Tensor,
        n_bootstrap: int = 10,
        dim_result: Optional[DimensionResult] = None,
    ) -> Tuple[SignatureResult, Dict]:
        """Signature search with bootstrap confidence estimate.

        Runs the search on *n_bootstrap* resampled datasets and reports
        the majority-vote signature plus the full distribution.

        Returns:
            ``(best_result, confidence)`` where *confidence* contains
            ``"distribution"`` (Counter) and ``"agreement"`` (float).
        """
        from .sampler import StatisticalSampler

        cfg = SamplingConfig(
            strategy="bootstrap",
            max_samples=min(data.shape[0], 500),
            n_bootstrap=n_bootstrap,
        )
        resamples, _ = StatisticalSampler.sample(data, cfg)

        signatures = []
        results = []
        for sample in resamples:
            r = self.analyze(sample, dim_result=dim_result)
            signatures.append(r.signature)
            results.append(r)

        counter = Counter(signatures)
        best_sig = counter.most_common(1)[0][0]
        agreement = counter[best_sig] / len(signatures)

        best_result = next(r for r in results if r.signature == best_sig)

        confidence = {
            "distribution": dict(counter),
            "agreement": agreement,
            "n_bootstrap": n_bootstrap,
        }
        return best_result, confidence
