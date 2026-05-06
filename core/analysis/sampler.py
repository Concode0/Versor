# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Statistical sampling strategies for geometric data analysis.

The stratified sampler uses geodesic-flow coherence as a universal
stratification criterion -- it does not assume a specific metric
signature for the data.
"""

from typing import Dict, List, Tuple, Union

import torch

from ._types import SamplingConfig


class StatisticalSampler:
    """Stateless sampler supporting multiple strategies.

    Supports ``"random"``, ``"stratified"``, ``"bootstrap"``, and
    ``"passthrough"`` strategies.  All methods are deterministic when a
    seed is provided.

    The ``"stratified"`` strategy uses per-point geodesic-flow coherence
    to partition data into geometrically meaningful strata (high-structure
    vs low-structure regions), ensuring coverage of diverse geometric
    neighbourhoods without assuming a specific metric signature.
    """

    @staticmethod
    def sample(
        data: torch.Tensor,
        config: SamplingConfig,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict]:
        """Sample from *data* according to *config*.

        Args:
            data: ``[N, D]`` tensor of observations.
            config: Sampling configuration.

        Returns:
            ``(sampled, metadata)`` where *sampled* is a tensor (or list
            of tensors for ``"bootstrap"``) and *metadata* is a dict with
            at least ``"strategy"``.
        """
        strategy = config.strategy
        if strategy == "passthrough":
            return data, {"strategy": "passthrough", "n_original": data.shape[0]}

        if strategy == "random":
            return StatisticalSampler._random(data, config)
        if strategy == "stratified":
            return StatisticalSampler._stratified(data, config)
        if strategy == "bootstrap":
            return StatisticalSampler._bootstrap(data, config)

        raise ValueError(f"Unknown sampling strategy: {strategy!r}")

    @staticmethod
    def recommend_size(n_features: int, n_total: int) -> int:
        """Heuristic recommendation for adequate sample size.

        Returns ``min(n_total, max(500, 20 * n_features))``.
        """
        return min(n_total, max(500, 20 * n_features))

    @staticmethod
    def _random(data: torch.Tensor, config: SamplingConfig) -> Tuple[torch.Tensor, Dict]:
        N = data.shape[0]
        k = min(N, config.max_samples)
        gen = torch.Generator(device="cpu").manual_seed(config.seed)
        indices = torch.randperm(N, generator=gen)[:k]
        return data[indices], {
            "strategy": "random",
            "indices": indices,
            "n_original": N,
        }

    @staticmethod
    def _stratified(data: torch.Tensor, config: SamplingConfig) -> Tuple[torch.Tensor, Dict]:
        """Coherence-based stratified sampling.

        Embeds data as grade-1 in a default Euclidean algebra, computes
        per-point geodesic-flow coherence, then partitions into quantile
        strata.  This ensures sampling covers both structured (high
        coherence) and unstructured (low coherence) regions of the data
        without assuming a specific metric signature.
        """
        from core.config import make_algebra

        from .geodesic import GeodesicFlow

        N, D = data.shape
        k = min(N, config.max_samples)
        n_strata = config.n_strata or max(2, min(k // 20, 10))
        gen = torch.Generator(device="cpu").manual_seed(config.seed)

        # Cap algebra dimension for tractability
        alg_dim = min(D, 6)
        algebra = make_algebra(alg_dim, 0, device=data.device)

        # Embed into algebra (truncate to alg_dim if needed)
        raw = data[:, :alg_dim].float()
        mv = algebra.embed_vector(raw)  # [N, 2^alg_dim]

        # Compute per-point coherence as the stratification metric
        gf_k = min(8, N - 1)
        gf = GeodesicFlow(algebra, k=gf_k)
        with torch.no_grad():
            scores = gf.per_point_coherence(mv)  # [N]

        # Assign to quantile strata
        quantiles = torch.linspace(0, 1, n_strata + 1, device=data.device)
        thresholds = torch.quantile(scores.float(), quantiles)

        labels = torch.bucketize(scores, thresholds[1:-1]).clamp(max=n_strata - 1)

        # Proportional draw from each stratum
        all_indices: List[torch.Tensor] = []
        for c in range(n_strata):
            mask = (labels == c).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                continue
            n_from = max(1, int(round(k * len(mask) / N)))
            n_from = min(n_from, len(mask))
            perm = torch.randperm(len(mask), generator=gen)[:n_from]
            all_indices.append(mask[perm])

        indices = torch.cat(all_indices)
        if len(indices) > k:
            indices = indices[:k]

        return data[indices], {
            "strategy": "stratified",
            "indices": indices,
            "n_strata": n_strata,
            "n_original": N,
            "coherence_scores": scores,
        }

    @staticmethod
    def _bootstrap(data: torch.Tensor, config: SamplingConfig) -> Tuple[List[torch.Tensor], Dict]:
        N = data.shape[0]
        k = min(N, config.max_samples)
        gen = torch.Generator(device="cpu").manual_seed(config.seed)

        resamples: List[torch.Tensor] = []
        for _ in range(config.n_bootstrap):
            indices = torch.randint(0, N, (k,), generator=gen)
            resamples.append(data[indices])

        return resamples, {
            "strategy": "bootstrap",
            "n_bootstrap": config.n_bootstrap,
            "sample_size": k,
            "n_original": N,
        }
